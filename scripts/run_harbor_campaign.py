#!/usr/bin/env python3
"""Run frozen Harbor tasks with audited, bounded infrastructure retries.

Run this with Harbor's Python environment. Report imports use --report-python.
Original jobs are immutable. Retry jobs never overwrite or count as new samples.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


MAX_RETRIES = 2
NETWORK_MARKERS = (
    'timeout awaiting response headers', 'temporary failure in name resolution',
    'connection reset by peer', 'connection timed out', 'tls handshake timeout',
    'network is unreachable', 'econnreset', 'eai_again', 'etimedout',
    '503 service unavailable', '502 bad gateway', 'connection reset',
)
BLOCKED_TYPES = {
    'AgentAuthenticationError', 'AuthenticationError', 'ModelNotFoundError',
    'ApiUsageLimitError', 'PermissionDeniedError',
}
MODEL_FAILURE_TYPES = {'AgentTimeoutError', 'AgentSafetyRefusalError'}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def artifacts_available(record: Path) -> bool:
    manifest = record / 'artifacts/manifest.json'
    if not manifest.is_file() or manifest.is_symlink():
        return False
    entries = read_json(manifest)
    if not isinstance(entries, list):
        return False
    for source, destination in [('/app/output', 'artifacts/site'),
                                ('/logs/artifacts', 'artifacts/logs/artifacts')]:
        matches = [entry for entry in entries
                   if entry.get('source') == source and entry.get('destination') == destination
                   and entry.get('service') in {None, 'main'}]
        if not matches or matches[-1].get('status') not in {'ok', 'empty'}:
            return False
        entry = matches[-1]
        if entry.get('type') != 'directory' or entry.get('exclude') not in (None, []):
            return False
        path = record / destination
        if entry['status'] == 'ok' and (not path.is_dir() or path.is_symlink()):
            return False
    return True


def classify(result: dict, trial_dir: Path | None = None,
             expected_steps: list[str] | None = None) -> tuple[str, str]:
    """Choose an action from harness exception evidence, never the pass score."""
    nodes = [result, *(result.get('step_results') or [])]
    exceptions = [(node, node['exception_info']) for node in nodes if node.get('exception_info')]
    # Harbor wraps nested regrade failures; the step exceptions carry the cause.
    if len(exceptions) > 1:
        exceptions = [(node, e) for node, e in exceptions
                      if not (node is result and e['exception_type'] == 'RegradeError'
                              and e.get('exception_message', '').startswith('Fresh verification failed'))]
    sidecar_errors = []
    if trial_dir is not None:
        for path in [trial_dir / 'verifier/a11y-result.json',
                     *trial_dir.glob('steps/*/verifier/a11y-result.json')]:
            if path.is_file() and not path.is_symlink():
                data = read_json(path)
                if data.get('error') or (data.get('testFunctionResult') or {}).get('status') == 'error':
                    sidecar_errors.append(path)
    if sidecar_errors:
        return 'review', 'Verifier reported an error in its sidecar; inspect before classifying or retrying.'
    if not exceptions:
        if trial_dir is not None:
            records = ([trial_dir / 'steps' / name for name in expected_steps]
                       if expected_steps else [trial_dir])
            if any(not (record / 'verifier/a11y-result.json').is_file()
                   or not artifacts_available(record) for record in records):
                return 'review', 'Incomplete verification or artifact record; inspect before declaring the run complete.'
        return 'terminal', 'No harness exception; keep the evaluated outcome, including failures.'
    types = {e['exception_type'] for _, e in exceptions}
    if types & BLOCKED_TYPES:
        return 'blocked', 'Authentication, model access, or usage limit needs intervention.'
    if types & MODEL_FAILURE_TYPES and not (types - MODEL_FAILURE_TYPES):
        return 'terminal', 'Agent failure or time budget exhausted; no fresh attempt.'
    if types & MODEL_FAILURE_TYPES:
        return 'review', 'Model failure and separate harness exception; preserve failure and inspect infrastructure.'
    actions = []
    for node, error in exceptions:
        kind = error['exception_type']
        message = error.get('exception_message', '').lower()
        execution = node.get('agent_execution') or {}
        setup = not execution.get('started_at') and not result.get('step_results')
        docker = message.startswith('docker compose command failed')
        transient = any(marker in message for marker in NETWORK_MARKERS)
        known_setup = setup and (
            kind == 'AgentSetupTimeoutError'
            or (kind == 'NonZeroAgentExitCodeError'
                and ('missing optional dependency @openai/codex-' in message or transient))
        )
        provider_network = kind in {'APIConnectionError', 'APITimeoutError', 'InternalServerError'}
        if not (known_setup or (docker and transient) or provider_network):
            return 'review', f'Unclassified {kind}; requires inspection, no automatic retry.'
        try:
            during_verifier = (datetime.fromisoformat(error['occurred_at']) >=
                               datetime.fromisoformat(node['verifier']['started_at']))
        except (KeyError, TypeError, ValueError):
            during_verifier = False
        verifier_only = (
            execution.get('finished_at') is not None
            and during_verifier and node.get('verifier_result') is None and docker
        )
        actions.append('regrade' if verifier_only else 'retry')
    if 'terminal' in actions:
        return 'terminal', 'A model execution failure is present; do not retry the generation.'
    if all(action == 'regrade' for action in actions):
        if trial_dir is not None:
            recorded = [step['step_name'] for step in result.get('step_results') or []]
            records = ([trial_dir / 'steps' / name for name in expected_steps]
                       if expected_steps else [trial_dir])
            if (recorded != (expected_steps or [])
                    or any(not artifacts_available(record) for record in records)):
                return 'retry', 'Verifier infrastructure interrupted a partial task; restart the whole task.'
        return 'regrade', 'Confirmed verifier infrastructure failure; reuse recorded outputs.'
    return 'retry', 'Confirmed transient infrastructure/setup failure; restart the whole task.'


def collect_trials(job: Path, expected: set[str]) -> dict[str, tuple[Path, dict]]:
    trials = {}
    for path in sorted(job.glob('*/result.json')):
        result = read_json(path)
        name = result['task_name']
        if name in trials or name not in expected:
            raise ValueError(f'Duplicate or unexpected task in {job}: {name}')
        if not result.get('finished_at'):
            raise ValueError(f'Unfinished trial requires inspection: {path}')
        trials[name] = (path.parent, result)
    if set(trials) != expected:
        raise ValueError(f'Incomplete job {job}; missing {sorted(expected - set(trials))}')
    return trials


def materialize(selection: dict[str, tuple[Path, dict]], destination: Path) -> None:
    """Immutable local copy for import/regrade; preserve but never follow symlinks."""
    if destination.exists():
        raise ValueError(f'Selection already exists: {destination}')
    temporary = destination.with_name(destination.name + '.partial')
    temporary.mkdir(parents=True)
    for name, (source, _) in sorted(selection.items()):
        shutil.copytree(source, temporary / name, symlinks=True)
    temporary.rename(destination)


def verify_tasks(checksums: dict[str, str]) -> None:
    from harbor.models.task.task import Task
    for path, expected in checksums.items():
        if Task(Path(path)).checksum != expected:
            raise ValueError(f'Frozen task changed: {path}')


async def execute_job(config: dict, config_path: Path) -> Path:
    from harbor.job import Job
    from harbor.models.job.config import JobConfig
    job_path = Path(config['jobs_dir']) / config['job_name']
    if job_path.exists():
        # Harbor resume deletes incomplete trial artifacts. Never invoke it here.
        if not (job_path / 'result.json').exists() or not read_json(job_path / 'result.json').get('finished_at'):
            raise ValueError(f'Interrupted job retained for inspection: {job_path}')
        if read_json(config_path) != config:
            raise ValueError(f'Job configuration changed: {config_path}')
        return job_path
    write_json(config_path, config)
    job = await Job.create(JobConfig.model_validate(config))
    await job.run()
    return job_path


async def run_round(args, protocol: dict, model: str, number: int) -> dict:
    label = model.split('/')[-1]
    work = args.out / label / f'run-{number}'
    work.mkdir(parents=True, exist_ok=True)
    if (work / 'summary.json').exists():
        return read_json(work / 'summary.json')
    tasks = protocol['tasks']
    selected = {}
    ledger = []
    pending = {'generate': set(tasks)}
    for attempt in range(MAX_RETRIES + 1):
        next_pending = defaultdict(set)
        for mode, names in sorted(pending.items()):
            if not names:
                continue
            verify_tasks(protocol['checksums'])
            job_name = f'{protocol["id"]}-{label}-r{number}-a{attempt}-{mode}'
            config = {
                'job_name': job_name, 'jobs_dir': str(args.out / 'jobs'),
                'n_attempts': 1, 'n_concurrent_trials': 4, 'retry': {'max_retries': 0},
                'environment': {'type': 'docker'},
                'agents': [{'name': 'codex', 'model_name': model, 'resume_trajectory': True,
                            'kwargs': {'version': '0.160.1', 'reasoning_effort': 'medium'}}],
                'datasets': [], 'tasks': [{'path': tasks[name]} for name in sorted(names)],
            }
            if mode == 'regrade':
                source = work / f'regrade-source-{attempt}'
                if not source.exists():
                    materialize({name: selected[name] for name in names}, source)
                config['source_jobs'] = [{'action': 'regrade', 'type': 'local', 'path': str(source)}]
            write_json(args.out / 'status.json', {
                'status': 'RUNNING', 'model': model, 'run': number, 'attempt': attempt + 1,
                'mode': mode, 'tasks_in_job': len(names), 'job': str(args.out / 'jobs' / job_name),
                'updated_at': stamp(),
            })
            print(f'{stamp()} {model} run {number}/5 {mode} attempt {attempt + 1}: {len(names)} tasks', flush=True)
            job = await execute_job(config, work / f'{mode}-{attempt}.json')
            outcomes = collect_trials(job, names)
            selected.update(outcomes)
            for name, (path, result) in sorted(outcomes.items()):
                action, reason = classify(result, path, protocol['steps'].get(name, []))
                if mode == 'regrade' and action == 'retry':
                    action = 'regrade'
                ledger.append({'task': name, 'attempt': attempt + 1, 'mode': mode,
                               'source_trial': str(path), 'action': action, 'reason': reason,
                               'raw_usage_preserved_in': str(path / 'result.json')})
                if action in {'retry', 'regrade'} and attempt < MAX_RETRIES:
                    next_pending['generate' if action == 'retry' else 'regrade'].add(name)
            write_json(work / 'attempts.json', {'attempts': ledger})
            if any(classify(result, path, protocol['steps'].get(name, []))[0] == 'blocked'
                   for name, (path, result) in outcomes.items()):
                raise ValueError(f'Model access/authentication/usage failure in {job}; campaign stopped.')
        if not next_pending:
            break
        pending = next_pending
        await asyncio.sleep(30 * (attempt + 1))
    if set(selected) != set(tasks):
        raise ValueError('Selected outcomes do not cover the frozen task set')
    unresolved = [name for name, (path, result) in selected.items()
                  if classify(result, path, protocol['steps'].get(name, []))[0] != 'terminal']
    selection = work / 'selected'
    if not selection.exists():
        materialize(selected, selection)
    write_json(work / 'selection.json', {name: str(path) for name, (path, _) in selected.items()})
    report = work / f'{label}-run-{number}'
    if not (report / 'results.json').exists():
        subprocess.run([str(args.report_python), '-m', 'a11y_llm_tests.harbor', 'import',
                        str(selection), '--dataset', str(args.dataset), '--out', str(report), '--k', '1'],
                       check=True)
    payload = read_json(report / 'results.json')
    if len(payload['results']) != protocol['outputs_per_run']:
        raise ValueError('Imported report has unexpected output count')
    payload.setdefault('meta', {})['campaign'] = {
        'id': protocol['id'], 'policy': 'infra-retry-v1', 'complete': not unresolved,
        'unresolved_tasks': unresolved, 'attempts': len(ledger),
    }
    write_json(report / 'results.json', payload)
    summary = {'model': model, 'run': number, 'complete': not unresolved,
               'unresolved_tasks': unresolved, 'attempts': len(ledger), 'report': str(report),
               'finished_at': stamp()}
    write_json(work / 'summary.json', summary)
    return summary


async def campaign(args) -> None:
    from importlib.metadata import version
    if version('harbor') != '0.23.0':
        raise ValueError('This campaign requires Harbor 0.23.0')
    args.dataset = args.dataset.resolve()
    args.out = args.out.resolve()
    # Keep the venv symlink: resolving it would execute the base interpreter.
    args.report_python = args.report_python.absolute()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = read_json(args.dataset / 'manifest.json')
    checksums = read_json(args.dataset / 'reproducibility-task-checksums.json')
    tasks = {name: str(args.dataset / name) for name in manifest['tasks']}
    if set(tasks.values()) != set(checksums):
        raise ValueError('Manifest and baseline checksum task sets differ')
    protocol = {
        'id': args.out.name, 'models': args.models, 'runs_per_model': 5,
        'tasks': tasks, 'checksums': checksums,
        'steps': {name: [turn['id'] for turn in task.get('turns') or []]
                  for name, task in manifest['tasks'].items()},
        'manifest_sha256': hashlib.sha256((args.dataset / 'manifest.json').read_bytes()).hexdigest(),
        'outputs_per_run': sum(max(1, len(t.get('turns') or [])) for t in manifest['tasks'].values()),
        'harbor_version': '0.23.0', 'codex_version': '0.160.1', 'reasoning_effort': 'medium',
        'max_infrastructure_retries': MAX_RETRIES, 'retry_policy': 'infra-retry-v1',
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'comparison_sha256': hashlib.sha256(Path('scripts/compare_harbor_runs.py').read_bytes()).hexdigest(),
        'aggregation': 'Arithmetic mean of five complete run pass rates, separately per model and variant/turn.',
    }
    protocol_path = args.out / 'protocol.json'
    if protocol_path.exists() and read_json(protocol_path) != protocol:
        raise ValueError('Campaign protocol changed; cannot resume')
    write_json(protocol_path, protocol)
    verify_tasks(checksums)
    os.environ['CODEX_FORCE_AUTH_JSON'] = '1'
    summaries = []
    for number in range(1, 6):
        # Alternate model order to spread temporal effects across models.
        models = args.models if number % 2 else list(reversed(args.models))
        for model in models:
            summaries.append(await run_round(args, protocol, model, number))
    for model in args.models:
        runs = sorted((row for row in summaries if row['model'] == model), key=lambda r: r['run'])
        if all(row['complete'] for row in runs):
            comparison = args.out / model.split('/')[-1] / 'comparison'
            if not comparison.exists():
                subprocess.run([str(args.report_python), 'scripts/compare_harbor_runs.py',
                                *(row['report'] for row in runs), '--out', str(comparison)], check=True)
    write_json(args.out / 'status.json', {
        'status': 'FINISHED' if all(row['complete'] for row in summaries) else 'INCOMPLETE',
        'runs': summaries, 'updated_at': stamp(),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('harbor-tasks'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--report-python', type=Path, default=Path('.venv/bin/python'))
    parser.add_argument('--models', nargs='+', default=['openai/gpt-6-luna', 'openai/gpt-6.1-sol'])
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        parser.error('Models must be distinct')
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            asyncio.run(campaign(args))
        except Exception as exc:
            write_json(args.out / 'status.json', {'status': 'STOPPED', 'reason': str(exc), 'updated_at': stamp()})
            raise


if __name__ == '__main__':
    main()
