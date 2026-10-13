import json
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_harbor_campaign import artifacts_available, classify, collect_trials, materialize
from scripts import run_harbor_campaign as runner


def trial(kind=None, message='', *, phase='agent'):
    result = {'task_name': 'task', 'finished_at': '2026-10-10T12:00:00Z'}
    if phase in {'agent', 'verifier'}:
        result['agent_execution'] = {'started_at': '2026-10-10T10:00:00Z', 'finished_at': '2026-10-10T10:01:00Z'}
    if phase == 'verifier':
        result['verifier'] = {'started_at': '2026-10-10T10:02:00Z'}
    if kind:
        result['exception_info'] = {'exception_type': kind, 'exception_message': message,
                                    'occurred_at': '2026-10-10T10:03:00Z'}
    return result


def artifacts(path):
    (path / 'artifacts/site').mkdir(parents=True)
    (path / 'artifacts/site/index.html').write_text('output')
    (path / 'artifacts/manifest.json').write_text(json.dumps([
        {'source': '/app/output', 'destination': 'artifacts/site', 'type': 'directory', 'status': 'ok'},
        {'source': '/logs/artifacts', 'destination': 'artifacts/logs/artifacts', 'type': 'directory', 'status': 'empty'},
    ]))


@pytest.mark.parametrize('kind,message,phase,action', [
    (None, '', 'agent', 'terminal'),
    ('AgentTimeoutError', 'timed out', 'agent', 'terminal'),
    ('AgentSetupTimeoutError', 'setup timed out', 'setup', 'retry'),
    ('RuntimeError', 'Docker compose command failed: timeout awaiting response headers', 'setup', 'retry'),
    ('RuntimeError', 'Docker compose command failed: timeout awaiting response headers', 'verifier', 'regrade'),
    ('RuntimeError', 'Docker compose command failed: syntax error', 'setup', 'review'),
    ('NonZeroAgentExitCodeError', 'Missing optional dependency @openai/codex-linux-x64', 'setup', 'retry'),
    ('NonZeroAgentExitCodeError', 'exit 1', 'agent', 'review'),
    ('VerifierTimeoutError', 'timed out', 'verifier', 'review'),
    ('AgentAuthenticationError', 'expired', 'agent', 'blocked'),
    ('ApiUsageLimitError', 'quota', 'agent', 'blocked'),
    ('ModelNotFoundError', 'unknown model', 'agent', 'blocked'),
    ('APIConnectionError', 'connection reset', 'agent', 'retry'),
])
def test_classification(kind, message, phase, action):
    assert classify(trial(kind, message, phase=phase))[0] == action


def test_skill_step_exception_is_seen_and_model_timeout_is_not_retried():
    data = trial()
    data['step_results'] = [trial('APIConnectionError'), trial('AgentTimeoutError')]
    assert classify(data)[0] == 'review'
    data['step_results'] = [trial('APIConnectionError')]
    assert classify(data)[0] == 'retry'


def test_score_never_triggers_retry():
    data = trial()
    data['verifier_result'] = {'rewards': {'reward': 0}}
    assert classify(data)[0] == 'terminal'


def test_selection_preserves_original_and_counts_only_selected_attempt(tmp_path):
    source = tmp_path / 'job' / 'attempt'
    source.mkdir(parents=True)
    (source / 'result.json').write_text(json.dumps(trial()))
    (source / 'agent.log').write_text('original evidence')
    selection = collect_trials(source.parent, {'task'})
    target = tmp_path / 'selected'
    materialize(selection, target)
    assert len(list(target.iterdir())) == 1
    assert (source / 'agent.log').read_text() == 'original evidence'
    assert (target / 'task' / 'agent.log').read_text() == 'original evidence'
    with pytest.raises(ValueError, match='already exists'):
        materialize(selection, target)


def test_missing_and_duplicate_outcomes_are_rejected(tmp_path):
    job = tmp_path / 'job'
    job.mkdir()
    with pytest.raises(ValueError, match='Incomplete'):
        collect_trials(job, {'task'})
    for name in ('one', 'two'):
        p = job / name
        p.mkdir()
        (p / 'result.json').write_text(json.dumps(trial()))
    with pytest.raises(ValueError, match='Duplicate'):
        collect_trials(job, {'task'})


@pytest.mark.parametrize('outcomes,expected_attempts,complete', [
    (['retry', 'pass'], 2, True),
    (['retry', 'fail'], 2, True),
    (['retry', 'retry', 'retry'], 3, False),
    (['regrade', 'pass'], 2, True),
])
def test_round_bounds_retries_and_selects_first_evaluated_result(tmp_path, monkeypatch,
                                                               outcomes, expected_attempts, complete):
    args = SimpleNamespace(out=tmp_path, dataset=tmp_path / 'dataset', report_python=Path('/python'))
    protocol = {'id': 'test', 'tasks': {'task': '/frozen/task'}, 'checksums': {}, 'steps': {}, 'outputs_per_run': 1}
    jobs = []

    async def execute(config, config_path):
        mode = outcomes[len(jobs)]
        jobs.append(config)
        path = Path(config['jobs_dir']) / config['job_name']
        p = path / 'trial'
        p.mkdir(parents=True)
        if mode == 'retry':
            data = trial('AgentSetupTimeoutError', phase='setup')
        elif mode == 'regrade':
            data = trial('RuntimeError', 'Docker compose command failed: timeout awaiting response headers', phase='verifier')
            artifacts(p)
        else:
            data = trial()
            data['verifier_result'] = {'rewards': {'reward': int(mode == 'pass')}}
            artifacts(p)
            (p / 'verifier').mkdir()
            (p / 'verifier/a11y-result.json').write_text(json.dumps({'testFunctionResult': {'status': mode}}))
        (p / 'result.json').write_text(json.dumps(data))
        return path

    def import_report(command, check):
        destination = Path(command[command.index('--out') + 1])
        destination.mkdir()
        (destination / 'results.json').write_text(json.dumps({'results': [{}]}))

    async def no_wait(_):
        pass

    monkeypatch.setattr(runner, 'verify_tasks', lambda _: None)
    monkeypatch.setattr(runner, 'execute_job', execute)
    monkeypatch.setattr(runner.subprocess, 'run', import_report)
    monkeypatch.setattr(runner.asyncio, 'sleep', no_wait)
    summary = asyncio.run(runner.run_round(args, protocol, 'openai/model', 1))
    assert len(jobs) == expected_attempts
    assert summary['attempts'] == expected_attempts
    assert summary['complete'] is complete
    assert all(job['retry']['max_retries'] == 0 for job in jobs)
    selected = list((tmp_path / 'model/run-1/selected').glob('*/result.json'))
    assert len(selected) == 1
    assert len(list((tmp_path / 'jobs').glob('*/*/result.json'))) == expected_attempts
    if outcomes[0] == 'regrade':
        assert jobs[1]['source_jobs'][0]['action'] == 'regrade'
    if outcomes[-1] == 'fail':
        assert json.loads(selected[0].read_text())['verifier_result']['rewards']['reward'] == 0
    # Re-entering a completed or exhausted round cannot reset its retry budget.
    assert asyncio.run(runner.run_round(args, protocol, 'openai/model', 1)) == summary
    assert len(jobs) == expected_attempts


def test_nested_regrade_error_does_not_hide_transient_failure():
    data = trial('RegradeError', 'Fresh verification failed for step review')
    data['step_results'] = [trial('RuntimeError', 'Docker compose command failed: timeout awaiting response headers', phase='verifier')]
    assert classify(data)[0] == 'regrade'


def test_partial_skill_restarts_whole_task(tmp_path):
    data = trial()
    step = trial('RuntimeError', 'Docker compose command failed: timeout awaiting response headers', phase='verifier')
    step['step_name'] = 'generate'
    data['step_results'] = [step]
    assert classify(data, tmp_path, ['generate', 'review'])[0] == 'retry'


def test_verifier_sidecar_error_prevents_complete_run(tmp_path):
    path = tmp_path / 'verifier/a11y-result.json'
    path.parent.mkdir()
    path.write_text(json.dumps({'error': 'browser launch failed', 'testFunctionResult': {'status': 'error'}}))
    assert classify(trial(), tmp_path)[0] == 'review'


@pytest.mark.parametrize('status', ['failed', 'skipped'])
def test_failed_artifact_capture_is_not_a_completed_evaluation(tmp_path, status):
    artifacts(tmp_path)
    manifest = tmp_path / 'artifacts/manifest.json'
    data = json.loads(manifest.read_text())
    data[0]['status'] = status
    manifest.write_text(json.dumps(data))
    assert not artifacts_available(tmp_path)
    assert classify(trial(), tmp_path)[0] == 'review'


def test_legitimate_empty_output_remains_model_failure(tmp_path):
    artifacts(tmp_path)
    manifest = tmp_path / 'artifacts/manifest.json'
    data = json.loads(manifest.read_text())
    data[0]['status'] = 'empty'
    manifest.write_text(json.dumps(data))
    (tmp_path / 'verifier').mkdir()
    (tmp_path / 'verifier/a11y-result.json').write_text(json.dumps({'error': None, 'testFunctionResult': {'status': 'fail', 'error': 'Missing output'}}))
    assert artifacts_available(tmp_path)
    assert classify(trial(), tmp_path)[0] == 'terminal'


def test_agent_transport_failure_followed_by_verification_restarts_agent():
    data = trial('RuntimeError', 'Docker compose command failed: connection reset by peer', phase='verifier')
    data['exception_info']['occurred_at'] = '2026-10-10T10:00:30Z'
    data['verifier_result'] = {'rewards': {'reward': 0}}
    assert classify(data)[0] == 'retry'
