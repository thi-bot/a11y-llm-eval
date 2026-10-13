"""Import Harbor trials into the existing accessibility report format.

Multi-step usage is retained in Harbor sidecars but is not assigned to report
turns: some agent adapters report cumulative session usage and others report
per-invocation usage. Harbor costs may be estimates, including for subscriptions.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from pydantic import ValidationError

from .metrics import compute_pass_at_k, format_pass_at_k
from .schema import AggregateRecord, AxeResult, GenerationMeta, ResultRecord, TestFunctionResult


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _redact(value: Any) -> Any:
    """Keep configuration provenance without publishing authentication values."""
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, dict):
        return {
            key: "[redacted]" if (
                key.lower() == "env"
                or any(word in key.lower() for word in ("api_key", "token", "secret", "password", "credential"))
                and not key.lower().startswith(("n_", "total_", "prompt_", "completion_", "cached_", "max_"))
                and not isinstance(item, (int, float, type(None)))
            ) else _redact(item)
            for key, item in value.items()
        }
    return value


def _safe_copy_tree(source: Path, destination: Path, boundary: Path) -> None:
    """Copy regular artifact files only; never follow generated symlinks."""
    if source.is_symlink() or not source.is_dir():
        return
    if not source.resolve().is_relative_to(boundary.resolve()):
        raise ValueError(f"Artifact path escapes trial: {source}")
    for child in sorted(source.iterdir()):
        if child.is_symlink():
            continue
        target = destination / child.name
        if child.is_dir():
            _safe_copy_tree(child, target, boundary)
        elif child.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(child, target)


def _safe_read(path: Path, boundary: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    if path.is_symlink() or not path.resolve().is_relative_to(boundary.resolve()):
        raise ValueError(f"Input path escapes trial: {path}")
    return _read_object(path)


def _agent_identity(trial: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    config = (trial.get("config") or {}).get("agent") or {}
    info = trial.get("agent_info") or {}
    name = info.get("name") or config.get("name") or config.get("import_path") or "unknown-agent"
    model = config.get("model_name") or (info.get("model_info") or {}).get("name") or "default-model"
    version = info.get("version") or (config.get("kwargs") or {}).get("version") or "unknown-version"
    # Different settings are different experimental conditions. Authentication
    # and concurrency settings do not change that condition.
    condition = {
        key: value for key, value in config.items()
        if key not in {"env", "n_concurrent", "concurrency_group", "include_logs", "exclude_logs"}
    }
    condition.update({"name": name, "model_name": model, "version": version})
    fingerprint = hashlib.sha256(json.dumps(condition, sort_keys=True).encode()).hexdigest()[:12]
    label = f"{name}/{model}@{version} [{fingerprint}]"
    return label, {"name": label, "display_name": label, "agent": _redact(condition)}


def _duration(timing: dict[str, Any] | None) -> float:
    if not timing or not timing.get("started_at") or not timing.get("finished_at"):
        return 0.0
    start = datetime.fromisoformat(timing["started_at"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(timing["finished_at"].replace("Z", "+00:00"))
    return max(0.0, (end - start).total_seconds())


def _conversation(trajectory: dict[str, Any]) -> dict[str, Any]:
    messages = []
    for step in trajectory.get("steps") or []:
        if not isinstance(step, dict):
            continue
        calls = [{"tool_name": call.get("function_name"), "arguments": call.get("arguments")}
                 for call in step.get("tool_calls") or [] if isinstance(call, dict)]
        messages.append({"role": step.get("source") or "assistant", "content": step.get("message"), "tool_calls": calls})
    return {"format": "harbor-atif-preview/v1", "messages": messages}


def _evaluation(node: dict[str, Any] | None, error: str | None) -> tuple[TestFunctionResult, AxeResult | None, str]:
    if node is None:
        return TestFunctionResult(status="error", error=error or "Missing verifier a11y-result.json"), None, "ERROR"
    if node.get("error"):
        error = error or str(node["error"])
    if (node.get("renderEvaluation") or {}).get("rendered") is False:
        error = error or "Browser did not render the generated submission"
    raw = node.get("testFunctionResult") or {}
    assertions = []
    for entry in raw.get("assertions") or []:
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status") or "fail").lower()
        if status in {"n/a", "not_applicable", "not-applicable", "not applicable"}:
            status = "na"
        assertions.append({"name": entry.get("name") or "unknown", "status": status if status in {"pass", "fail", "na"} else "fail", "type": entry.get("type") or "R", "message": entry.get("message")})
    test = TestFunctionResult(**{**raw, "assertions": assertions, "status": raw.get("status") or "error"})
    for kind, suffix in (("R", ""), ("BP", "_bp")):
        for state, count_suffix in (("fail", "failures"), ("na", "na")):
            field = f"total_assertion{suffix}_{count_suffix}"
            setattr(test, field, sum(a.type == kind and a.status == state for a in test.assertions))
    axe_data = node.get("axeResult") or node.get("axe_result") or node.get("axe")
    axe = AxeResult.model_validate(axe_data) if axe_data else None
    if error or test.status not in {"pass", "fail"} or axe is None:
        test.status = "error"
        test.error = error or test.error or "Verifier returned incomplete accessibility results"
        return test, axe, "ERROR"
    passed = test.status == "pass" and test.total_assertion_failures == 0 and axe.failure_count == 0
    return test, axe, "PASS" if passed else "FAIL"


def import_jobs(
    job_dirs: list[Path], dataset_dir: Path, out: Path, ks: list[int], generate_report: bool = True,
) -> Path:
    """Import all recorded trials, including interrupted and missing skill turns.

    Returns the output directory. The destination must be absent or empty.
    Authentication-bearing configuration fields are redacted in copied results.
    """
    dataset_dir, out = Path(dataset_dir).resolve(), Path(out).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"Import destination must be empty: {out}")
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("k values must be positive integers")
    manifest = _read_object(dataset_dir / "manifest.json")
    if manifest.get("format") != "a11y-harbor/v1":
        raise ValueError("Unsupported or missing Harbor dataset manifest format")
    tasks = manifest.get("tasks") or {}
    trials: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    seen: set[Path] = set()
    for job in job_dirs:
        job = Path(job).resolve()
        if not job.is_dir():
            raise ValueError(f"Harbor job directory does not exist: {job}")
        for directory in sorted(job.iterdir()):
            if directory.is_symlink() or not directory.is_dir() or directory.resolve() in seen:
                continue
            trial = _safe_read(directory / "result.json", directory)
            if trial is None:
                config = _safe_read(directory / "config.json", directory)
                if config is None or "task" not in config:
                    continue
                trial = {"config": config, "exception_info": {"exception_message": "Harbor trial has no result.json (interrupted or unfinished)"}}
            if "config" not in trial:
                continue
            task_path = str(((trial.get("config") or {}).get("task") or {}).get("path") or "")
            task_name = Path(task_path).name if task_path else trial.get("task_name")
            task = tasks.get(task_name) or tasks.get(trial.get("task_name"))
            if not isinstance(task, dict):
                raise ValueError(f"Trial {directory.name} does not match this dataset: {task_name}")
            if task.get("variant_kind") == "skill" and not task.get("turns"):
                raise ValueError(f"Skill task has no turns: {task_name}")
            for turn in task.get("turns") or []:
                name = turn["id"]
                if not isinstance(name, str) or name in {".", ".."} or Path(name).name != name or "/" in name or "\\" in name:
                    raise ValueError(f"Unsafe skill turn id: {name!r}")
            trials.append((directory, trial, task))
            seen.add(directory.resolve())
    if not trials:
        raise ValueError("No Harbor trials found in the supplied job directories")
    out.mkdir(parents=True, exist_ok=True)
    prompt_variants = []
    for index, original in enumerate(manifest.get("prompt_variants") or []):
        variant = dict(original)
        instructions = variant.get("custom_instructions")
        source_path = variant.pop("custom_instructions_path", None)
        if not instructions and source_path:
            source = Path(source_path)
            source = source if source.is_absolute() else dataset_dir / source
            if source.resolve().is_relative_to(dataset_dir) and source.is_file():
                instructions = source.read_text(encoding="utf-8")
        if instructions:
            local_instructions = out / "instructions" / f"variant-{index}.md"
            local_instructions.parent.mkdir(parents=True, exist_ok=True)
            local_instructions.write_text(instructions, encoding="utf-8")
            variant["custom_instructions_path"] = str(local_instructions)
        if variant.get("skill_path"):
            source = Path(variant["skill_path"])
            source = source if source.is_absolute() else dataset_dir / source
            local_skill = out / "skills" / f"variant-{index}"
            _safe_copy_tree(source, local_skill, dataset_dir)
            variant["skill_path"] = str(local_skill)
        prompt_variants.append(variant)
    now = datetime.now(timezone.utc)
    results = []
    models: dict[str, dict[str, Any]] = {}
    sample_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for trial_index, (directory, trial, task) in enumerate(trials):
        model, model_info = _agent_identity(trial)
        models[model] = model_info
        variant = task.get("variant_id") or "control"
        kind = task.get("variant_kind") or "control"
        case = task["prompt_case_id"]
        sample_key = (case, variant, model)
        sample_index = sample_counts[sample_key]
        sample_counts[sample_key] += 1
        local_trial = out / "harbor" / f"trial-{trial_index:06d}"
        _write_json(local_trial / "result.json", _redact(trial))
        turns = task.get("turns") or []
        steps = {step["step_name"]: step for step in trial.get("step_results") or []}
        expected = turns if kind == "skill" else [None]
        for turn_index, turn in enumerate(expected):
            source = directory / "steps" / turn["id"] if turn else directory
            target = local_trial / "steps" / turn["id"] if turn else local_trial
            context = steps.get(turn["id"]) if turn else trial
            error = None
            if context is None:
                context = {}
                error = f"Harbor did not execute skill step {turn['id']}"
            # A failure in a later skill step must not invalidate a completed
            # earlier step. The trial-level exception is already reflected in
            # missing steps, or applies directly to a single-step trial.
            exception = context.get("exception_info")
            if exception:
                error = exception.get("exception_message") or exception.get("exception_type") or "Harbor trial failed"
            _safe_copy_tree(source / "artifacts" / "site", target / "site", directory)
            html = target / "site" / "index.html"
            if not html.is_file():
                error = error or "Missing generated site/index.html artifact"
                html.parent.mkdir(parents=True, exist_ok=True)
                html.write_text("", encoding="utf-8")
            try:
                node = _safe_read(source / "verifier" / "a11y-result.json", directory)
            except (json.JSONDecodeError, UnicodeError) as exc:
                node = None
                error = error or f"Invalid verifier JSON: {exc}"
            if node is not None:
                _write_json(target / "a11y-result.json", node)
            screenshot = source / "verifier" / "screenshot.png"
            local_screenshot = None
            if screenshot.is_file() and not screenshot.is_symlink() and screenshot.resolve().is_relative_to(directory.resolve()):
                local_screenshot = target / "screenshot.png"
                shutil.copyfile(screenshot, local_screenshot)
            try:
                trajectory = _safe_read(source / "agent" / "trajectory.json", directory)
            except (json.JSONDecodeError, UnicodeError):
                trajectory = None
            conversation_path = None
            if trajectory:
                _write_json(target / "trajectory.json", trajectory)
                conversation_path = target / "conversation.json"
                _write_json(conversation_path, _conversation(trajectory))
            try:
                test_result, axe, status = _evaluation(node, error)
            except (ValidationError, TypeError, AttributeError) as exc:
                test_result, axe, status = _evaluation(None, f"Invalid verifier result: {exc}")
            usage = (context.get("agent_result") or {}) if not turn else {}
            tokens_in, tokens_out = usage.get("n_input_tokens"), usage.get("n_output_tokens")
            generation = GenerationMeta(
                latency_s=_duration(context.get("agent_execution")),
                prompt_hash=hashlib.sha256(task.get("prompt_text", "").encode()).hexdigest(), cached=False,
                tokens_in=tokens_in, tokens_out=tokens_out,
                total_tokens=tokens_in + tokens_out if tokens_in is not None and tokens_out is not None else None,
                cost_usd=usage.get("cost_usd"), generation_mode="harbor", agent_sandbox="harbor",
                agent_limit_error=error, browser_smoke=(node or {}).get("renderEvaluation"),
            )
            record = ResultRecord(
                test_name=task["test_name"], base_test_name=task["base_test_name"], prompt_case_id=case,
                prompt_dimensions=task.get("prompt_dimensions") or [], model_name=model,
                timestamp=trial.get("started_at") or now, generation_html_path=str(html),
                generation_conversation_path=str(conversation_path) if conversation_path else None,
                generation_eval_path=str(target / "a11y-result.json") if node is not None else str(local_trial / "result.json"),
                screenshot_path=str(local_screenshot) if local_screenshot else None,
                test_function=test_result, axe=axe, result=status, generation=generation,
                sample_index=sample_index, prompt_variant_id=variant, prompt_variant_kind=kind,
                turn_id=turn["id"] if turn else None, turn_index=turn_index if turn else None,
                turn_count_total=len(turns) if turn else None,
            )
            result = record.model_dump(mode="json")
            result["harbor"] = {"trial_result_path": str(local_trial / "result.json"), "source_trial": str(directory), "usage_scope": "unclassified_multi_step" if turn else "trial"}
            results.append(result)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        groups[(result["prompt_case_id"], result["model_name"], result["prompt_variant_id"], result["turn_id"])].append(result)
    aggregates = []
    for records in groups.values():
        first = records[0]
        passed = sum(record["result"] == "PASS" for record in records)
        fields = {key: first[key] for key in ("test_name", "base_test_name", "prompt_case_id", "prompt_dimensions", "model_name", "prompt_variant_id", "prompt_variant_kind", "turn_id", "turn_index")}
        applicable_ks = [k for k in ks if k <= len(records)]
        aggregates.append(AggregateRecord(**fields, n_samples=len(records), n_applicable=len(records), n_pass=passed, pass_at_k=format_pass_at_k(compute_pass_at_k(passed, len(records), applicable_ks)), k_values=applicable_ks, computed_at=now).model_dump(mode="json"))
    payload = {
        "run_id": out.name, "models": list(models), "tests": sorted({r["test_name"] for r in results}),
        "prompts": manifest.get("prompts") or {}, "results": results, "aggregates": aggregates,
        "meta": {"status": "EVALUATED", "sampling": {"k_values": ks},
                 "prompt_variants": prompt_variants, "prompt_cases": manifest.get("prompt_cases") or [],
                 "models_info": list(models.values()),
                 "runtime": {"engine": "harbor", "dataset_manifest": str(dataset_dir / "manifest.json"),
                             "job_dirs": [str(Path(job).resolve()) for job in job_dirs],
                             "usage_note": "Multi-step usage is preserved only in Harbor result sidecars because agent adapters differ in cumulative versus per-step accounting. Costs may be estimates, not billed subscription charges."}},
    }
    _write_json(out / "results.json", payload)
    if generate_report:
        from .report import render_report
        render_report(out / "results.json", out / "index.html", {"models": list(models.values())})
    return out
