from __future__ import annotations

import json
from pathlib import Path

import pytest

from a11y_llm_tests.harbor_import import import_jobs
from a11y_llm_tests.schema import AggregateRecord, ResultRecord


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    path = tmp_path / "dataset"
    control = {
        "prompt_case_id": "single-checkbox", "test_name": "single-checkbox", "base_test_name": "single-checkbox",
        "prompt_dimensions": [], "prompt_text": "Build a checkbox.", "variant_id": "control", "variant_kind": "control", "turns": [],
    }
    skill = {**control, "variant_id": "review", "variant_kind": "skill", "turns": [{"id": "generate", "name": "Generate", "prompt": "Generate"}, {"id": "review", "name": "Review", "prompt": "Review"}]}
    write_json(path / "manifest.json", {
        "format": "a11y-harbor/v1", "tasks": {"checkbox-control": control, "checkbox-review": skill},
        "prompt_variants": [{"id": "control", "kind": "control"}, {"id": "review", "kind": "skill", "name": "Review", "turns": skill["turns"]}],
        "prompt_cases": [{"id": "single-checkbox", "test_name": "single-checkbox", "base_test_name": "single-checkbox"}],
        "prompts": {"single-checkbox": "Build a checkbox."},
    })
    return path


def make_trial(job: Path, name: str, *, task: str = "checkbox-control", agent: str = "codex", version: str = "1.0", kwargs: dict | None = None, steps: list | None = None) -> Path:
    path = job / name
    write_json(path / "result.json", {
        "task_name": task, "config": {"task": {"path": f"/export/{task}"}, "agent": {"name": agent, "model_name": "openai/gpt-test", "kwargs": kwargs or {}, "env": {"SECRET": "never-publish-this"}}},
        "agent_info": {"name": agent, "version": version}, "started_at": "2026-10-06T10:00:00Z",
        "agent_execution": {"started_at": "2026-10-06T10:00:00Z", "finished_at": "2026-10-06T10:00:02Z"},
        "agent_result": {"n_input_tokens": 100, "n_cache_tokens": 40, "n_output_tokens": 20, "cost_usd": 0.01},
        "step_results": steps,
    })
    return path


def make_output(path: Path, *, passed: bool = True) -> None:
    site = path / "artifacts" / "site"
    site.mkdir(parents=True, exist_ok=True)
    (site / "index.html").write_text('<!doctype html><html lang="en"><title>Checkbox</title><link rel="stylesheet" href="style.css"><input aria-label="Agree" type="checkbox"></html>')
    (site / "style.css").write_text("body {color: black}")
    write_json(path / "verifier" / "a11y-result.json", {
        "testFunctionResult": {"status": "pass" if passed else "fail", "assertions": [{"name": "Checkbox labelled", "status": "pass" if passed else "fail"}]},
        "axeResult": {"failure_count": 0, "failures": []},
    })
    (path / "verifier" / "screenshot.png").write_bytes(b"fake screenshot")
    write_json(path / "agent" / "trajectory.json", {"schema_version": "ATIF-v1.8", "steps": [{"source": "user", "message": "Build a checkbox."}, {"source": "assistant", "message": "Created the checkbox", "tool_calls": [{"function_name": "shell", "arguments": {"command": "pwd"}}]}]})


def load_results(out: Path) -> dict:
    payload = json.loads((out / "results.json").read_text())
    for record in payload["results"]:
        ResultRecord.model_validate(record)
    for aggregate in payload["aggregates"]:
        AggregateRecord.model_validate(aggregate)
    return payload


def test_import_single_trial_and_render_report(dataset: Path, tmp_path: Path) -> None:
    job = tmp_path / "job"
    trial = make_trial(job, "trial-1")
    make_output(trial)
    out = import_jobs([job], dataset, tmp_path / "report", [1, 5])
    payload = load_results(out)
    record = payload["results"][0]
    assert record["result"] == "PASS"
    assert record["generation"]["total_tokens"] == 120  # Cache tokens already included.
    assert record["generation"]["latency_s"] == 2
    assert Path(record["generation_html_path"]).with_name("style.css").is_file()
    assert Path(record["screenshot_path"]).is_file()
    conversation = json.loads(Path(record["generation_conversation_path"]).read_text())
    assert conversation["messages"][1]["tool_calls"][0]["tool_name"] == "shell"
    assert payload["aggregates"][0]["pass_at_k"]["1"] == 1
    assert "5" not in payload["aggregates"][0]["pass_at_k"]
    assert (out / "index.html").is_file()
    assert "never-publish-this" not in (out / "harbor" / "trial-000000" / "result.json").read_text()


def test_missing_skill_step_is_error_and_usage_is_not_double_counted(dataset: Path, tmp_path: Path) -> None:
    job = tmp_path / "job"
    trial = make_trial(job, "skill", task="checkbox-review", steps=[{"step_name": "generate", "agent_result": {"n_input_tokens": 100, "n_output_tokens": 20}}])
    make_output(trial / "steps" / "generate", passed=False)
    out = import_jobs([job], dataset, tmp_path / "report", [1], generate_report=True)
    payload = load_results(out)
    assert [r["result"] for r in payload["results"]] == ["FAIL", "ERROR"]
    assert [r["turn_id"] for r in payload["results"]] == ["generate", "review"]
    assert all(r["generation"]["tokens_in"] is None for r in payload["results"])
    assert all(r["sample_index"] == 0 for r in payload["results"])
    assert len(payload["aggregates"]) == 2
    assert "did not execute" in payload["results"][1]["test_function"]["error"]


def test_agents_versions_kwargs_and_samples_stay_separate(dataset: Path, tmp_path: Path) -> None:
    job = tmp_path / "job"
    for index, config in enumerate([{}, {}, {"agent": "claude-code"}, {"version": "2.0"}, {"kwargs": {"reasoning_effort": "high"}}]):
        trial = make_trial(job, f"trial-{index}", **config)
        make_output(trial, passed=index != 0)
    out = import_jobs([job, job], dataset, tmp_path / "report", [1], generate_report=False)
    payload = load_results(out)
    assert len(payload["models"]) == 4
    assert len(payload["results"]) == 5
    assert [r["sample_index"] for r in payload["results"][:2]] == [0, 1]
    assert payload["aggregates"][0]["n_samples"] == 2
    assert payload["aggregates"][0]["pass_at_k"]["1"] == 0.5


def test_interrupted_trial_is_retained(dataset: Path, tmp_path: Path) -> None:
    job = tmp_path / "job"
    write_json(job / "interrupted" / "config.json", {"task": {"path": "/export/checkbox-control"}, "agent": {"name": "codex", "model_name": "openai/gpt-test"}})
    out = import_jobs([job], dataset, tmp_path / "report", [1], generate_report=False)
    record = load_results(out)["results"][0]
    assert record["result"] == "ERROR"
    assert "interrupted" in record["test_function"]["error"]


def test_unknown_tasks_and_nonempty_output_are_rejected(dataset: Path, tmp_path: Path) -> None:
    job = tmp_path / "job"
    make_trial(job, "unknown", task="different-dataset")
    with pytest.raises(ValueError, match="does not match"):
        import_jobs([job], dataset, tmp_path / "report", [1])
    assert not (tmp_path / "report").exists()
    out = tmp_path / "occupied"
    out.mkdir()
    (out / "keep.txt").write_text("retain")
    with pytest.raises(ValueError, match="must be empty"):
        import_jobs([job], dataset, out, [1])
    assert (out / "keep.txt").read_text() == "retain"


def test_generated_symlinks_are_not_followed(dataset: Path, tmp_path: Path) -> None:
    job = tmp_path / "job"
    trial = make_trial(job, "trial")
    make_output(trial)
    secret = tmp_path / "private.txt"
    secret.write_text("private")
    try:
        (trial / "artifacts" / "site" / "outside.txt").symlink_to(secret)
    except OSError:
        pytest.skip("Symlinks unavailable on this platform")
    out = import_jobs([job], dataset, tmp_path / "report", [1], generate_report=False)
    html = Path(load_results(out)["results"][0]["generation_html_path"])
    assert not html.with_name("outside.txt").exists()


@pytest.mark.parametrize("extra", [{"error": "Browser failed"}, {"renderEvaluation": {"rendered": False}}])
def test_browser_errors_cannot_be_imported_as_passes(dataset: Path, tmp_path: Path, extra: dict) -> None:
    job = tmp_path / "job"
    trial = make_trial(job, "trial")
    make_output(trial)
    evaluation = trial / "verifier" / "a11y-result.json"
    write_json(evaluation, {**json.loads(evaluation.read_text()), **extra})
    out = import_jobs([job], dataset, tmp_path / "report", [1], generate_report=False)
    assert load_results(out)["results"][0]["result"] == "ERROR"


def test_corrupt_verifier_json_is_error(dataset: Path, tmp_path: Path) -> None:
    job = tmp_path / "job"
    trial = make_trial(job, "trial")
    make_output(trial)
    (trial / "verifier" / "a11y-result.json").write_text('{"partial":')
    out = import_jobs([job], dataset, tmp_path / "report", [1], generate_report=False)
    assert load_results(out)["results"][0]["result"] == "ERROR"


def test_instruction_text_is_copied_into_report(dataset: Path, tmp_path: Path) -> None:
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["prompt_variants"][0]["custom_instructions"] = "Use explicit form labels."
    write_json(manifest_path, manifest)
    job = tmp_path / "job"
    make_output(make_trial(job, "trial"))
    out = import_jobs([job], dataset, tmp_path / "report", [1], generate_report=False)
    variant = load_results(out)["meta"]["prompt_variants"][0]
    instructions_path = Path(variant["custom_instructions_path"])
    assert instructions_path.is_relative_to(out)
    assert instructions_path.read_text() == "Use explicit form labels."


def test_later_step_exception_preserves_completed_generation(dataset: Path, tmp_path: Path) -> None:
    job = tmp_path / "job"
    trial = make_trial(job, "skill", task="checkbox-review", steps=[
        {"step_name": "generate"},
        {"step_name": "review", "exception_info": {"exception_message": "Agent timed out"}},
    ])
    result_path = trial / "result.json"
    result = json.loads(result_path.read_text())
    result["exception_info"] = {"exception_message": "Agent timed out"}
    write_json(result_path, result)
    make_output(trial / "steps" / "generate")
    make_output(trial / "steps" / "review")
    out = import_jobs([job], dataset, tmp_path / "report", [1], generate_report=False)
    assert [r["result"] for r in load_results(out)["results"]] == ["PASS", "ERROR"]


def test_skill_files_are_copied_from_relative_manifest_path(dataset: Path, tmp_path: Path) -> None:
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["prompt_variants"][1]["skill_path"] = "skills/review"
    write_json(manifest_path, manifest)
    skill = dataset / "skills" / "review"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("Review accessibility.", encoding="utf-8")
    job = tmp_path / "job"
    make_output(make_trial(job, "trial"))
    out = import_jobs([job], dataset, tmp_path / "report", [1], generate_report=False)
    local_skill = Path(load_results(out)["meta"]["prompt_variants"][1]["skill_path"])
    assert local_skill.is_relative_to(out)
    assert (local_skill / "SKILL.md").read_text() == "Review accessibility."
