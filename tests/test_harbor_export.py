"""Contract tests for dataset expansion and the isolated accessibility verifier."""
import json
from pathlib import Path
import tomllib

import pytest
from typer.testing import CliRunner

from a11y_llm_tests.harbor import ROOT, _copy_tree, app, export_dataset
from a11y_llm_tests.harbor_verifier import passed, verify


def test_full_matrix_and_isolation(tmp_path):
    out = tmp_path / "tasks"
    manifest = export_dataset(out)
    assert len(manifest["tasks"]) == 128
    assert len(manifest["prompt_cases"]) == 32
    assert len(manifest["prompt_variants"]) == 4
    assert json.loads((out / "manifest.json").read_text()) == manifest
    for name, meta in manifest["tasks"].items():
        task = out / name
        config = tomllib.loads((task / "task.toml").read_text())
        assert config["artifacts"] == [{"source": "/app/output", "destination": "site"}]
        assert config["verifier"]["environment_mode"] == "separate"
        assert (task / "tests/test_cases" / meta["base_test_name"] / "test.js").read_bytes() == (
            ROOT / "test_cases" / meta["base_test_name"] / "test.js"
        ).read_bytes()
        assert not (task / "environment/test_cases").exists()
        assert not (task / "environment/node_runner").exists()
        if meta["variant_kind"] == "skill":
            assert [s["name"] for s in config["steps"]] == ["generate", "review"]
            assert all("min_reward" not in step for step in config["steps"])
            assert config["multi_step_reward_strategy"] == "final"
            assert config["environment"]["skills_dir"] == "/skills"
            assert (task / "environment/skills/building-accessible-ui/SKILL.md").is_file()
            assert "{{" not in (task / "steps/generate/instruction.md").read_text()
        else:
            assert "steps" not in config
            assert "skills_dir" not in config["environment"]
            assert meta["prompt_text"] in (task / "instruction.md").read_text()
        if meta["variant_kind"] == "control":
            assert not (task / "environment/skills").exists()
            assert "accessible" not in (task / "instruction.md").read_text().lower()


def test_cli_filtered_control_and_existing_destination(tmp_path):
    out = tmp_path / "tasks"
    args = ["export", "--out", str(out), "--control-only", "--test-cases", "simple-contact-form"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert len(json.loads((out / "manifest.json").read_text())["tasks"]) == 8
    assert CliRunner().invoke(app, args).exit_code != 0


@pytest.mark.parametrize("extra", ["samples: 2", "id: ../escape", "agent: {limits: {max_turns: 1}}"])
def test_reject_unportable_legacy_options_atomically(tmp_path, extra):
    (tmp_path / "instructions.md").write_text("Make the page accessible.")
    sets = tmp_path / "instructions.yaml"
    sets.write_text("instruction_sets:\n  - id: basic\n    instructions_markdown: instructions.md\n    " + extra + "\n")
    with pytest.raises(ValueError):
        export_dataset(tmp_path / "out", instruction_sets_file=sets, skills_file=None)
    assert not (tmp_path / "out").exists()


def test_skill_previous_submission_requires_explicit_file_reference(tmp_path):
    (tmp_path / "skill").mkdir()
    (tmp_path / "skill/SKILL.md").write_text("A skill.")
    skills = tmp_path / "skills.yaml"
    skills.write_text('skills:\n  - id: example\n    skill_dir: skill\n    turns:\n'
                      '      - id: generate\n        prompt: "{{test_case_prompt}}"\n'
                      '      - id: review\n        prompt: "Review {{previous_submission}}"\n')
    with pytest.raises(ValueError, match="previous_submission"):
        export_dataset(tmp_path / "out", instruction_sets_file=None, skills_file=skills)
    assert not (tmp_path / "out").exists()


def test_verifier_reward_semantics():
    result = {
        "testFunctionResult": {"status": "pass", "assertions": [{"status": "fail", "type": "BP"}]},
        "axeResult": {"failure_count": 0, "best_practice_count": 3},
        "renderEvaluation": {"rendered": True},
    }
    assert passed(result)
    result["testFunctionResult"]["assertions"][0]["type"] = "R"
    assert not passed(result)
    result["testFunctionResult"]["assertions"] = []
    result["axeResult"]["failure_count"] = 1
    assert not passed(result)
    result["axeResult"]["failure_count"] = 0
    result["renderEvaluation"]["rendered"] = False
    assert not passed(result)
    result["renderEvaluation"]["rendered"] = True
    result["error"] = "test script crashed after returning assertions"
    assert not passed(result)


def test_missing_submission_replaces_stale_reward(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "reward.txt").write_text("1")
    result = verify(tmp_path / "site", tmp_path / "test.js", logs)
    assert result["testFunctionResult"]["status"] == "fail"
    assert (logs / "reward.txt").read_text() == "0\n"
    assert json.loads((logs / "a11y-result.json").read_text()) == result


def test_verifier_exception_records_failure(tmp_path, monkeypatch):
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<html></html>")
    def broken(*args, **kwargs):
        raise RuntimeError("Browser failed to start")
    monkeypatch.setattr("a11y_llm_tests.harbor_verifier.node_bridge.run", broken)
    result = verify(site, tmp_path / "test.js", tmp_path / "logs")
    assert result["error"] == "Browser failed to start"
    assert (tmp_path / "logs/reward.txt").read_text() == "0\n"


def test_export_ignores_installed_dependencies_and_their_links(tmp_path):
    source = tmp_path / "source"
    (source / "node_modules/.bin").mkdir(parents=True)
    try:
        (source / "node_modules/.bin/tool").symlink_to("../tool/bin.js")
    except OSError:
        pytest.skip("Symlinks unavailable on this platform")
    (source / "runner.js").write_text("// runner")
    destination = tmp_path / "copy"
    _copy_tree(source, destination)
    assert (destination / "runner.js").is_file()
    assert not (destination / "node_modules").exists()
