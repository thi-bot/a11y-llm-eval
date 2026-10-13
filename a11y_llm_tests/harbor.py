"""Export native Harbor tasks and import their accessibility reports.

Harbor owns agent selection, credentials, containers, concurrency and retries.
This adapter owns prompt composition and the existing deterministic verifier.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

import typer

from .prompt_specs import load_prompt_specs
from .variants import _load_instruction_sets, _load_skills

ROOT = Path(__file__).resolve().parent.parent
app = typer.Typer(add_completion=False, help=__doc__)
OUTPUT_INSTRUCTIONS = (
    "Save your answer to /app/output/index.html. Feel free to use separate CSS "
    "and JS files within /app/output. The site must work when served as static "
    "files over HTTP, without a build step or a development server."
)


def _safe_id(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", value):
        raise ValueError(f"Harbor variant and turn IDs must be portable directory names: {value!r}")
    return value


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _copy_tree(source: Path, destination: Path) -> None:
    # Do not follow links into host files or copy local caches/dependencies.
    if source.is_symlink():
        raise ValueError(f"Export sources must not contain symlinks: {source}")
    ignored = shutil.ignore_patterns(
        "node_modules", "__pycache__", ".git", ".cache", "*.pyc",
    )
    def ignore(directory: str, names: list[str]) -> set[str]:
        skipped = ignored(directory, names)
        for name in set(names) - skipped:
            if (Path(directory) / name).is_symlink():
                raise ValueError(f"Export sources must not contain symlinks: {Path(directory) / name}")
        return skipped
    shutil.copytree(source, destination, ignore=ignore)


def export_dataset(
    out: Path,
    test_cases_dir: Path = ROOT / "test_cases",
    dimensions_file: Path = ROOT / "config/prompt_dimensions.yaml",
    instruction_sets_file: Path | None = ROOT / "config/default_instruction_sets.yaml",
    skills_file: Path | None = ROOT / "config/default_skills.yaml",
    test_cases: list[str] | None = None,
    agent_timeout: int = 600,
) -> dict:
    """Create a self-contained dataset, refusing to overwrite existing files."""
    if out.exists():
        raise ValueError(f"Export destination already exists: {out}. Choose a new directory.")
    if agent_timeout < 1:
        raise ValueError("agent_timeout must be positive")
    specs = load_prompt_specs(test_cases_dir, dimensions_file, test_case_filter=test_cases)
    if not specs.prompt_cases:
        raise ValueError("No prompt cases found")
    variants = [{"id": "control", "name": "Control", "kind": "control"}]
    if instruction_sets_file is not None:
        variants.extend(
            {**v, "kind": "instruction_set"}
            for v in _load_instruction_sets(str(instruction_sets_file), instruction_sets_file.resolve().parent)
        )
    if skills_file is not None:
        variants.extend(
            {**v, "kind": "skill"}
            for v in _load_skills(str(skills_file), skills_file.resolve().parent,
                                  existing_ids={v["id"] for v in variants})
        )
    for variant in variants:
        _safe_id(variant["id"])
        if variant.get("samples") is not None:
            raise ValueError(
                f"Variant {variant['id']!r} sets samples. Use Harbor --n-attempts, "
                "with separate jobs filtered by variant for different sample counts."
            )
        agent = variant.get("agent") or {}
        if set(agent) - {"limits"} or set(agent.get("limits", {})) - {"timeout_s"}:
            raise ValueError(f"Variant {variant['id']!r}: only agent.limits.timeout_s maps to Harbor; "
                             "configure other agent options in the Harbor job.")
        for turn in variant.get("turns", []):
            _safe_id(turn["id"])
            if "{{previous_submission}}" in turn["prompt"]:
                raise ValueError("Harbor step instructions are static: replace {{previous_submission}} "
                                 "with an instruction to read /app/output/index.html.")

    dockerfile = (ROOT / "config/harbor/Dockerfile").read_text(encoding="utf-8")
    manifest = {
        "format": "a11y-harbor/v1",
        "harbor_version": "0.23.0",
        "instruction_delivery": "task_prompt",
        "prompts": specs.prompts_map,
        "prompt_cases": specs.prompt_cases_meta,
        "prompt_variants": [],
        "tasks": {},
    }
    # Export atomically so validation/copy failures cannot leave a runnable,
    # partially populated dataset behind.
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".harbor-export-", dir=out.parent) as staging:
        dataset = Path(staging) / "dataset"
        dataset.mkdir()
        for variant in variants:
            vid = variant["id"]
            report_variant = {
                "id": vid, "name": variant["name"], "kind": variant["kind"],
                "description": variant.get("description"), "url": variant.get("url"),
                "generation_mode": "harbor", "turns": variant.get("turns"),
                "custom_instructions": variant.get("markdown_text"),
            }
            if variant.get("markdown_text"):
                rel = Path("instructions") / f"{vid}.md"
                _write(dataset / rel, variant["markdown_text"])
                report_variant["custom_instructions_path"] = rel.as_posix()
            if variant["kind"] == "skill":
                rel = Path("skills") / vid
                _copy_tree(Path(variant["skill_dir_abs_path"]), dataset / rel)
                report_variant["skill_path"] = rel.as_posix()
            manifest["prompt_variants"].append(report_variant)
            for prompt in specs.prompt_cases:
                _safe_id(prompt.base_test_name)
                digest = hashlib.sha256(f"{prompt.prompt_case_id}\0{vid}".encode()).hexdigest()[:12]
                name = f"{prompt.base_test_name}--{vid}--{digest}"
                task = dataset / name
                environment = task / "environment"
                tests = task / "tests"
                environment.mkdir(parents=True)
                tests.mkdir()
                for folder in (environment, tests):
                    for filename in ("package.json", "package-lock.json"):
                        shutil.copyfile(ROOT / "node_runner" / filename, folder / filename)
                _write(environment / "Dockerfile", dockerfile)
                # Keep all private assertions and runner code in the fresh
                # verifier image, never in the agent's environment.
                _copy_tree(ROOT / "node_runner", tests / "node_runner")
                case_dir = tests / "test_cases" / prompt.base_test_name
                case_dir.mkdir(parents=True)
                shutil.copyfile(test_cases_dir / prompt.base_test_name / "test.js", case_dir / "test.js")
                _write(tests / "a11y_llm_tests/__init__.py", "")
                for filename in ("node_bridge.py", "harbor_verifier.py"):
                    shutil.copyfile(ROOT / "a11y_llm_tests" / filename, tests / "a11y_llm_tests" / filename)
                _write(tests / "Dockerfile", dockerfile + "\nCOPY . /tests/\n")
                _write(tests / "test.sh", "#!/bin/bash\nset -euo pipefail\n"
                       "cd /tests\npython3 -m a11y_llm_tests.harbor_verifier "
                       f"--test-js /tests/test_cases/{prompt.base_test_name}/test.js\n")

                turns = variant.get("turns") or []
                timeout = (variant.get("agent", {}).get("limits") or {}).get("timeout_s", agent_timeout)
                if not isinstance(timeout, (int, float)) or timeout <= 0:
                    raise ValueError(f"Invalid timeout for variant {vid!r}")
                config = [
                    'schema_version = "1.4"',
                    'artifacts = [{ source = "/app/output", destination = "site" }]',
                ]
                if turns:
                    config.append('multi_step_reward_strategy = "final"')
                config.extend([
                    "\n[metadata]", 'benchmark = "a11y-llm-eval"',
                    f"prompt_case_id = {json.dumps(prompt.prompt_case_id)}",
                    f"variant_id = {json.dumps(vid)}",
                    f"variant_kind = {json.dumps(variant['kind'])}",
                    '\n[agent]', f"timeout_sec = {float(timeout)}",
                    '\n[environment]', 'build_timeout_sec = 1200.0',
                    'cpus = 2', 'memory_mb = 4096', 'storage_mb = 10240',
                ])
                if turns:
                    config.append('skills_dir = "/skills"')
                    _copy_tree(Path(variant["skill_dir_abs_path"]), environment / "skills" / vid)
                    with (environment / "Dockerfile").open("a", encoding="utf-8") as handle:
                        handle.write("\nCOPY skills/ /skills/\n")
                config.extend([
                    '\n[verifier]', 'timeout_sec = 180.0', 'environment_mode = "separate"',
                    '\n[verifier.environment]', 'build_timeout_sec = 1200.0',
                    'cpus = 2', 'memory_mb = 4096', 'storage_mb = 10240',
                ])
                if turns:
                    for turn in turns:
                        instruction = turn["prompt"].replace("{{test_case_prompt}}", prompt.prompt_text)
                        instruction = instruction.replace("{{skill_id}}", vid).replace("{{skill_path}}", f"/skills/{vid}")
                        if re.search(r"\{\{[^}]+\}\}", instruction):
                            raise ValueError(f"Unresolved template token in skill {vid!r}")
                        _write(task / "steps" / turn["id"] / "instruction.md", instruction.strip() + "\n\n" + OUTPUT_INSTRUCTIONS + "\n")
                        config.extend(['\n[[steps]]', f"name = {json.dumps(turn['id'])}",
                                       '[steps.agent]', f"timeout_sec = {float(timeout)}"])
                else:
                    instruction = prompt.prompt_text + "\n\n" + OUTPUT_INSTRUCTIONS
                    if variant.get("markdown_text"):
                        instruction += "\n\n" + variant["markdown_text"].strip()
                    _write(task / "instruction.md", instruction + "\n")
                _write(task / "task.toml", "\n".join(config) + "\n")
                manifest["tasks"][name] = {
                    **asdict(prompt), "variant_id": vid, "variant_kind": variant["kind"], "turns": turns,
                }
        _write(dataset / "manifest.json", json.dumps(manifest, indent=2) + "\n")
        dataset.rename(out)
    return manifest


@app.command("export")
def export_command(
    out: Path = typer.Option(Path("harbor-tasks"), help="New dataset directory; never overwritten."),
    test_cases_dir: Path = typer.Option(ROOT / "test_cases"),
    prompt_dimensions_file: Path = typer.Option(ROOT / "config/prompt_dimensions.yaml"),
    instruction_sets_file: Path = typer.Option(ROOT / "config/default_instruction_sets.yaml"),
    skills_file: Path = typer.Option(ROOT / "config/default_skills.yaml"),
    control_only: bool = typer.Option(False, help="Export only control tasks."),
    test_cases: str | None = typer.Option(None, help="Comma-separated base test names."),
    agent_timeout: int = typer.Option(600, min=1),
) -> None:
    """Expand YAML prompts and variants into native Harbor tasks (no model calls)."""
    try:
        manifest = export_dataset(
            out, test_cases_dir, prompt_dimensions_file,
            None if control_only else instruction_sets_file,
            None if control_only else skills_file,
            [name.strip() for name in test_cases.split(",") if name.strip()] if test_cases else None,
            agent_timeout,
        )
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Exported {len(manifest['tasks'])} tasks to {out}")
    typer.echo("Run with: harbor run --path " + str(out) + " --agent <agent> --model <model> --resume-trajectory")


@app.command("import")
def import_command(
    jobs: list[Path] = typer.Argument(..., help="One or more Harbor job directories."),
    dataset: Path = typer.Option(Path("harbor-tasks"), help="Dataset used for these jobs."),
    out: Path = typer.Option(..., help="New directory for the combined accessibility report."),
    k: str = typer.Option("1,5,10", help="pass@k values; values exceeding sample count are omitted."),
    report: bool = typer.Option(True, "--report/--no-report"),
) -> None:
    """Combine agent jobs into the existing detailed accessibility report."""
    from .harbor_import import import_jobs
    try:
        ks = [int(value.strip()) for value in k.split(",")]
        if not ks or any(value < 1 for value in ks):
            raise ValueError("k values must be positive integers")
        result = import_jobs(jobs, dataset, out, ks, generate_report=report)
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Imported Harbor results: {result}")


if __name__ == "__main__":
    app()
