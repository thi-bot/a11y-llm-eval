"""Shared instruction-set and skill configuration loaders."""
from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

def _load_instruction_sets(instruction_sets_file: str, base_dir: Path) -> list[dict]:
    """Load instruction sets YAML.

    Expected format:
      instruction_sets:
        - id: concise
          name: Concise
          description: ...
          url: https://example.com/full-instructions
          instructions_markdown: path/to/file.md
          samples: 10

    The legacy key ``system_prompt_append_markdown`` is still accepted as an
    alias for ``instructions_markdown`` for back-compat with older configs.
    """
    path = Path(instruction_sets_file)
    if not path.is_absolute():
        cwd_path = path.resolve()
        if cwd_path.exists():
            path = cwd_path
        else:
            path = (base_dir / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Instruction sets file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    sets = data.get("instruction_sets") or []
    if not isinstance(sets, list):
        raise ValueError("instruction_sets must be a list")
    out = []
    for s in sets:
        if not isinstance(s, dict):
            continue
        sid = (s.get("id") or "").strip()
        if not sid:
            raise ValueError("Each instruction set must include a non-empty 'id'")
        if sid.lower() == "control":
            raise ValueError("'control' is reserved; choose a different instruction set id")
        if "generation_mode" in s:
            raise ValueError(
                f"Instruction set '{sid}' cannot specify generation_mode; instruction sets always use the copilot_agent path"
            )
        md_path = s.get("instructions_markdown") or s.get("system_prompt_append_markdown")
        if not md_path:
            raise ValueError(f"Instruction set '{sid}' must include instructions_markdown")
        mdp = Path(md_path)
        if not mdp.is_absolute():
            mdp = (base_dir / mdp).resolve()
        if not mdp.exists():
            raise FileNotFoundError(f"Instruction markdown not found for '{sid}': {mdp}")
        out.append({
            "id": sid,
            "name": (s.get("name") or sid).strip(),
            "description": (s.get("description") or "").strip() or None,
            "url": (s.get("url") or "").strip() or None,
            "markdown_path": str(mdp),
            "markdown_text": mdp.read_text(encoding="utf-8"),
            "samples": s.get("samples"),
            "generation_mode": "copilot_agent",
            "agent": s.get("agent") if isinstance(s.get("agent"), dict) else {},
        })
    return out


# Exactly one turn per skill must reference this token. Earlier turns that produce
# HTML the evaluator will score must embed the user's test case prompt.
SKILL_TOKEN_TEST_CASE_PROMPT = "{{test_case_prompt}}"


def _hash_skill_files(skill_dir: Path) -> str:
    """Compute a stable digest over every file under the skill directory."""
    sha = hashlib.sha256()
    for rel in sorted(p.relative_to(skill_dir).as_posix() for p in skill_dir.rglob("*") if p.is_file()):
        sha.update(rel.encode("utf-8"))
        sha.update(b"\0")
        sha.update((skill_dir / rel).read_bytes())
        sha.update(b"\0")
    return sha.hexdigest()[:16]


def _load_skills(skills_file: str, base_dir: Path, existing_ids: set[str] | None = None) -> list[dict]:
    """Load a skills YAML file.

    Expected format::

        skills:
          - id: a11y-reviewer
            name: Accessibility Reviewer
            description: ...
            url: https://example.com/full-skill
            skill_dir: skills/a11y-reviewer   # relative to base_dir or absolute
            samples: 10                       # optional
            agent:                            # optional; same shape as instruction sets
              sandbox: [docker, path/to/compose.yaml]
              limits: {...}
            turns:
              - id: generate
                name: Generate
                prompt: "{{test_case_prompt}}"
              - id: review
                name: Review & remediate
                prompt: "Review using {{skill_path}}/SKILL.md ..."
    """
    reserved = set(existing_ids or set())
    path = Path(skills_file)
    if not path.is_absolute():
        cwd_path = path.resolve()
        if cwd_path.exists():
            path = cwd_path
        else:
            path = (base_dir / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Skills file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw = data.get("skills") or []
    if not isinstance(raw, list):
        raise ValueError("skills must be a list")
    out: list[dict] = []
    seen_ids: set[str] = set()
    for s in raw:
        if not isinstance(s, dict):
            continue
        sid = (s.get("id") or "").strip()
        if not sid:
            raise ValueError("Each skill must include a non-empty 'id'")
        if sid.lower() == "control":
            raise ValueError("'control' is reserved; choose a different skill id")
        if sid in seen_ids or sid in reserved:
            raise ValueError(f"Duplicate variant id '{sid}' (skills and instruction sets share a namespace)")
        seen_ids.add(sid)
        if "generation_mode" in s:
            raise ValueError(
                f"Skill '{sid}' cannot specify generation_mode; skills always use the copilot_agent path"
            )
        skill_dir_raw = s.get("skill_dir")
        if not skill_dir_raw:
            raise ValueError(f"Skill '{sid}' must include skill_dir")
        sdp = Path(skill_dir_raw)
        if not sdp.is_absolute():
            sdp = (base_dir / sdp).resolve()
        if not sdp.exists() or not sdp.is_dir():
            raise FileNotFoundError(f"Skill directory not found for '{sid}': {sdp}")
        if not (sdp / "SKILL.md").exists():
            raise FileNotFoundError(f"Skill '{sid}' is missing SKILL.md at {sdp / 'SKILL.md'}")

        turns_raw = s.get("turns")
        if not isinstance(turns_raw, list) or not turns_raw:
            raise ValueError(f"Skill '{sid}' must include a non-empty 'turns' list")
        turns: list[dict] = []
        turn_ids: set[str] = set()
        test_case_token_count = 0
        for i, t in enumerate(turns_raw):
            if not isinstance(t, dict):
                raise ValueError(f"Skill '{sid}' turn #{i} must be a mapping")
            tid = (t.get("id") or "").strip()
            if not tid:
                raise ValueError(f"Skill '{sid}' turn #{i} must include a non-empty 'id'")
            if tid in turn_ids:
                raise ValueError(f"Skill '{sid}' has duplicate turn id '{tid}'")
            turn_ids.add(tid)
            prompt_tmpl = t.get("prompt")
            if not isinstance(prompt_tmpl, str) or not prompt_tmpl.strip():
                raise ValueError(f"Skill '{sid}' turn '{tid}' must include a non-empty 'prompt'")
            if SKILL_TOKEN_TEST_CASE_PROMPT in prompt_tmpl:
                test_case_token_count += 1
            turns.append({
                "id": tid,
                "name": (t.get("name") or tid).strip(),
                "prompt": prompt_tmpl,
            })
        if test_case_token_count != 1:
            raise ValueError(
                f"Skill '{sid}' must reference {SKILL_TOKEN_TEST_CASE_PROMPT} in exactly one turn "
                f"(found {test_case_token_count})."
            )

        name = (s.get("name") or sid).strip()
        files_hash = _hash_skill_files(sdp)
        out.append({
            "id": sid,
            "name": name,
            "description": (s.get("description") or "").strip() or None,
            "url": (s.get("url") or "").strip() or None,
            "skill_dir_abs_path": str(sdp),
            "skill_files_hash": files_hash,
            "turns": turns,
            "samples": s.get("samples"),
            "generation_mode": "copilot_agent",
            "agent": s.get("agent") if isinstance(s.get("agent"), dict) else {},
        })
    return out
