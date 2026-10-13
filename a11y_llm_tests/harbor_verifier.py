"""Harbor verifier entrypoint, copied into the isolated verifier image.

Uses only the standard library and the existing Node bridge. Grading code is
never installed in the agent image. A missing submission is a failed attempt;
an exception or broken browser is an infrastructure error, never a pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from a11y_llm_tests import node_bridge


def passed(result: dict) -> bool:
    test = result.get("testFunctionResult") or {}
    axe = result.get("axeResult")
    return (
        not result.get("error")
        and test.get("status") == "pass"
        and not any(
            a.get("status") == "fail" and a.get("type", "R").upper() == "R"
            for a in test.get("assertions", [])
        )
        and isinstance(axe, dict)
        and axe.get("failure_count") == 0
        and (result.get("renderEvaluation") or {}).get("rendered") is not False
    )


def verify(site: Path, test_js: Path, logs: Path) -> dict:
    logs.mkdir(parents=True, exist_ok=True)
    # Never leave a stale success behind when a verifier crashes on a rerun.
    (logs / "reward.txt").write_text("0\n", encoding="utf-8")
    try:
        submission = site / "index.html"
        if not submission.is_file() or not submission.read_text(encoding="utf-8").strip():
            result = {
                "testFunctionResult": {
                    "status": "fail", "assertions": [],
                    "error": "Missing or empty /app/output/index.html",
                },
                "axeResult": None,
                "error": None,
            }
        else:
            result = node_bridge.run(
                submission.read_text(encoding="utf-8"), str(test_js),
                str(logs / "screenshot.png"), html_dir=str(site),
            )
    except Exception as exc:
        result = {
            "testFunctionResult": {"status": "error", "assertions": [], "error": str(exc)},
            "axeResult": None, "error": str(exc),
        }
    (logs / "a11y-result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (logs / "reward.txt").write_text("1\n" if passed(result) else "0\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=Path("/app/output"))
    parser.add_argument("--test-js", type=Path, required=True)
    parser.add_argument("--logs", type=Path, default=Path("/logs/verifier"))
    args = parser.parse_args()
    result = verify(args.site, args.test_js, args.logs)
    if result.get("error") or (result.get("testFunctionResult") or {}).get("status") == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
