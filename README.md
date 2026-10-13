# A11y LLM Evaluation Harness and Dataset

Benchmark how well coding agents generate accessible web pages, using
[Harbor](https://docs.harborframework.com/) to swap agents and run isolated trials.
The benchmark keeps its existing prompt dimensions, instruction variants, skills,
Playwright/axe-core checks, and detailed accessibility reports.

## Quick start

You need Python 3.12+, Docker Engine/Desktop with Compose v2, and
[uv](https://docs.astral.sh/uv/). Harbor installs coding agents inside containers;
you do not need their CLIs on the host for API-key runs.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
uv tool install harbor==0.23.0

# Export all prompts, instruction variants, and skills. No model calls.
python -m a11y_llm_tests.harbor export --out harbor-tasks

# Start small: one attempt per task. Set OPENAI_API_KEY in your environment.
harbor run --path harbor-tasks \
  --agent codex --model openai/gpt-5.4 \
  --resume-trajectory --n-attempts 1 --n-concurrent 4 \
  --job-name codex-pilot

# Import the completed job and build the accessibility report.
python -m a11y_llm_tests.harbor import jobs/codex-pilot \
  --dataset harbor-tasks --out runs/codex-pilot --k 1

python -m a11y_llm_tests.cli serve runs/codex-pilot --open
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`.
Use PowerShell line continuation or put each command on one line.

Harbor runs generation and verification together. Exported datasets and report
output directories must be new, so an export or import cannot overwrite a prior
experiment. The report is self-contained, including generated assets, screenshots,
verifier details and conversation previews.

## Swap agents

Use the same exported tasks with a different Harbor agent and model:

```bash
# Set ANTHROPIC_API_KEY in your environment.
harbor run --path harbor-tasks \
  --agent claude-code --model anthropic/claude-sonnet-4-6 \
  --resume-trajectory --n-attempts 1 --n-concurrent 4 \
  --job-name claude-pilot

# Combine jobs; agent/model/version/settings remain separate report entries.
python -m a11y_llm_tests.harbor import jobs/codex-pilot jobs/claude-pilot \
  --dataset harbor-tasks --out runs/agent-comparison --k 1
```

Harbor also provides `copilot-cli`, `gemini-cli`, `opencode`, and other
[agent integrations](https://docs.harborframework.com/agents/pre-integrated-agents).
Use `harbor agent schema <agent>` for agent-specific options and credential
requirements. Multi-turn skill tasks require native resume support when
`--resume-trajectory` is enabled; Harbor rejects unsupported agents. Agents also
need to support Harbor's `skills_dir` to evaluate skill discovery fairly.

For an agent/model matrix, edit [config/harbor.example.yaml](config/harbor.example.yaml)
and run:

```bash
harbor run --config config/harbor.example.yaml
```

That example requests **20 attempts** per task and agent. Start with one attempt
while validating credentials and cost. Harbor controls concurrency, retries,
timeouts, environment providers and agent versions. Pin agent versions in job
configuration when comparing runs. Keep credentials in your environment; do not
put secrets in task files or Docker build contexts.

Switching to Harbor does not itself provide subscription authentication. Follow
the selected agent's Harbor authentication options; the commands above use direct
API keys. A Codex subscription, OpenAI API billing and Copilot billing remain
separate choices.

## Benchmark matrix and methodology

The current dataset expands to 32 prompt cases:

- Six base test cases × two frameworks × two styles: 24 prompts.
- Contact form × two frameworks × two styles × two validation states: 8 prompts.

The default export creates 128 tasks: 32 prompts × control, minimal instructions,
basic instructions, and the accessibility skill. The skill has two steps, so one
attempt across all tasks produces 160 evaluated outputs **per agent/model**.

- **Control:** the composed test prompt and common artifact instructions only.
- **Instruction variants:** the same prompt with each instruction set appended
  separately. This delivery is consistent across agents. It differs from the
  legacy Copilot workflow's `.github/copilot-instructions.md`; record this
  methodology change when comparing old and new runs.
- **Skills:** Harbor makes the skill available through `environment.skills_dir`.
  Generate and review run as separate steps in one persistent environment.
  `--resume-trajectory` preserves the agent conversation. Both steps are scored;
  failing generation does not prevent review. Harbor's trial-level reward is the
  final step's score; the accessibility report preserves every step.

Agents write a static site to `/app/output/index.html`, with optional sibling
CSS/JS/assets. Harbor snapshots the whole output directory after each step. A
fresh verifier container renders that snapshot over localhost HTTP and runs the
existing assertions and axe-core. The agent image contains public Playwright and
axe-core tooling for self-testing, but none of the benchmark's private assertions.
The verifier's tests and results are not fed back to the review step.

A reward of **1** requires zero axe-core WCAG failures and all requirement
assertions passing. Best Practice failures and not-applicable assertions remain
visible but do not fail the sample. Missing HTML, browser failures, and verifier
exceptions cannot receive a passing reward. Interrupted trials and missing skill
steps remain in the imported report as errors, rather than disappearing from the
sample denominator.

Use `--n-attempts 20` for repeated sampling. The importer computes pass@k only for
k values no larger than the number of samples in a group. For example, use
`--k 1,5,10` after collecting at least ten samples. There is no legacy generation
cache on the Harbor path: new trials make fresh agent runs. See Harbor's
[regrading documentation](https://docs.harborframework.com/jobs/regrade) for
reusing generated artifacts without paying for another generation.

For five independent runs of the frozen dataset with the agreed infrastructure
retry policy, use Harbor's Python environment to run the campaign script:

This campaign runner requires Linux or macOS (it uses `fcntl` for locking) and
a host Codex login saved in `auth.json`. It sets `CODEX_FORCE_AUTH_JSON=1` to use
that login. The general Harbor commands above use API keys instead.

For a new experiment, export the dataset first, then freeze its checksums once
with the following command. Keep this file and the dataset at the same path
throughout the campaign; do not regenerate the baseline when resuming a run.

```sh
~/.local/share/uv/tools/harbor/bin/python - <<'PY'
import json
from pathlib import Path
from harbor.models.task.task import Task

dataset = Path("harbor-tasks").resolve()
manifest = json.loads((dataset / "manifest.json").read_text())
checksums = {
    str(dataset / name): Task(dataset / name).checksum
    for name in manifest["tasks"]
}
with (dataset / "reproducibility-task-checksums.json").open("x") as output:
    json.dump(checksums, output, indent=2)
    output.write("\n")
PY
```

```sh
~/.local/share/uv/tools/harbor/bin/python scripts/run_harbor_campaign.py \
  --out runs/codex-luna-sol-five-runs \
  --models openai/gpt-6-luna openai/gpt-6.1-sol
```

The campaign uses Codex 0.160.1, medium reasoning, and four concurrent tasks.
It requires `harbor-tasks/reproducibility-task-checksums.json` from the frozen
baseline. Each model receives five fresh runs of all variants. Model order
alternates between rounds. Earlier runs are kept separate.

Confirmed transient network/setup failures receive at most two additional
attempts. Verifier infrastructure failures reuse saved outputs when the full
task was recorded; otherwise the affected task restarts in a fresh environment.
Accessibility failures, broken generated pages, missing output, and agent time
limits do not earn additional attempts. Unknown errors require inspection;
authentication, unavailable models, and usage limits stop the campaign.

Each attempt retains its original logs and usage data. Selection manifests
identify exactly one final outcome per task; retries never increase the sample
count. Exhausted infrastructure errors mark a run incomplete and prevent a final
five-run comparison. The headline metric is the arithmetic mean of five run
pass rates per model and variant/turn, with median and sample standard deviation
as secondary statistics. Raw multi-step usage is retained without summing
potentially cumulative token or cost counters.

The output directory contains `protocol.json`, `status.json`, original Harbor
jobs, per-run attempt/selection records and HTML reports, and per-model
`comparison/comparison.json` and CSV files when all five runs are complete.
Restarting the script reuses finished jobs and preserves the retry budget;
interrupted jobs stop for inspection rather than deleting partial artifacts.

## Customize the dataset

The existing YAML files remain the source of truth:

- [config/prompt_dimensions.yaml](config/prompt_dimensions.yaml): global framework/style dimensions.
- `test_cases/<case>/prompt.yaml` and `test.js`: prompts and accessibility assertions.
- [config/default_instruction_sets.yaml](config/default_instruction_sets.yaml): instruction variants.
- [config/default_skills.yaml](config/default_skills.yaml): skill directory and ordered user turns.

Export a small control-only dataset, or supply alternate configurations:

```bash
python -m a11y_llm_tests.harbor export \
  --out harbor-pilot --control-only \
  --test-cases simple-contact-form,modal-dialog

python -m a11y_llm_tests.harbor export \
  --out harbor-custom \
  --instruction-sets-file config/default_instruction_sets.yaml \
  --skills-file config/default_skills.yaml \
  --agent-timeout 900
```

An empty `instruction_sets: []` or `skills: []` file disables that family while
retaining control. Each export includes `manifest.json` for the report importer.
Retain the exported dataset used by a job; changing source YAML does not mutate
that experiment.

Harbor uses native `task.toml`, `instruction.md`, Docker build contexts and verifier
scripts. There is no custom agent wrapper. Per-variant `agent.limits.timeout_s`
maps to Harbor's timeout; other legacy agent options are rejected rather than
silently ignored. Set sample counts with Harbor's `--n-attempts`. For different
counts by variant, run separate jobs with `--include-task-name '*--control--*'`
or the corresponding variant name, then import those jobs together. Remove
legacy `samples` overrides from YAML before exporting.

Skill turn templates support `{{test_case_prompt}}`, `{{skill_id}}` and
`{{skill_path}}`. Harbor instructions are static, so `{{previous_submission}}`
is rejected: instruct the agent to read `/app/output/index.html` instead.

## Results and cost accounting

Harbor retains its native trial results and trajectories under `jobs/`. The
importer creates the existing `results.json` and HTML report under `runs/`,
qualifying each model with its agent, version and configuration. This prevents
scores from different agent configurations being merged.

Single-step token/cost fields are imported when Harbor provides them. For
multi-step skills, token and cost fields are kept only in Harbor result sidecars
because agent adapters differ in cumulative versus per-step accounting. Reported
costs can be estimates rather than subscription charges; use provider billing for
actual spend. Missing usage is left unknown, not treated as free usage.

## Development and legacy runs

For host-side verifier development and the existing fixture suite:

```bash
python scripts/install_node_deps.py
python -m pytest
```

The previous `python -m a11y_llm_tests.cli run` path is retained for reproducing
older Copilot SDK experiments. Install `requirements-copilot.txt` for that path.
See [the legacy Copilot workflow](docs/legacy-copilot.md). New runs should use the
Harbor workflow above.

## Adding a Test Case
Create a new folder under `test_cases/`:
```
test_cases/
  form-labels/
    prompt.yaml
    test.js
    examples/
```

`prompt.yaml` defines the base prompt plus any prompt dimensions for the test case. A minimal example is:

```yaml
base_prompt: |
  Build a simple form.
common_requirements:
  - Include a submit button.
dimensions:
  validation-message:
    label: Validation Message
    values:
      present:
        label: Error Message Present
        prompt_fragment: Include an inline validation message.
      absent:
        label: No Error Message
        prompt_fragment: Do not show an inline validation message on initial render.
```

Global prompt dimensions such as framework and style live in `config/prompt_dimensions.yaml` and are combined with each test case automatically.

`test.js` must export:

```js
module.exports.run = async ({ page, assert, utils }) => {
  await assert("Has an h1", async () => {
    const count = await page.$$eval('h1', els => els.length);
    return count >= 1; // truthy => pass, falsy => fail
  });
  await assert("Sequential heading levels", async () => {
    // Return object form to include custom message
    const ok = await page.$$eval('h1 + h2', els => els.length) > 0;
    return { pass: ok, message: ok ? undefined : 'h2 does not follow h1' };
  }, { type: 'BP' });
  return {}; // assertions collected automatically
};
```

The runner injects an `assert(name, fn, opts?)` helper:

| Parameter | Description |
|-----------|-------------|
| `name` | Human-readable assertion label |
| `fn` | Async/Sync function returning boolean OR `{ pass, message? }` |
| `opts.type` | `'R'` (Requirement, default) or `'BP'` (Best Practice) |

Return shape from `run` can be empty.

### Assertion Types

Each assertion may now include a `type` field:

| Type | Meaning | Affects Test Pass/Fail | Aggregated Separately |
|------|---------|------------------------|-----------------------|
| `R`  | Requirement (default) | Yes (any failing R => test fails) | Requirement Pass Rate |
| `BP` | Best Practice | No (ignored for pass/fail) | Best Practice Pass Rate |

If `type` is omitted it defaults to `R` for backward compatibility. The HTML report shows both Requirement Pass Rate (percentage of tests whose requirement assertions passed) and Best Practice Pass Rate (percentage of tests containing BP assertions where all BP assertions passed).

The `fn` callback can also return `{ status: "pass" | "fail" | "na", message?: string }` for assertions that may not be applicable to a given page.

## Report
Generated at `runs/<timestamp>/index.html` with:
- Summary stats per model
- Detailed per model/test breakdown
- Axe violations
- Assertions & statuses
- Pass@k aggregate table and per-sample cards when multiple samples are collected

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
