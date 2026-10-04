# Running an AgentPrivArena experiment

This guide covers installation, task preparation, live execution, and evaluation.
It is a procedure for running the harness, not a promise to reproduce paper
numbers. Exact comparisons require the same task manifest, source revision,
model deployments, prompts, service versions, and evaluation settings.

AgentPrivArena is the framework described in **AgentPrivArena: Evaluating and
Auditing Real-world AI Agent Privacy**; AgentPrivAudit is its runtime privacy
auditor. The public CLI is `python -m agentprivarena`; the historical
`python -m privacylens_live` entry point uses the same implementation and
remains compatible. The manuscript's source benchmark is PrivacyLens.

All commands below run from the repository root. Paths under `.agent_tmp/` and
`results/` are local experiment artifacts, excluded from the public source
snapshot.

## 1. Install and validate the code

Use Python 3.13, uv 0.8.13 or newer, and Git. For live runs, install Docker
Engine with the Compose plugin and permission to create containers, networks,
and volumes. The stack runs several databases and applications; available
memory and disk space must accommodate those services and the agent image.
Model endpoints are separate from this stack.

For an exported source directory without Git metadata, first run `git init`
locally; `make build` installs Git pre-commit hooks.

```bash
make build
uv sync --frozen --group dev
uv run python -m agentprivarena --help
uv run pytest privacylens_live/tests tests/privacylens_live tests/sdk/privacy \
  tests/sdk/agent/test_privacy_extraction.py
```

The unit tests use mocks. They check conversion, runner logic, tool wrappers,
audit behavior, and evaluators without starting Docker or calling model APIs.
Passing them does not validate service provisioning or a model deployment.

## 2. Configure private local settings

Copy `privacylens_live/.env.example` to `privacylens_live/.env` on a fresh
checkout and fill in the settings for your deployment. Preserve an existing
`.env` rather than replacing it. This file is ignored by Git.

| Setting | Purpose |
| --- | --- |
| `LLM_MODEL` | Execution model with a provider prefix, such as `openai/<deployment>` |
| `LLM_API_KEY` | Execution endpoint credential; `OPENAI_API_KEY` is also accepted |
| `LLM_BASE_URL` | Custom endpoint, or `none` for the provider's normal routing |
| `LLM_API_VERSION` | Needed only for deployments requiring an API version |
| `EXTRACTION_LLM_MODEL` | Audit model for `--enable-privacy-analyzer` |
| `EXTRACTION_LLM_API_KEY` | Audit endpoint credential; falls back to the execution key |
| `EXTRACTION_LLM_BASE_URL` | Audit endpoint, configured independently of the execution endpoint |
| `AGENT_SERVER_IMAGE` | Local agent image tag; default `privacylens-agent-server:local` |
| `GOOGLE_DRIVE_ARTIFACT_ROOT` | Host-side directory for seeded file/image artifacts |

Use an available model identifier and credential from your own deployment. An
empty endpoint uses provider routing; no project-specific remote endpoint is
required. The audit model must be explicitly configured before running an
audit condition. Leaving an empty value in the example does not disable the
corresponding runtime default.

`Config.from_env()` loads `.env` over code defaults, then gives already-set
process environment variables precedence. `setup` writes service settings and
provisioned tokens back to `.env`, preserving other existing entries. It does
not add process-only model credentials to that file. Treat the resulting file
as private because it contains service tokens and any secrets you placed in it.

## 3. Build and start the service stack

Build the agent server from this checkout so its privacy hooks match the
host-side runner:

```bash
docker build \
  -f openhands-agent-server/openhands/agent_server/docker/Dockerfile \
  --target source-minimal \
  -t privacylens-agent-server:local .
export OH_PRELOAD_TOOLS=false
uv run python -m agentprivarena setup
```

The MCP-only experiments do not need browser-tool preloading, so keep
`OH_PRELOAD_TOOLS=false` set in the shell used for runs. Rebuild the agent image
after changing SDK audit code; editing host source alone does not update an
already-built image.

On a fresh BookStack database, the first `setup` starts the infrastructure and
then stops when it cannot validate an API token. Open the local BookStack UI
at `http://localhost:3000`, provision an API token for a user with access to the
benchmark books/pages, and set `BOOKSTACK_TOKEN_ID` and
`BOOKSTACK_TOKEN_SECRET` in `.env`. Rerun `setup` to continue provisioning the
remaining services and start the MCP containers. Existing valid tokens are
reused. Bootstrap does not create the BookStack token for you.

```bash
uv run python -m agentprivarena setup
docker compose -f privacylens_live/docker-compose.yml ps
```

The default Compose project creates the
`privacylens_live_privacylens-net` network used by agent workspaces. If you
change the Compose project name, update `DOCKER_NETWORK` to the actual network
name. Host service URLs and published ports must also agree if you customize
them.

The Compose stack uses local demonstration credentials and publishes ports on
`127.0.0.1` by default. Keep it on a dedicated local research machine. Seeding
and cleanup modify service contents; use an isolated stack for these
experiments. Run only one seeding/execution batch per stack at a time,
including `live-readiness`.

## 4. Prepare benchmark tasks

Acquire PrivacyLens source data under its original distribution terms and put
its `main_data.json` in a private local directory. The public snapshot excludes
both source records and the derived task collection pending a separate data
release review. In the commands below, replace
`/path/to/privacylens/main_data.json` with the actual input path.

Generate into a fresh output directory:

```bash
uv run python -m agentprivarena generate \
  --data /path/to/privacylens/main_data.json \
  --output .agent_tmp/tasks
uv run python -m agentprivarena verify \
  --data /path/to/privacylens/main_data.json \
  --tasks-dir .agent_tmp/tasks \
  --report-out .agent_tmp/tasks/verify_report.json
```

Each task directory contains `task.json`, `seed_data/<service>.json`, and a
`seed_manifest.json`. Conversion drops tasks marked unsuitable by default and
records the reasons in `dropped_tasks.json`. Review the generated task names,
repair/drop reasons, and source version before deciding the experiment subset.
Do not assume that a newly generated subset matches a historical task count.
`--include-drop-recommended` is a separate conversion choice, not the default.

The optional MPCI-Bench adapter uses the same task contract:

```bash
uv run python -m privacylens_live.benchmarks.mpci \
  --data /path/to/mpci_bench.json \
  --image-root /path/to/authorized/images \
  --output .agent_tmp/mpci_tasks \
  --quality-filter --require-images --paired-only
```

It copies available image artifacts and emits benchmark-only `oracle.json`
metadata. Source records, image rights, and derived artifacts need separate
provenance and redistribution review. Keep oracle fields out of execution
prompts and service seeds.

For image tasks, the default host artifact root is
`.agent_tmp/privacylens_artifacts/google_drive`; Compose mounts that directory
into `google-drive-mcp` as `/artifacts`. `setup` writes the corresponding host
mount path. Set `GOOGLE_DRIVE_VISION_MODEL`, `GOOGLE_DRIVE_VISION_BASE_URL`,
`GOOGLE_DRIVE_VISION_API_KEY`, and, if needed,
`GOOGLE_DRIVE_VISION_API_VERSION` for image-description calls. These settings
are distinct from the execution model. The backend serves local artifacts and
does not require Google account credentials.

## 5. Check live readiness and run a small condition

Once setup has completed, validate that the seeded records can be read:

```bash
uv run python -m agentprivarena live-readiness \
  --tasks-dir .agent_tmp/tasks --range 0-3 \
  --report-out .agent_tmp/readiness.json
```

This command changes service state and must finish before execution starts.
Task ranges are zero-based slices of sorted task directories with an exclusive
end; `0-3` selects three tasks. Use an explicit `--names` list when preserving
an exact subset across conversions.

Run a baseline smoke test, then a separate audit condition:

```bash
uv run python -m agentprivarena run \
  --tasks-dir .agent_tmp/tasks --range 0-3 \
  --results-dir results/baseline_smoke \
  --prompt-variant baseline --read-policy natural
uv run python -m agentprivarena run \
  --tasks-dir .agent_tmp/tasks --range 0-3 \
  --results-dir results/audit_smoke \
  --prompt-variant ci_audit_contextual --enable-privacy-analyzer \
  --read-policy natural
```

Execution and evaluation call the configured models and may incur charges.
Inspect the smoke results before expanding the range. Each condition should
have its own directory. Leave `--enable-privacy-analyzer` off for baseline and
prompt-only variants; it changes tool schemas and audit execution as well as
enabling a model.

`--resume` skips tasks with any existing valid result, including `error` and
`no_action`. Add `--retry-errors` to rerun errors and unfinished tasks while
preserving completed results. Retried task files are replaced, so retain the
original attempt artifacts separately when reporting retry policy. Never
resume a directory with different prompts, models, or conditions.

```bash
uv run python -m agentprivarena status \
  --tasks-dir .agent_tmp/tasks --results-dir results/audit_smoke
```

`_summary.json` covers only the latest invocation; `status` reads the result
directory for a cumulative view. A status of `ok` is an execution outcome,
not a judgment that the action preserved privacy or completed every task goal.

## 6. Evaluate a smoke run

Use a consistent, explicitly named judge for comparisons. These examples assume
you have separately deployed an OpenAI-compatible endpoint at
`http://localhost:8001/v1` serving the alias `Qwen3-14B-Judge`. The Compose stack
does not deploy this model. `not-needed` is a placeholder for a local server
without authentication; use appropriate private credentials for other servers.
This single-judge smoke check differs from the manuscript's three-judge
outcome protocol described below.

```bash
uv run python -m agentprivarena evaluate \
  --results-dir results/audit_smoke --tasks-dir .agent_tmp/tasks \
  --judge-model openai/Qwen3-14B-Judge \
  --judge-base-url http://localhost:8001/v1 \
  --judge-api-key not-needed \
  --sanitize-trajectory --no-cache \
  --report-out results/audit_smoke/report_qwen3_14b.json
uv run python -m agentprivarena trajectory-evaluate \
  --results-dir results/audit_smoke --tasks-dir .agent_tmp/tasks \
  --judge-model openai/Qwen3-14B-Judge \
  --judge-base-url http://localhost:8001/v1 \
  --judge-api-key not-needed --no-cache \
  --report-out results/audit_smoke/report_trajectory_qwen3_14b.json \
  --judgments-out results/audit_smoke/trajectory_judgments_qwen3_14b.json
```

Repeat with `results/baseline_smoke` to evaluate the baseline. Use `--no-cache`
when changing judge deployments or prompts because cache identity does not
fully describe all judge settings. Keep separate output names per judge.

Outcome evaluation produces aggregate reports and per-task judgments covering
leakage, helpfulness, routing, and behavior. Trajectory evaluation additionally
examines retrieval coverage and process metrics. Judge errors, model refusals,
and missing actions should be reported alongside denominators instead of
silently omitted. `--sanitize-trajectory` removes selected long observation
fields from helpfulness-judge input; it is an evaluation setting, not a
de-identification or publication scrubber for the underlying artifacts.

## 7. Map the manuscript's conditions and outcome protocol

The main conditions in the supplied manuscript map to these CLI options:

| Paper condition | `--prompt-variant` | Additional options |
| --- | --- | --- |
| C0: no mitigation | `baseline` | None |
| C1: privacy prompt | `privacy_enhanced` | None |
| C2: PII audit | `ci_audit_contextual` | `--enable-privacy-analyzer --audit-policy pii` |
| C3: data-minimization audit | `ci_audit_contextual` | `--enable-privacy-analyzer --audit-policy data_minimization` |
| C4: contextual-integrity audit | `ci_audit_contextual` | `--enable-privacy-analyzer --audit-policy contextual_integrity` |

Keep `--read-policy natural`, `--privacy-audit-mode full`, and
`--audit-strictness balanced` for this mapping. The manuscript uses GPT-5.4 as
the auditor in its main outcome comparison: explicitly set
`EXTRACTION_LLM_MODEL=openai/gpt-5.4` with your corresponding credentials and
endpoint for C2–C4. Historical result-directory names are not authoritative
condition labels; inspect the run metadata.

The manuscript reports two-of-three majority-vote leakage and mean helpfulness
from independent outcome judges. Run `evaluate` separately for each model
below against the same committed actions, with `--sanitize-trajectory` and
`--no-cache`, retaining each judge's output files:

| Judge | `--judge-model` |
| --- | --- |
| OpenAI | `openai/gpt-5` |
| Anthropic | `anthropic/claude-sonnet-4-5-20250929` |
| Google | `gemini/gemini-2.5-pro` |

Supply the correct private credential for each judge. Use
`--judge-base-url none` for the provider's normal endpoint or an explicit
endpoint for your deployment. Model aliases and provider availability must be
checked for the experiment date; sharing a provider prefix does not imply two
models use the same gateway.

Once all three per-task judgment files are present, the ensemble API can write
the combined leakage judgments without any additional model calls:

```bash
uv run python - <<'PY'
from pathlib import Path
from privacylens_live.base.judge_ensemble import write_vote_file

summary = write_vote_file(Path("results/audit_smoke"))
print(summary)
PY
```

This creates `judgments_vote3.json`. Missing or unusable votes and ties can
remain unresolved; report their count and the resolved-task denominator. The
majority-vote file does not calculate helpfulness: the aggregation API in
`privacylens_live.base.vote_tables.load_cell` computes the per-task mean across
available judge scores and then averages those task means. Check judge
completeness before comparing conditions.

These commands describe the available execution and outcome-evaluation paths.
The manuscript's full experiment additionally fixes the 389-task selection,
model settings, and trajectory-level protected-item matching protocol. A
newly generated task subset, the coverage report from `trajectory-evaluate`,
and a single-judge smoke score are not substitutes for those artifacts. The
public source snapshot does not yet bundle the cleared task manifest and
complete paper reproduction artifacts.

## 8. Record provenance and stop services

Before a paper comparison, retain the source commit, lockfile, dataset source
revision/checksum and terms, generated task-name list, conversion/drop manifests,
prompt version, model deployment versions, service image digests, command lines
with secrets removed, retry policy, and judge configuration. Record retrieval
policy, clarification limit, analyzer settings, audit policy/strictness, and
tool-call sequencing. The result metadata captures several of these fields but
does not replace an experiment manifest.

The six application images in Compose are pinned to full digests recovered
from local Docker metadata, matching the digest prefixes in the manuscript's
Appendix B. Database images still use major-version tags, and MCP builds
resolve dependency ranges from `requirements.txt`. Those dependencies and
provider models can change without a code change. Record the exact images,
resolved dependencies, and deployments used; the application pins and a
matching `uv.lock` alone do not make a live experiment deterministic.
Ground-truth sensitive items belong to evaluation, not default execution
prompts. Report prompt and audit ablations as distinct conditions.

Stop the Compose services when finished:

```bash
uv run python -m agentprivarena teardown
```

This keeps the named data volumes for later use. Review
[public-release.md](public-release.md) before exporting source or sharing any
experiment artifact.
