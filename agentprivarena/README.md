# AgentPrivArena evaluation harness

AgentPrivArena evaluates how language-model agents handle private information
while completing tasks through live services. It converts benchmark trajectories
into service records, runs an OpenHands agent against MCP tools, and evaluates
both outgoing actions and the agent's execution trajectory.

This harness accompanies **AgentPrivArena: Evaluating and Auditing Real-world
AI Agent Privacy**. The runtime auditing module is **AgentPrivAudit**. Both are
built on the full OpenHands SDK fork included in this repository; the privacy
audit hooks run inside its agent server.

The implementation package and CLI use the `agentprivarena` name; run
`python -m agentprivarena` from the repository root. PrivacyLens is the
upstream source benchmark, not the name of this framework.

## Start here

Run commands from the repository root. Use Python 3.13 and uv 0.8.13 or newer:

```bash
make build
uv sync --frozen --group dev
uv run python -m agentprivarena --help
uv run pytest agentprivarena/tests tests/agentprivarena tests/sdk/privacy \
  tests/sdk/agent/test_privacy_extraction.py
```

The focused unit suite uses mocks and does not require live services or model
credentials. Full experiments additionally require Docker Engine with Compose,
benchmark inputs obtained under their original terms, an execution-model
endpoint, and an evaluation-model endpoint. Audit conditions need an audit
model as well.

Follow the [reproduction guide](docs/reproduction.md) to configure
[.env.example](.env.example), build the local agent image, prepare task data,
bootstrap services, and run a small experiment before scaling up. Fresh
BookStack instances require a manually provisioned API token.

## Code map

| Path | Responsibility |
| --- | --- |
| [tasks/](tasks/) | PrivacyLens conversion, seed manifests, static verification, live readiness checks |
| [benchmarks/](benchmarks/) | Additional dataset adapters, including MPCI-Bench |
| [runner/](runner/) | Task contract, agent profiles, prompts, execution, events, and result persistence |
| [mcp_servers/](mcp_servers/) | MCP wrappers for service APIs and local image artifacts |
| [base/](base/) | Service seeding, outcome and trajectory evaluation, and statistical analysis |
| [docker-compose.yml](docker-compose.yml) | Local service stack and MCP containers |
| [tests/](tests/) | Focused platform unit tests |
| [../packages/sdk/openhands/sdk/privacy/](../packages/sdk/openhands/sdk/privacy/) | Reusable privacy analysis and audit hooks |
| [../packages/workspace/](../packages/workspace/) | Docker workspace implementation |

The service stack provides BookStack for knowledge pages, Mattermost for direct
messages, RocketChat for chat channels, Mailpit for captured email, GoToSocial
for social posts, and Radicale for calendar data. The `google_drive` MCP backend
serves local benchmark file/image artifacts; it does not connect to a Google
account.

For each task, the runner seeds the required services, connects an agent inside
a Docker workspace to their MCP servers, collects tool calls and observations,
and writes per-task results. Evaluation then measures leakage and helpfulness,
routing, retrieval coverage, and audit behavior. These stages can be inspected
and tested separately. AgentPrivAudit attaches at two boundaries: it extracts
information flows after content reads, then judges proposed outgoing writes
under a configurable policy. PASS permits the write; ABSTRACT and BLOCK can
reject it and let the agent recompose. The supported criteria are contextual
integrity, PII protection, and data minimization.

## Experimental conditions

The manuscript's main conditions are C0 (no mitigation), C1 (privacy-conscious
prompt), and C2–C4 (AgentPrivAudit under PII, data minimization, and contextual
integrity). These condition labels are distinct from the historical L0–L3
variant names. The CLI also exposes additional prompt and audit ablations.
The [reproduction guide](docs/reproduction.md#7-map-the-manuscripts-conditions-and-outcome-protocol)
maps C0–C4 to CLI flags and describes the three-judge outcome protocol. See
[prompt_builder.py](runner/prompt_builder.py) and `run --help` for all prompt
definitions and additional ablation flags.

| Variant | Mechanism |
| --- | --- |
| `baseline` | Stock SDK system prompt with neutral task-completion guidance |
| `reflect_freeform` | Baseline plus a generic privacy-reflection instruction |
| `contextual_integrity`, `data_minimization`, `pii_redaction` | Matched prompt-only arms stating different privacy criteria |
| `air_gap` | Prompt-only extract-then-compose instruction |
| `privacy_enhanced` | Privacy-conscious system-prompt framing |
| `ci_reasoning` | Privacy-conscious framing plus structured information-flow reasoning |
| `ci_audit` | External read-boundary guidance and a write-time judge |
| `ci_audit_flows` | External audit with structured flow guidance |
| `ci_audit_plan` | Structured audit guidance plus an explicit write-plan instruction |
| `ci_audit_contextual` | External audit with audience-aware share decisions and suggested safe renderings |

The four `ci_audit*` variants require `--enable-privacy-analyzer`. Leave that
flag off for prompt-only comparisons. `privacy_enhanced` is a framing control;
use `contextual_integrity` for the criterion-as-prompt arm.

The audit design leaves raw read observations visible to the execution model.
It steers behavior and can reject outgoing tool calls; it does not provide
strict information isolation. Likewise, `air_gap` is a prompt condition, not
an isolated execution environment. Tool serialization, retrieval policy, and
judge configuration are additional experimental variables that must be held
constant or reported as separate conditions.

## Data, outputs, and publication

Benchmark sources, derived task collections, images, raw trajectories, model
credentials, and historical experiment directories are excluded from the
default public snapshot. Acquire dataset inputs from their original publishers
and verify their redistribution terms before sharing derivatives. The code
license does not grant rights to third-party datasets or service images.

Runs produce `<task>.json`, `<task>.events.json`, and `_summary.json` in the
chosen results directory. These files can contain full source observations,
private facts, prompts, and model responses. Publish reviewed aggregates and
documented provenance; keep raw artifacts private until individually cleared.

Use the [public-release guide](docs/public-release.md) for the export boundary
and remaining publication checks. [CITATION.cff](../CITATION.cff) records the
authors and paper title verified by the existing project website and author
publication list; no arXiv identifier or acceptance status is inferred.
The commands in this repository are not a claim of exact paper reproduction.
