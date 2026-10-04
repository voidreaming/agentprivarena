<div align="center">

<h1>AgentPrivArena</h1>
<h3>Evaluating and Auditing Real-world AI Agent Privacy</h3>

<p>
  <a href="https://voidreaming.github.io/agentprivarena/assets/agentprivarena.pdf"><img src="https://img.shields.io/badge/Paper-PDF-b31b1b" alt="Read the paper"></a>
  <a href="https://voidreaming.github.io/agentprivarena/"><img src="https://img.shields.io/badge/Project-Website-2563eb" alt="Project website"></a>
  <a href="agentprivarena/docs/reproduction.md"><img src="https://img.shields.io/badge/Guide-Reproduction-15803d" alt="Reproduction guide"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/Code-MIT-64748b" alt="Code license: MIT"></a>
</p>

<p><b>389 executable tasks · 6 self-hosted services · 28 MCP tools · 5 executor models</b></p>

<p>
  <a href="#what-agentprivarena-contributes">Highlights</a> ·
  <a href="#real-service-environment">Services</a> ·
  <a href="#agentprivaudit-a-modular-runtime-auditor">Method</a> ·
  <a href="#results">Results</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#repository-map">Code &amp; docs</a>
</p>

</div>

**AgentPrivArena brings privacy evaluation to live agent workflows:** real MCP
services, trajectory-level privacy metrics, and **AgentPrivAudit**—a runtime
auditor that checks information flows after reads and before writes under
configurable privacy policies.

<p align="center">
  <img src="dist/assets/figures/figure-1-overview.png" width="100%" alt="AgentPrivArena converts scenarios into seeded services. An agent discovers records through real MCP calls, accumulates observations, and commits an action.">
</p>

*Figure 1 from the paper. Sensitive records live in services, not in a supplied
trajectory: the agent must discover them through its own tool calls.*

## What AgentPrivArena contributes

### Real service environment

Run privacy-sensitive tasks against six self-hosted applications through real
MCP tools. Synthetic records are seeded into service state, so agents discover
information and act on live applications.

| Service | Role | Version |
| --- | --- | ---: |
| [BookStack](https://www.bookstackapp.com) | Knowledge base | `26.03.3` |
| [Mattermost](https://mattermost.com) | Direct messages | `11.6.0` |
| [Rocket.Chat](https://www.rocket.chat) | Team chat | `6.13` |
| [Mailpit](https://mailpit.axllent.org) | Email | `1.29.7` |
| [GoToSocial](https://gotosocial.org) | Social media | `0.21.2` |
| [Radicale](https://radicale.org) | Calendar | `3.6.1` |

Together, these services expose **28 MCP tools: 13 discovery, 8 access, and
7 write tools**. These classes describe tool intent, not a privacy boundary:
search results can themselves contain sensitive record content.

Versions come from [Table 2 of the paper](https://voidreaming.github.io/agentprivarena/assets/agentprivarena.pdf#page=3).
[Docker Compose](agentprivarena/docker-compose.yml) pins all six images by
**full SHA-256 digest**. See the [MCP adapters](agentprivarena/mcp_servers/) and
[tool inventory](https://voidreaming.github.io/agentprivarena/assets/agentprivarena.pdf#page=12)
for implementation and provenance.

### Trajectory-level privacy evaluation

Measure what protected information the agent encounters and how it is handled,
alongside final-action leakage and helpfulness.

| Metric | Definition |
| --- | :---: |
| Exposure rate | $\frac{E}{R}$ |
| Extraction recall | $\frac{X}{E}$ |
| Disposition share | $\frac{F_d}{F}$ |

Counts are pooled over task executions within a condition: $R$ reference
protected items, $E$ exposed items, and $X$ exposed items represented by the
auditor. $F$ counts extracted flows, and $F_d$ those assigned
$d\in\lbrace\mathrm{PASS},\mathrm{ABSTRACT},\mathrm{BLOCK}\rbrace$.
An item may produce multiple flows; the denominators are deliberately distinct.

### AgentPrivAudit: a modular runtime auditor

Audit information flows after reads and before writes. Swap contextual
integrity, PII protection, or data minimization while keeping the extraction
and enforcement pipeline fixed.

<a href="dist/assets/figures/figure-2-audit-framework.png"><img align="left" src="dist/assets/figures/figure-2-audit-framework.png" width="340" alt="AgentPrivAudit extracts flows after reads and reviews proposed writes using the accumulated inventory."></a>

**After reads**<br>
Extract potentially sensitive flows into a shared inventory and provide audit feedback.

**Before writes**<br>
Review proposed transmissions under the selected privacy policy. Feedback lets the agent revise its action.

**PASS** — permit the flow.<br>
**ABSTRACT** — request less-specific content.<br>
**BLOCK** — reject the flow.

<br clear="both">

| Criterion | Audit question |
| --- | --- |
| PII | Does this identify someone other than the user? |
| Data minimization | Is this information necessary for the task? |
| Contextual integrity | Is sharing appropriate for this recipient and context? |

Raw tool observations still reach the executor unchanged. This is runtime
auditing, **not a mechanism that hides sensitive context from the model**.

## Results

> **46.8% → 17.8% leakage** (−29.0 percentage points), with average helpfulness
> at 2.60 → 2.59 / 3 under contextual-integrity auditing.

Reported results from [Table 4 of the paper](https://voidreaming.github.io/agentprivarena/assets/agentprivarena.pdf#page=6),
pooled across five executors. Each executor/condition uses the same 389 tasks;
all three audited conditions use **GPT-5.4 as the auditor**.

<a href="dist/assets/figures/figure-4-leakage-results.png"><img align="right" src="dist/assets/figures/figure-4-leakage-results.png" width="380" alt="Figure 4. Leakage across five executors under C0 through C4. The range narrows from 25.4 percentage points without mitigation to 3.1 with contextual-integrity auditing."></a>

| Condition | Leakage ↓ | Helpfulness ↑ |
| --- | ---: | ---: |
| C0 · No mitigation | 46.8% | 2.60 |
| C1 · Privacy prompt | 43.0% | **2.64** |
| C2 · PII audit | 26.5% | 2.55 |
| C3 · Data min. audit | 19.7% | 2.59 |
| C4 · CI audit | **17.8%** | 2.59 |

<p><sub>Bold marks the best value per metric.<br>
C2–C4 use AgentPrivAudit; CI = contextual integrity.</sub></p>

<br clear="both">

**Why the mechanism matters.** In the separate criterion ablation, stating the
same three privacy criteria as instructions yields **40.7%** average leakage;
enforcing them through AgentPrivAudit yields **21.7%** (Figure 5). Merely naming
a privacy principle is not equivalent to checking information flows at runtime.

<details>
<summary><b>Beyond final-action leakage: trajectory diagnostics</b></summary>

<p align="center">
  <img src="dist/assets/figures/figure-3-trajectory-metrics.png" width="100%" alt="Exposure, extraction recall, and pass/abstract/block dispositions are measured separately because they count protected items, exposed items, and extracted flows respectively.">
</p>

Under contextual integrity, extraction recall is **70.5%**: the inventory misses
29.5% of exposed protected items. Among extracted flows, **38% pass, 52% are
abstracted, and 10% are blocked**. These diagnostics locate failure points that
a final-action score alone cannot explain (Figure 3, §5.4).

</details>

<details>
<summary><b>Evaluation scope and limitations</b></summary>

- The main experiment covers **389 of 493 source scenarios**. The remaining 104
  could not be instantiated because of service-wrapper and seed-mapping
  constraints; this is not an evaluation of all 493 scenarios.
- Leakage requires agreement from at least **two of three LLM judges**.
  Helpfulness is scored on a **0–3** scale, averaged across judges and then
  across tasks producing an action. The paper does not report completed human
  verification of these judgments.
- Contextual-integrity auditing still leaves **16–19% leakage** across
  executors. Average utility can also hide individual over-abstraction failures.
- The audit is **fail-open**, not an adversarial security guarantee. Its C4
  configuration uses **2.07×** the baseline's total tokens (Table 8).
- These are the paper's reported measurements, not a new reproduction run of
  this source release. Exact replication needs the approved task subset,
  explicit executor/auditor/judge settings, and matching service versions.

</details>

## Quick start

Use **Python 3.13** and **uv ≥ 0.8.13**. Install the code and run the offline
research tests without starting services or calling model APIs:

```bash
git clone https://github.com/voidreaming/agentprivarena.git
cd agentprivarena
uv sync --frozen --dev
uv run python -m agentprivarena --help
uv run pytest agentprivarena/tests tests/agentprivarena tests/sdk/privacy
```

For live experiments, follow the [reproduction guide](agentprivarena/docs/reproduction.md):
configure private credentials, build the agent image, start the Docker Compose
services, prepare licensed benchmark inputs, then run a small baseline/audit
comparison. The guide maps **C0–C4** to CLI settings and distinguishes paper
conditions from runtime defaults.

**Release contents:** code, configuration templates, evaluation tooling, and
published paper figures. Credentials, generated benchmark payloads, raw
trajectories/results, human-evaluation responses, and private research notes are
not included. Benchmark inputs require separate acquisition and preparation;
this is not a bundled dataset release.

## Repository map

**`agentprivarena/` runs and evaluates experiments; `packages/` supplies the
agent runtime and reusable AgentPrivAudit implementation.**

```text
.
├── agentprivarena/                      Research environment and evaluation
│   ├── cli.py                          setup / generate / run / evaluate commands
│   ├── config.py                       Service, model, and path settings
│   ├── bootstrap.py                    Service provisioning and local tokens
│   ├── docker-compose.yml              Applications, databases, and MCP servers
│   ├── tasks/                          Scenario conversion and readiness checks
│   ├── mcp_servers/                    Service-specific MCP tools
│   ├── runner/                         Task execution, prompts, and trace collection
│   ├── base/                           Service seeding, evaluators, and statistics
│   ├── benchmarks/                     Optional benchmark adapters
│   ├── tests/                          Research-module tests
│   └── docs/                           Reproduction and release guides
├── packages/                           Reusable agent infrastructure
│   ├── sdk/openhands/sdk/
│   │   ├── privacy/                    AgentPrivAudit flows, policies, and decisions
│   │   └── agent/tool_audit.py          Agent-loop audit extension points
│   ├── tools/                          Built-in agent tools
│   ├── workspace/                      Container and remote workspace backends
│   └── agent-server/openhands/agent_server/
│       ├── api.py                      Container-side execution API
│       └── docker/                     Agent image Dockerfile and build helpers
├── tests/                              Runtime and release regression tests
├── examples/                           SDK usage examples
├── scripts/                            Development checks and public source export
├── docs/                               SDK and website documentation
├── dist/                               Project website, paper, and original figures
├── public-release.toml                 Public source-export allowlist
├── pyproject.toml                      Workspace, dependencies, and test settings
└── uv.lock                             Locked dependency versions
```

| To understand… | Start with… |
| --- | --- |
| How a task is seeded, executed, and recorded | [agent_runner.py](agentprivarena/runner/agent_runner.py) |
| How AgentPrivAudit controls read/write boundaries | [audit.py](packages/sdk/openhands/sdk/privacy/audit.py) |
| How flows are extracted and privacy criteria applied | [llm_analyzer.py](packages/sdk/openhands/sdk/privacy/llm_analyzer.py) |
| How leakage and helpfulness are scored | [evaluator.py](agentprivarena/base/evaluator.py) |

The SDK's Python namespace remains `openhands.*`. The optional MPCI-Bench
adapter and additional experimental variants are outside the main 389-task
comparison.

## Documentation

| Start here | What you will find |
| --- | --- |
| [Research framework](agentprivarena/README.md) | CLI, service adapters, and evaluation components |
| [Reproduction guide](agentprivarena/docs/reproduction.md) | Setup, task preparation, execution, and paper-condition mapping |
| [Development guide](DEVELOPMENT.md) | Environment setup, testing, and contribution workflow |
| [Public release policy](agentprivarena/docs/public-release.md) | Private-data boundaries and allowlisted source exports |
| [SDK reference](docs/openhands-sdk.md) | Underlying agent APIs and usage examples |

## Citation and acknowledgments

If you use AgentPrivArena or AgentPrivAudit, please cite
*AgentPrivArena: Evaluating and Auditing Real-world AI Agent Privacy*.
Use [CITATION.cff](CITATION.cff) or GitHub's **Cite this repository** button for
citation metadata; the manuscript is also available on
[OpenReview](https://openreview.net/forum?id=zxllNfqsYS).

AgentPrivArena builds on the [OpenHands Software Agent SDK](https://github.com/OpenHands/software-agent-sdk)
for agent execution and adapts [PrivacyLens](https://github.com/SALT-NLP/PrivacyLens)
scenarios and evaluation prompts. Upstream copyright and license notices are
preserved. Code is distributed under the [MIT license](LICENSE); dataset,
paper, and other third-party asset terms are separate. See
[third-party notices](THIRD_PARTY_NOTICES.md) and
[figure provenance](dist/assets/figures/manifest.json).
