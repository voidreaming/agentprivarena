# AgentPrivArena

**Evaluating and Auditing Real-world AI Agent Privacy**

Shouju Wang and Haopeng Zhang

[Project website](https://voidreaming.github.io/agentprivarena/) ·
[Paper](https://voidreaming.github.io/agentprivarena/assets/agentprivarena.pdf) ·
[OpenReview](https://openreview.net/forum?id=zxllNfqsYS)

AgentPrivArena is a research fork of the [OpenHands Software Agent SDK](https://github.com/OpenHands/software-agent-sdk).
It runs agents against self-hosted services through real MCP tools and evaluates
privacy across the execution trajectory and the final action.
**AgentPrivAudit** adds runtime auditing at the read and write boundaries, with
configurable contextual-integrity, PII, and data-minimization policies.

The repository contains the full SDK, built-in tools, workspace implementations,
and agent server alongside the research harness. Python imports remain
`openhands.*`. Use `python -m agentprivarena` for the research CLI; the historical
`python -m privacylens_live` command remains supported for existing experiments.

## Start here

- [Research framework and tool services](privacylens_live/README.md)
- [Setup and reproduction guide](privacylens_live/docs/reproduction.md)
- [Public release and private-data policy](privacylens_live/docs/public-release.md)
- [Upstream SDK overview and API examples](docs/openhands-sdk.md)
- [Development guide](DEVELOPMENT.md)
- [Project website maintenance](docs/website.md)
- [Third-party notices and dataset provenance](THIRD_PARTY_NOTICES.md)

## Installation

Use Python 3.13 and uv 0.8.13 or newer. From the repository root:

```bash
git clone https://github.com/voidreaming/agentprivarena.git
cd agentprivarena
uv sync --frozen --dev
uv run python -m agentprivarena --help
uv run pytest privacylens_live/tests tests/privacylens_live tests/sdk/privacy
```

Development setup is also available through `make build`. Live evaluations need
Docker Compose, a locally built agent-server image, provider credentials, and
benchmark inputs obtained under their source licenses. Follow the
[reproduction guide](privacylens_live/docs/reproduction.md) before starting services.
The source release excludes raw experiment outputs, human responses, credentials,
and generated benchmark payloads.

## Repository layout

| Path | Purpose |
| --- | --- |
| `agentprivarena/` | Public CLI entry point, sharing the existing harness implementation |
| `openhands-sdk/` | Agent APIs, execution hooks, and AgentPrivAudit implementation in `openhands.sdk.privacy` |
| `openhands-tools/` | Built-in SDK tools and presets |
| `openhands-workspace/` | Local and remote workspace implementations |
| `openhands-agent-server/` | Container-side runtime and REST APIs |
| `privacylens_live/runner/` | Task execution, agent profiles, and result collection |
| `privacylens_live/mcp_servers/` | Service adapters for the local benchmark environment |
| `privacylens_live/tasks/` | Task conversion, seed validation, and readiness checks |
| `privacylens_live/base/` | Outcome and trajectory evaluation, aggregation, and human-evaluation tooling |
| `tests/`, `privacylens_live/tests/` | SDK and research regression tests |
| `examples/` | Upstream SDK usage examples |
| `dist/` | Existing project website, paper, and published figures |

## Research scope

The paper introduces AgentPrivArena for live privacy evaluation and
AgentPrivAudit for runtime information-flow auditing. Its main experiment uses
389 executable tasks converted from PrivacyLens and compares no mitigation,
privacy prompting, and three audit criteria. The source also contains additional
experimental variants and an MPCI-Bench adapter; their presence does not imply
they are part of the paper's main comparison.

Read observations reach the executor unchanged. The audit extracts a flow
inventory and can provide guidance after reads; the write boundary reviews
proposed transmissions before execution. Outcome leakage, helpfulness, exposure,
extraction recall, and flow dispositions answer different questions and must
retain their respective denominators.

The reproduction guide distinguishes the paper's conditions from code defaults.
Exact reproduction additionally needs an approved task manifest, model
configuration, and judge settings. The six application images are pinned to full
digests recovered from the local images and matched to the manuscript's prefixes;
database and model-provider versions also affect reproduction.

## Preparing a public snapshot

```bash
uv run python scripts/export_public_release.py --list
uv run python scripts/export_public_release.py --output .agent_tmp/agentprivarena-public
```

The exporter reads `public-release.toml`, copies the selected working-tree files,
and writes SHA-256 checksums. It excludes Git history and private research
material. Review and secret-scan the snapshot before creating the public
repository; changing ignore rules does not remove files from existing history.

## Citation and attribution

Paper title: *AgentPrivArena: Evaluating and Auditing Real-world AI Agent Privacy*.

The [author's publication list](https://hpzhang94.github.io/research/) names
Shouju Wang and Haopeng Zhang. See [CITATION.cff](CITATION.cff) for citation
metadata. The project links to its OpenReview record; no arXiv identifier or
conference acceptance is inferred.

This project builds on OpenHands and adapts PrivacyLens evaluation prompts and
task-conversion ideas. The existing [MIT license](LICENSE) and upstream notices
are retained. Dataset and image terms are separate from the code license; see
[third-party notices](THIRD_PARTY_NOTICES.md).
