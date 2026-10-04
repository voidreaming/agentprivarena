# Publishing AgentPrivArena

AgentPrivArena publishes AgentPrivAudit, the live evaluation harness, the four
customized OpenHands packages, source tests, and usage examples. The research
package and CLI use `agentprivarena`; the underlying SDK keeps its `openhands.*`
Python namespaces. The root README introduces the paper;
`docs/openhands-sdk.md` preserves the upstream SDK overview and citation.

The canonical repository is
[voidreaming/agentprivarena](https://github.com/voidreaming/agentprivarena).
Its pre-existing project website, paper, and published figures remain under
`dist/`, with maintenance instructions in `docs/website.md`. The code import
preserves that website history; it does not import the SDK development history.
The exporter below includes the four published paper figures used in the README
and their provenance manifest, but omits the full website and paper PDF. The
figure manifest describes the original six-figure extraction; its source PDF
and the other figures remain available on the project website.

The public organization follows the useful separation of code, services,
evaluation, and documentation in
[OpenAgentSafety](https://github.com/Open-Agent-Safety/OpenAgentSafety).
AgentPrivArena uses its own research package name while preserving the SDK's
public Python interfaces.

## Public and private material

| Material | Release treatment |
| --- | --- |
| SDK, audit, runner, MCP, and evaluation source | Included by `public-release.toml` |
| Runtime templates, dependency manifests, unit fixtures, SDK examples | Included |
| Setup, reproduction, attribution, and these release instructions | Included |
| Published overview, audit, trajectory, and leakage figures with provenance metadata | Included; full website and paper PDF remain outside the source export |
| API keys, service tokens, populated `.env*`, deployment configuration | Private; only the reviewed `.env.example` is included |
| Raw experiment outputs, event logs, model completions, service databases | Private |
| Human-evaluation responses, annotator assignments, private manifests | Private; generic evaluation server/tooling remains public |
| Rebuttal notes, internal experiment notes, project memory, unpublished drafts | Private |
| Generated benchmark task/seed payloads and multimodal images | Excluded from this source snapshot; obtain inputs from their sources |
| Existing Git history, remotes, and inherited upstream workflow automation | Excluded |

The original working tree remains intact. Exclusion means the files are absent
from the exported directory, not deleted from local storage. The public release
includes upstream synthetic/recorded unit fixtures; it does not include local
research trajectories. Review any future fixture additions before export.

The code license does not replace dataset or image licenses. See
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) for PrivacyLens, MPCI-Bench,
and upstream code attribution. A future data release should have its own source
revision, conversion configuration, attribution, task-ID manifest, and terms.

## Export a reviewable snapshot

From the repository root:

```bash
uv run python scripts/export_public_release.py --list
uv run python scripts/export_public_release.py --output .agent_tmp/agentprivarena-public
```

The first command is a dry run. The second requires a new destination and copies
the selected **current working-tree contents**, including uncommitted source
edits. It does not stage, commit, push, create a remote, or include `.git`.
`PUBLIC_RELEASE_MANIFEST.json` records each exported file's SHA-256 checksum.
Review the outstanding source diffs as well as the file list.

The allowlist is a file-selection boundary, not a secret detector. It rejects
symlinks, Git metadata, non-example environment files, and missing selections.
Source files can still contain private strings. Run a secret scanner over the
snapshot, inspect its findings, and manually review documentation, URLs,
examples, and fixtures before publishing. Keep scan reports private.

Verify the exported snapshot from its own directory, using a fresh environment:

```bash
uv sync --frozen --dev
uv run python -m agentprivarena --help
uv run pytest agentprivarena/tests tests/agentprivarena tests/sdk/privacy \
  tests/sdk/agent/test_privacy_extraction.py
uv run python scripts/export_public_release.py
```

The included `AgentPrivArena tests` workflow uses public GitHub runners and
requires no provider credentials or running services. SDK-wide test commands
remain available through the upstream development guide. Live runs are
separate from this offline validation and require the reproduction setup.

## Credential history and deployment

The development history previously contained provisioned service credentials.
Removing a default or adding `.gitignore` does not erase old commits or revoke a
token. Rotate any reused credentials on the original services. Publish from the
reviewed snapshot as a source import; do not merge or push the original SDK
development branch, which would also publish its old contents. No history
rewriting or credential rotation is performed by the exporter.

The Compose environment is a local research sandbox with disposable service
accounts. Published ports bind to `127.0.0.1` by default. Container-to-container
traffic still uses the Docker network. `AGENTPRIVARENA_BIND_HOST` can override the
host binding for an intentionally configured deployment; these defaults are not
a production authentication setup. Keep real user data out of the sandbox.

## Paper metadata and reproducibility

The supplied manuscript is titled **AgentPrivArena: Evaluating and Auditing
Real-world AI Agent Privacy** and names **AgentPrivAudit** as the auditing
method. Although the manuscript is anonymous, the
[author's publication list](https://hpzhang94.github.io/research/) and existing
project website identify Shouju Wang and Haopeng Zhang. `CITATION.cff` records
that attribution and the public OpenReview record. An arXiv identifier or
conference acceptance is not inferred.

Before tagging the paper release, record the reviewed source revision, approved
task manifest, and exact model and judge configurations. The six application
images in Compose are pinned to full digests recovered from local Docker image
metadata; each matches the manuscript's twelve-character digest prefix.
Database images still use major-version tags, and the MCP build resolves version
ranges from `requirements.txt`. These remaining dependencies and provider-side
model changes prevent claiming an exact reconstruction from image pins alone.

Keep the paper-associated tag immutable and label subsequent fixes separately,
as OpenAgentSafety does for its paper snapshot and later corrections. Publish
aggregate results only after reviewing them separately; exporting source does
not authorize publication of raw outputs or annotations.
