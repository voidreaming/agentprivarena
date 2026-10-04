# Development Guide

AgentPrivArena retains the OpenHands workspace packages and Python namespaces.
The research harness and CLI live in the `agentprivarena` package.

## Setup

From a clone of this repository, run:

```bash
make build
```

For the locked environment used by a source release, run `uv sync --frozen --dev`.
An exported snapshot has no Git history; initialize a new local repository
before installing pre-commit hooks. See the
[reproduction guide](agentprivarena/docs/reproduction.md) for Docker and model
configuration.

## Code Quality

```bash
make format                              # Format code
make lint                                # Lint code
uv run pre-commit run --all-files        # Run all checks
```

Pre-commit hooks run automatically on commit with type checking and linting.

## Testing

```bash
uv run pytest                            # Upstream SDK/workspace/tools/server tests
uv run pytest tests/sdk/                 # SDK tests only
uv run pytest tests/tools/               # Tools tests only
uv run pytest agentprivarena/tests tests/agentprivarena tests/sdk/privacy \
  tests/sdk/agent/test_privacy_extraction.py  # Offline research regression suite
```

## Project Structure

```
agentprivarena/
├── openhands-sdk/          # Core SDK package
├── openhands-tools/        # Built-in tools
├── openhands-workspace/    # Workspace management
├── openhands-agent-server/ # Agent server
├── examples/               # Usage examples
├── agentprivarena/         # Research harness, MCP services, and evaluators
└── tests/                  # Test suites
```

## Contributing

1. Create a new branch
2. Make your changes
3. Run tests and checks
4. Push and create a pull request

Keep credentials, research outputs, annotations, and draft/rebuttal material
private. Review the [public export policy](agentprivarena/docs/public-release.md)
before preparing a release. Preserve upstream license notices and distinguish
paper conditions from new experiment variants.

The [upstream OpenHands community](https://openhands.dev/joinslack) supports the
underlying SDK; research-specific changes belong in this fork.
