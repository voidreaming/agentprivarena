---
title: AgentPrivArena Tests
description: Research regression tests and inherited runtime test suites.
---

# AgentPrivArena Tests

This directory contains research release checks and the inherited runtime test
suites. The focused research unit tests also live in `agentprivarena/tests/`.

## Test Structure

```
tests/
├── agentprivarena/  # Research CLI, configuration, and public release checks
├── agent_server/   # Agent-server unit tests
├── cross/          # Cross-package tests
├── integration/    # Integration tests
├── sdk/            # SDK unit tests
├── tools/          # Tools unit tests
└── workspace/      # Workspace unit tests
```

## Test Categories

### Integration Tests (`integration`)

End-to-end tests cover large parts of the code base and may require model
credentials or external services. They are not run by the offline research CI
workflow. Some behavior tests deliberately clone a pinned upstream repository;
paths in those fixtures describe that upstream layout, not this checkout.

### Unit Tests (`cross`, `sdk`, `tools`)

Component-specific tests prevent regressions in core functionality. Runtime
sources live under `packages/sdk`, `packages/tools`, `packages/workspace`, and
`packages/agent-server`; their test directories retain the corresponding domain
names.

The [research CI workflow](../.github/workflows/agentprivarena-tests.yml) runs the
offline research and privacy regression suite on pushes to `main` and relevant
pull requests, including changes under `packages/`. Other inherited runtime
tests can be run locally; see the [development guide](../DEVELOPMENT.md).
