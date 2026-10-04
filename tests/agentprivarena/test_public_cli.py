"""The AgentPrivArena module exposes the research evaluation CLI."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [((), "AgentPrivArena"), (("run",), "--prompt-variant")],
    ids=["command-help", "run-help"],
)
def test_public_cli_help(arguments: tuple[str, ...], expected: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "agentprivarena", *arguments, "--help"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert expected in result.stdout
