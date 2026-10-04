"""The public and legacy module commands expose the same evaluation CLI."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [((), "AgentPrivArena"), (("run",), "--prompt-variant")],
    ids=["command-help", "run-help"],
)
def test_public_cli_preserves_legacy_help(
    arguments: tuple[str, ...], expected: str
) -> None:
    outputs = []
    for module in ("agentprivarena", "privacylens_live"):
        result = subprocess.run(
            [sys.executable, "-m", module, *arguments, "--help"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        assert expected in result.stdout
        outputs.append(result.stdout)

    assert outputs[0] == outputs[1]
