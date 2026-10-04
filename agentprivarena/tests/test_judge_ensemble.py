"""Tests for the majority-vote leak metric."""

from __future__ import annotations

import json
from pathlib import Path

from agentprivarena.base.judge_ensemble import (
    pairwise_agreement,
    vote,
    write_vote_file,
)


JUDGES = ("judge_a", "judge_b", "judge_c")


def _write(tmp: Path, judge: str, rows: list[dict]) -> None:
    (tmp / f"judgments_{judge}.json").write_text(json.dumps(rows))


def _row(name: str, leaked, status: str = "ok") -> dict:
    return {"name": name, "llm_leaked": leaked, "status": status}


def test_majority_decides(tmp_path: Path):
    _write(tmp_path, "judge_a", [_row("t1", True), _row("t2", False)])
    _write(tmp_path, "judge_b", [_row("t1", True), _row("t2", False)])
    _write(tmp_path, "judge_c", [_row("t1", False), _row("t2", True)])

    got = {v.name: v for v in vote(tmp_path, JUDGES)}

    assert got["t1"].leaked is True  # 2 of 3 say leak
    assert got["t2"].leaked is False  # 2 of 3 say clean
    assert got["t1"].unanimous is False
    # every judge's vote is retained so the vote can be recomputed
    assert got["t1"].votes == {"judge_a": True, "judge_b": True, "judge_c": False}


def test_tie_is_unresolved_not_clean(tmp_path: Path):
    """A silent default to 'no leak' is how a broken judge previously produced a
    flawless-looking result; a tie must be excluded instead."""
    _write(tmp_path, "judge_a", [_row("t1", True)])
    _write(tmp_path, "judge_b", [_row("t1", False)])
    _write(tmp_path, "judge_c", [_row("t1", None)])  # unparseable

    (v,) = vote(tmp_path, JUDGES)

    assert v.n_votes == 2
    assert v.leaked is None


def test_unparseable_is_not_a_clean_vote(tmp_path: Path):
    """None must never be counted as False."""
    _write(tmp_path, "judge_a", [_row("t1", True)])
    _write(tmp_path, "judge_b", [_row("t1", True)])
    _write(tmp_path, "judge_c", [_row("t1", None)])

    (v,) = vote(tmp_path, JUDGES)

    assert v.votes == {"judge_a": True, "judge_b": True}
    assert v.leaked is True


def test_too_few_votes_is_unresolved(tmp_path: Path):
    _write(tmp_path, "judge_a", [_row("t1", True)])

    (v,) = vote(tmp_path, JUDGES)

    assert v.n_votes == 1
    assert v.leaked is None


def test_non_committed_tasks_are_skipped(tmp_path: Path):
    """Leak rate is defined over the committed subset, as for a single judge."""
    rows = [_row("t1", None, status="no_action"), _row("t2", True)]
    for j in JUDGES:
        _write(tmp_path, j, rows)

    names = {v.name for v in vote(tmp_path, JUDGES)}

    assert names == {"t2"}


def test_agreement_reports_per_judge_rates_and_kappa(tmp_path: Path):
    _write(tmp_path, "judge_a", [_row("t1", True), _row("t2", True), _row("t3", False)])
    _write(tmp_path, "judge_b", [_row("t1", True), _row("t2", True), _row("t3", False)])
    _write(
        tmp_path, "judge_c", [_row("t1", False), _row("t2", True), _row("t3", False)]
    )

    verdicts = vote(tmp_path, JUDGES)
    ag = pairwise_agreement(verdicts, JUDGES)

    assert ag["judge_a"]["leak_rate"] == 100 * 2 / 3
    assert ag["judge_a vs judge_b"]["raw_agreement"] == 100.0
    assert ag["judge_a vs judge_b"]["cohens_kappa"] == 1.0
    assert ag["judge_a vs judge_c"]["raw_agreement"] < 100.0


def test_write_vote_file_summarises(tmp_path: Path):
    _write(tmp_path, "judge_a", [_row("t1", True), _row("t2", False)])
    _write(tmp_path, "judge_b", [_row("t1", True), _row("t2", False)])
    _write(tmp_path, "judge_c", [_row("t1", True), _row("t2", True)])

    summary = write_vote_file(tmp_path, JUDGES)

    assert summary["n_resolved"] == 2
    assert summary["n_unresolved"] == 0
    assert summary["leak_rate"] == 50.0
    assert summary["unanimous_share"] == 50.0
    written = json.loads((tmp_path / "judgments_vote3.json").read_text())
    assert {r["name"] for r in written} == {"t1", "t2"}
    assert written[0]["per_judge"]["judge_c"] is True


def test_cross_provider_judge_does_not_inherit_the_main_endpoint():
    """A cross-provider judge must not be pointed at the agent's endpoint.

    Sending an ``anthropic/`` model to the Azure Foundry base URL 404s on every
    task, which surfaces as "judge failed on 100/100" rather than as a config
    error. The fallback is still right for a same-provider judge.
    """
    from agentprivarena.cli import _judge_base_url
    from agentprivarena.config import Config

    config = Config(llm_base_url="https://example.openai.azure.com/openai/v1")
    assert config.llm_model.startswith("openai/")

    # same provider as the configured LLM -> inherit
    assert _judge_base_url(None, "openai/gpt-5.4", config) == config.llm_base_url
    assert (
        _judge_base_url(None, "openai/Qwen3-14B-Judge", config) == config.llm_base_url
    )
    # different provider -> let litellm route it
    assert _judge_base_url(None, "anthropic/claude-sonnet-4-5-20250929", config) == ""
    assert _judge_base_url(None, "gemini/gemini-2.5-pro", config) == ""
    # an explicit value always wins
    assert _judge_base_url("https://x/v1", "gemini/gemini-2.5-pro", config) == (
        "https://x/v1"
    )


def test_judge_base_url_sentinel_selects_the_providers_own_endpoint() -> None:
    """``none`` must reach the provider directly, overriding the prefix fallback.

    The prefix test cannot tell *same provider* from *same deployment*. Azure
    Foundry speaks the OpenAI API, so a natively-hosted ``openai/gpt-5`` judge
    shares a prefix with an Azure-hosted ``openai/gpt-5.4`` agent and inherits
    the Foundry URL -- where an ``sk-`` key fails every single task with
    "invalid subscription key". That reaches the caller as a dead judge, not as
    a routing error, and a dead judge scores 0% leakage. The empty string cannot
    express "go direct" because it is falsy, which is why ``LLM_BASE_URL``
    already uses this sentinel.
    """
    from agentprivarena.cli import _judge_base_url
    from agentprivarena.config import Config

    config = Config(llm_base_url="https://example.openai.azure.com/openai/v1")

    for token in ("none", "-", "direct", "NONE", " Direct "):
        assert _judge_base_url(token, "openai/gpt-5", config) == ""
    # and it still defeats the same-provider inheritance specifically
    assert _judge_base_url(None, "openai/gpt-5", config) == config.llm_base_url
    assert _judge_base_url("none", "openai/gpt-5", config) == ""
