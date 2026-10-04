"""Tests for the result-cell pre-flight validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentprivarena.base.cell_validation import (
    CellValidationError,
    describe_cell,
    find_duplicate_conditions,
    problems_for,
    require_healthy,
)


def _cell(
    tmp_path: Path,
    name: str,
    *,
    n: int = 10,
    analyzer: bool = True,
    flows_on: int | None = None,
    exec_model: str = "openai/gpt-5.4",
    audit_mode: str = "full",
    status: str = "ok",
    judges: tuple[str, ...] = (),
) -> Path:
    """Write a minimal results dir. ``flows_on`` = how many tasks get a flow."""
    d = tmp_path / name / "results"
    d.mkdir(parents=True)
    if flows_on is None:
        flows_on = n if analyzer else 0
    for i in range(n):
        (d / f"main{i}.json").write_text(
            json.dumps(
                {
                    "name": f"main{i}",
                    "status": status,
                    "prompt_variant": "ci_audit_contextual" if analyzer else "baseline",
                    "privacy_flows": (
                        [{"data_type": "x", "data_subject": "y"}]
                        if i < flows_on
                        else []
                    ),
                    "run_metadata": {
                        "execution_model": exec_model,
                        "audit_model": "openai/gpt-5.4" if analyzer else None,
                        "privacy_analyzer_enabled": analyzer,
                        "privacy_audit_mode": audit_mode if analyzer else None,
                        "agent_server_image": "agentprivarena-agent-server:local",
                    },
                }
            )
        )
        # An events sibling must not be counted as a result.
        (d / f"main{i}.events.json").write_text("[]")
    for j in judges:
        (d / f"judgments_{j}.json").write_text("[]")
    return d


def test_healthy_audit_cell_has_no_problems(tmp_path):
    d = _cell(tmp_path, "good")
    health = describe_cell(d)
    assert health.n_results == 10
    assert health.extraction_rate == 1.0
    assert health.signature is not None
    assert health.signature.analyzer_enabled is True
    assert problems_for(health) == []


def test_events_files_are_not_counted_as_results(tmp_path):
    """20 files on disk, 10 of which are event logs."""
    d = _cell(tmp_path, "c", n=10)
    assert len(list(d.glob("main*.json"))) == 20
    assert describe_cell(d).n_results == 10


def test_stale_stack_cell_is_rejected(tmp_path):
    """The Mistral incident: analyzer on, but extraction mostly missing.

    88% is the rate the stale-stack cells actually had, and it silently
    inflated every audit row it appeared in.
    """
    d = _cell(tmp_path, "stale", n=100, analyzer=True, flows_on=88)
    problems = problems_for(describe_cell(d))
    assert len(problems) == 1
    assert "analyzer was ON" in problems[0]
    assert "88%" in problems[0]


def test_broken_read_steer_arm_is_rejected(tmp_path):
    d = _cell(tmp_path, "read_steer", n=100, analyzer=True, flows_on=21)
    assert any("analyzer was ON" in p for p in problems_for(describe_cell(d)))


def test_prompt_only_cell_must_have_no_flows(tmp_path):
    """A prompt baseline that recorded flows is a mislabelled audit run."""
    clean = _cell(tmp_path, "prompt_clean", analyzer=False)
    assert problems_for(describe_cell(clean)) == []

    dirty = _cell(tmp_path, "prompt_dirty", analyzer=False, flows_on=5)
    problems = problems_for(describe_cell(dirty))
    assert any("analyzer was OFF" in p for p in problems)


def test_partial_cell_is_rejected(tmp_path):
    """The 26/100 PII cell."""
    d = _cell(tmp_path, "partial", n=26)
    problems = problems_for(describe_cell(d), expected_n=100)
    assert any("partial cell" in p for p in problems)


def test_missing_judges_are_reported(tmp_path):
    d = _cell(tmp_path, "c", judges=("gpt_5",))
    problems = problems_for(
        describe_cell(d), required_judges=("gpt_5", "gemini_2_5_pro")
    )
    assert any("gemini_2_5_pro" in p for p in problems)


def test_low_ok_rate_is_reported(tmp_path):
    d = _cell(tmp_path, "errored", n=10, status="error")
    assert any("tasks ok" in p for p in problems_for(describe_cell(d)))


def test_duplicate_condition_is_detected(tmp_path):
    """Same condition in two directories -- the root cause of all three bugs."""
    a = _cell(tmp_path, "policy_pilot_audit_ci_mistral", exec_model="m")
    b = _cell(tmp_path, "policy_pilot_fixed_audit_ci_mistral", exec_model="m")
    dups = find_duplicate_conditions([describe_cell(a), describe_cell(b)])
    assert len(dups) == 1
    assert sorted(next(iter(dups.values()))) == sorted([a, b])


def test_differing_conditions_are_not_duplicates(tmp_path):
    a = _cell(tmp_path, "read_steer", audit_mode="read_steer_only")
    b = _cell(tmp_path, "full", audit_mode="full")
    assert find_duplicate_conditions([describe_cell(a), describe_cell(b)]) == {}


def test_different_scopes_are_not_duplicates(tmp_path):
    """first100 and full389 share a signature but are different experiments.

    Keying duplicates on signature alone flagged 20 such pairs in the real
    tree, which would have trained everyone to ignore the check.
    """
    small = _cell(tmp_path, "first100", n=10)
    large = _cell(tmp_path, "full389", n=40)
    assert find_duplicate_conditions([describe_cell(small), describe_cell(large)]) == {}


def test_same_condition_same_tasks_is_a_duplicate(tmp_path):
    """The Mistral case: two dirs, same condition, same 100 tasks."""
    a = _cell(tmp_path, "stale_copy", n=10, flows_on=9)
    b = _cell(tmp_path, "fixed_copy", n=10, flows_on=10)
    dups = find_duplicate_conditions([describe_cell(a), describe_cell(b)])
    assert len(dups) == 1


def test_require_healthy_raises_and_names_every_problem(tmp_path):
    good = _cell(tmp_path, "good")
    stale = _cell(tmp_path, "stale", n=100, flows_on=50)
    with pytest.raises(CellValidationError) as excinfo:
        require_healthy([good, stale])
    assert "stale" in str(excinfo.value)


def test_require_healthy_passes_on_clean_input(tmp_path):
    cells = require_healthy(
        [
            _cell(tmp_path, "a", exec_model="m1"),
            _cell(tmp_path, "b", exec_model="m2"),
        ]
    )
    assert len(cells) == 2


def _strip_metadata(dir_: Path) -> list[Path]:
    """Drop run_metadata from every result file, returning those files.

    Globs before writing anything: an events file written mid-loop would match
    ``main*.json`` on a later iteration and be parsed as a result.
    """
    files = [p for p in sorted(dir_.glob("main*.json")) if ".events." not in p.name]
    for f in files:
        row = json.loads(f.read_text())
        del row["run_metadata"]
        f.write_text(json.dumps(row))
    return files


def _events(dir_: Path, name: str, exec_model: str, audit_model: str | None) -> None:
    """Write an events file shaped like the runner's, for metadata recovery."""
    calls = [{"usage_id": "default", "model": exec_model}]
    if audit_model:
        calls.append({"usage_id": "agentprivarena-extraction", "model": audit_model})
    (dir_ / f"{name}.events.json").write_text(
        json.dumps([{"key": "full_state", "value": {"stats": {"usages": calls}}}])
    )


def test_models_are_recovered_from_events_when_metadata_is_absent(tmp_path):
    """Cells predating run_metadata are verifiable, not merely trusted.

    The 2x2 exec x audit cells were produced before run_metadata existed. Their
    directory names claim which executor and auditor ran, and a directory name
    is exactly what the duplicate-condition incidents showed cannot be trusted
    -- but the events record which model billed each call.
    """
    d = _cell(tmp_path, "old", n=3, analyzer=True)
    for f in _strip_metadata(d):
        _events(d, f.stem, "openai/grok-4", "openai/gpt-5.4")

    sig = describe_cell(d).signature
    assert sig is not None
    assert sig.recovered_from_events
    assert sig.execution_model == "openai/grok-4"
    assert sig.audit_model == "openai/gpt-5.4"
    assert sig.analyzer_enabled
    assert problems_for(describe_cell(d)) == []


def test_recovery_marks_analyzer_off_when_no_auditor_billed(tmp_path):
    d = _cell(tmp_path, "old_l0", n=3, analyzer=False)
    for f in _strip_metadata(d):
        _events(d, f.stem, "openai/gpt-5.4", None)

    sig = describe_cell(d).signature
    assert sig is not None
    assert not sig.analyzer_enabled
    assert sig.audit_model is None
    assert problems_for(describe_cell(d)) == []


def test_missing_metadata_and_missing_events_is_still_a_problem(tmp_path):
    """Recovery must not paper over a cell with nothing to recover from."""
    d = _cell(tmp_path, "opaque", n=3, analyzer=False)
    _strip_metadata(d)

    assert describe_cell(d).signature is None
    problems = problems_for(describe_cell(d))
    assert any("directory name alone" in p for p in problems)


def _audited_cell(tmp_path, *, n, with_flows, steers_per_task):
    """A cell where the analyzer was on, with controllable health signals."""
    results = tmp_path / "results"
    results.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (results / f"main{i}.json").write_text(
            json.dumps(
                {
                    "name": f"main{i}",
                    "status": "ok",
                    "privacy_flows": [{"data_type": "x"}] if i < with_flows else [],
                    "stats": {"audit_instructions_emitted": steers_per_task},
                    "run_metadata": {
                        "execution_model": "openai/gpt-5.4",
                        "audit_model": "openai/gpt-5.4",
                        "prompt_variant": "ci_audit_contextual",
                        "privacy_analyzer_enabled": True,
                    },
                }
            )
        )
    return results


def test_low_extraction_is_refused_only_when_the_audit_was_not_firing(tmp_path):
    """Separate the two causes of a low extraction rate.

    A stale MCP stack and an executor that answers without opening records both
    depress the extraction rate, but only the first corrupts the cell. Every
    observation-content measure we tried put the archived stale-stack cells and
    Gemini-Flash-2.5 in the same band; steering volume separates them cleanly
    (0.16--0.18 against 1.23--2.22), because it measures whether the audit ran
    at all rather than what it found.
    """
    from agentprivarena.base.cell_validation import describe_cell, problems_for

    def verdict(steers):
        results = _audited_cell(
            tmp_path / f"s{steers}", n=100, with_flows=88, steers_per_task=steers
        )
        notes: list[str] = []
        problems = problems_for(describe_cell(results), notes=notes)
        return [p for p in problems if "read-time flow" in p], notes

    # audit barely ran -> the failure the gate exists for
    refused, _ = verdict(0)
    assert refused and "stale MCP stack" in refused[0]

    # audit ran normally and still found little -> executor behaviour
    refused, notes = verdict(2)
    assert not refused
    assert notes and "executor behaviour" in notes[0]


def test_the_steering_rule_never_introduces_a_new_refusal(tmp_path):
    """It is an escape hatch, so a healthy extraction rate is untouched.

    An arm that disables read steering by design emits no notes at all, and must
    keep passing on the strength of its extraction rate alone.
    """
    from agentprivarena.base.cell_validation import describe_cell, problems_for

    results = _audited_cell(tmp_path / "we", n=100, with_flows=100, steers_per_task=0)
    assert not problems_for(describe_cell(results))
