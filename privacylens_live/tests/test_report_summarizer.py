"""Tests for PrivacyLens report-table summarization."""

from __future__ import annotations

import json

from privacylens_live.base.report_summarizer import (
    RunSpec,
    build_report_summary,
    parse_run_spec,
    render_markdown,
)


def test_parse_run_spec_uses_explicit_or_directory_label() -> None:
    explicit = parse_run_spec("L3=results/l3")
    implicit = parse_run_spec("results/l0")

    assert explicit.label == "L3"
    assert str(explicit.results_dir) == "results/l3"
    assert implicit.label == "l0"


def test_build_report_summary_groups_four_sections(tmp_path) -> None:
    results_dir = tmp_path / "l3_run"
    results_dir.mkdir()
    (results_dir / "report_qwen3_14b_behavior.json").write_text(
        json.dumps(
            {
                "total": 10,
                "status_counts": {"ok": 9, "no_action": 1},
                "LR": 11.1,
                "LR_h": 9.1,
                "helpfulness_avg": 2.4,
                "helpful_rate": 80.0,
                "routing_correct_rate": 100.0,
            }
        )
    )
    (results_dir / "report_trajectory.json").write_text(
        json.dumps(
            {
                "total": 10,
                "commit_rate": 90.0,
                "no_action_rate": 10.0,
                "error_rate": 0.0,
                "content_access": {
                    "record_any_coverage_rate": 95.0,
                    "record_full_coverage_rate": 85.0,
                    "record_weighted_coverage_rate": 90.0,
                    "complete_full_case_rate": 60.0,
                },
                "process": {
                    "clarification_case_rate": 5.0,
                    "avg_agent_read_tool_calls": 8.5,
                    "read_error_case_rate": 2.5,
                    "avg_privacy_flow_count": 1.7,
                    "privacy_flow_case_rate": 55.0,
                    "flow_coverage_rate": 70.0,
                    "avg_privacy_judgment_count": 1.0,
                    "judge_intervention_case_rate": 3.0,
                    "judge_intervention_given_flow_rate": 6.0,
                },
                "trajectory_conditioned_outcome": {
                    "complete_full_coverage": {
                        "case_count": 6,
                        "evaluated_case_count": 6,
                        "LR": 8.0,
                        "LR_h": 7.0,
                        "helpfulness_avg": 2.5,
                        "helpful_rate": 83.3,
                    }
                },
            }
        )
    )

    summary = build_report_summary([RunSpec("L3", results_dir)])
    markdown = render_markdown(summary)

    assert summary["outcome"][0]["LR_h"] == 9.1
    assert summary["intermediate"][0]["record_full_coverage_rate"] == 85.0
    assert summary["conditional_outcome"][0]["subset"] == "complete_full_coverage"
    assert summary["l3_privacy_pipeline"][0]["flow_coverage_rate"] == 70.0
    assert "## Outcome Report" in markdown
    assert "| L3 | 10 | 9 | 11.10 | 9.10 | 2.40 | 80.00 |" in markdown
