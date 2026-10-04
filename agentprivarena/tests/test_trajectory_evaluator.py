"""Tests for trajectory-level evaluation helpers."""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

from agentprivarena.base.trajectory_evaluator import (
    BEHAVIOR_JUDGMENTS_FILENAME,
    _resolve_behavior_judgments_path,
    build_trajectory_report,
    collect_seed_records,
    evaluate_trajectory_results_dir,
    parse_coverage_response,
    read_trajectory,
)


def test_resolve_behavior_judgments_prefers_slug(tmp_path) -> None:
    # slug-based file (what `evaluate` writes) wins when present
    slug_file = tmp_path / "judgments_gpt_5_4.json"
    slug_file.write_text("[]")
    (tmp_path / BEHAVIOR_JUDGMENTS_FILENAME).write_text("[]")
    resolved = _resolve_behavior_judgments_path(tmp_path, "openai/gpt-5.4")
    assert resolved == slug_file


def test_resolve_behavior_judgments_historical_fallback(tmp_path) -> None:
    # no slug file for this judge -> fall back to the historical fixed name
    historical = tmp_path / BEHAVIOR_JUDGMENTS_FILENAME
    historical.write_text("[]")
    resolved = _resolve_behavior_judgments_path(tmp_path, "some/other-judge")
    assert resolved == historical


def test_resolve_behavior_judgments_glob_fallback(tmp_path) -> None:
    # neither slug nor historical -> any judgments_*.json
    other = tmp_path / "judgments_qwen3_14b_full389.json"
    other.write_text("[]")
    resolved = _resolve_behavior_judgments_path(tmp_path, None)
    assert resolved == other


def test_collect_seed_records_excludes_synthetic_contacts(tmp_path) -> None:
    task_dir = tmp_path / "main1"
    seed_dir = task_dir / "seed_data"
    seed_dir.mkdir(parents=True)
    (seed_dir / "bookstack.json").write_text(
        json.dumps(
            [
                {"id": "123", "title": "Page", "content": "Full page body"},
                {"type": "contact", "name": "Alex", "email": "alex@example.com"},
            ]
        )
    )
    (seed_dir / "mailpit.json").write_text(
        json.dumps(
            [
                {
                    "type": "contact",
                    "name": "Target",
                    "email": "target@example.com",
                    "info": (
                        "Synthetic addressability record from the user instruction."
                    ),
                }
            ]
        )
    )

    records = collect_seed_records(task_dir)

    assert [record.record_id for record in records] == ["bookstack:123"]
    assert records[0].content == "Full page body"


def test_read_trajectory_filters_writes_finish_and_errors() -> None:
    reads = read_trajectory(
        [
            {
                "step": 1,
                "tool": "bookstack_list_pages",
                "input": {},
                "observation": {"pages": [{"name": "A"}]},
            },
            {
                "step": 2,
                "tool": "bookstack_read_page",
                "input": {"page_id": 1},
                "observation": "failed",
                "is_error": True,
            },
            {
                "step": 3,
                "tool": "gotosocial_create_post",
                "input": {"content": "post"},
                "observation": {"success": True},
            },
            {"step": 4, "tool": "finish", "input": {}, "observation": "done"},
        ]
    )

    assert [read["tool"] for read in reads] == ["bookstack_list_pages"]


def test_parse_coverage_response_normalizes_missing_and_extra(tmp_path) -> None:
    task_dir = tmp_path / "main1"
    seed_dir = task_dir / "seed_data"
    seed_dir.mkdir(parents=True)
    (seed_dir / "bookstack.json").write_text(
        json.dumps(
            [
                {"id": "123", "title": "Page A", "content": "Body A"},
                {"id": "456", "title": "Page B", "content": "Body B"},
            ]
        )
    )
    records = collect_seed_records(task_dir)

    parsed = parse_coverage_response(
        json.dumps(
            {
                "records": [
                    {
                        "record_id": "bookstack:123",
                        "coverage": "full",
                        "matched_steps": [1],
                        "reason": "read",
                    },
                    {
                        "record_id": "extra",
                        "coverage": "full",
                        "matched_steps": [2],
                        "reason": "extra",
                    },
                ]
            }
        ),
        records,
    )

    by_id = {row["record_id"]: row for row in parsed["records"]}
    assert by_id["bookstack:123"]["coverage"] == "full"
    assert by_id["bookstack:456"]["coverage"] == "none"
    assert parsed["unmatched_annotations"] == ["extra"]


def test_evaluate_trajectory_tolerates_non_json_coverage_response(tmp_path) -> None:
    tasks_dir = tmp_path / "tasks"
    task_dir = tasks_dir / "main1"
    seed_dir = task_dir / "seed_data"
    seed_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps({"name": "main1", "user_instruction": "Read the page"})
    )
    (seed_dir / "bookstack.json").write_text(
        json.dumps([{"id": "123", "title": "Page A", "content": "Body A"}])
    )

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "main1.json").write_text(
        json.dumps(
            {
                "name": "main1",
                "status": "ok",
                "tool_calls": [
                    {
                        "step": 1,
                        "tool": "bookstack_list_pages",
                        "input": {},
                        "observation": {"pages": [{"title": "Page A"}]},
                    }
                ],
            }
        )
    )

    class BadCoverageJudge:
        def __init__(self) -> None:
            self.calls = 0
            self.prompts: list[str] = []

        def complete(self, prompt: str) -> str:
            self.calls += 1
            self.prompts.append(prompt)
            return "I would rate this as partial."

    judge = BadCoverageJudge()
    report, rows = asyncio.run(
        evaluate_trajectory_results_dir(
            results_dir=results_dir,
            tasks_dir=tasks_dir,
            judge=cast(Any, judge),
            max_concurrency=1,
            use_cache=False,
        )
    )

    assert judge.calls == 2
    assert "Previous invalid response" in judge.prompts[1]
    assert report["coverage_eval_error_case_count"] == 1
    assert report["coverage_eval_error_rate"] == 100.0
    assert rows[0]["coverage_eval_error"] is True
    assert rows[0]["coverage_eval_retried"] is True
    assert rows[0]["coverage_summary"]["none"] == 1
    assert rows[0]["coverage_records"][0]["reason"].startswith("coverage_eval_error:")


def test_build_trajectory_report_aggregates_process_and_coverage() -> None:
    report = build_trajectory_report(
        [
            {
                "status": "ok",
                "tool_call_count": 4,
                "read_tool_call_count": 2,
                "read_error_count": 0,
                "write_action_attempts": 1,
                "error_tool_call_count": 0,
                "clarification_rounds": 0,
                "sensitive_item_count": 1,
                "privacy_flow_count": 3,
                "judge_interventions": 1,
                "audit_errors": 2,
                "coverage_summary": {
                    "seed_record_count": 2,
                    "full": 1,
                    "partial": 1,
                    "none": 0,
                },
                "unmatched_annotations": [],
            },
            {
                "status": "no_action",
                "tool_call_count": 1,
                "read_tool_call_count": 1,
                "read_error_count": 1,
                "write_action_attempts": 0,
                "error_tool_call_count": 1,
                "clarification_rounds": 1,
                "sensitive_item_count": 1,
                "privacy_flow_count": 0,
                "judge_interventions": 0,
                "coverage_summary": {
                    "seed_record_count": 2,
                    "full": 0,
                    "partial": 0,
                    "none": 2,
                },
                "unmatched_annotations": ["extra"],
            },
        ]
    )

    assert report["total"] == 2
    assert report["commit_rate"] == 50.0
    assert report["process"]["avg_read_tool_calls"] == 1.5
    assert report["process"]["flow_coverage_rate"] == 50.0
    assert report["process"]["judge_intervention_given_flow_rate"] == 100.0
    assert report["content_access"]["record_full_coverage_rate"] == 25.0
    assert report["content_access"]["any_coverage_rate"] == 50.0
    assert report["coverage_eval_error_case_count"] == 0
    assert report["unmatched_annotation_case_count"] == 1
    # One of the two cases had the fail-open audit degrade to a no-op. Reported
    # separately because a crashed audit otherwise reads as a clean one.
    assert report["audit_error_case_count"] == 1
    assert report["audit_error_case_rate"] == 50.0


def test_build_trajectory_report_adds_conditioned_outcome() -> None:
    base = {
        "tool_call_count": 1,
        "read_tool_call_count": 1,
        "read_error_count": 0,
        "write_action_attempts": 1,
        "error_tool_call_count": 0,
        "clarification_rounds": 0,
        "sensitive_item_count": 1,
        "privacy_judgment_count": 0,
        "unmatched_annotations": [],
    }
    report = build_trajectory_report(
        [
            {
                **base,
                "status": "ok",
                "privacy_flow_count": 1,
                "judge_interventions": 1,
                "helpfulness_score": 3,
                "llm_leaked": True,
                "coverage_summary": {
                    "seed_record_count": 1,
                    "full": 1,
                    "partial": 0,
                    "none": 0,
                },
            },
            {
                **base,
                "status": "ok",
                "privacy_flow_count": 0,
                "judge_interventions": 0,
                "helpfulness_score": 2,
                "llm_leaked": False,
                "coverage_summary": {
                    "seed_record_count": 1,
                    "full": 0,
                    "partial": 1,
                    "none": 0,
                },
            },
        ]
    )

    conditioned = report["trajectory_conditioned_outcome"]
    assert conditioned["all_evaluated"]["LR_h"] == 50.0
    assert conditioned["complete_full_coverage"]["LR_h"] == 100.0
    assert conditioned["partial_or_missing_coverage"]["LR_h"] == 0.0
