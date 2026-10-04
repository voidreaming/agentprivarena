"""Build compact PrivacyLens-Live report tables from existing run reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


JsonObject = dict[str, Any]

BEHAVIOR_REPORT_GLOB = "report_*behavior*.json"
TRAJECTORY_REPORT_FILENAME = "report_trajectory.json"


@dataclass(frozen=True)
class RunSpec:
    """One results directory to include in a cross-run summary."""

    label: str
    results_dir: Path


def parse_run_spec(raw: str) -> RunSpec:
    """Parse ``LABEL=results/dir`` or use the directory name as the label."""
    if "=" in raw:
        label, path = raw.split("=", 1)
        label = label.strip()
        results_dir = Path(path.strip())
    else:
        results_dir = Path(raw.strip())
        label = results_dir.name
    if not label:
        raise ValueError(f"Run label is empty in {raw!r}")
    if not str(results_dir):
        raise ValueError(f"Run results directory is empty in {raw!r}")
    return RunSpec(label=label, results_dir=results_dir)


def build_report_summary(runs: list[RunSpec]) -> JsonObject:
    """Return four report sections for the provided run directories."""
    outcome: list[JsonObject] = []
    intermediate: list[JsonObject] = []
    conditional: list[JsonObject] = []
    l3_pipeline: list[JsonObject] = []
    run_inputs: list[JsonObject] = []

    for run in runs:
        behavior_path = _find_behavior_report(run.results_dir)
        trajectory_path = run.results_dir / TRAJECTORY_REPORT_FILENAME
        behavior = _read_json_object(behavior_path) if behavior_path else None
        trajectory = (
            _read_json_object(trajectory_path) if trajectory_path.exists() else None
        )
        run_inputs.append(
            {
                "setting": run.label,
                "results_dir": str(run.results_dir),
                "behavior_report": str(behavior_path) if behavior_path else None,
                "trajectory_report": (
                    str(trajectory_path) if trajectory_path.exists() else None
                ),
            }
        )
        if behavior is not None:
            outcome.append(_build_outcome_row(run.label, behavior))
        if trajectory is not None:
            intermediate.append(_build_intermediate_row(run.label, trajectory))
            conditional.extend(_build_conditional_rows(run.label, trajectory))
            l3_pipeline.append(_build_l3_pipeline_row(run.label, trajectory))

    return {
        "runs": run_inputs,
        "outcome": outcome,
        "intermediate": intermediate,
        "conditional_outcome": conditional,
        "l3_privacy_pipeline": l3_pipeline,
    }


def render_markdown(summary: JsonObject) -> str:
    """Render the summary as Markdown tables suitable for notes or slides."""
    sections = [
        (
            "Outcome Report",
            summary.get("outcome", []),
            [
                "setting",
                "N",
                "ok",
                "LR",
                "LR_h",
                "helpfulness_avg",
                "helpful_rate",
            ],
        ),
        (
            "Intermediate Metrics",
            summary.get("intermediate", []),
            [
                "setting",
                "commit_rate",
                "clarification_case_rate",
                "record_any_coverage_rate",
                "record_full_coverage_rate",
                "avg_agent_read_tool_calls",
                "read_error_case_rate",
            ],
        ),
        (
            "Conditional Outcome",
            summary.get("conditional_outcome", []),
            [
                "setting",
                "subset",
                "case_count",
                "evaluated_case_count",
                "LR",
                "LR_h",
                "helpfulness_avg",
                "helpful_rate",
            ],
        ),
        (
            "L3 Privacy Extractor And Judge",
            summary.get("l3_privacy_pipeline", []),
            [
                "setting",
                "avg_privacy_flow_count",
                "privacy_flow_case_rate",
                "flow_coverage_rate",
                "judge_intervention_case_rate",
                "judge_intervention_given_flow_rate",
                "avg_privacy_judgment_count",
                "audit_error_case_rate",
            ],
        ),
    ]
    rendered: list[str] = []
    for title, rows, columns in sections:
        rendered.append(f"## {title}")
        rendered.append(_render_table(_as_rows(rows), columns))
        rendered.append("")
    return "\n".join(rendered).rstrip() + "\n"


def _find_behavior_report(results_dir: Path) -> Path | None:
    preferred = results_dir / "report_qwen3_14b_behavior.json"
    if preferred.exists():
        return preferred
    candidates = sorted(results_dir.glob(BEHAVIOR_REPORT_GLOB))
    return candidates[0] if candidates else None


def _read_json_object(path: Path) -> JsonObject:
    loaded = json.loads(path.read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return loaded


def _get_number(data: JsonObject, key: str) -> float | int | None:
    value = data.get(key)
    return value if isinstance(value, (int, float)) else None


def _nested_number(data: JsonObject, section: str, key: str) -> float | int | None:
    value = data.get(section)
    if not isinstance(value, dict):
        return None
    return _get_number(value, key)


def _ok_count(report: JsonObject) -> int | None:
    status_counts = report.get("status_counts")
    if not isinstance(status_counts, dict):
        return None
    ok = status_counts.get("ok")
    return int(ok) if isinstance(ok, (int, float)) else None


def _build_outcome_row(setting: str, report: JsonObject) -> JsonObject:
    return {
        "setting": setting,
        "N": _get_number(report, "total"),
        "ok": _ok_count(report),
        "LR": _get_number(report, "LR"),
        "LR_h": _get_number(report, "LR_h"),
        "helpfulness_avg": _get_number(report, "helpfulness_avg"),
        "helpful_rate": _get_number(report, "helpful_rate"),
        "routing_correct_rate": _get_number(report, "routing_correct_rate"),
    }


def _build_intermediate_row(setting: str, report: JsonObject) -> JsonObject:
    content = report.get("content_access")
    content = content if isinstance(content, dict) else {}
    process = report.get("process")
    process = process if isinstance(process, dict) else {}
    return {
        "setting": setting,
        "N": _get_number(report, "total"),
        "commit_rate": _get_number(report, "commit_rate")
        or _get_number(report, "committed_rate"),
        "no_action_rate": _get_number(report, "no_action_rate"),
        "error_rate": _get_number(report, "error_rate"),
        "clarification_case_rate": _get_number(process, "clarification_case_rate"),
        "record_any_coverage_rate": _get_number(content, "record_any_coverage_rate")
        or _get_number(content, "any_coverage_rate"),
        "record_full_coverage_rate": _get_number(content, "record_full_coverage_rate")
        or _get_number(content, "full_coverage_rate"),
        "record_weighted_coverage_rate": _get_number(
            content, "record_weighted_coverage_rate"
        )
        or _get_number(content, "weighted_coverage_rate"),
        "complete_full_case_rate": _get_number(content, "complete_full_case_rate"),
        "avg_agent_read_tool_calls": _get_number(process, "avg_agent_read_tool_calls"),
        "read_error_case_rate": _get_number(process, "read_error_case_rate"),
    }


def _build_conditional_rows(setting: str, report: JsonObject) -> list[JsonObject]:
    conditioned = report.get("trajectory_conditioned_outcome")
    if not isinstance(conditioned, dict):
        return []

    rows: list[JsonObject] = []
    for subset, values in conditioned.items():
        if not isinstance(values, dict):
            continue
        row: JsonObject = {
            "setting": setting,
            "subset": subset,
        }
        for key in (
            "case_count",
            "evaluated_case_count",
            "helpful_case_count",
            "leaked_case_count",
            "leaked_helpful_case_count",
            "LR",
            "LR_h",
            "helpfulness_avg",
            "helpful_rate",
        ):
            row[key] = _get_number(values, key)
        rows.append(row)
    return rows


def _build_l3_pipeline_row(setting: str, report: JsonObject) -> JsonObject:
    process = report.get("process")
    process = process if isinstance(process, dict) else {}
    return {
        "setting": setting,
        "avg_privacy_flow_count": _get_number(process, "avg_privacy_flow_count"),
        "privacy_flow_case_rate": _get_number(process, "privacy_flow_case_rate"),
        "flow_coverage_rate": _get_number(process, "flow_coverage_rate"),
        "sensitive_case_count": _get_number(process, "sensitive_case_count"),
        "sensitive_case_flow_count": _get_number(process, "sensitive_case_flow_count"),
        "avg_privacy_judgment_count": _get_number(
            process, "avg_privacy_judgment_count"
        ),
        "avg_judge_interventions": _get_number(process, "avg_judge_interventions"),
        "judge_intervention_case_rate": _get_number(
            process, "judge_intervention_case_rate"
        ),
        "judge_intervention_given_flow_rate": _get_number(
            process, "judge_intervention_given_flow_rate"
        ),
        # Fail-open rate: a non-zero value means this arm's reported mitigation
        # was absent on some cases, which otherwise looks like a clean audit.
        "audit_error_case_rate": _get_number(report, "audit_error_case_rate"),
    }


def _as_rows(value: Any) -> list[JsonObject]:
    return (
        [row for row in value if isinstance(row, dict)]
        if isinstance(value, list)
        else []
    )


def _render_table(rows: list[JsonObject], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    if not rows:
        return "\n".join([header, separator])
    body = [
        "| " + " | ".join(_format_cell(row.get(column)) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
