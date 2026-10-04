"""Build structured scenario results from collected agent events."""

from __future__ import annotations

from typing import Any, Literal

from agentprivarena.runner.event_collector import EventCollector, ScenarioResult
from agentprivarena.runner.task_spec import TaskSpec


ResultStatus = Literal["ok", "no_action", "error"]


def build_scenario_result(
    *,
    task: TaskSpec,
    collector: EventCollector,
    clarifications: list[dict[str, Any]],
    read_policy: str,
    planned_initial_read_count: int,
    initial_read_actions_planned: bool,
    status: ResultStatus | None = None,
    error: str | None = None,
    include_final_action: bool = True,
) -> ScenarioResult:
    """Construct the stable result object persisted by the runner."""
    final_action = collector.extract_final_action() if include_final_action else None
    stats = collector.extract_stats()
    stats["clarification_rounds"] = len(clarifications)
    error_category = collector.extract_error_category(error)
    resolved_status = status or _status_from_final_action(final_action)
    return ScenarioResult(
        name=task.name,
        status=resolved_status,
        final_action=final_action,
        expected_final_action_type=task.final_action_type,
        tool_calls=collector.extract_tool_calls(),
        clarifications=clarifications,
        final_message=collector.extract_final_message(),
        sensitive_info_items=list(task.sensitive_info_items or []),
        privacy_flows=collector.extract_privacy_flows(),
        privacy_judgments=collector.extract_privacy_judgments(),
        audit_instructions=collector.extract_audit_instructions(),
        stats=stats,
        read_policy=read_policy,
        forced_read_success=forced_read_success(
            read_policy,
            stats,
            planned_initial_read_count,
            initial_read_actions_planned,
        ),
        events=collector.events,
        error_category=error_category,
        error=error,
    )


def forced_read_success(
    read_policy: str,
    stats: dict[str, Any],
    planned_count: int,
    planned_successfully: bool,
) -> bool | None:
    if read_policy != "forced_oracle":
        return None
    if not planned_successfully:
        return False
    return (
        stats.get("forced_read_tool_call_count", 0) == planned_count
        and stats.get("forced_read_error_count", 0) == 0
    )


def _status_from_final_action(final_action: dict[str, Any] | None) -> ResultStatus:
    if final_action and not final_action.get("is_error"):
        return "ok"
    return "no_action"
