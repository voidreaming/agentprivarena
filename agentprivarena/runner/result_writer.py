"""Persistence helpers for AgentPrivArena runner results."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agentprivarena.runner.event_collector import ScenarioResult


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """Execution settings needed to reproduce and compare a run."""

    execution_model: str
    audit_model: str | None
    privacy_analyzer_enabled: bool
    privacy_audit_mode: str | None
    agent_server_image: str
    privacy_audit_strictness: str | None = None
    privacy_audit_policy: str | None = None
    audit_policy: str | None = None
    # Ablation switches. ``None`` when no analyzer ran; otherwise the value the
    # cell was produced with, so a cell's condition is recoverable from its
    # results rather than from the script that launched it.
    audit_judge_blind: bool | None = None
    audit_verify_recompose: bool | None = None
    privacy_sequential_tool_calls: bool | None = None


def write_scenario_result(
    result: ScenarioResult,
    *,
    results_dir: Path,
    prompt_variant: str,
    prompt_version: str,
    security_analyzer_disabled: bool,
    run_metadata: RunMetadata | None = None,
) -> None:
    """Write the main result JSON and raw-event sidecar."""
    result_file = results_dir / f"{result.name}.json"
    result_file.write_text(
        json.dumps(
            scenario_result_payload(
                result,
                prompt_variant=prompt_variant,
                prompt_version=prompt_version,
                security_analyzer_disabled=security_analyzer_disabled,
                run_metadata=run_metadata,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )

    events_file = results_dir / f"{result.name}.events.json"
    events_file.write_text(json.dumps(result.events, indent=2, ensure_ascii=False))


def scenario_result_payload(
    result: ScenarioResult,
    *,
    prompt_variant: str,
    prompt_version: str,
    security_analyzer_disabled: bool,
    run_metadata: RunMetadata | None = None,
) -> dict[str, Any]:
    """Return the backward-compatible result-file payload."""
    return {
        "name": result.name,
        "status": result.status,
        "prompt_variant": prompt_variant,
        "read_policy": result.read_policy,
        "prompt_version": prompt_version,
        "security_analyzer_disabled": security_analyzer_disabled,
        "run_metadata": asdict(run_metadata) if run_metadata is not None else None,
        "forced_read_success": result.forced_read_success,
        "error_category": result.error_category,
        "final_action": result.final_action,
        "expected_final_action_type": result.expected_final_action_type,
        "tool_calls": result.tool_calls,
        "clarifications": result.clarifications,
        "final_message": result.final_message,
        "sensitive_info_items": result.sensitive_info_items,
        "privacy_flows": result.privacy_flows,
        "privacy_judgments": result.privacy_judgments,
        "audit_instructions": result.audit_instructions,
        "stats": result.stats,
        "error": result.error,
    }


def write_summary(
    *,
    results: list[ScenarioResult],
    results_dir: Path,
    prompt_variant: str,
    read_policy: str,
    prompt_version: str,
    security_analyzer_disabled: bool,
    run_metadata: RunMetadata | None = None,
    batch_start: float,
    batch_end: float | None = None,
) -> None:
    batch_end = batch_end if batch_end is not None else time.time()
    total = len(results)
    total_elapsed = batch_end - batch_start
    summary = {
        "total_in_this_run": total,
        "prompt_variant": prompt_variant,
        "read_policy": read_policy,
        "prompt_version": prompt_version,
        "security_analyzer_disabled": security_analyzer_disabled,
        "run_metadata": asdict(run_metadata) if run_metadata is not None else None,
        "ok": sum(1 for r in results if r.status == "ok"),
        "no_action": sum(1 for r in results if r.status == "no_action"),
        "error": sum(1 for r in results if r.status == "error"),
        "provider_content_filter_errors": sum(
            int((r.stats or {}).get("provider_content_filter_errors", 0) or 0)
            for r in results
        ),
        "total_privacy_flows": sum(len(r.privacy_flows) for r in results),
        "total_elapsed_seconds": round(total_elapsed, 1),
        "mean_seconds_per_task": round(total_elapsed / total, 1) if total else 0,
        "started_at_unix": round(batch_start),
        "finished_at_unix": round(batch_end),
    }
    (results_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
