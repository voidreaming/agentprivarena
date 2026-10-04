"""Tests for runner result construction and persistence payloads."""

from __future__ import annotations

from privacylens_live.runner.event_collector import EventCollector
from privacylens_live.runner.result_builder import build_scenario_result
from privacylens_live.runner.result_writer import RunMetadata, scenario_result_payload
from privacylens_live.runner.task_spec import TaskSpec


def test_build_scenario_result_preserves_baseline_result_shape() -> None:
    collector = EventCollector()
    collector.events = [
        {
            "kind": "ActionEvent",
            "tool_name": "mailpit_send_email",
            "tool_call_id": "send-1",
            "action": {
                "data": {"to": "a@example.com", "body": "Done"},
                "kind": "MCPToolAction",
            },
        },
        {
            "kind": "ObservationEvent",
            "tool_name": "mailpit_send_email",
            "tool_call_id": "send-1",
            "observation": {
                "kind": "MCPToolObservation",
                "is_error": False,
                "content": [{"type": "text", "text": '{"success": true}'}],
            },
        },
    ]
    task = TaskSpec(
        name="case1",
        user_instruction="Email the update.",
        dependencies=["mailpit"],
        final_action_type="email",
        sensitive_info_items=["diagnosis"],
    )

    result = build_scenario_result(
        task=task,
        collector=collector,
        clarifications=[],
        read_policy="natural",
        planned_initial_read_count=0,
        initial_read_actions_planned=True,
    )
    payload = scenario_result_payload(
        result,
        prompt_variant="baseline",
        prompt_version="v-test",
        security_analyzer_disabled=False,
        run_metadata=RunMetadata(
            execution_model="openai/gpt-5.4",
            audit_model=None,
            privacy_analyzer_enabled=False,
            privacy_audit_mode=None,
            agent_server_image="agent:test",
        ),
    )

    assert result.status == "ok"
    assert payload["prompt_variant"] == "baseline"
    assert payload["final_action"]["tool"] == "mailpit_send_email"
    assert payload["expected_final_action_type"] == "email"
    assert payload["sensitive_info_items"] == ["diagnosis"]
    assert payload["forced_read_success"] is None
    assert payload["run_metadata"] == {
        "execution_model": "openai/gpt-5.4",
        "audit_model": None,
        "privacy_analyzer_enabled": False,
        "privacy_audit_mode": None,
        "agent_server_image": "agent:test",
        "privacy_audit_strictness": None,
        "privacy_audit_policy": None,
        "audit_policy": None,
        # Ablation switches are None with no analyzer, so a baseline cell's
        # metadata stays unambiguous about which condition produced it.
        "audit_judge_blind": None,
        "audit_verify_recompose": None,
        "privacy_sequential_tool_calls": None,
    }


def test_build_scenario_result_preserves_error_category() -> None:
    collector = EventCollector()
    task = TaskSpec(
        name="case-filtered",
        user_instruction="Reply to the message.",
        dependencies=["mattermost"],
        final_action_type="message",
    )

    result = build_scenario_result(
        task=task,
        collector=collector,
        clarifications=[],
        read_policy="natural",
        planned_initial_read_count=0,
        initial_read_actions_planned=True,
        status="error",
        error="litellm.BadRequestError: code=content_filter",
    )
    payload = scenario_result_payload(
        result,
        prompt_variant="ci_audit",
        prompt_version="v-test",
        security_analyzer_disabled=False,
    )

    assert result.error_category == "provider_content_filter"
    assert payload["error_category"] == "provider_content_filter"
    assert payload["run_metadata"] is None
