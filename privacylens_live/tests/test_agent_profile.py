"""Tests for runner agent-profile construction."""

from __future__ import annotations

import pytest

from openhands.sdk.privacy import (
    PrivacyAuditGuidanceMode,
    PrivacyAuditMode,
    PrivacyReadAuditMode,
    PrivacyStrictness,
)
from privacylens_live.config import Config
from privacylens_live.runner.agent_profile import AgentProfile
from privacylens_live.runner.prompt_builder import get_system_prompt_path
from privacylens_live.runner.task_spec import TaskSpec


def test_baseline_profile_uses_stock_prompt_without_prompt_volume() -> None:
    profile = AgentProfile(prompt_variant="baseline")
    workspace_kwargs = profile.build_workspace_kwargs(Config())

    assert profile.uses_custom_prompt_file is False
    assert "volumes" not in workspace_kwargs


def test_non_baseline_profile_mounts_prompt_directory() -> None:
    profile = AgentProfile(prompt_variant="privacy_enhanced")
    workspace_kwargs = profile.build_workspace_kwargs(Config())
    prompt_dir = str(get_system_prompt_path("privacy_enhanced").parent)

    assert profile.uses_custom_prompt_file is True
    assert workspace_kwargs["volumes"] == [f"{prompt_dir}:{prompt_dir}:ro"]


def test_ci_audit_profile_uses_stock_prompt_without_prompt_volume() -> None:
    profile = AgentProfile(prompt_variant="ci_audit", enable_privacy_analyzer=True)
    workspace_kwargs = profile.build_workspace_kwargs(Config())

    assert profile.uses_custom_prompt_file is False
    assert "volumes" not in workspace_kwargs


@pytest.mark.parametrize(
    ("variant", "expected_mode"),
    [
        ("ci_audit", PrivacyAuditGuidanceMode.INSTRUCTION_ONLY),
        ("ci_audit_flows", PrivacyAuditGuidanceMode.ANNOTATED_FLOWS),
        (
            "ci_audit_plan",
            PrivacyAuditGuidanceMode.ANNOTATED_FLOWS_WITH_PLAN,
        ),
        (
            "ci_audit_contextual",
            PrivacyAuditGuidanceMode.CONTEXTUAL_SHARE_DECISIONS,
        ),
    ],
)
def test_ci_audit_profiles_set_guidance_mode(
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    expected_mode: PrivacyAuditGuidanceMode,
) -> None:
    monkeypatch.setattr(
        "privacylens_live.runner.privacy_tool_semantics."
        "register_privacylens_tool_privacy_semantics",
        lambda: None,
    )
    profile = AgentProfile(prompt_variant=variant, enable_privacy_analyzer=True)
    task = TaskSpec(
        name="sample",
        user_instruction="Send an email",
        dependencies=["mailpit"],
        user_name="Jane",
    )

    agent_kwargs = profile.build_agent_kwargs(
        task=task,
        config=Config(),
        mcp_config={},
    )

    assert profile.uses_custom_prompt_file is False
    assert agent_kwargs["privacy_audit_guidance_mode"] == expected_mode


def test_ci_audit_contextual_profile_sets_contextual_read_audit_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "privacylens_live.runner.privacy_tool_semantics."
        "register_privacylens_tool_privacy_semantics",
        lambda: None,
    )
    profile = AgentProfile(
        prompt_variant="ci_audit_contextual",
        enable_privacy_analyzer=True,
    )
    task = TaskSpec(
        name="sample",
        user_instruction="Send an email",
        dependencies=["mailpit"],
        user_name="Jane",
    )

    agent_kwargs = profile.build_agent_kwargs(
        task=task,
        config=Config(),
        mcp_config={},
    )

    assert (
        agent_kwargs["privacy_analyzer"].read_audit_mode
        is PrivacyReadAuditMode.CONTEXTUAL_SHARE
    )


def test_ci_audit_profile_keeps_judge_prompt_out_of_execution_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "privacylens_live.runner.privacy_tool_semantics."
        "register_privacylens_tool_privacy_semantics",
        lambda: None,
    )
    profile = AgentProfile(prompt_variant="ci_audit", enable_privacy_analyzer=True)
    task = TaskSpec(
        name="sample",
        user_instruction="Send an email",
        dependencies=["mailpit"],
        user_name="Jane",
        expected_recipient="Mike",
        expected_channel="Mattermost DM",
    )

    agent_kwargs = profile.build_agent_kwargs(
        task=task,
        config=Config(),
        mcp_config={},
    )

    assert "system_prompt_filename" not in agent_kwargs
    assert agent_kwargs.get("system_prompt_kwargs") is None
    assert agent_kwargs["privacy_task_purpose"] == "Send an email"
    assert agent_kwargs["privacy_expected_recipient"] == "Mike"
    assert agent_kwargs["privacy_expected_channel"] == "Mattermost DM"


def test_ci_audit_profile_passes_component_ablation_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "privacylens_live.runner.privacy_tool_semantics."
        "register_privacylens_tool_privacy_semantics",
        lambda: None,
    )
    profile = AgentProfile(
        prompt_variant="ci_audit_contextual",
        enable_privacy_analyzer=True,
        privacy_audit_mode=PrivacyAuditMode.WRITE_ENFORCE_ONLY,
    )
    task = TaskSpec(
        name="sample",
        user_instruction="Send an email",
        dependencies=["mailpit"],
        user_name="Jane",
    )

    agent_kwargs = profile.build_agent_kwargs(
        task=task,
        config=Config(),
        mcp_config={},
    )

    assert agent_kwargs["privacy_audit_mode"] is PrivacyAuditMode.WRITE_ENFORCE_ONLY


def test_agent_profile_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="Unknown prompt_variant"):
        AgentProfile(prompt_variant="unknown")


def test_ablation_switches_default_to_existing_behaviour(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defaults must reproduce the condition every cell on disk was run under,
    so enabling one creates a new condition instead of invalidating the old."""
    monkeypatch.setattr(
        "privacylens_live.runner.privacy_tool_semantics."
        "register_privacylens_tool_privacy_semantics",
        lambda: None,
    )
    task = TaskSpec(
        name="sample",
        user_instruction="Send an email",
        dependencies=["mailpit"],
        user_name="Jane",
    )

    def kwargs_for(
        *,
        audit_judge_blind: bool = False,
        audit_verify_recompose: bool = False,
        privacy_sequential_tool_calls: bool = True,
    ) -> dict:
        profile = AgentProfile(
            prompt_variant="ci_audit_contextual",
            enable_privacy_analyzer=True,
            audit_judge_blind=audit_judge_blind,
            audit_verify_recompose=audit_verify_recompose,
            privacy_sequential_tool_calls=privacy_sequential_tool_calls,
        )
        return profile.build_agent_kwargs(task=task, config=Config(), mcp_config={})

    default = kwargs_for()
    analyzer = default["privacy_analyzer"]
    assert analyzer.judge_blind_to_read_disposition is False
    assert analyzer.verify_recompose is False
    assert default["privacy_sequential_tool_calls"] is True

    flipped = kwargs_for(
        audit_judge_blind=True,
        audit_verify_recompose=True,
        privacy_sequential_tool_calls=False,
    )
    flipped_analyzer = flipped["privacy_analyzer"]
    assert flipped_analyzer.judge_blind_to_read_disposition is True
    assert flipped_analyzer.verify_recompose is True
    assert flipped["privacy_sequential_tool_calls"] is False
    # Serialization is a separate knob from execution concurrency: turning it
    # off must not quietly re-enable parallel tool execution.
    assert flipped["tool_concurrency_limit"] == 1


@pytest.mark.parametrize(
    "level",
    [
        PrivacyStrictness.LENIENT,
        PrivacyStrictness.BALANCED,
        PrivacyStrictness.STRICT,
    ],
)
def test_ci_audit_profile_passes_strictness(
    monkeypatch: pytest.MonkeyPatch,
    level: PrivacyStrictness,
) -> None:
    monkeypatch.setattr(
        "privacylens_live.runner.privacy_tool_semantics."
        "register_privacylens_tool_privacy_semantics",
        lambda: None,
    )
    profile = AgentProfile(
        prompt_variant="ci_audit_contextual",
        enable_privacy_analyzer=True,
        audit_strictness=level,
    )
    task = TaskSpec(
        name="sample",
        user_instruction="Send an email",
        dependencies=["mailpit"],
        user_name="Jane",
    )

    agent_kwargs = profile.build_agent_kwargs(
        task=task,
        config=Config(),
        mcp_config={},
    )

    assert agent_kwargs["privacy_analyzer"].strictness is level


def test_agent_profile_defaults_to_balanced_strictness() -> None:
    assert AgentProfile().audit_strictness is PrivacyStrictness.BALANCED
    profile = AgentProfile(
        prompt_variant="ci_audit_contextual",
        enable_privacy_analyzer=True,
        audit_strictness=PrivacyStrictness.STRICT,
    )
    assert profile.audit_strictness is PrivacyStrictness.STRICT


def test_ci_audit_profile_passes_privacy_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "privacylens_live.runner.privacy_tool_semantics."
        "register_privacylens_tool_privacy_semantics",
        lambda: None,
    )
    policy = "Never share the principal's photo; address is acceptable."
    profile = AgentProfile(
        prompt_variant="ci_audit_contextual",
        enable_privacy_analyzer=True,
        privacy_policy=policy,
    )
    task = TaskSpec(
        name="sample",
        user_instruction="Send an email",
        dependencies=["mailpit"],
        user_name="Jane",
    )

    agent_kwargs = profile.build_agent_kwargs(
        task=task,
        config=Config(),
        mcp_config={},
    )

    assert agent_kwargs["privacy_analyzer"].privacy_policy == policy
