"""Agent profile construction for PrivacyLens-Live runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import SecretStr

from openhands.sdk.privacy import (
    AuditPolicy,
    PrivacyAuditGuidanceMode,
    PrivacyAuditMode,
    PrivacyReadAuditMode,
    PrivacyStrictness,
)
from privacylens_live.config import Config
from privacylens_live.runner.prompt_builder import (
    VALID_PROMPT_VARIANTS,
    build_system_suffix,
    get_system_prompt_path,
)
from privacylens_live.runner.task_spec import TaskSpec


@dataclass(frozen=True)
class AgentProfile:
    """Prompt and analyzer settings for one runner variant."""

    prompt_variant: str = "baseline"
    disable_security_analyzer: bool = False
    enable_privacy_analyzer: bool = False
    privacy_audit_mode: PrivacyAuditMode = PrivacyAuditMode.FULL
    audit_strictness: PrivacyStrictness = PrivacyStrictness.BALANCED
    audit_policy: AuditPolicy = AuditPolicy.CONTEXTUAL_INTEGRITY
    privacy_policy: str = ""
    # Ablation switches. Each default reproduces the behaviour every existing
    # run was produced with, so turning one on creates a new condition rather
    # than invalidating the cells already on disk.
    audit_judge_blind: bool = False
    audit_verify_recompose: bool = False
    privacy_sequential_tool_calls: bool = True

    def __post_init__(self) -> None:
        if self.prompt_variant not in VALID_PROMPT_VARIANTS:
            raise ValueError(
                f"Unknown prompt_variant {self.prompt_variant!r}; "
                f"expected one of {VALID_PROMPT_VARIANTS}"
            )

    @property
    def uses_custom_prompt_file(self) -> bool:
        return self.prompt_variant in {"privacy_enhanced", "ci_reasoning"}

    def build_agent_kwargs(
        self,
        *,
        task: TaskSpec,
        config: Config,
        mcp_config: dict[str, Any],
        initial_read_actions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build kwargs for ``openhands.sdk.Agent``.

        The runner still owns orchestration. This profile owns the Baseline vs
        mitigation differences so the execution pipeline does not need to know
        prompt-template details.
        """
        from openhands.sdk import LLM
        from openhands.sdk.context.agent_context import AgentContext

        llm = LLM(**self._llm_kwargs(config))
        agent_kwargs: dict[str, Any] = {
            "llm": llm,
            "tools": [],
            "mcp_config": mcp_config,
        }
        if initial_read_actions is not None:
            agent_kwargs["initial_read_actions"] = initial_read_actions

        sys_kwargs = self._system_prompt_kwargs()
        if sys_kwargs:
            agent_kwargs["system_prompt_kwargs"] = sys_kwargs

        if self.enable_privacy_analyzer:
            from openhands.sdk.privacy import LLMPrivacyAnalyzer
            from privacylens_live.runner.privacy_tool_semantics import (
                register_privacylens_tool_privacy_semantics,
            )

            register_privacylens_tool_privacy_semantics()
            extraction_llm = LLM(
                model=config.extraction_llm_model,
                api_key=SecretStr(config.extraction_llm_api_key),
                base_url=config.extraction_llm_base_url,
                usage_id="privacylens-live-extraction",
            )
            agent_kwargs["privacy_analyzer"] = LLMPrivacyAnalyzer(
                llm=extraction_llm,
                read_audit_mode=self._privacy_read_audit_mode(),
                strictness=self.audit_strictness,
                audit_policy=self.audit_policy,
                privacy_policy=self.privacy_policy,
                judge_blind_to_read_disposition=self.audit_judge_blind,
                verify_recompose=self.audit_verify_recompose,
            )
            agent_kwargs["privacy_principal"] = task.user_name
            agent_kwargs["privacy_task_purpose"] = task.user_instruction
            agent_kwargs["privacy_expected_recipient"] = task.expected_recipient
            agent_kwargs["privacy_expected_channel"] = task.expected_channel
            agent_kwargs["privacy_audit_guidance_mode"] = (
                self._privacy_audit_guidance_mode()
            )
            agent_kwargs["privacy_audit_mode"] = self.privacy_audit_mode
            # Off, the agent may emit several tool calls per planning turn, so a
            # write composed in the same turn as its reads cannot be steered by
            # the read-boundary note and falls to the write gate alone. This is
            # also the main cost lever: on, each task takes as many planning
            # turns as it makes tool calls.
            agent_kwargs["privacy_sequential_tool_calls"] = (
                self.privacy_sequential_tool_calls
            )
            # Unrelated to the above: execution stays serial either way, so the
            # ablation isolates steering opportunity, not concurrency.
            agent_kwargs["tool_concurrency_limit"] = 1

        suffix = build_system_suffix(task.prompt_payload(), self.prompt_variant)
        if suffix:
            agent_kwargs["agent_context"] = AgentContext(system_message_suffix=suffix)

        if self.uses_custom_prompt_file:
            agent_kwargs["system_prompt_filename"] = str(
                get_system_prompt_path(self.prompt_variant)
            )

        return agent_kwargs

    def build_workspace_kwargs(self, config: Config) -> dict[str, Any]:
        """Build kwargs for ``DockerWorkspace``."""
        workspace_kwargs: dict[str, Any] = {
            "server_image": config.agent_server_image,
            "network": config.docker_network,
            # Forward OH_PRELOAD_TOOLS so we can disable the agent-server's
            # eager chromium/browser preload. PrivacyLens agents use only MCP
            # tools; the browser preload is dead weight and, in the local
            # source-minimal image (no chromium), its failing init can delay
            # server readiness and cause spurious connection-refused errors.
            "forward_env": ["DEBUG", "OH_PRELOAD_TOOLS"],
        }
        if self.uses_custom_prompt_file:
            prompt_path = get_system_prompt_path(self.prompt_variant)
            host_prompt_dir = str(prompt_path.parent)
            workspace_kwargs["volumes"] = [f"{host_prompt_dir}:{host_prompt_dir}:ro"]
        return workspace_kwargs

    def _llm_kwargs(self, config: Config) -> dict[str, Any]:
        llm_kwargs: dict[str, Any] = {
            "model": config.llm_model,
            "api_key": SecretStr(config.llm_api_key),
        }
        # An empty base_url means "use the provider's own endpoint". Passing the
        # Azure gateway URL to a natively-routed provider such as
        # ``gemini/gemini-2.5-flash`` would send Google traffic to Azure and
        # fail, so the key is omitted rather than passed as an empty string.
        if config.llm_base_url:
            llm_kwargs["base_url"] = config.llm_base_url
        if config.llm_model.startswith("azure/") and config.llm_api_version:
            llm_kwargs["api_version"] = config.llm_api_version
        return llm_kwargs

    def _system_prompt_kwargs(self) -> dict[str, bool]:
        sys_kwargs: dict[str, bool] = {}
        if self.disable_security_analyzer:
            sys_kwargs["llm_security_analyzer"] = False
        return sys_kwargs

    def _privacy_audit_guidance_mode(self) -> PrivacyAuditGuidanceMode:
        if self.prompt_variant == "ci_audit_flows":
            return PrivacyAuditGuidanceMode.ANNOTATED_FLOWS
        if self.prompt_variant == "ci_audit_plan":
            return PrivacyAuditGuidanceMode.ANNOTATED_FLOWS_WITH_PLAN
        if self.prompt_variant == "ci_audit_contextual":
            return PrivacyAuditGuidanceMode.CONTEXTUAL_SHARE_DECISIONS
        # ci_audit (L3a) and any other CI-audit variant default to instruction-only.
        return PrivacyAuditGuidanceMode.INSTRUCTION_ONLY

    def _privacy_read_audit_mode(self) -> PrivacyReadAuditMode:
        if self.prompt_variant == "ci_audit_contextual":
            return PrivacyReadAuditMode.CONTEXTUAL_SHARE
        return PrivacyReadAuditMode.INFORMATION_FLOWS
