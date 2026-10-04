"""Agent runner — orchestrates seed → run → collect for each task.

Uses OpenHands SDK with DockerWorkspace on the agentprivarena-net network
so the agent container can reach MCP servers via Docker DNS.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from agentprivarena.base._util import format_duration as _fmt_duration
from agentprivarena.base.seeder import Seeder
from agentprivarena.config import Config
from agentprivarena.runner.agent_profile import AgentProfile
from agentprivarena.runner.conversation_executor import ConversationExecutor
from agentprivarena.runner.event_collector import (
    EventCollector,
    ScenarioResult,
)
from agentprivarena.runner.forced_read import (
    build_initial_read_actions,
    validate_read_policy,
)
from agentprivarena.runner.mcp_config_builder import build_mcp_config
from agentprivarena.runner.prompt_builder import PROMPT_VERSION
from agentprivarena.runner.result_builder import build_scenario_result
from agentprivarena.runner.result_writer import (
    RunMetadata,
    write_scenario_result,
    write_summary,
)
from agentprivarena.runner.task_spec import TaskSpec
from openhands.sdk.privacy import AuditPolicy, PrivacyAuditMode, PrivacyStrictness


logger = logging.getLogger("agent_runner")


class AgentPrivArenaRunner:
    """Runs AgentPrivArena tasks against real services via OpenHands SDK."""

    def __init__(
        self,
        config: Config,
        max_clarification_rounds: int = 3,
        prompt_variant: str = "baseline",
        read_policy: str = "natural",
        disable_security_analyzer: bool = False,
        enable_privacy_analyzer: bool = False,
        privacy_audit_mode: PrivacyAuditMode | str = PrivacyAuditMode.FULL,
        audit_strictness: PrivacyStrictness | str = PrivacyStrictness.BALANCED,
        privacy_policy: str = "",
        audit_policy: AuditPolicy | str = AuditPolicy.CONTEXTUAL_INTEGRITY,
        audit_judge_blind: bool = False,
        audit_verify_recompose: bool = False,
        privacy_sequential_tool_calls: bool = True,
    ):
        normalized_audit_mode = PrivacyAuditMode(privacy_audit_mode)
        normalized_strictness = PrivacyStrictness(audit_strictness)
        normalized_policy = AuditPolicy(audit_policy)
        self.agent_profile = AgentProfile(
            prompt_variant=prompt_variant,
            disable_security_analyzer=disable_security_analyzer,
            enable_privacy_analyzer=enable_privacy_analyzer,
            privacy_audit_mode=normalized_audit_mode,
            audit_strictness=normalized_strictness,
            audit_policy=normalized_policy,
            privacy_policy=privacy_policy,
            audit_judge_blind=audit_judge_blind,
            audit_verify_recompose=audit_verify_recompose,
            privacy_sequential_tool_calls=privacy_sequential_tool_calls,
        )
        self.read_policy = validate_read_policy(read_policy)
        self.config = config
        self.max_clarification_rounds = max_clarification_rounds
        self.conversation_executor = ConversationExecutor(
            max_clarification_rounds=max_clarification_rounds
        )
        self.prompt_variant = self.agent_profile.prompt_variant
        self.disable_security_analyzer = self.agent_profile.disable_security_analyzer
        self.enable_privacy_analyzer = self.agent_profile.enable_privacy_analyzer
        self.privacy_audit_mode = self.agent_profile.privacy_audit_mode
        self.audit_strictness = self.agent_profile.audit_strictness
        self.run_metadata = RunMetadata(
            execution_model=config.llm_model,
            audit_model=(
                config.extraction_llm_model if self.enable_privacy_analyzer else None
            ),
            privacy_analyzer_enabled=self.enable_privacy_analyzer,
            privacy_audit_mode=(
                self.privacy_audit_mode.value if self.enable_privacy_analyzer else None
            ),
            agent_server_image=config.agent_server_image,
            privacy_audit_strictness=(
                self.audit_strictness.value if self.enable_privacy_analyzer else None
            ),
            privacy_audit_policy=(
                privacy_policy
                if (self.enable_privacy_analyzer and privacy_policy)
                else None
            ),
            audit_policy=(
                self.agent_profile.audit_policy.value
                if self.enable_privacy_analyzer
                else None
            ),
            audit_judge_blind=(
                audit_judge_blind if self.enable_privacy_analyzer else None
            ),
            audit_verify_recompose=(
                audit_verify_recompose if self.enable_privacy_analyzer else None
            ),
            privacy_sequential_tool_calls=(
                privacy_sequential_tool_calls if self.enable_privacy_analyzer else None
            ),
        )
        self.seeder = Seeder(
            bookstack_url=config.bookstack_url,
            bookstack_token_id=config.bookstack_token_id,
            bookstack_token_secret=config.bookstack_token_secret,
            mattermost_url=config.mattermost_url,
            mattermost_user=config.mattermost_user,
            mattermost_password=config.mattermost_password,
            rocketchat_url=config.rocketchat_url,
            rocketchat_user=config.rocketchat_user,
            rocketchat_password=config.rocketchat_password,
            mailpit_api_url=config.mailpit_api_url,
            mailpit_smtp_host=config.mailpit_smtp_host,
            mailpit_smtp_port=config.mailpit_smtp_port,
            gotosocial_url=config.gotosocial_url,
            gotosocial_token=config.gotosocial_token,
            radicale_url=config.radicale_url,
            radicale_user=config.radicale_user,
            radicale_password=config.radicale_password,
            google_drive_artifact_root=config.google_drive_artifact_root,
        )

    async def run_task(self, task_dir: Path) -> ScenarioResult:
        """Run a single AgentPrivArena task.

        1. Load task.json and seed data
        2. Seed services with observation data
        3. Build MCP config for this task's dependencies
        4. Create OpenHands agent with MCP tools
        5. Run in Docker sandbox on shared network
        6. Extract structured result (final action + tool_calls + stats)
        7. Cleanup seeded data
        """
        # Import OpenHands SDK here to allow running without it
        from openhands.sdk import Agent, Conversation
        from openhands.workspace import DockerWorkspace

        # 1. Load task spec
        task_json = task_dir / "task.json"
        if not task_json.exists():
            return ScenarioResult(
                name=task_dir.name,
                status="error",
                error=f"task.json not found in {task_dir}",
            )
        task = TaskSpec.from_file(task_json)
        name = task.name
        logger.info(f"Running task: {name}")

        # 2. Seed services
        seed_dir = task_dir / "seed_data"
        seed_result = None
        await self.seeder.reset_services(task.dependencies)
        logger.info("Reset task dependency services for %s", name)
        if seed_dir.exists():
            seed_result = await self.seeder.seed_from_dir(seed_dir)
            logger.info(f"Seeded {len(seed_result.handles)} services for {name}")

        # Pre-initialize so the except block can always introspect them.
        collector = EventCollector()
        clarifications: list[dict] = []
        initial_read_actions: list[dict] = []
        initial_read_actions_planned = self.read_policy != "forced_oracle"
        try:
            # 3. Build MCP config
            mcp_config = build_mcp_config(task.dependencies, self.config)
            logger.info(f"MCP config: {list(mcp_config['mcpServers'].keys())}")
            if self.read_policy == "forced_oracle":
                initial_read_actions = await build_initial_read_actions(
                    task_dir=task_dir,
                    seed_result=seed_result,
                    config=self.config,
                )
                initial_read_actions_planned = True
                logger.info(
                    "Planned %d forced initial read action(s) for %s",
                    len(initial_read_actions),
                    name,
                )

            initial_actions_for_agent = (
                initial_read_actions if self.read_policy == "forced_oracle" else None
            )
            agent_kwargs = self.agent_profile.build_agent_kwargs(
                task=task,
                config=self.config,
                mcp_config=mcp_config,
                initial_read_actions=initial_actions_for_agent,
            )
            agent = Agent(**agent_kwargs)

            # 5. Run in Docker sandbox.
            # For non-baseline prompt variants, bind-mount the host prompts
            # directory into the agent container at the same absolute path.
            # The SDK resolves ``system_prompt_filename`` with ``open()``
            # inside the agent server process (which runs in the container),
            # so the host path must also exist inside the container for
            # Jinja to load it. Mounting read-only to the identical path
            # keeps the path valid in both worlds with no translation.
            workspace_kwargs = self.agent_profile.build_workspace_kwargs(self.config)
            with DockerWorkspace(**workspace_kwargs) as workspace:
                conversation = Conversation(
                    agent=agent,
                    workspace=workspace,
                    callbacks=[collector.on_event],
                    max_iteration_per_run=self.config.max_iterations,
                )
                self.conversation_executor.run(
                    conversation=conversation,
                    initial_message=task.user_instruction,
                    collector=collector,
                    task_name=name,
                    clarifications=clarifications,
                )

            # 6. Build structured result
            return build_scenario_result(
                task=task,
                collector=collector,
                clarifications=clarifications,
                read_policy=self.read_policy,
                planned_initial_read_count=len(initial_read_actions),
                initial_read_actions_planned=initial_read_actions_planned,
            )

        except Exception as e:
            # Try to surface the actual ConversationErrorEvent body —
            # the default ``str(e)`` for a remote conversation failure
            # is just "Remote conversation ended with error" with no
            # detail.
            detail = collector.extract_error_detail()
            full_error = f"{e} | {detail}" if detail else str(e)
            logger.error(f"Task {name} failed: {full_error}")
            return build_scenario_result(
                task=task,
                collector=collector,
                clarifications=clarifications,
                read_policy=self.read_policy,
                planned_initial_read_count=len(initial_read_actions),
                initial_read_actions_planned=initial_read_actions_planned,
                status="error",
                error=full_error,
                include_final_action=False,
            )

        finally:
            # 7. Cleanup
            if seed_result:
                await self.seeder.cleanup(seed_result)
                logger.info(f"Cleaned up seed data for {name}")

    async def run_tasks(
        self,
        task_dirs: list[Path],
        results_dir: Path | None = None,
    ) -> list[ScenarioResult]:
        """Run multiple tasks sequentially and save results.

        For each task, writes two files:

        - ``<results_dir>/<name>.json`` — clean structured result
          (final_action, tool_calls, stats, etc.)
        - ``<results_dir>/<name>.events.json`` — raw event dump from
          the SDK, kept as a sidecar for deep debugging

        After all tasks finish, writes ``<results_dir>/_summary.json``
        with batch-level aggregates (counts by status, total elapsed,
        means per task).

        Each ``run_task`` call is wrapped in a defensive try/except so
        that an unexpected error in any single task can never kill the
        entire batch — the failed task is recorded as ``status="error"``
        and the loop continues with the next one.
        """
        if results_dir:
            results_dir.mkdir(parents=True, exist_ok=True)

        results: list[ScenarioResult] = []
        total = len(task_dirs)
        batch_start = time.time()

        for i, task_dir in enumerate(task_dirs):
            task_start = time.time()
            elapsed_batch = task_start - batch_start
            progress_pct = (i / total * 100) if total else 0.0
            eta_str = (
                f"ETA {_fmt_duration((elapsed_batch / i) * (total - i))}"
                if i > 0
                else "ETA --:--"
            )
            logger.info(
                f"[{i + 1:3d}/{total} | {progress_pct:5.1f}%] "
                f"{task_dir.name}  "
                f"(batch elapsed {_fmt_duration(elapsed_batch)}, {eta_str})"
            )

            # Defensive: any unexpected exception below run_task's own
            # try/except still produces an error result so the batch
            # continues. Without this, an asyncio cancellation or an
            # error in seed_from_dir would crash the whole loop.
            try:
                result = await self.run_task(task_dir)
            except Exception as e:
                logger.error(f"Task {task_dir.name} crashed at runner level: {e}")
                result = ScenarioResult(
                    name=task_dir.name,
                    status="error",
                    read_policy=self.read_policy,
                    forced_read_success=False
                    if self.read_policy == "forced_oracle"
                    else None,
                    error=f"runner-level exception: {type(e).__name__}: {e}",
                )

            results.append(result)
            task_elapsed = time.time() - task_start

            if results_dir:
                write_scenario_result(
                    result,
                    results_dir=results_dir,
                    prompt_variant=self.prompt_variant,
                    prompt_version=PROMPT_VERSION,
                    security_analyzer_disabled=self.disable_security_analyzer,
                    run_metadata=self.run_metadata,
                )

            stats = result.stats or {}
            logger.info(
                f"  → {result.status:9}  "
                f"clarif={stats.get('clarification_rounds', 0)} "
                f"calls={stats.get('tool_call_count', 0)} "
                f"errs={stats.get('errors_recovered', 0)}  "
                f"task elapsed {_fmt_duration(task_elapsed)}"
            )

        batch_end = time.time()

        if results_dir:
            write_summary(
                results=results,
                results_dir=results_dir,
                prompt_variant=self.prompt_variant,
                read_policy=self.read_policy,
                prompt_version=PROMPT_VERSION,
                security_analyzer_disabled=self.disable_security_analyzer,
                run_metadata=self.run_metadata,
                batch_start=batch_start,
                batch_end=batch_end,
            )

        return results
