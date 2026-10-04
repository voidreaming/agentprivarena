from __future__ import annotations

from abc import ABC, abstractmethod

from openhands.sdk.privacy.flow import (
    AuditDecision,
    InformationFlow,
    PrivacyJudgment,
    TransmissionContext,
)
from openhands.sdk.tool.schema import Observation
from openhands.sdk.utils.models import DiscriminatedUnionMixin


class PrivacyAnalyzerBase(DiscriminatedUnionMixin, ABC):
    """Base class for privacy analyzers.

    Analyzers run at two well-defined points in the agent loop:

    1. **Decision boundary** (end of a read-tool batch, before the agent
       composes its next action): :meth:`audit_decision` receives the buffered
       raw read observations plus any prior accumulated inventory and returns
       an :class:`AuditDecision`. The inventory delta is attached to the batch's
       ObservationEvents so write-time judging continues to see a complete
       inventory; the optional instruction is emitted as a proactive
       ``MessageEvent`` to steer the agent before the next write.
    2. **Pre-write** (before a write tool executes): :meth:`judge_write_action`
       (via :meth:`judge_write_action_with_context`) returns a
       :class:`PrivacyJudgment` (PASS / ABSTRACT / BLOCK). Non-PASS short-
       circuits the tool with a ``UserRejectObservation``.

    This is the v2 contract. The legacy per-read ``extract_flows`` hook has
    been removed: extraction now happens once per read batch as part of
    :meth:`audit_decision`.
    """

    @abstractmethod
    def audit_decision(
        self,
        buffered_reads: list[tuple[Observation, str]],
        accumulated_flows: list[InformationFlow],
        transmission_context: TransmissionContext,
    ) -> AuditDecision:
        """Run one auditor pass at the end of a read batch.

        Args:
            buffered_reads: ``(observation, tool_name)`` pairs for the reads
                in this batch, in execution order. Empty inputs are valid;
                implementations should return an empty :class:`AuditDecision`.
            accumulated_flows: CI tuples collected from prior read batches in
                this conversation. Used to deduplicate the new inventory delta.
            transmission_context: Read-boundary context. ``data_recipient`` is
                typically empty (no pending write yet); ``principal`` and
                ``task_purpose`` are the audience-neutral steering inputs.

        Returns:
            :class:`AuditDecision` with a deduplicated ``inventory_delta`` and
            an optional ``instruction`` for proactive steering.
        """
        ...

    @abstractmethod
    def judge_write_action(
        self,
        write_content: str,
        accumulated_flows: list[InformationFlow],
        recipient_descriptor: str,
        principal: str,
    ) -> PrivacyJudgment:
        """Judge a write action against accumulated information flows.

        Called before write-tool execution. The agent loop short-circuits the
        tool on ABSTRACT/BLOCK, emitting a ``UserRejectObservation`` whose
        ``rejection_reason`` is the judgment rationale.
        """
        ...

    def judge_write_action_with_context(
        self,
        write_content: str,
        accumulated_flows: list[InformationFlow],
        transmission_context: TransmissionContext,
    ) -> PrivacyJudgment:
        """Judge a write action using a structured CI transmission context.

        The default implementation delegates to the legacy
        ``judge_write_action`` hook. New analyzers can override this method to
        use the full context directly.
        """
        return self.judge_write_action(
            write_content=write_content,
            accumulated_flows=accumulated_flows,
            recipient_descriptor=transmission_context.recipient_descriptor,
            principal=transmission_context.principal,
        )
