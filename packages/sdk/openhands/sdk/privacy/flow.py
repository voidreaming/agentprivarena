from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field


def information_flow_key(data_type: str, data_subject: str) -> tuple[str, str]:
    """Canonical key for deduplicating equivalent extracted flows."""
    normalized_type = data_type.casefold().replace("hospitalisation", "hospitalization")
    type_tokens = sorted(set(re.findall(r"[a-z0-9]+", normalized_type)))
    normalized_subject = re.sub(r"\s+", " ", data_subject.strip()).casefold()
    return (" ".join(type_tokens), normalized_subject)


class FlowDisposition(str, Enum):
    """Per-flow judgment of a single transmitted CI tuple."""

    PASS = "pass"
    ABSTRACT = "abstract"
    GENERALIZE = "generalize"
    OMIT = "omit"
    BLOCK = "block"


class InformationFlow(BaseModel):
    """A single audited CI tuple extracted from a tool observation.

    Represents one unit of information found in a tool result, decomposed into
    the subject of the information and the concrete attribute/fact being
    transmitted about that subject. The read-boundary auditor also attaches a
    proactive per-flow disposition and short reason so the execution model can
    decide what to use before drafting a write.
    """

    data_type: str = Field(
        description=(
            "The concrete attribute or fact about the subject "
            "(e.g., 'flu diagnosis', 'Friday project deadline', "
            "'Saturday parade break near the main square')."
        ),
    )
    data_subject: str = Field(
        description="The person or entity this information is about.",
    )
    disposition: FlowDisposition = Field(
        default=FlowDisposition.PASS,
        description=(
            "Read-boundary per-flow guidance for the likely task audience. "
            "PASS = usable as-is; ABSTRACT = use only at a more abstract "
            "level; BLOCK = do not send this flow to the likely audience. "
            "GENERALIZE and OMIT remain for write-time judge compatibility. "
            "Old serialized flows without this field load as PASS."
        ),
    )
    reason: str = Field(
        default="",
        description=(
            "Short audience-focused reason for the read-boundary disposition."
        ),
    )
    safe_use: str = Field(
        default="",
        description=(
            "Short instruction for how the execution model may use this flow. "
            "For example: use as-is, use only at a higher level, or omit."
        ),
    )
    share_question: str = Field(
        default="",
        description=(
            "Optional contextual share question asked at read time, for "
            "example whether this principal should share this fact with the "
            "expected recipient through the expected channel."
        ),
    )
    safe_rendering: str = Field(
        default="",
        description=(
            "Optional safe replacement wording for ABSTRACT flows. Empty for "
            "PASS when exact wording is acceptable and for BLOCK when the "
            "fact should be omitted entirely."
        ),
    )
    semantic_recipient: str = Field(
        default="",
        description=(
            "Read-boundary semantic recipient/audience inferred by the "
            "auditor from the task purpose, expected context, and reads. "
            "This is a human-meaningful audience label, not necessarily a "
            "tool argument such as an email address or username."
        ),
    )
    semantic_channel: str = Field(
        default="",
        description=(
            "Read-boundary semantic channel inferred by the auditor, such as "
            "email, Mattermost direct message, or public social post."
        ),
    )
    semantic_recipient_role: str = Field(
        default="",
        description=(
            "Optional inferred relationship or role of the recipient relative "
            "to the principal or data subject."
        ),
    )
    context_confidence: str = Field(
        default="",
        description=(
            "Auditor confidence in the inferred semantic context, usually "
            "high, medium, or low."
        ),
    )
    context_evidence: str = Field(
        default="",
        description=(
            "Short evidence string explaining where the inferred semantic "
            "recipient/channel came from."
        ),
    )
    source_action_ids: list[str] = Field(
        default_factory=list,
        description=(
            "ActionEvent ids of the reads whose observations were audited in "
            "the batch that produced this flow. Recorded at BATCH granularity: "
            "the auditor is given the batch's observations together and does "
            "not attribute a flow to an individual read, so claiming per-read "
            "provenance would be inventing attribution. Under the default "
            "sequential mode each batch contains exactly one eligible read, so "
            "in practice this is exact; it widens only when batching is on."
        ),
    )
    source_tools: list[str] = Field(
        default_factory=list,
        description=(
            "Tool names of the reads audited in the batch that produced this "
            "flow, in the same batch order as ``source_action_ids``."
        ),
    )


class JudgmentDecision(str, Enum):
    """Action-level judgment outcome, derived from per-flow dispositions."""

    PASS = "pass"
    ABSTRACT = "abstract"
    ESCALATE = "escalate"
    BLOCK = "block"


class TransmittedFlow(BaseModel):
    """An information flow identified as being transmitted by a write action."""

    data_type: str = Field(
        description="The specific information being transmitted.",
    )
    data_subject: str = Field(
        description="Whose data is being transmitted.",
    )
    data_recipient: str = Field(
        description="Who will receive this information.",
    )
    disposition: FlowDisposition = Field(
        default=FlowDisposition.PASS,
        description=(
            "Per-flow judgment. PASS = flow is appropriate for this "
            "recipient; GENERALIZE = flow should be swapped for a superclass "
            "on re-compose; OMIT = flow should be dropped entirely on "
            "re-compose; BLOCK = flow is inappropriate for this recipient "
            "under any generalization."
        ),
    )


class TransmissionContext(BaseModel):
    """Write-time CI context for judging an attempted transmission.

    This context is derived from the actual write tool call and the current
    task. It is intentionally policy-neutral: the analyzer still decides
    whether the attempted flow is appropriate.
    """

    data_recipient: str = Field(
        default="",
        description=(
            "Actual recipient or audience extracted from write-tool arguments "
            "and channel visibility hints."
        ),
    )
    transmission_channel: str = Field(
        default="",
        description=(
            "Tool/channel through which the content will be transmitted, "
            "including visibility when known."
        ),
    )
    principal: str = Field(
        default="",
        description="The user the agent is acting on behalf of.",
    )
    task_purpose: str = Field(
        default="",
        description=(
            "The user instruction or short purpose for the current task, used "
            "to judge whether this recipient needs the exact information."
        ),
    )
    recipient_role: str | None = Field(
        default=None,
        description="Optional role or relationship of the recipient when known.",
    )
    tool_name: str = Field(
        default="",
        description="The write tool that would transmit the content.",
    )
    previously_rejected: list[TransmittedFlow] = Field(
        default_factory=list,
        description=(
            "Items a prior judgment rejected for this task, when this write is "
            "a re-composition attempt. Lets the judge verify that those "
            "specific items were resolved instead of re-running the general "
            "criterion on the rewritten message -- the latter passes on the "
            "first retry essentially always, including when the item is still "
            "there. Empty on a first attempt."
        ),
    )

    @property
    def recipient_descriptor(self) -> str:
        """Backward-compatible audience label for legacy analyzer hooks."""
        if self.data_recipient and self.transmission_channel:
            if self.tool_name and self.tool_name not in self.transmission_channel:
                return (
                    f"{self.transmission_channel} via {self.tool_name} "
                    f"({self.data_recipient})"
                )
            return f"{self.transmission_channel} ({self.data_recipient})"
        return self.data_recipient or self.transmission_channel or self.tool_name


class PrivacyCheckResult(BaseModel):
    """Back-compat: transmitted flows identified for a write action.

    This model is retained for serialization compatibility with
    extraction-only L3 result files.  New code should use
    :class:`PrivacyJudgment`, which extends this shape with an action-level
    decision and rationale.
    """

    transmitted_flows: list[TransmittedFlow] = Field(
        default_factory=list,
        description=(
            "Information flows from the accumulated inventory "
            "that are being transmitted by this write action."
        ),
    )


class AuditDecision(BaseModel):
    """Result of a single decision-boundary audit pass.

    Produced once per read-tool batch by :meth:`PrivacyAnalyzerBase.audit_decision`.
    Carries both the information-flow inventory delta extracted from the batch
    and an optional proactive steering instruction that the agent loop surfaces
    before the agent composes its next action. The default agent loop attaches
    this text to the triggering read tool result instead of inserting a
    separate user-role message, so strict tool-call/tool-result histories stay
    valid. ``instruction`` is ``None`` (or empty) when the auditor sees nothing
    worth steering on.
    """

    inventory_delta: list[InformationFlow] = Field(
        default_factory=list,
        description=(
            "Information-flow inventory extracted from the buffered read "
            "observations in this batch. Deduplicated against any prior "
            "accumulated flows by the auditor."
        ),
    )
    instruction: str | None = Field(
        default=None,
        description=(
            "Optional proactive steering text. Audience-focused, scenario-"
            "neutral. When non-empty, the agent loop attaches it to the "
            "triggering read tool result before the agent's next step."
        ),
    )


class PrivacyJudgment(PrivacyCheckResult):
    """Result of the extract+judge pass over a write action.

    Extends :class:`PrivacyCheckResult` with an action-level decision
    derived from the per-flow dispositions on ``transmitted_flows``,
    plus a one-sentence rationale surfaced to the agent on
    ABSTRACT/BLOCK.
    """

    decision: JudgmentDecision = Field(
        default=JudgmentDecision.PASS,
        description=(
            "Action-level outcome. PASS = execute the write as-is. "
            "ABSTRACT = reject and ask the agent to re-compose without "
            "specifics. BLOCK = reject; recipient is not an appropriate "
            "audience for this content. ESCALATE = defer to human (reserved "
            "for interactive use; disabled in automated eval)."
        ),
    )
    rationale: str = Field(
        default="",
        description=(
            "One-sentence explanation shown to the agent on ABSTRACT/BLOCK. "
            "Reasons about audience appropriateness, not content sensitivity."
        ),
    )
    transmission_context: TransmissionContext | None = Field(
        default=None,
        description=(
            "Write-time context used by the audit model when judging this "
            "action. Optional for backward compatibility with older results."
        ),
    )
    unresolved_items: list[TransmittedFlow] = Field(
        default_factory=list,
        description=(
            "Previously rejected items the judge did not confirm as resolved in "
            "this re-composition. Non-empty forces a non-PASS decision in code, "
            "so a recompose cannot be waved through by the model simply "
            "restating the criterion. Always empty unless verification of "
            "re-composed writes is enabled."
        ),
    )
    recompose_exhausted: bool = Field(
        default=False,
        description=(
            "Set when the re-composition retry budget for this write tool ran "
            "out. The gate stays closed; the rejection directive changes to ask "
            "the agent to send without the flagged content or consult the user, "
            "instead of inviting another rewrite indefinitely."
        ),
    )
