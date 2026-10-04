from typing import Literal

from pydantic import Field
from rich.text import Text

from openhands.sdk.event.base import N_CHAR_PREVIEW, LLMConvertibleEvent
from openhands.sdk.event.types import EventID, SourceType, ToolCallID
from openhands.sdk.llm import Message, TextContent, content_to_str
from openhands.sdk.privacy.flow import (
    FlowDisposition,
    InformationFlow,
    PrivacyJudgment,
)
from openhands.sdk.tool.schema import Observation


# Source of action rejection - used to distinguish user rejections from hook blocks
RejectionSource = Literal["user", "hook"]


class ObservationBaseEvent(LLMConvertibleEvent):
    """Base class for anything as a response to a tool call.

    Examples include tool execution, error, user reject.
    """

    source: SourceType = "environment"
    tool_name: str = Field(
        ..., description="The tool name that this observation is responding to"
    )
    tool_call_id: ToolCallID = Field(
        ..., description="The tool call id that this observation is responding to"
    )


class ObservationEvent(ObservationBaseEvent):
    observation: Observation = Field(
        ..., description="The observation (tool call) sent to LLM"
    )
    action_id: EventID = Field(
        ..., description="The action id that this observation is responding to"
    )
    information_flows: list[InformationFlow] | None = Field(
        default=None,
        description=(
            "CI tuples extracted from this observation by the privacy analyzer."
        ),
    )
    information_flows_visible_to_llm: bool = Field(
        default=True,
        description=(
            "Whether extracted information flows are rendered into the LLM's "
            "tool observation. Flows remain persisted for downstream auditing "
            "when this is false."
        ),
    )
    audit_instruction: str | None = Field(
        default=None,
        description=(
            "Read-boundary privacy-auditor guidance attached to this tool "
            "observation. Keeping this in the tool response preserves strict "
            "LLM tool-call/tool-result message ordering."
        ),
    )
    privacy_check_result: PrivacyJudgment | None = Field(
        default=None,
        description=(
            "Privacy judgment on the write action: action-level decision "
            "(PASS / ABSTRACT / BLOCK), per-flow dispositions on each "
            "transmitted flow, and rationale. Old result files storing a "
            "plain PrivacyCheckResult deserialize here with the default "
            "PASS decision because PrivacyJudgment extends that shape."
        ),
    )
    privacy_audit_error: str | None = Field(
        default=None,
        description=(
            "Set when a privacy audit at this step degraded to a no-op because "
            "it raised. The audit is fail-open, so without this a crashed "
            "audit is indistinguishable from one that approved the action. "
            "Metadata only: never rendered into the LLM message, so recording "
            "it cannot change agent behaviour."
        ),
    )

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation of this observation event."""
        to_viz = self.observation.visualize
        content = Text()
        if to_viz.plain.strip():
            content.append("Tool: ", style="bold")
            content.append(self.tool_name)
            content.append("\nResult:\n", style="bold")
            content.append(to_viz)
        return content

    def to_llm_message(self) -> Message:
        content = list(self.observation.to_llm_content)
        if self.information_flows and self.information_flows_visible_to_llm:
            lines: list[str] = []
            for flow in self.information_flows:
                disposition = _display_flow_disposition(flow.disposition)
                line = (
                    f"- {flow.data_type} (subject: {flow.data_subject}); "
                    f"disposition: {disposition}"
                )
                reason = flow.reason.strip()
                if reason:
                    line += f"; reason: {reason}"
                lines.append(line)
            annotation = (
                "\n\n[INFORMATION INVENTORY]\n"
                "The following information was found in this result:\n"
                + "\n".join(lines)
            )
            content.append(TextContent(text=annotation))
        if self.audit_instruction:
            instruction = self.audit_instruction.strip()
            if instruction:
                content.append(
                    TextContent(
                        text=(
                            "\n\n[PRIVACY AUDITOR GUIDANCE]\n"
                            "The following guidance applies to this tool "
                            "result before you plan any write action:\n"
                            f"{instruction}"
                        )
                    )
                )
        return Message(
            role="tool",
            content=content,
            name=self.tool_name,
            tool_call_id=self.tool_call_id,
        )

    def __str__(self) -> str:
        """Plain text string representation for ObservationEvent."""
        base_str = f"{self.__class__.__name__} ({self.source})"
        content_str = "".join(content_to_str(self.observation.to_llm_content))
        obs_preview = (
            content_str[:N_CHAR_PREVIEW] + "..."
            if len(content_str) > N_CHAR_PREVIEW
            else content_str
        )
        return f"{base_str}\n  Tool: {self.tool_name}\n  Result: {obs_preview}"


def _display_flow_disposition(disposition: FlowDisposition) -> str:
    if disposition in (
        FlowDisposition.ABSTRACT,
        FlowDisposition.GENERALIZE,
        FlowDisposition.OMIT,
    ):
        return "abstract"
    return disposition.value


class UserRejectObservation(ObservationBaseEvent):
    """Observation when an action is rejected by user or hook.

    This event is emitted when:
    - User rejects an action during confirmation mode (rejection_source="user")
    - A PreToolUse hook blocks an action (rejection_source="hook")
    """

    rejection_reason: str = Field(
        default="User rejected the action",
        description="Reason for rejecting the action",
    )
    rejection_source: RejectionSource = Field(
        default="user",
        description=(
            "Source of the rejection: 'user' for confirmation mode rejections, "
            "'hook' for PreToolUse hook blocks"
        ),
    )
    action_id: EventID = Field(
        ..., description="The action id that this observation is responding to"
    )
    privacy_check_result: PrivacyJudgment | None = Field(
        default=None,
        description=(
            "Privacy judgment that caused this hook rejection, when the "
            "privacy analyzer rejected a write before execution."
        ),
    )

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation of this user rejection event."""
        content = Text()
        content.append("Tool: ", style="bold")
        content.append(self.tool_name)
        content.append("\n\nRejection Reason:\n", style="bold")
        content.append(self.rejection_reason)
        return content

    def to_llm_message(self) -> Message:
        return Message(
            role="tool",
            content=[TextContent(text=f"Action rejected: {self.rejection_reason}")],
            name=self.tool_name,
            tool_call_id=self.tool_call_id,
        )

    def __str__(self) -> str:
        """Plain text string representation for UserRejectObservation."""
        base_str = f"{self.__class__.__name__} ({self.source})"
        reason_preview = (
            self.rejection_reason[:N_CHAR_PREVIEW] + "..."
            if len(self.rejection_reason) > N_CHAR_PREVIEW
            else self.rejection_reason
        )
        return f"{base_str}\n  Tool: {self.tool_name}\n  Reason: {reason_preview}"


class AgentErrorEvent(ObservationBaseEvent):
    """Error triggered by the agent.

    Note: This event should not contain model "thought" or "reasoning_content". It
    represents an error produced by the agent/scaffold, not model output.
    """

    source: SourceType = "agent"
    error: str = Field(..., description="The error message from the scaffold")

    @property
    def visualize(self) -> Text:
        """Return Rich Text representation of this agent error event."""
        content = Text()
        content.append("Error Details:\n", style="bold")
        content.append(self.error)
        return content

    def to_llm_message(self) -> Message:
        # Provide plain string error content; serializers handle Chat vs Responses.
        # For Responses API, output is a string; JSON is not required.
        return Message(
            role="tool",
            content=[TextContent(text=self.error)],
            name=self.tool_name,
            tool_call_id=self.tool_call_id,
        )

    def __str__(self) -> str:
        """Plain text string representation for AgentErrorEvent."""
        base_str = f"{self.__class__.__name__} ({self.source})"
        error_preview = (
            self.error[:N_CHAR_PREVIEW] + "..."
            if len(self.error) > N_CHAR_PREVIEW
            else self.error
        )
        return f"{base_str}\n  Error: {error_preview}"
