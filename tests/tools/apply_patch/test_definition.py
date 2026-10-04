"""Native patch tools accept the shared privacy-aware serialization interface."""

from openhands.sdk.tool import ToolDefinition
from openhands.tools.apply_patch.definition import (
    ApplyPatchAction,
    ApplyPatchObservation,
    ApplyPatchTool,
)


def test_native_patch_schema_accepts_privacy_context_option() -> None:
    tool: ToolDefinition = ApplyPatchTool(
        description="Apply a patch",
        action_type=ApplyPatchAction,
        observation_type=ApplyPatchObservation,
    )

    schema = tool.to_responses_tool(
        add_privacy_context=True,
        action_type=ApplyPatchAction,
    )

    assert schema["type"] == "function"
    assert schema["parameters"] == {
        "type": "object",
        "properties": {"patch": {"type": "string"}},
        "required": ["patch"],
    }
