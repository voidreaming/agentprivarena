"""Typed task contract for PrivacyLens-Live runner inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    """Task metadata consumed by the execution runner.

    The on-disk ``task.json`` file stays backward compatible. This class gives
    the runner a small typed boundary so benchmark adapters can target one
    contract instead of relying on ad hoc dict access throughout the pipeline.
    """

    name: str
    user_instruction: str
    dependencies: list[str]
    user_name: str | None = None
    expected_recipient: str | None = None
    expected_channel: str | None = None
    final_action_type: str | None = None
    sensitive_info_items: list[str] | None = None
    raw: dict[str, Any] | None = None

    @classmethod
    def from_task_dir(cls, task_dir: Path) -> TaskSpec:
        return cls.from_file(task_dir / "task.json")

    @classmethod
    def from_file(cls, task_json: Path) -> TaskSpec:
        data = json.loads(task_json.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{task_json} must contain a JSON object")

        name = _required_str(data, "name", task_json)
        user_instruction = _required_str(data, "user_instruction", task_json)
        dependencies = _string_list(data.get("dependencies", []), "dependencies")
        sensitive_info_items = _string_list(
            data.get("sensitive_info_items", []),
            "sensitive_info_items",
        )

        return cls(
            name=name,
            user_instruction=user_instruction,
            dependencies=dependencies,
            user_name=_optional_str(data.get("user_name"), "user_name"),
            expected_recipient=_optional_str(
                data.get("expected_recipient"),
                "expected_recipient",
            ),
            expected_channel=_optional_str(
                data.get("expected_channel"),
                "expected_channel",
            ),
            final_action_type=_optional_str(
                data.get("final_action_type"),
                "final_action_type",
            ),
            sensitive_info_items=sensitive_info_items,
            raw=dict(data),
        )

    def prompt_payload(self) -> dict[str, Any]:
        """Return the legacy dict shape expected by prompt builders."""
        if self.raw is not None:
            return dict(self.raw)
        return {
            "name": self.name,
            "user_instruction": self.user_instruction,
            "dependencies": list(self.dependencies),
            "user_name": self.user_name,
            "expected_recipient": self.expected_recipient,
            "expected_channel": self.expected_channel,
            "final_action_type": self.final_action_type,
            "sensitive_info_items": list(self.sensitive_info_items or []),
        }


def _required_str(data: dict[str, Any], key: str, source: Path) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source} is missing required string field {key!r}")
    return value


def _optional_str(value: Any, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Field {key!r} must be a string when present")
    return value


def _string_list(value: Any, key: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"Field {key!r} must be a list")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"Field {key!r} must contain only strings")
    return list(value)
