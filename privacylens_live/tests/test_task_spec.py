"""Tests for the runner task contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from privacylens_live.runner.task_spec import TaskSpec


def test_task_spec_loads_legacy_task_json(tmp_path: Path) -> None:
    task_dir = tmp_path / "case1"
    task_dir.mkdir()
    (task_dir / "task.json").write_text(
        json.dumps(
            {
                "name": "case1",
                "user_instruction": "Email the update.",
                "dependencies": ["mailpit"],
                "user_name": "Jane",
                "expected_recipient": "Mike",
                "expected_channel": "Mattermost DM",
                "final_action_type": "email",
                "sensitive_info_items": ["diagnosis"],
                "extra_field": {"kept": True},
            }
        )
    )

    spec = TaskSpec.from_task_dir(task_dir)

    assert spec.name == "case1"
    assert spec.user_instruction == "Email the update."
    assert spec.dependencies == ["mailpit"]
    assert spec.user_name == "Jane"
    assert spec.expected_recipient == "Mike"
    assert spec.expected_channel == "Mattermost DM"
    assert spec.final_action_type == "email"
    assert spec.sensitive_info_items == ["diagnosis"]
    assert spec.prompt_payload()["extra_field"] == {"kept": True}


def test_task_spec_rejects_invalid_dependencies(tmp_path: Path) -> None:
    task_json = tmp_path / "task.json"
    task_json.write_text(
        json.dumps(
            {
                "name": "bad",
                "user_instruction": "Do it.",
                "dependencies": "mailpit",
            }
        )
    )

    with pytest.raises(ValueError, match="dependencies"):
        TaskSpec.from_file(task_json)
