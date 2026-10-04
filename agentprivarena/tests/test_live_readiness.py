"""Tests for live-readiness record checks."""

from __future__ import annotations

import asyncio

from agentprivarena.tasks.live_readiness import _check_mattermost


class _MattermostModule:
    async def read_messages(self, max_count: int) -> dict:
        assert max_count == 200
        return {
            "messages": [
                {
                    "text": "Seeded chat message",
                    "sender": "alex",
                    "sender_display": "Alex",
                }
            ]
        }

    async def list_users(self) -> dict:
        return {
            "users": [
                {
                    "username": "mom",
                    "name": "Mom",
                    "email": "mom@agentprivarena.local",
                }
            ]
        }


def test_mattermost_readiness_validates_user_registry_records() -> None:
    failures = asyncio.run(
        _check_mattermost(
            _MattermostModule(),
            [
                {
                    "type": "user",
                    "name": "mom",
                    "username": "mom",
                    "email": "mom@agentprivarena.local",
                },
                {
                    "message": "Seeded chat message",
                    "sender_id": "alex",
                    "sender_display": "Alex",
                },
            ],
        )
    )

    assert failures == []


def test_mattermost_readiness_reports_missing_user_registry_record() -> None:
    failures = asyncio.run(
        _check_mattermost(
            _MattermostModule(),
            [
                {
                    "type": "user",
                    "name": "emily",
                    "username": "emily",
                    "email": "emily@agentprivarena.local",
                },
            ],
        )
    )

    assert failures == ["mattermost[0] user not listed: 'emily'"]
