"""Service setup diagnostics must keep configured credentials private."""

from unittest.mock import patch

import httpx
import pytest

from agentprivarena.bootstrap import bootstrap_bookstack
from agentprivarena.config import Config


def test_rejected_bookstack_token_is_not_disclosed() -> None:
    config = Config(
        bookstack_token_id="test-id-do-not-log",
        bookstack_token_secret="test-secret-do-not-log",
    )
    with (
        patch(
            "agentprivarena.bootstrap.httpx.get",
            return_value=httpx.Response(401),
        ),
        pytest.raises(RuntimeError) as error,
    ):
        bootstrap_bookstack(config)

    message = str(error.value)
    assert config.bookstack_token_id not in message
    assert config.bookstack_token_secret not in message
    assert "HTTP 401" in message
    assert "BookStack UI" in message
    assert "BOOKSTACK_TOKEN_ID and BOOKSTACK_TOKEN_SECRET" in message
    assert "agentprivarena/.env" in message
    assert "config.py" not in message
