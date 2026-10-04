"""Scripted LLM calls remain compatible with the base LLM privacy arguments."""

import pytest

from openhands.sdk.llm import Message, TextContent
from openhands.sdk.testing import TestLLM


@pytest.mark.parametrize("use_responses_api", [False, True])
def test_test_llm_accepts_positional_privacy_context(use_responses_api: bool) -> None:
    scripted = Message(role="assistant", content=[TextContent(text="Done")])
    llm = TestLLM.from_messages([scripted])

    if use_responses_api:
        response = llm.responses([], None, None, None, False, False, True, None)
    else:
        response = llm.completion([], None, False, False, True, None)

    assert response.message == scripted
    assert llm.call_count == 1
