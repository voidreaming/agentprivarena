"""Portable runtime configuration and preservation of local user settings."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from agentprivarena.config import Config


@patch.dict(os.environ, {}, clear=True)
def test_clean_config_needs_no_private_endpoint_or_provisioned_tokens(
    tmp_path: Path,
) -> None:
    config = Config.from_env(tmp_path / "missing.env")

    assert config.llm_base_url == ""
    assert config.extraction_llm_base_url == ""
    assert config.bookstack_token_id == ""
    assert config.bookstack_token_secret == ""
    assert config.llm_api_key == ""
    assert not config.google_drive_artifact_root.is_absolute()


@pytest.mark.parametrize(
    ("base_url", "expected_model"),
    [
        ("", "openai/test-model"),
        ("none", "openai/test-model"),
        ("direct", "openai/test-model"),
        ("-", "openai/test-model"),
        ("https://example.openai.azure.com/openai/v1", "openai/test-model"),
        ("https://example.openai.azure.com/", "azure/test-model"),
    ],
)
@patch.dict(os.environ, {}, clear=True)
def test_bare_model_uses_portable_or_explicit_legacy_provider(
    tmp_path: Path,
    base_url: str,
    expected_model: str,
) -> None:
    os.environ["LLM_MODEL"] = "test-model"
    os.environ["LLM_BASE_URL"] = base_url

    config = Config.from_env(tmp_path / "missing.env")

    assert config.llm_model == expected_model


@pytest.mark.parametrize("base_url", ["none", "direct", "-", ""])
@patch.dict(os.environ, {}, clear=True)
def test_extraction_model_accepts_provider_default_endpoint(
    tmp_path: Path,
    base_url: str,
) -> None:
    os.environ["EXTRACTION_LLM_MODEL"] = "gemini/test-model"
    os.environ["EXTRACTION_LLM_BASE_URL"] = base_url

    config = Config.from_env(tmp_path / "missing.env")

    assert config.extraction_llm_base_url == ""
    assert config.extraction_llm_model == "gemini/test-model"


@patch.dict(os.environ, {}, clear=True)
def test_setup_preserves_user_settings_and_environment_precedence(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    user_settings = (
        'LLM_MODEL="openai/test-agent"\n'
        "LLM_BASE_URL=https://example.invalid/v1\n"
        "LLM_API_KEY=test-file-key\n"
        "EXTRACTION_LLM_MODEL=openai/test-auditor\n"
        "EXTRACTION_LLM_BASE_URL=http://localhost:8001/v1\n"
        "AGENT_SERVER_IMAGE=test-agent:custom\n"
        "BOOKSTACK_HOST_PORT=3300\n"
    )
    env_file.write_text(user_settings + "BOOKSTACK_TOKEN_ID=old-token\n")
    os.environ["BOOKSTACK_TOKEN_ID"] = "provisioned-token"
    os.environ["LLM_API_KEY"] = "test-process-key"

    config = Config.from_env(env_file)

    assert config.llm_model == "openai/test-agent"
    assert config.llm_base_url == "https://example.invalid/v1"
    assert config.llm_api_key == "test-process-key"
    assert config.extraction_llm_base_url == "http://localhost:8001/v1"
    assert config.agent_server_image == "test-agent:custom"
    assert config.bookstack_token_id == "provisioned-token"

    config.write_env_file(env_file)
    first_write = env_file.read_text()
    assert user_settings in first_write
    assert "BOOKSTACK_TOKEN_ID=provisioned-token\n" in first_write
    assert "old-token" not in first_write
    assert "test-process-key" not in first_write

    config.write_env_file(env_file)
    assert env_file.read_text() == first_write


def test_new_env_file_does_not_copy_llm_keys_from_config(tmp_path: Path) -> None:
    config = Config(
        llm_api_key="test-agent-secret",
        extraction_llm_api_key="test-auditor-secret",
    )

    env_file = config.write_env_file(tmp_path / ".env")

    assert "test-agent-secret" not in env_file.read_text()
    assert "test-auditor-secret" not in env_file.read_text()
