"""Focused tests for PrivacyLens-Live runtime config."""

from __future__ import annotations

from pathlib import Path

import pytest

from privacylens_live.config import Config


def test_from_env_accepts_openai_api_key_fallback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("EXTRACTION_LLM_API_KEY", raising=False)

    config = Config.from_env(tmp_path / "missing.env")

    assert config.llm_api_key == "test-key"
    assert config.extraction_llm_api_key == "test-key"


def test_from_env_keeps_specific_llm_api_keys_preferred(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("LLM_API_KEY", "main-key")
    monkeypatch.setenv("EXTRACTION_LLM_API_KEY", "extract-key")

    config = Config.from_env(tmp_path / "missing.env")

    assert config.llm_api_key == "main-key"
    assert config.extraction_llm_api_key == "extract-key"


def test_from_env_uses_openai_provider_for_foundry_v1_bare_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("LLM_MODEL", "gpt-5.4")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.openai.azure.com/openai/v1")

    config = Config.from_env(tmp_path / "missing.env")

    assert config.llm_model == "openai/gpt-5.4"


def test_native_provider_with_gateway_base_url_is_refused(monkeypatch):
    """Routing Gemini through the Azure gateway must fail loudly.

    The request would reach a gateway that does not serve that model; the run
    would either error or be served by something else, and because results
    record only the model name the substitution would be invisible afterwards.
    """
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "gemini/gemini-2.5-flash")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.openai.azure.com/openai/v1")
    with pytest.raises(ValueError, match="routed natively"):
        Config.from_env(env_file=Path("/nonexistent"))


def test_native_provider_with_base_url_none_uses_the_provider_endpoint(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "gemini/gemini-2.5-flash")
    monkeypatch.setenv("LLM_BASE_URL", "none")
    cfg = Config.from_env(env_file=Path("/nonexistent"))
    assert cfg.llm_base_url == ""
    assert cfg.llm_model == "gemini/gemini-2.5-flash"


def test_gateway_models_are_unaffected(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-5.4")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.openai.azure.com/openai/v1")
    cfg = Config.from_env(env_file=Path("/nonexistent"))
    assert cfg.llm_base_url.endswith("/openai/v1")


def test_audit_model_routing_is_checked_too(monkeypatch):
    """The auditor has its own credentials and its own way to be misrouted."""
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("EXTRACTION_LLM_MODEL", "gemini/gemini-2.5-pro")
    monkeypatch.setenv(
        "EXTRACTION_LLM_BASE_URL", "https://example.openai.azure.com/openai/v1"
    )
    with pytest.raises(ValueError, match="audit model"):
        Config.from_env(env_file=Path("/nonexistent"))
