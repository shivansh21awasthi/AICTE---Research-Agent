"""
tests/test_config.py — Unit tests for Config loader (no IBM API required).
"""
import os
import pytest


def test_config_loads():
    from research_agent.config import Config
    cfg = Config()
    # Defaults should be present even without a filled .env
    assert cfg.watsonx_url.startswith("https://")
    assert cfg.granite_model_id == "ibm/granite-13b-instruct-v2"
    assert cfg.max_search_results > 0
    assert cfg.max_new_tokens > 0


def test_config_validate_raises_without_credentials():
    from research_agent.config import Config
    cfg = Config()
    # Default placeholder values → should raise
    with pytest.raises(ValueError, match="IBM_CLOUD_API_KEY|WATSONX_PROJECT_ID"):
        cfg.validate()


def test_config_validate_passes_with_credentials(monkeypatch):
    monkeypatch.setenv("IBM_CLOUD_API_KEY", "real_key_abc123")
    monkeypatch.setenv("WATSONX_PROJECT_ID", "real_project_xyz")
    from research_agent.config import Config
    cfg = Config()
    # Should not raise
    cfg.validate()
