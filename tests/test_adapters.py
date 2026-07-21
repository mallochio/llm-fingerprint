"""Unit tests for adapters (Mock, CLI, OpenAI)."""

from fingerprint.adapters import load_adapter, MockAdapter, CLIAdapter, OpenAICompatAdapter


def test_mock_adapter():
    adapter = MockAdapter(target_profile="gpt-4o")
    res = adapter.complete("Name a random color.")
    assert res["valid"] is True
    assert isinstance(res["raw_text"], str)
    assert res["latency_ms"] > 0


def test_load_builtin_adapter():
    ad = load_adapter("cursor_auto")
    assert isinstance(ad, CLIAdapter)
    assert ad.name == "cursor_auto"
    assert ad.environment == "cursor-cli"


def test_load_openai_adapter():
    ad = load_adapter("openai", overrides={"model": "gpt-4o-mini"})
    assert isinstance(ad, OpenAICompatAdapter)
    assert ad.model == "gpt-4o-mini"
    assert ad.environment == "openai-api"
