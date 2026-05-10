"""
Tests for the LLM provider factory.

Key principle: we test the factory's behavior, not the LLM itself.
We never make real API calls in unit tests — that's slow, costly, and flaky.
We test: correct type returned, correct model name, error on bad config.
"""
import os
import pytest
from unittest.mock import patch
from langchain_core.language_models.chat_models import BaseChatModel


def test_get_triage_llm_returns_openai():
    """Factory returns ChatOpenAI when LLM_PROVIDER=openai."""
    with patch.dict(os.environ, {
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-test-fake-key",
    }):
        from src.llm.providers import get_triage_llm
        llm = get_triage_llm()
        # Check it's a valid LangChain chat model
        assert isinstance(llm, BaseChatModel)
        # Check it's specifically OpenAI
        assert "ChatOpenAI" in type(llm).__name__


def test_get_reasoning_llm_returns_openai():
    """Factory returns ChatOpenAI reasoning model for openai provider."""
    with patch.dict(os.environ, {
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-test-fake-key",
    }):
        from src.llm.providers import get_reasoning_llm
        llm = get_reasoning_llm()
        assert isinstance(llm, BaseChatModel)
        assert "ChatOpenAI" in type(llm).__name__


def test_get_triage_llm_returns_anthropic():
    """Factory returns ChatAnthropic when LLM_PROVIDER=anthropic."""
    with patch.dict(os.environ, {
        "LLM_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": "sk-ant-test-fake",
    }):
        from src.llm.providers import get_triage_llm
        llm = get_triage_llm()
        assert isinstance(llm, BaseChatModel)
        assert "ChatAnthropic" in type(llm).__name__


def test_invalid_provider_raises_value_error():
    """
    Factory must fail loudly on invalid provider.
    Silent defaults hide misconfiguration — we'd rather crash early.
    """
    with patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}):
        with pytest.raises(ValueError, match="LLM_PROVIDER must be"):
            from src.llm.providers import get_triage_llm
            get_triage_llm()


def test_custom_model_name_respected():
    """TRIAGE_MODEL env var overrides the default model name."""
    with patch.dict(os.environ, {
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "sk-test-fake-key",
        "TRIAGE_MODEL": "gpt-4o-mini",
    }):
        from src.llm.providers import get_triage_llm
        llm = get_triage_llm()
        # ChatOpenAI stores model name as .model_name
        assert llm.model_name == "gpt-4o-mini"