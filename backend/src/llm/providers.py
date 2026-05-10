"""
LLM provider factory — model-agnostic, env-driven.

Two tiers:
- Triage LLM:    Fast, cheap. For parsing, classification, extraction.
- Reasoning LLM: Capable. For anomaly analysis, explanation generation.

Switch provider via LLM_PROVIDER env var: openai | anthropic
Switch models via TRIAGE_MODEL / REASONING_MODEL env vars.

Why a factory and not module-level instances?
  Module-level: instantiated at import time → env vars must be set before import
  Factory:      instantiated at call time  → env vars can be set after import
  The factory pattern is safer in test environments and FastAPI startup.
"""
import os
from functools import lru_cache
from langchain_core.language_models.chat_models import BaseChatModel


def _get_provider() -> str:
    """
    Read LLM_PROVIDER from env. Fail loudly if missing or invalid.
    Silent defaults hide misconfiguration bugs in production.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower().strip()
    if provider not in ("openai", "anthropic"):
        raise ValueError(
            f"LLM_PROVIDER must be 'openai' or 'anthropic', got: '{provider}'. "
            "Check your .env file."
        )
    return provider


def get_triage_llm() -> BaseChatModel:
    """
    Returns the fast/cheap model for lightweight tasks.
    Default: gpt-4o-mini (OpenAI) or claude-haiku-4-5 (Anthropic)

    Use for:
    - Structured data extraction
    - Classification tasks
    - Any node where speed > reasoning depth
    """
    provider = _get_provider()
    model_name = os.getenv("TRIAGE_MODEL")

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        model_name = model_name or "gpt-4o-mini"
        return ChatOpenAI(
            model=model_name,
            temperature=0,       # deterministic output for triage tasks
        )

    # provider == "anthropic"
    from langchain_anthropic import ChatAnthropic
    model_name = model_name or "claude-haiku-4-5-20251001"
    return ChatAnthropic(
        model=model_name,
        temperature=0,
    )


def get_reasoning_llm() -> BaseChatModel:
    """
    Returns the capable model for complex reasoning tasks.
    Default: gpt-4o (OpenAI) or claude-sonnet-4-6 (Anthropic)

    Use for:
    - Anomaly explanation generation
    - Multi-step analysis
    - Any node where quality > cost
    """
    provider = _get_provider()
    model_name = os.getenv("REASONING_MODEL")

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        model_name = model_name or "gpt-4o"
        return ChatOpenAI(
            model=model_name,
            temperature=0,
        )

    # provider == "anthropic"
    from langchain_anthropic import ChatAnthropic
    model_name = model_name or "claude-sonnet-4-6"
    return ChatAnthropic(
        model=model_name,
        temperature=0,
    )