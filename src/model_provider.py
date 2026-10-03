from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_PROVIDERS = ("openai", "custom", "gemini", "anthropic", "ollama", "openrouter")

_PROVIDER_ALIASES = {
    "openai": "openai",
    "gpt": "openai",
    "custom": "custom",
    "openai-compatible": "custom",
    "openai_compatible": "custom",
    "compatible": "custom",
    "gemini": "gemini",
    "google": "gemini",
    "google-genai": "gemini",
    "google_genai": "gemini",
    "anthropic": "anthropic",
    "anthorpic": "anthropic",
    "antropic": "anthropic",
    "claude": "anthropic",
    "ollama": "ollama",
    "openrouter": "openrouter",
    "open-router": "openrouter",
    "open_router": "openrouter",
}


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Supported providers:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None

    def is_live_ready(self) -> bool:
        """True when enough settings exist to call a real model."""

        if not self.model_name:
            return False
        if self.provider == "ollama":
            return True
        if self.provider == "custom":
            return bool(self.base_url)
        return bool(self.api_key)


def normalize_provider(value: str) -> str:
    """Map aliases like `anthorpic` -> `anthropic`; reject unknown providers."""

    key = (value or "").strip().lower()
    if key not in _PROVIDER_ALIASES:
        raise ValueError(f"Unsupported provider '{value}'. Supported: {', '.join(SUPPORTED_PROVIDERS)}")
    return _PROVIDER_ALIASES[key]


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider.

    Imports are lazy so offline mode never needs provider SDKs to be importable.
    """

    provider = normalize_provider(config.provider)
    common = {"model": config.model_name, "temperature": config.temperature}

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(**common, api_key=config.api_key, base_url=config.base_url)
    if provider == "custom":
        from langchain_openai import ChatOpenAI

        if not config.base_url:
            raise ValueError("Provider 'custom' requires a base_url (CUSTOM_BASE_URL).")
        return ChatOpenAI(**common, api_key=config.api_key or "not-needed", base_url=config.base_url)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(**common, api_key=config.api_key)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(**common, api_key=config.api_key)
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        if config.base_url:
            return ChatOllama(**common, base_url=config.base_url)
        return ChatOllama(**common)
    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        return ChatOpenRouter(**common, api_key=config.api_key)

    raise ValueError(f"Unsupported provider '{config.provider}'")
