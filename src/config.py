from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider

DEFAULT_COMPACT_THRESHOLD_TOKENS = 600
DEFAULT_COMPACT_KEEP_MESSAGES = 4
DEFAULT_PROFILE_CONFIDENCE_THRESHOLD = 0.6

# Only OpenAI gets a default model name (the README example). Other providers
# must set LLM_MODEL or their own *_MODEL variable; otherwise live mode stays off.
_DEFAULT_MODELS = {"openai": "gpt-4o-mini"}

_API_KEY_VARS = {
    "openai": ("OPENAI_API_KEY",),
    "custom": ("CUSTOM_API_KEY", "OPENAI_API_KEY"),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "ollama": (),
    "openrouter": ("OPENROUTER_API_KEY",),
}

_BASE_URL_VARS = {
    "openai": ("OPENAI_BASE_URL",),
    "custom": ("CUSTOM_BASE_URL",),
    "gemini": (),
    "anthropic": (),
    "ollama": ("OLLAMA_BASE_URL",),
    "openrouter": (),
}

_MODEL_VARS = {
    "openai": ("OPENAI_MODEL",),
    "custom": ("CUSTOM_MODEL",),
    "gemini": ("GEMINI_MODEL",),
    "anthropic": ("ANTHROPIC_MODEL",),
    "ollama": ("OLLAMA_MODEL",),
    "openrouter": ("OPENROUTER_MODEL",),
}


@dataclass
class LabConfig:
    """Shared configuration for the lab: paths, compact-memory knobs, providers."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    profile_confidence_threshold: float = DEFAULT_PROFILE_CONFIDENCE_THRESHOLD


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


def _provider_from_env(prefix: str, fallback: ProviderConfig | None = None) -> ProviderConfig:
    """Build a ProviderConfig from `<prefix>_PROVIDER`, `<prefix>_MODEL`, ..."""

    raw_provider = os.getenv(f"{prefix}_PROVIDER") or (fallback.provider if fallback else "openai")
    provider = normalize_provider(raw_provider)
    model_name = (
        os.getenv(f"{prefix}_MODEL")
        or _first_env(_MODEL_VARS[provider])
        or (fallback.model_name if fallback and fallback.provider == provider else None)
        or _DEFAULT_MODELS.get(provider, "")
    )
    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=_env_float(f"{prefix}_TEMPERATURE", 0.0),
        api_key=_first_env(_API_KEY_VARS[provider]),
        base_url=_first_env(_BASE_URL_VARS[provider]),
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load `.env` (if present) and return a populated LabConfig.

    Environment variables:
    - LLM_PROVIDER / LLM_MODEL / LLM_TEMPERATURE
    - JUDGE_PROVIDER / JUDGE_MODEL / JUDGE_TEMPERATURE (default: same as LLM)
    - OPENAI_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY
    - CUSTOM_BASE_URL / CUSTOM_API_KEY, OLLAMA_BASE_URL
    - COMPACT_THRESHOLD_TOKENS, COMPACT_KEEP_MESSAGES, PROFILE_CONFIDENCE_THRESHOLD
    """

    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()

    try:
        from dotenv import load_dotenv

        load_dotenv(root / ".env", override=False)
    except ImportError:
        pass

    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    model = _provider_from_env("LLM")
    judge_model = _provider_from_env("JUDGE", fallback=model)

    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=_env_int("COMPACT_THRESHOLD_TOKENS", DEFAULT_COMPACT_THRESHOLD_TOKENS),
        compact_keep_messages=_env_int("COMPACT_KEEP_MESSAGES", DEFAULT_COMPACT_KEEP_MESSAGES),
        model=model,
        judge_model=judge_model,
        profile_confidence_threshold=_env_float(
            "PROFILE_CONFIDENCE_THRESHOLD", DEFAULT_PROFILE_CONFIDENCE_THRESHOLD
        ),
    )
