"""LLM provider package — factory entry point."""

from __future__ import annotations

import logging

from mira.config import LLMConfig
from mira.llm.base import LLMProviderProtocol

logger = logging.getLogger(__name__)

# Model-id prefixes natively served by the Mistral API. Anything else pointed
# at the Mistral endpoint is almost certainly a misconfiguration (e.g. an
# OpenRouter-style id for another vendor's model) — warn, don't block, since
# free-form ids are a supported feature.
_MISTRAL_MODEL_PREFIXES = (
    "mistral",
    "magistral",
    "codestral",
    "devstral",
    "ministral",
    "pixtral",
    "open-mistral",
    "open-mixtral",
    # Third-party models hosted on La Plateforme (e.g. Z.ai GLM).
    "zai-",
)


def _warn_if_model_profile_mismatch(config: LLMConfig, profile: dict) -> None:
    """Warn when the model id looks wrong for the resolved provider profile.

    Currently only checks the Mistral endpoint: bare ids from other vendors
    (e.g. ``glm-5-2``) or other vendors' ``prefix/id`` forms will not resolve
    to a servable Mistral model. Warning-only — custom gateways may accept
    any id.
    """
    if profile.get("name") != "mistral":
        return
    candidate = config.model.split("/", 1)[1] if "/" in config.model else config.model
    if not candidate.lower().startswith(_MISTRAL_MODEL_PREFIXES):
        logger.warning(
            "Model %r does not look like a Mistral model id for base_url %r; "
            "check llm.model (e.g. mistral/mistral-medium-3-5)",
            config.model,
            config.base_url,
        )


def create_llm(config: LLMConfig) -> LLMProviderProtocol:
    """Create the appropriate LLM provider based on config.provider.

    Returns an instance satisfying LLMProviderProtocol.
    """
    if config.provider == "bedrock":
        from mira.llm.bedrock import BedrockProvider

        return BedrockProvider(config)

    if config.provider in {"codex-cli", "codex_cli", "codex"}:
        from mira.llm.codex_cli import CodexCLIProvider

        return CodexCLIProvider(config)

    if config.api_style == "responses":
        from mira.llm.responses import ResponsesProvider

        provider = ResponsesProvider(config)
        _warn_if_model_profile_mismatch(config, provider.profile)
        return provider

    # Default: OpenAI-compatible endpoint (OpenRouter, vLLM, Ollama, etc.)
    from mira.llm.provider import LLMProvider

    provider = LLMProvider(config)
    _warn_if_model_profile_mismatch(config, provider.profile)
    return provider
