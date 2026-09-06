"""Provider protocol — the interface that all LLM backends must satisfy."""

from __future__ import annotations

import logging
import os
import random
from datetime import UTC, datetime
from typing import ClassVar, Protocol, runtime_checkable

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from mira.config import LLMConfig
from mira.exceptions import LLMError, NonRetriableLLMError
from mira.llm import provider_profiles as profiles
from mira.llm.tool_schemas import SUBMIT_REVIEW_TOOL, SUBMIT_WALKTHROUGH_TOOL

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMProviderProtocol(Protocol):
    """Structural interface for LLM providers.

    Both the OpenAI-compatible provider and direct-API providers
    (Bedrock, Anthropic, Vertex, etc.) satisfy this protocol.

    Capability annotations:
        supports_json_mode: Provider natively supports response_format=json_object.
        supports_tool_calling: Provider supports function/tool calling.
    """

    supports_json_mode: bool
    supports_tool_calling: bool

    total_prompt_tokens: int
    total_completion_tokens: int

    async def complete(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = True,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str: ...

    async def complete_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        temperature: float | None = None,
    ) -> str: ...

    async def complete_agentic(
        self,
        messages: list,
        tools: list[dict],
        temperature: float | None = None,
    ) -> dict: ...

    async def review(
        self, messages: list[dict[str, str]], temperature: float | None = None
    ) -> str: ...

    async def walkthrough(self, messages: list[dict[str, str]]) -> str: ...

    def count_tokens(self, text: str) -> int: ...

    @property
    def usage(self) -> dict[str, int]: ...


# ── Module-level helpers (shared by both providers) ─────────────────


def _get_api_key(config: LLMConfig, profile: dict | None = None) -> str:
    """Resolve the API key for the configured endpoint.

    Reads `config.api_key_env` first, then the matched provider profile's
    `api_key_env`, then the legacy `OPENROUTER_API_KEY` / `OPENAI_API_KEY`
    lookup for backward compatibility. If `api_key_env` is explicitly "" the
    empty string is returned without error — useful for local endpoints
    (Ollama, llama.cpp server) that don't require auth.
    """
    if config.api_key_env == "":
        return ""
    key = os.environ.get(config.api_key_env, "")
    if not key and profile and profile.get("api_key_env"):
        key = os.environ.get(profile["api_key_env"], "")
    if not key:
        # Back-compat with pre-`api_key_env` setups.
        key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise LLMError("no_api_key", api_key_env=config.api_key_env)
    return key


def _strip_model_prefix(model: str, base_url: str) -> str:
    """Apply the endpoint's model-prefix policy from its provider profile.

    'keep' (OpenRouter) routes on the full `vendor/model` string and only sheds
    a redundant self-prefix (`openrouter/…`). 'strip' (the default for other
    endpoints) sends the bare model name (e.g. 'minimax/MiniMax-M2.7' →
    'MiniMax-M2.7').
    """
    profile = profiles.resolve(base_url)
    if profile.get("model_prefix") == "keep":
        self_prefix = f"{profile['name']}/"
        return model[len(self_prefix) :] if model.startswith(self_prefix) else model
    return model.split("/", 1)[1] if "/" in model else model


def _retriable(exception: BaseException) -> bool:
    """Return True for transient errors; False for 4xx non-retriable errors."""
    if isinstance(exception, NonRetriableLLMError):
        return False
    return isinstance(
        exception,
        (httpx.TimeoutException, httpx.NetworkError, LLMError),
    )


# Accepted HTTP-date shapes for Retry-After (RFC 7231 §7.1.3 prefers the
# IMF-fixdate form; the obsolete two-digit-year variants are vanishingly
# rare from real gateways and fall back to exponential backoff).
_RETRY_AFTER_DATE_FORMATS = ("%a, %d %b %Y %H:%M:%S GMT", "%a, %d %b %Y %H:%M:%S UTC")


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a ``Retry-After`` header value into seconds, or None.

    Accepts delay-seconds (``"120"``) and HTTP-dates. Returns None for
    missing, malformed, or non-positive values — callers fall back to the
    configured exponential backoff.
    """
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        seconds = float(value)
        return seconds if seconds > 0 else None
    for fmt in _RETRY_AFTER_DATE_FORMATS:
        try:
            retry_at = datetime.strptime(value, fmt).replace(tzinfo=UTC)
            break
        except ValueError:
            continue
    else:
        return None
    seconds = (retry_at - datetime.now(UTC)).total_seconds()
    return seconds if seconds > 0 else None


def _rate_limit_hint(exception: BaseException) -> float | None:
    """Server-provided retry delay in seconds, following ``__cause__`` chains.

    ``_handle_error`` attaches ``retry_after_hint`` to the 429 ``LLMError``;
    public API wrappers re-raise chained errors, so walk the chain.
    """
    seen: set[int] = set()
    current: BaseException | None = exception
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        hint = getattr(current, "retry_after_hint", None)
        if isinstance(hint, (int, float)) and hint > 0:
            return float(hint)
        current = current.__cause__ or current.__context__
    return None
# ── Shared base for OpenAI-compatible providers ─────────────────────


class OpenAICompatibleProvider:
    """Protocol-agnostic base for OpenAI-compatible API providers.

    Captures the code shared by ``LLMProvider`` (chat/completions) and
    ``ResponsesProvider`` (/responses protocol): retry setup, header
    building, reasoning, fallback model logic, token accounting, and the
    public API surface.  Protocol-specific paths (URL, input/output
    format, tool transformation) remain in each subclass.
    """

    supports_json_mode: ClassVar[bool] = True
    supports_tool_calling: ClassVar[bool] = True

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.profile = profiles.resolve(config.base_url)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._no_forced_tool_choice: set[str] = set()
        self._no_reasoning: set[str] = set()

        # Apply retry decorator imperatively so it reads config values
        # (max_retries, retry_min_wait, retry_max_wait) at instance time.
        base_wait = wait_exponential(
            multiplier=1,
            min=config.retry_min_wait,
            max=config.retry_max_wait,
        )

        def _wait(retry_state) -> float:
            """Exponential backoff, honoring the server's Retry-After hint.

            The exponential wait is the floor; a parsed ``Retry-After`` delay
            (capped at ``retry_max_wait``) extends it so concurrent lanes back
            off together instead of re-hammering a rate-limited endpoint. A
            small jitter decorrelates lanes. Zero-config waits stay zero so
            unit tests with ``retry_min_wait=retry_max_wait=0`` stay fast.
            """
            wait = base_wait(retry_state)
            try:
                hint = _rate_limit_hint(retry_state.outcome.exception())
            except Exception:
                hint = None
            if hint is not None:
                wait = max(wait, min(hint, config.retry_max_wait))
            if wait > 0:
                wait += random.uniform(0, min(1.0, wait))
            return wait

        self._retry = retry(
            stop=stop_after_attempt(config.max_retries),
            wait=_wait,
            retry=retry_if_exception(_retriable),
            reraise=True,
        )
        # Decorate the concrete subclass methods with retry logic.
        self._call_llm = self._retry(self._call_llm)
        self._call_llm_with_tools = self._retry(self._call_llm_with_tools)
        self._call_llm_agentic = self._retry(self._call_llm_agentic)

    # ── Shared helpers ─────────────────────────────────────────────

    def _build_headers(self) -> dict[str, str]:
        """Build request headers: Content-Type, optional Bearer auth, and any
        provider-specific extras from the profile. Authorization is omitted
        entirely if the endpoint needs no key (Ollama, llama.cpp, etc.)."""
        if hasattr(self, "_cached_headers"):
            return dict(self._cached_headers)
        headers: dict[str, str] = {"Content-Type": "application/json"}
        key = _get_api_key(self.config, self.profile)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        headers.update(self.profile.get("extra_headers", {}))
        self._cached_headers = headers
        return dict(headers)

    def _apply_reasoning(self, body: dict) -> None:
        """Enable extended thinking when a reasoning effort is configured.

        The effort is passed via the unified ``reasoning.effort`` knob, after
        any per-provider remap from the profile. Anthropic models reject a
        custom ``temperature`` while thinking is on, so we drop it.
        No-op when reasoning is off, keeping the request unchanged.
        """
        effort = self.config.reasoning_effort
        if not effort or effort == "off":
            return
        if body.get("model") in self._no_reasoning:
            return
        effort = self.profile.get("reasoning_effort_map", {}).get(effort, effort)
        body["reasoning"] = {"effort": effort}
        body.pop("temperature", None)

    def _account_usage(self, data: dict) -> None:
        """Accumulate token counts. Default: chat/completions key names.

        Subclasses override when the API uses different keys
        (e.g. Responses API uses ``input_tokens`` / ``output_tokens``).
        """
        usage = data.get("usage")
        if usage:
            self.total_prompt_tokens += usage.get("prompt_tokens", 0)
            self.total_completion_tokens += usage.get("completion_tokens", 0)

    @staticmethod
    def _handle_error(resp: httpx.Response) -> None:
        """Raise LLMError or NonRetriableLLMError on non-200 responses.

        On 429 the ``Retry-After`` header (when present and parseable) is
        attached to the error as ``retry_after_hint`` so the retry wait can
        honor the server's backoff instead of re-hammering the endpoint.
        """
        if resp.status_code != 200:
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                raise NonRetriableLLMError("api_error", status=resp.status_code, body=resp.text)
            err = LLMError("api_error", status=resp.status_code, body=resp.text)
            if resp.status_code == 429:
                try:
                    hint = _parse_retry_after(resp.headers.get("retry-after"))
                except Exception:
                    hint = None
                remaining = None
                try:
                    remaining = resp.headers.get("x-ratelimit-remaining")
                except Exception:
                    remaining = None
                if hint is not None:
                    err.retry_after_hint = hint
                logger.warning(
                    "LLM rate-limited (429)%s%s",
                    f"; retry after {hint:g}s" if hint is not None else "",
                    f"; X-RateLimit-Remaining: {remaining}" if remaining else "",
                )
            raise err

    # ── Subclass hooks (abstract) ──────────────────────────────────

    async def _call_llm(
        self,
        model: str,
        messages: list[dict[str, str]],
        json_mode: bool,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        raise NotImplementedError

    async def _call_llm_with_tools(
        self,
        model: str,
        messages: list[dict[str, str]],
        tools: list[dict],
        temperature: float | None = None,
    ) -> str:
        raise NotImplementedError

    async def _call_llm_agentic(
        self,
        model: str,
        messages: list,
        tools: list[dict],
        temperature: float | None = None,
    ) -> dict:
        raise NotImplementedError

    # ── Public API (shared across chat and responses providers) ─────

    async def complete(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = True,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Complete a prompt using JSON mode, with fallback model support."""
        try:
            return await self._call_llm(
                self.config.model,
                messages,
                json_mode,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except NonRetriableLLMError:
            raise
        except Exception as primary_err:
            if self.config.fallback_model:
                logger.warning(
                    "Primary model %s failed (%s), trying fallback %s",
                    self.config.model,
                    primary_err,
                    self.config.fallback_model,
                )
                try:
                    return await self._call_llm(
                        self.config.fallback_model,
                        messages,
                        json_mode,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except Exception as fallback_err:
                    raise LLMError(
                        "both_models_failed",
                        primary_model=self.config.model,
                        fallback_model=self.config.fallback_model,
                        error=fallback_err,
                    ) from fallback_err
            raise LLMError(
                "completion_failed", model=self.config.model, error=primary_err
            ) from primary_err

    async def complete_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict],
        temperature: float | None = None,
    ) -> str:
        """Complete a prompt using tool calling for structured output."""
        try:
            return await self._call_llm_with_tools(
                self.config.model, messages, tools, temperature=temperature
            )
        except NonRetriableLLMError:
            raise
        except Exception as primary_err:
            if self.config.fallback_model:
                logger.warning(
                    "Primary model %s failed (%s), trying fallback %s",
                    self.config.model,
                    primary_err,
                    self.config.fallback_model,
                )
                try:
                    return await self._call_llm_with_tools(
                        self.config.fallback_model,
                        messages,
                        tools,
                        temperature=temperature,
                    )
                except Exception as fallback_err:
                    raise LLMError(
                        "both_models_failed",
                        primary_model=self.config.model,
                        fallback_model=self.config.fallback_model,
                        error=fallback_err,
                    ) from fallback_err
            raise LLMError(
                "tool_call_failed", model=self.config.model, error=primary_err
            ) from primary_err

    async def complete_agentic(
        self,
        messages: list,
        tools: list[dict],
        temperature: float | None = None,
    ) -> dict:
        """Single hop of an agentic loop. Returns the assistant message dict."""
        try:
            return await self._call_llm_agentic(
                self.config.model, messages, tools, temperature=temperature
            )
        except NonRetriableLLMError:
            raise
        except Exception as primary_err:
            if self.config.fallback_model:
                logger.warning(
                    "Primary model %s failed (%s), trying fallback %s",
                    self.config.model,
                    primary_err,
                    self.config.fallback_model,
                )
                try:
                    return await self._call_llm_agentic(
                        self.config.fallback_model, messages, tools, temperature=temperature
                    )
                except Exception as fallback_err:
                    raise LLMError(
                        "both_models_failed",
                        primary_model=self.config.model,
                        fallback_model=self.config.fallback_model,
                        error=fallback_err,
                    ) from fallback_err
            raise LLMError(
                "agentic_failed", model=self.config.model, error=primary_err
            ) from primary_err

    async def review(self, messages: list[dict[str, str]], temperature: float | None = None) -> str:
        """Submit a review using tool calling."""
        return await self.complete_with_tools(
            messages, tools=[SUBMIT_REVIEW_TOOL], temperature=temperature
        )

    async def walkthrough(self, messages: list[dict[str, str]]) -> str:
        """Submit a walkthrough using tool calling."""
        return await self.complete_with_tools(messages, tools=[SUBMIT_WALKTHROUGH_TOOL])

    def count_tokens(self, text: str) -> int:
        """Estimate token count. Uses ~4 chars per token heuristic."""
        return len(text) // 4

    @property
    def usage(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }
