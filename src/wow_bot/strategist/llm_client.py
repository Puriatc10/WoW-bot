"""Local asynchronous LLM client for Ollama's OpenAI-compatible API (Task 4.1).

Provides an asynchronous client wrapper around ``openai.AsyncOpenAI`` with strict local-only
endpoint validation, disabled environment-proxy resolution (``trust_env=False``), explicit
project-level retries with bounded exponential backoff, lightweight observability, and
a health check probe.

Public API:
    - :class:`LLMClient`
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from urllib.parse import urlparse

import httpx
import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from wow_bot.shared.config import LLMConfig, Settings
from wow_bot.shared.logger import get_logger

logger = get_logger("LLM")

# Accepted loopback hostnames and IPs for local endpoint safety invariant.
LOCAL_HOSTNAMES: frozenset[str] = frozenset(
    {"127.0.0.1", "localhost", "::1", "[::1]"}
)


class LLMResponseError(RuntimeError):
    """The endpoint returned no usable completion content."""


def _validate_local_endpoint(base_url: str) -> None:
    """Ensure base_url targets a local loopback service.

    Raises:
        ValueError: If base_url scheme is invalid or hostname is non-local.
    """
    if not base_url:
        raise ValueError("base_url must be a non-empty string")

    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Invalid URL scheme '{parsed.scheme}' in base_url: {base_url}")

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ValueError(f"Could not parse hostname from base_url: {base_url}")

    if hostname not in LOCAL_HOSTNAMES:
        raise ValueError(
            f"Non-local LLM endpoint rejected for safety: '{hostname}'. "
            f"Must be one of {sorted(LOCAL_HOSTNAMES)}."
        )


def _is_transient_error(exc: Exception) -> bool:
    """Determine whether an exception represents a transient network/server failure."""
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError)):
        return True

    if isinstance(exc, openai.APIStatusError):
        # 5xx server errors are transient; 4xx client errors are permanent
        return exc.status_code >= 500

    return isinstance(exc, httpx.HTTPError)


class LLMClient:
    """Async client for local Ollama OpenAI-compatible chat completions API.

    Args:
        config: :class:`LLMConfig` or top-level :class:`Settings` containing LLM parameters.
        _async_openai_client: Optional injected AsyncOpenAI instance (used primarily in tests).
        _async_http_client: Optional injected httpx.AsyncClient instance (used primarily in tests).
    """

    def __init__(
        self,
        config: LLMConfig | Settings | Any,
        _async_openai_client: AsyncOpenAI | None = None,
        _async_http_client: httpx.AsyncClient | None = None,
    ) -> None:
        # Resolve LLMConfig if top-level Settings was passed
        if isinstance(config, Settings):
            self._cfg: LLMConfig = config.llm
        elif hasattr(config, "llm") and isinstance(config.llm, LLMConfig):
            self._cfg = config.llm
        elif isinstance(config, LLMConfig) or (
            hasattr(config, "base_url") and hasattr(config, "model")
        ):
            self._cfg = config
        else:
            raise TypeError(f"Invalid config type provided to LLMClient: {type(config)}")

        _validate_local_endpoint(self._cfg.base_url)

        self.base_url: str = self._cfg.base_url
        self.model: str = self._cfg.model
        self.timeout_seconds: float = float(self._cfg.timeout_seconds)
        self.max_retries: int = int(self._cfg.max_retries)
        self.api_key: str = self._cfg.api_key

        self._closed: bool = False

        if _async_http_client is not None:
            self._http_client = _async_http_client
            self._owns_http_client = False
        else:
            # trust_env=False is MANDATORY to prevent system proxy env vars
            # (HTTP_PROXY, HTTPS_PROXY) from intercepting localhost traffic.
            self._http_client = httpx.AsyncClient(
                trust_env=False,
                timeout=httpx.Timeout(self.timeout_seconds),
            )
            self._owns_http_client = True

        if _async_openai_client is not None:
            self._client = _async_openai_client
        else:
            # Disable SDK automatic retries so LLMClient explicitly owns retry policy
            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                http_client=self._http_client,  # type: ignore[arg-type]
                max_retries=0,
            )

    async def query(self, system_prompt: str, user_prompt: str) -> str:
        """Send system and user prompts to the local LLM and return assistant text.

        Args:
            system_prompt: System prompt defining behavior/persona.
            user_prompt: User prompt containing current state/inputs.

        Returns:
            Stripped text content from the assistant response.

        Raises:
            TypeError: If prompt inputs are not strings.
            RuntimeError: If response choices or content are missing/malformed.
            Exception: Propagates the final transient exception if retry budget is exhausted,
                or permanent exception on client errors.
        """
        if self._closed:
            raise RuntimeError("LLMClient is closed")

        if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
            raise TypeError(
                f"Prompts must be strings, got system={type(system_prompt)}, user={type(user_prompt)}"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        total_attempts = 1 + max(0, self.max_retries)
        attempt = 0

        while attempt < total_attempts:
            attempt += 1
            start_time = time.perf_counter()

            try:
                response: ChatCompletion = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                )
                latency = time.perf_counter() - start_time

                if not response.choices:
                    raise LLMResponseError("LLM response contained no choices")

                choice = response.choices[0]
                message_content = choice.message.content

                if message_content is None:
                    raise LLMResponseError("LLM response choice content was None")

                result_text = message_content.strip()

                # Extract token usage metadata if available
                prompt_tokens = getattr(response.usage, "prompt_tokens", None) if response.usage else None
                completion_tokens = getattr(response.usage, "completion_tokens", None) if response.usage else None
                total_tokens = getattr(response.usage, "total_tokens", None) if response.usage else None

                logger.info(
                    f"LLM query completed: model={self.model} system_chars={len(system_prompt)} "
                    f"user_chars={len(user_prompt)} latency_s={latency:.3f} attempt={attempt}/{total_attempts} "
                    f"prompt_tokens={prompt_tokens if prompt_tokens is not None else 'N/A'} "
                    f"completion_tokens={completion_tokens if completion_tokens is not None else 'N/A'} "
                    f"total_tokens={total_tokens if total_tokens is not None else 'N/A'}"
                )

                return result_text

            except asyncio.CancelledError:
                # Always preserve cancellation; do not retry or swallow
                raise

            except Exception as exc:
                latency = time.perf_counter() - start_time
                is_transient = _is_transient_error(exc)

                if is_transient and attempt < total_attempts:
                    backoff = 0.5 * (2 ** (attempt - 1))
                    logger.warning(
                        f"Transient failure on LLM query (attempt {attempt}/{total_attempts}): "
                        f"{type(exc).__name__}. Retrying in {backoff:.2f}s..."
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error(
                        f"LLM query failed permanently or exhausted retry budget (attempt {attempt}/{total_attempts}): "
                        f"{type(exc).__name__}"
                    )
                    raise

        raise RuntimeError("Unreachable retry loop exit point")  # pragma: no cover

    async def health_check(self) -> bool:
        """Return True if the local Ollama LLM endpoint is reachable and responsive."""
        if self._closed:
            return False

        try:
            await self._client.models.list()
            logger.debug(f"Health check succeeded for local LLM endpoint {self.base_url}")
            return True
        except asyncio.CancelledError:
            raise
        except (openai.APIError, httpx.HTTPError) as exc:
            logger.debug(f"Health check failed for local LLM endpoint {self.base_url}: {type(exc).__name__}")
            return False
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Unexpected error during LLM health check: {type(exc).__name__}")
            return False

    async def close(self) -> None:
        """Close owned HTTP and OpenAI resources idempotently."""
        if self._closed:
            return

        self._closed = True

        if self._owns_http_client:
            try:
                await self._http_client.aclose()
            except Exception as exc:  # noqa: BLE001 - pragma: no cover
                logger.warning(f"Error closing HTTP client: {exc}")

        logger.debug("Closed LLMClient resources")
