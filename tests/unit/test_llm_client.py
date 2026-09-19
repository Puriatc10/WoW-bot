"""Unit tests for the local async LLM client (Task 4.1).

Covers all required behavioral test cases (A through P):
  - A: Correct construction & trust_env=False
  - B: Successful query & response content extraction
  - C: System and user message payload format
  - D: Configured model selection
  - E: Response whitespace stripping
  - F: Malformed response (no choices) failure
  - G: Missing/None choice content failure
  - H: Transient error retry backoff & eventual success
  - I: Retry exhaustion propagates final exception
  - J: Permanent client error not retried
  - K: Timeout error is retryable
  - L: Health check success returns True
  - M: Health check failure returns False
  - N: Non-local endpoint URL rejected loudly
  - O: Resource cleanup and idempotent close()
  - P: Cancellation propagates immediately without retries
"""

from __future__ import annotations

import asyncio
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from wow_bot.shared.config import LLMConfig, Settings
from wow_bot.strategist.llm_client import LLMClient


def _make_mock_chat_completion(
    content: str | None = "hello world",
    role: Literal["assistant"] = "assistant",
    has_choices: bool = True,
    usage: CompletionUsage | None = None,
) -> ChatCompletion:
    """Helper to build a realistic ChatCompletion response object."""
    if not has_choices:
        choices = []
    else:
        msg = ChatCompletionMessage(role=role, content=content)
        choice = Choice(finish_reason="stop", index=0, message=msg)
        choices = [choice]

    if usage is None:
        usage = CompletionUsage(completion_tokens=10, prompt_tokens=20, total_tokens=30)

    return ChatCompletion(
        id="chatcmpl-test",
        choices=choices,
        created=1234567890,
        model="qwen2.5:7b",
        object="chat.completion",
        usage=usage,
    )


# --- Test A: Construction & trust_env=False ---
def test_construction_defaults_and_http_client_trust_env() -> None:
    config = LLMConfig(
        base_url="http://127.0.0.1:11434/v1/",
        api_key="ollama",
        model="qwen2.5:7b",
        timeout_seconds=45.0,
        max_retries=2,
    )
    client = LLMClient(config)

    try:
        assert client.base_url == "http://127.0.0.1:11434/v1/"
        assert client.model == "qwen2.5:7b"
        assert client.timeout_seconds == 45.0
        assert client.max_retries == 2
        assert client._http_client.trust_env is False
    finally:
        asyncio.run(client.close())


def test_construction_from_top_level_settings() -> None:
    settings = Settings()
    client = LLMClient(settings)
    try:
        assert client.model == settings.llm.model
    finally:
        asyncio.run(client.close())


def test_invalid_config_type_raises_type_error() -> None:
    with pytest.raises(TypeError, match="Invalid config type"):
        LLMClient(123)


# --- Test B: Successful query ---
@pytest.mark.asyncio
async def test_successful_query_and_response_extraction() -> None:
    config = LLMConfig()
    mock_openai = MagicMock(spec=openai.AsyncOpenAI)
    mock_completion = _make_mock_chat_completion(content="  Sample Strategy Response  ")
    mock_openai.chat.completions.create = AsyncMock(return_value=mock_completion)

    client = LLMClient(config, _async_openai_client=mock_openai)
    result = await client.query("system prompt", "user prompt")

    assert result == "Sample Strategy Response"
    await client.close()


# --- Test C: Correct messages ---
@pytest.mark.asyncio
async def test_query_message_format() -> None:
    config = LLMConfig()
    mock_openai = MagicMock(spec=openai.AsyncOpenAI)
    mock_openai.chat.completions.create = AsyncMock(
        return_value=_make_mock_chat_completion("ok")
    )

    client = LLMClient(config, _async_openai_client=mock_openai)
    await client.query("sys_text", "usr_text")

    mock_openai.chat.completions.create.assert_called_once()
    kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert kwargs["messages"] == [
        {"role": "system", "content": "sys_text"},
        {"role": "user", "content": "usr_text"},
    ]
    await client.close()


# --- Test D: Configured model ---
@pytest.mark.asyncio
async def test_configured_model_selection() -> None:
    config = LLMConfig(model="custom-qwen-model:14b")
    mock_openai = MagicMock(spec=openai.AsyncOpenAI)
    mock_openai.chat.completions.create = AsyncMock(
        return_value=_make_mock_chat_completion("ok")
    )

    client = LLMClient(config, _async_openai_client=mock_openai)
    await client.query("sys", "usr")

    kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "custom-qwen-model:14b"
    await client.close()


# --- Test E: Response stripping ---
@pytest.mark.asyncio
async def test_response_content_stripping() -> None:
    config = LLMConfig()
    mock_openai = MagicMock(spec=openai.AsyncOpenAI)
    mock_openai.chat.completions.create = AsyncMock(
        return_value=_make_mock_chat_completion("\n  {\n  \"goal\": \"explore\"\n}  \n")
    )

    client = LLMClient(config, _async_openai_client=mock_openai)
    res = await client.query("sys", "usr")

    assert res == '{\n  "goal": "explore"\n}'
    await client.close()


# --- Test F: Malformed response (no choices) ---
@pytest.mark.asyncio
async def test_malformed_response_no_choices() -> None:
    config = LLMConfig(max_retries=0)
    mock_openai = MagicMock(spec=openai.AsyncOpenAI)
    mock_openai.chat.completions.create = AsyncMock(
        return_value=_make_mock_chat_completion(has_choices=False)
    )

    client = LLMClient(config, _async_openai_client=mock_openai)
    with pytest.raises(RuntimeError, match="no choices"):
        await client.query("sys", "usr")

    await client.close()


# --- Test G: Missing/None content ---
@pytest.mark.asyncio
async def test_missing_or_none_content_handling() -> None:
    config = LLMConfig(max_retries=0)
    mock_openai = MagicMock(spec=openai.AsyncOpenAI)
    mock_openai.chat.completions.create = AsyncMock(
        return_value=_make_mock_chat_completion(content=None)
    )

    client = LLMClient(config, _async_openai_client=mock_openai)
    with pytest.raises(RuntimeError, match="choice content was None"):
        await client.query("sys", "usr")

    await client.close()


# --- Test H: Retry succeeds ---
@pytest.mark.asyncio
async def test_retry_on_transient_failure_and_eventual_success() -> None:
    config = LLMConfig(max_retries=2)
    mock_openai = MagicMock(spec=openai.AsyncOpenAI)

    req = httpx.Request("POST", "http://127.0.0.1:11434/v1/chat/completions")
    transient_err = openai.APIConnectionError(request=req)  # type: ignore[arg-type]
    success_resp = _make_mock_chat_completion("retry succeeded")

    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[transient_err, success_resp]
    )

    client = LLMClient(config, _async_openai_client=mock_openai)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        res = await client.query("sys", "usr")

        assert res == "retry succeeded"
        assert mock_openai.chat.completions.create.call_count == 2
        mock_sleep.assert_called_once_with(0.5)

    await client.close()


# --- Test I: Retries exhaust ---
@pytest.mark.asyncio
async def test_retry_exhaustion_propagates_final_exception() -> None:
    config = LLMConfig(max_retries=2)  # total 3 attempts
    mock_openai = MagicMock(spec=openai.AsyncOpenAI)

    req = httpx.Request("POST", "http://127.0.0.1:11434/v1/chat/completions")
    transient_err = openai.APITimeoutError(request=req)  # type: ignore[arg-type]

    mock_openai.chat.completions.create = AsyncMock(side_effect=transient_err)

    client = LLMClient(config, _async_openai_client=mock_openai)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        with pytest.raises(openai.APITimeoutError):
            await client.query("sys", "usr")

        assert mock_openai.chat.completions.create.call_count == 3
        assert mock_sleep.call_count == 2

    await client.close()


# --- Test J: Permanent error not retried ---
@pytest.mark.asyncio
async def test_permanent_client_error_not_retried() -> None:
    config = LLMConfig(max_retries=3)
    mock_openai = MagicMock(spec=openai.AsyncOpenAI)

    req = httpx.Request("POST", "http://127.0.0.1:11434/v1/chat/completions")
    resp = httpx.Response(400, request=req)
    perm_err = openai.BadRequestError(
        message="Invalid request body", response=resp, body=None  # type: ignore[arg-type]
    )

    mock_openai.chat.completions.create = AsyncMock(side_effect=perm_err)

    client = LLMClient(config, _async_openai_client=mock_openai)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        with pytest.raises(openai.BadRequestError):
            await client.query("sys", "usr")

        assert mock_openai.chat.completions.create.call_count == 1
        mock_sleep.assert_not_called()

    await client.close()


# --- Test K: Timeout is retryable ---
@pytest.mark.asyncio
async def test_timeout_error_is_retryable() -> None:
    config = LLMConfig(max_retries=1)
    mock_openai = MagicMock(spec=openai.AsyncOpenAI)

    req = httpx.Request("POST", "http://127.0.0.1:11434/v1/chat/completions")
    timeout_err = openai.APITimeoutError(request=req)  # type: ignore[arg-type]
    success_resp = _make_mock_chat_completion("after timeout")

    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[timeout_err, success_resp]
    )

    client = LLMClient(config, _async_openai_client=mock_openai)

    with patch("asyncio.sleep", new_callable=AsyncMock):
        res = await client.query("sys", "usr")
        assert res == "after timeout"
        assert mock_openai.chat.completions.create.call_count == 2

    await client.close()


# --- Test L: Health check success ---
@pytest.mark.asyncio
async def test_health_check_success() -> None:
    config = LLMConfig()
    mock_openai = MagicMock(spec=openai.AsyncOpenAI)
    mock_openai.models.list = AsyncMock(return_value=MagicMock())

    client = LLMClient(config, _async_openai_client=mock_openai)
    ok = await client.health_check()

    assert ok is True
    mock_openai.models.list.assert_called_once()
    await client.close()


# --- Test M: Health check unavailable ---
@pytest.mark.asyncio
async def test_health_check_unavailable() -> None:
    config = LLMConfig()
    mock_openai = MagicMock(spec=openai.AsyncOpenAI)

    req = httpx.Request("GET", "http://127.0.0.1:11434/v1/models")
    conn_err = openai.APIConnectionError(request=req)  # type: ignore[arg-type]
    mock_openai.models.list = AsyncMock(side_effect=conn_err)

    client = LLMClient(config, _async_openai_client=mock_openai)
    ok = await client.health_check()

    assert ok is False
    await client.close()


# --- Test N: Non-local endpoint rejected ---
@pytest.mark.parametrize(
    "external_url",
    [
        "https://api.openai.com/v1/",
        "http://example.com/v1/",
        "http://localhost.attacker.com:11434/v1/",
        "http://192.168.1.100:11434/v1/",
    ],
)
def test_non_local_endpoint_rejected(external_url: str) -> None:
    config = LLMConfig(base_url=external_url)
    with pytest.raises(ValueError, match="Non-local LLM endpoint rejected"):
        LLMClient(config)


@pytest.mark.parametrize("local_url", ["http://127.0.0.1:11434/v1/", "http://localhost:11434/v1/", "http://[::1]:11434/v1/"])
def test_valid_local_endpoint_accepted(local_url: str) -> None:
    config = LLMConfig(base_url=local_url)
    client = LLMClient(config)
    assert client.base_url == local_url
    asyncio.run(client.close())


# --- Test O: Close lifecycle ---
@pytest.mark.asyncio
async def test_close_lifecycle_idempotent() -> None:
    config = LLMConfig()
    mock_http = AsyncMock(spec=httpx.AsyncClient)
    client = LLMClient(config, _async_http_client=mock_http)
    client._owns_http_client = True

    await client.close()
    mock_http.aclose.assert_called_once()

    # Second close should be harmless no-op
    await client.close()
    mock_http.aclose.assert_called_once()

    # Query on closed client fails
    with pytest.raises(RuntimeError, match="LLMClient is closed"):
        await client.query("sys", "usr")

    # Health check on closed client returns False
    assert await client.health_check() is False


# --- Test P: Cancellation propagates ---
@pytest.mark.asyncio
async def test_cancellation_propagates_without_retry() -> None:
    config = LLMConfig(max_retries=3)
    mock_openai = MagicMock(spec=openai.AsyncOpenAI)
    mock_openai.chat.completions.create = AsyncMock(side_effect=asyncio.CancelledError())

    client = LLMClient(config, _async_openai_client=mock_openai)

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        with pytest.raises(asyncio.CancelledError):
            await client.query("sys", "usr")

        assert mock_openai.chat.completions.create.call_count == 1
        mock_sleep.assert_not_called()

    await client.close()


# --- Additional edge cases ---
@pytest.mark.asyncio
async def test_query_invalid_prompt_types() -> None:
    config = LLMConfig()
    client = LLMClient(config)
    try:
        with pytest.raises(TypeError, match="Prompts must be strings"):
            await client.query(123, "user")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="Prompts must be strings"):
            await client.query("system", None)  # type: ignore[arg-type]
    finally:
        await client.close()
