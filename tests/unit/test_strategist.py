"""Unit tests for Strategist Orchestrator (Task 4.4).

Verifies strategy generation, fallback behavior, lifecycle management,
time handling, contract preservation, and test cases A through Y.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import numpy as np
import openai
import pytest

from wow_bot.shared.config import Settings
from wow_bot.shared.interfaces import MetaState, Strategy
from wow_bot.strategist.llm_client import LLMClient, LLMResponseError
from wow_bot.strategist.orchestrator import Strategist
from wow_bot.strategist.parser import StrategyParseError
from wow_bot.strategist.prompts import DynamicContext

# Canonical test timestamp: fixed deterministic reference time
NOW: float = 1_750_000_000.0

VALID_LLM_JSON: str = """{
  "reasoning": "Focus on farming herbs safely in Elwynn Forest.",
  "goal": "farm_herbs",
  "region": "Elwynn Forest",
  "risk_tolerance": 0.3,
  "priority": ["herbs", "safety"],
  "constraints": {
    "max_deaths_per_hour": 2,
    "max_session_minutes": 45,
    "avoid_pvp": true
  }
}"""

VALID_LLM_JSON_B: str = """{
  "reasoning": "Switching to exploration mode.",
  "goal": "explore",
  "region": "Duskwood",
  "risk_tolerance": 0.6,
  "priority": ["map_reveal", "chests"],
  "constraints": {
    "max_deaths_per_hour": 5,
    "max_session_minutes": 30,
    "avoid_pvp": false
  }
}"""


@pytest.fixture
def sample_meta_state() -> MetaState:
    return MetaState(
        vector=np.array([0.5, 0.4, 0.6, 0.2, 0.3], dtype=np.float64),
        recent_events=[],
        timestamp=NOW,
    )


@pytest.fixture
def sample_dynamic_context() -> DynamicContext:
    return DynamicContext(
        now=NOW,
        session_start=NOW - 600.0,
        available_regions=("Elwynn Forest", "Duskwood"),
    )


@pytest.fixture
def mock_llm_client() -> MagicMock:
    client = MagicMock(spec=LLMClient)
    client.query = AsyncMock(return_value=VALID_LLM_JSON)
    return client


@pytest.fixture
def mock_memory() -> MagicMock:
    memory = MagicMock()
    memory.recall_similar = AsyncMock()
    memory.add = AsyncMock()
    memory.decay_old = AsyncMock()
    return memory


@pytest.fixture
def mock_config() -> MagicMock:
    return MagicMock(spec=Settings)


# ---------------------------------------------------------------------------
# Test A: Initial state
# ---------------------------------------------------------------------------
def test_initial_state(mock_config: MagicMock, mock_llm_client: MagicMock, mock_memory: MagicMock) -> None:
    strategist = Strategist(mock_config, mock_llm_client, mock_memory)
    assert strategist.current_strategy is None
    assert strategist.is_expired(NOW) is True


# ---------------------------------------------------------------------------
# Test B: Successful generation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_successful_generation(
    mock_config: MagicMock,
    mock_llm_client: MagicMock,
    mock_memory: MagicMock,
    sample_meta_state: MetaState,
    sample_dynamic_context: DynamicContext,
) -> None:
    strategist = Strategist(mock_config, mock_llm_client, mock_memory)
    strategy = await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)

    assert isinstance(strategy, Strategy)
    assert strategist.current_strategy is strategy
    assert strategy.goal == "farm_herbs"
    assert strategy.region == "Elwynn Forest"
    assert strategy.risk_tolerance == 0.3
    assert strategy.raw_llm_output == VALID_LLM_JSON


# ---------------------------------------------------------------------------
# Test C: Prompt pipeline integration
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_prompt_pipeline_integration(
    mock_config: MagicMock,
    mock_llm_client: MagicMock,
    mock_memory: MagicMock,
    sample_meta_state: MetaState,
    sample_dynamic_context: DynamicContext,
) -> None:
    strategist = Strategist(mock_config, mock_llm_client, mock_memory)
    await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)

    mock_llm_client.query.assert_called_once()
    system_prompt, user_prompt = mock_llm_client.query.call_args.args

    assert "<context>" in user_prompt
    assert "<internal_state>" in user_prompt
    assert "<available_regions>" in user_prompt
    assert "<persona_identity>" in system_prompt


# ---------------------------------------------------------------------------
# Test D: Previous strategy injected into prompt
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_previous_strategy_injected(
    mock_config: MagicMock,
    mock_llm_client: MagicMock,
    mock_memory: MagicMock,
    sample_meta_state: MetaState,
    sample_dynamic_context: DynamicContext,
) -> None:
    mock_llm_client.query.side_effect = [VALID_LLM_JSON, VALID_LLM_JSON_B]
    strategist = Strategist(mock_config, mock_llm_client, mock_memory)

    # First generation
    strategy_a = await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)

    # Second generation
    await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)

    _, second_user_prompt = mock_llm_client.query.call_args.args
    assert "<previous_strategy>" in second_user_prompt
    assert strategy_a.goal in second_user_prompt
    assert strategy_a.region in second_user_prompt


# ---------------------------------------------------------------------------
# Test E: Caller's previous_strategy in DynamicContext is overridden
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_caller_previous_strategy_overridden(
    mock_config: MagicMock,
    mock_llm_client: MagicMock,
    mock_memory: MagicMock,
    sample_meta_state: MetaState,
    sample_dynamic_context: DynamicContext,
) -> None:
    mock_llm_client.query.side_effect = [VALID_LLM_JSON, VALID_LLM_JSON_B]
    strategist = Strategist(mock_config, mock_llm_client, mock_memory)

    # 1. Establish Strategy A
    strategy_a = await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)

    # 2. Construct caller context containing a different/stale previous_strategy
    stale_strategy = Strategy(
        goal="flee",
        region="Stranglethorn Vale",
        risk_tolerance=0.9,
        priority=["escape"],
        constraints={"max_deaths_per_hour": 10, "max_session_minutes": 10, "avoid_pvp": False},
        valid_until=NOW + 500,
        raw_llm_output="{}",
    )
    stale_context = DynamicContext(
        now=NOW + 100,
        session_start=NOW - 500,
        available_regions=("Elwynn Forest",),
        previous_strategy=stale_strategy,
    )

    await strategist.generate_strategy(sample_meta_state, stale_context)

    _, second_user_prompt = mock_llm_client.query.call_args.args
    assert strategy_a.goal in second_user_prompt
    assert "flee" not in second_user_prompt


# ---------------------------------------------------------------------------
# Test F: valid_until calculation using context time + TTL
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_valid_until_calculation(
    mock_config: MagicMock,
    mock_llm_client: MagicMock,
    mock_memory: MagicMock,
    sample_meta_state: MetaState,
) -> None:
    context = DynamicContext(now=1000.0, session_start=0.0, available_regions=())
    strategist = Strategist(mock_config, mock_llm_client, mock_memory)

    strategy = await strategist.generate_strategy(sample_meta_state, context)

    # Default TTL is 30 minutes (1800 seconds)
    assert strategy.valid_until == 1000.0 + 1800.0


# ---------------------------------------------------------------------------
# Test G: is_expired before expiry
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_is_expired_before_expiry(
    mock_config: MagicMock,
    mock_llm_client: MagicMock,
    mock_memory: MagicMock,
    sample_meta_state: MetaState,
    sample_dynamic_context: DynamicContext,
) -> None:
    strategist = Strategist(mock_config, mock_llm_client, mock_memory)
    strategy = await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)

    assert strategist.is_expired(strategy.valid_until - 0.001) is False


# ---------------------------------------------------------------------------
# Test H: is_expired exact boundary
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_is_expired_exact_boundary(
    mock_config: MagicMock,
    mock_llm_client: MagicMock,
    mock_memory: MagicMock,
    sample_meta_state: MetaState,
    sample_dynamic_context: DynamicContext,
) -> None:
    strategist = Strategist(mock_config, mock_llm_client, mock_memory)
    strategy = await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)

    assert strategist.is_expired(strategy.valid_until) is True


# ---------------------------------------------------------------------------
# Test I: is_expired after expiry
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_is_expired_after_expiry(
    mock_config: MagicMock,
    mock_llm_client: MagicMock,
    mock_memory: MagicMock,
    sample_meta_state: MetaState,
    sample_dynamic_context: DynamicContext,
) -> None:
    strategist = Strategist(mock_config, mock_llm_client, mock_memory)
    strategy = await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)

    assert strategist.is_expired(strategy.valid_until + 10.0) is True


# ---------------------------------------------------------------------------
# Test J: Invalid is_expired now parameter validation
# ---------------------------------------------------------------------------
def test_is_expired_invalid_now_inputs(
    mock_config: MagicMock, mock_llm_client: MagicMock, mock_memory: MagicMock
) -> None:
    strategist = Strategist(mock_config, mock_llm_client, mock_memory)

    invalid_inputs = [float("nan"), float("inf"), float("-inf"), True, False, "123", None]
    for invalid in invalid_inputs:
        with pytest.raises(ValueError, match="must be a finite real number"):
            strategist.is_expired(invalid)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test K: LLM timeout fallback
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_llm_timeout_fallback(
    mock_config: MagicMock,
    mock_llm_client: MagicMock,
    mock_memory: MagicMock,
    sample_meta_state: MetaState,
    sample_dynamic_context: DynamicContext,
) -> None:
    strategist = Strategist(mock_config, mock_llm_client, mock_memory)
    strategy_a = await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)

    # Cause query timeout error
    request = httpx.Request("POST", "http://127.0.0.1:11434/v1/chat/completions")
    mock_llm_client.query = AsyncMock(
        side_effect=openai.APITimeoutError(request=request)  # type: ignore[arg-type]
    )

    fallback_strategy = await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)

    assert fallback_strategy is strategy_a
    assert strategist.current_strategy is strategy_a


# ---------------------------------------------------------------------------
# Test L: LLM connection failure fallback
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_llm_connection_failure_fallback(
    mock_config: MagicMock,
    mock_llm_client: MagicMock,
    mock_memory: MagicMock,
    sample_meta_state: MetaState,
    sample_dynamic_context: DynamicContext,
) -> None:
    strategist = Strategist(mock_config, mock_llm_client, mock_memory)
    strategy_a = await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)

    # Cause connection error
    mock_llm_client.query = AsyncMock(
        side_effect=httpx.ConnectError("Failed to connect")
    )

    fallback_strategy = await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)

    assert fallback_strategy is strategy_a
    assert strategist.current_strategy is strategy_a


# ---------------------------------------------------------------------------
# Test M: Parser failure fallback
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_parser_failure_fallback(
    mock_config: MagicMock,
    mock_llm_client: MagicMock,
    mock_memory: MagicMock,
    sample_meta_state: MetaState,
    sample_dynamic_context: DynamicContext,
) -> None:
    strategist = Strategist(mock_config, mock_llm_client, mock_memory)
    strategy_a = await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)

    # Return invalid JSON
    mock_llm_client.query = AsyncMock(return_value="NOT_VALID_JSON")

    fallback_strategy = await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)

    assert fallback_strategy is strategy_a
    assert strategist.current_strategy is strategy_a


# ---------------------------------------------------------------------------
# Test N: Failed generation never replaces current strategy
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_failed_generation_preserves_current_strategy(
    mock_config: MagicMock,
    mock_llm_client: MagicMock,
    mock_memory: MagicMock,
    sample_meta_state: MetaState,
    sample_dynamic_context: DynamicContext,
) -> None:
    strategist = Strategist(mock_config, mock_llm_client, mock_memory)
    strategy_a = await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)

    mock_llm_client.query = AsyncMock(side_effect=LLMResponseError("Missing completion"))

    res = await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)

    assert res is strategy_a
    assert strategist.current_strategy is strategy_a


# ---------------------------------------------------------------------------
# Test O: Failure with no previous strategy propagates original error
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_failure_with_no_previous_strategy_propagates(
    mock_config: MagicMock,
    mock_llm_client: MagicMock,
    mock_memory: MagicMock,
    sample_meta_state: MetaState,
    sample_dynamic_context: DynamicContext,
) -> None:
    mock_llm_client.query = AsyncMock(side_effect=StrategyParseError("Parse failed"))
    strategist = Strategist(mock_config, mock_llm_client, mock_memory)

    with pytest.raises(StrategyParseError, match="Parse failed"):
        await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)

    assert strategist.current_strategy is None


# ---------------------------------------------------------------------------
# Test P: Expired previous strategy still returned on failure
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_expired_previous_strategy_returned_on_failure(
    mock_config: MagicMock,
    mock_llm_client: MagicMock,
    mock_memory: MagicMock,
    sample_meta_state: MetaState,
) -> None:
    context_initial = DynamicContext(now=1000.0, session_start=0.0, available_regions=())
    strategist = Strategist(mock_config, mock_llm_client, mock_memory)

    # Strategy valid until 1000 + 1800 = 2800
    strategy_a = await strategist.generate_strategy(sample_meta_state, context_initial)

    # Now simulated time moves forward to 3000.0 (strategy_a is now expired)
    context_later = DynamicContext(now=3000.0, session_start=0.0, available_regions=())
    assert strategist.is_expired(context_later.now) is True

    # Cause query failure
    mock_llm_client.query = AsyncMock(side_effect=LLMResponseError("Missing completion"))

    fallback_strategy = await strategist.generate_strategy(sample_meta_state, context_later)

    assert fallback_strategy is strategy_a
    assert fallback_strategy.valid_until == 2800.0
    assert strategist.is_expired(context_later.now) is True


# ---------------------------------------------------------------------------
# Test Q: No orchestrator-level double retry
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_orchestrator_level_retry(
    mock_config: MagicMock,
    mock_llm_client: MagicMock,
    mock_memory: MagicMock,
    sample_meta_state: MetaState,
    sample_dynamic_context: DynamicContext,
) -> None:
    mock_llm_client.query = AsyncMock(side_effect=RuntimeError("LLM fail"))
    strategist = Strategist(mock_config, mock_llm_client, mock_memory)

    with pytest.raises(RuntimeError):
        await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)

    assert mock_llm_client.query.call_count == 1


# ---------------------------------------------------------------------------
# Test R: Successful second generation replaces current strategy
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_successful_second_generation_replaces_current(
    mock_config: MagicMock,
    mock_llm_client: MagicMock,
    mock_memory: MagicMock,
    sample_meta_state: MetaState,
    sample_dynamic_context: DynamicContext,
) -> None:
    mock_llm_client.query.side_effect = [VALID_LLM_JSON, VALID_LLM_JSON_B]
    strategist = Strategist(mock_config, mock_llm_client, mock_memory)

    strategy_a = await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)
    strategy_b = await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)

    assert strategy_b is not strategy_a
    assert strategy_b.goal == "explore"
    assert strategist.current_strategy is strategy_b


# ---------------------------------------------------------------------------
# Test S: Explicit generation ignores non-expired status
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_explicit_generation_ignores_non_expired_status(
    mock_config: MagicMock,
    mock_llm_client: MagicMock,
    mock_memory: MagicMock,
    sample_meta_state: MetaState,
    sample_dynamic_context: DynamicContext,
) -> None:
    strategist = Strategist(mock_config, mock_llm_client, mock_memory)

    await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)
    assert strategist.is_expired(sample_dynamic_context.now) is False

    await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)

    assert mock_llm_client.query.call_count == 2


# ---------------------------------------------------------------------------
# Test T: Cancellation propagation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_cancellation_propagation(
    mock_config: MagicMock,
    mock_llm_client: MagicMock,
    mock_memory: MagicMock,
    sample_meta_state: MetaState,
    sample_dynamic_context: DynamicContext,
) -> None:
    mock_llm_client.query = AsyncMock(side_effect=asyncio.CancelledError())
    strategist = Strategist(mock_config, mock_llm_client, mock_memory)

    # Establish initial strategy
    mock_llm_client.query = AsyncMock(return_value=VALID_LLM_JSON)
    strategy_a = await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)

    # Cause cancellation
    mock_llm_client.query = AsyncMock(side_effect=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)

    assert strategist.current_strategy is strategy_a


# ---------------------------------------------------------------------------
# Test U: Unexpected programming error propagation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [TypeError, RuntimeError])
async def test_unexpected_programming_error_propagation(
    mock_config: MagicMock,
    mock_llm_client: MagicMock,
    mock_memory: MagicMock,
    sample_meta_state: MetaState,
    sample_dynamic_context: DynamicContext,
    error_type: type[Exception],
) -> None:
    mock_llm_client.query = AsyncMock(return_value=VALID_LLM_JSON)
    strategist = Strategist(mock_config, mock_llm_client, mock_memory)
    strategy_a = await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)

    mock_llm_client.query = AsyncMock(side_effect=error_type("Unexpected code bug"))

    with pytest.raises(error_type, match="Unexpected code bug"):
        await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)

    assert strategist.current_strategy is strategy_a


# ---------------------------------------------------------------------------
# Test V: Raw model output preserved
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_raw_model_output_preserved(
    mock_config: MagicMock,
    mock_llm_client: MagicMock,
    mock_memory: MagicMock,
    sample_meta_state: MetaState,
    sample_dynamic_context: DynamicContext,
) -> None:
    strategist = Strategist(mock_config, mock_llm_client, mock_memory)
    strategy = await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)

    assert strategy.raw_llm_output == VALID_LLM_JSON


# ---------------------------------------------------------------------------
# Test W: Reasoning remains diagnostic only
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_reasoning_remains_diagnostic_only(
    mock_config: MagicMock,
    mock_llm_client: MagicMock,
    mock_memory: MagicMock,
    sample_meta_state: MetaState,
    sample_dynamic_context: DynamicContext,
) -> None:
    strategist = Strategist(mock_config, mock_llm_client, mock_memory)
    strategy = await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)

    assert not hasattr(strategy, "reasoning")
    assert "Focus on farming herbs safely in Elwynn Forest." in strategy.raw_llm_output


# ---------------------------------------------------------------------------
# Test X: No MemoryStore activity in Task 4.4
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_memorystore_activity(
    mock_config: MagicMock,
    mock_llm_client: MagicMock,
    mock_memory: MagicMock,
    sample_meta_state: MetaState,
    sample_dynamic_context: DynamicContext,
) -> None:
    strategist = Strategist(mock_config, mock_llm_client, mock_memory)
    await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)

    mock_memory.recall_similar.assert_not_called()
    mock_memory.add.assert_not_called()
    mock_memory.decay_old.assert_not_called()


# ---------------------------------------------------------------------------
# Test Y: No wall-clock access
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_wall_clock_access(
    mock_config: MagicMock,
    mock_llm_client: MagicMock,
    mock_memory: MagicMock,
    sample_meta_state: MetaState,
    sample_dynamic_context: DynamicContext,
) -> None:
    strategist = Strategist(mock_config, mock_llm_client, mock_memory)

    with patch("time.time", side_effect=AssertionError("Wall clock read prohibited!")), patch(
        "datetime.datetime", side_effect=AssertionError("Wall clock read prohibited!")
    ):
        strategy = await strategist.generate_strategy(sample_meta_state, sample_dynamic_context)
        is_exp = strategist.is_expired(sample_dynamic_context.now)

    assert strategy is not None
    assert is_exp is False
