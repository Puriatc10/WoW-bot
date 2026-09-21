"""# Pre-lab (MOCK_MODE)

Strategist Orchestrator (Task 4.4).

Composes the prompt builder, local LLM client, and strategy response parser into
an integrated high-level strategy generator.

Pipeline:
    MetaState + DynamicContext
              ↓
        Prompt Builder
              ↓
          LLMClient
              ↓
       Strategy Parser
              ↓
           Strategy

Public API:
    - :class:`Strategist`
"""

from __future__ import annotations

import dataclasses
import math

import httpx
import openai

from wow_bot.shared.interfaces import MetaState, Strategy
from wow_bot.shared.logger import get_logger
from wow_bot.strategist.llm_client import LLMClient, LLMResponseError
from wow_bot.strategist.parser import StrategyParseError, parse_strategy_response
from wow_bot.strategist.prompts import DynamicContext, build_user_prompt, load_system_prompt

logger = get_logger("STRATEGIST")

DEFAULT_STRATEGY_TTL_SECONDS: float = 30 * 60  # 30 minutes (1800 seconds)

_EXPECTED_GENERATION_ERRORS = (
    openai.APIError,
    httpx.HTTPError,
    StrategyParseError,
    LLMResponseError,
)


def _validate_timestamp(ts: float, name: str) -> float:
    """Ensure timestamp parameter is a finite real number."""
    if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts):
        raise ValueError(f"{name} must be a finite real number, got {ts!r}")
    return float(ts)


# DEPRECATED: see STRATEGIST_RECONCILIATION.md
class Strategist:
    """Orchestrates high-level strategy generation via local LLM.

    Args:
        config: Application or component configuration.
        llm_client: Injected LLMClient instance for model calls.
        memory: Injected MemoryStore instance (reserved for future integration).
    """

    def __init__(
        self,
        config: object,
        llm_client: LLMClient,
        memory: object,
    ) -> None:
        self._config = config
        self._llm_client = llm_client
        self._memory = memory
        self._current_strategy: Strategy | None = None

    @property
    def current_strategy(self) -> Strategy | None:
        """Return the currently active Strategy or None if no strategy exists."""
        return self._current_strategy

    def is_expired(self, now: float) -> bool:
        """Check if the current strategy is expired as of timestamp ``now``.

        Args:
            now: Floating-point Unix timestamp from authoritative runtime/simulation.

        Returns:
            True if no strategy exists or if now >= current_strategy.valid_until,
            False otherwise.

        Raises:
            ValueError: If now is boolean, NaN, Infinity, or not a finite real number.
        """
        valid_now = _validate_timestamp(now, "now")
        if self._current_strategy is None:
            return True
        return valid_now >= self._current_strategy.valid_until

    async def generate_strategy(
        self,
        meta_state: MetaState,
        dynamic_context: DynamicContext,
    ) -> Strategy:
        """Generate a new Strategy using current state and dynamic context.

        Logical sequence:
            1. Validate runtime inputs
            2. Create effective DynamicContext with current Strategy
            3. Load system prompt
            4. Build user prompt
            5. Calculate valid_until
            6. Call LLMClient.query(...)
            7. Parse raw response
            8. Set _current_strategy
            9. Return Strategy

        On expected operational failure (LLM transport timeout/error or StrategyParseError):
            - If _current_strategy exists, fallback to returning _current_strategy intact.
            - If _current_strategy is None, re-raise the failure exception.

        Args:
            meta_state: Current internal state vector and recent events.
            dynamic_context: Authoritative caller-supplied runtime time and context.

        Returns:
            A newly generated Strategy or the fallback previous Strategy.
        """
        valid_now = _validate_timestamp(dynamic_context.now, "dynamic_context.now")

        has_prev = self._current_strategy is not None
        logger.info(
            f"Strategy generation started: meta_state_ts={meta_state.timestamp:.3f} "
            f"context_now={valid_now:.3f} has_previous_strategy={has_prev}"
        )

        # Ensure effective context uses Orchestrator-owned _current_strategy regardless of caller context
        effective_context = dataclasses.replace(
            dynamic_context,
            previous_strategy=self._current_strategy,
        )

        system_prompt = load_system_prompt()
        user_prompt = build_user_prompt(meta_state, effective_context)

        ttl_seconds = float(
            getattr(self._config, "strategy_ttl_seconds", DEFAULT_STRATEGY_TTL_SECONDS)
        )
        valid_until = valid_now + ttl_seconds

        try:
            raw_response = await self._llm_client.query(system_prompt, user_prompt)
            strategy = parse_strategy_response(raw_response, valid_until=valid_until)
            self._current_strategy = strategy

            logger.info(
                f"Strategy generation succeeded: goal={strategy.goal!r} region={strategy.region!r} "
                f"risk_tolerance={strategy.risk_tolerance:.2f} valid_until={strategy.valid_until:.3f}"
            )
            return strategy

        except _EXPECTED_GENERATION_ERRORS as exc:
            is_expired = self.is_expired(valid_now) if self._current_strategy is not None else True
            logger.warning(
                f"Strategy generation failed ({type(exc).__name__}). "
                f"has_previous={has_prev} previous_expired_at_now={is_expired}"
            )
            if self._current_strategy is not None:
                return self._current_strategy
            raise
