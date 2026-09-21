"""Strategist Orchestrator v2 for WoW-bot slow decision loop (Task 9.4).

Coordinates prompt building, cooldown gating, local LLM invocation via injected
LlmClient protocol, JSON strategy parsing, and vocabulary validation.
Produces deterministic OrchestratorResult decisions and emits structured session events.
Isolated from network I/O, threads, async, and forbidden domain modules.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from wow_bot.strategist.cooldown_v2 import CooldownDecision, CooldownGate
from wow_bot.strategist.prompts_v2 import (
    GameStateView,
    MetaStateLike,
    build_prompt,
)
from wow_bot.strategist.vocab_v2 import (
    StrategyCandidate,
    ValidatedStrategy,
    VocabularyGuard,
)
from wow_bot.world.summary import WorldSummary

if TYPE_CHECKING:
    from wow_bot.session import Session

__all__ = [
    "GameStateView",
    "JsonStrategy",
    "LlmClient",
    "MetaStateLike",
    "OrchestratorConfig",
    "OrchestratorError",
    "OrchestratorOutcome",
    "OrchestratorResult",
    "OrchestratorV2",
    "parse_strategy_json",
]


class OrchestratorError(Exception):
    """Raised when an unrecoverable error occurs in orchestrator operations."""


class LlmClient(Protocol):
    """Synchronous protocol for local LLM completion requests."""

    def complete(self, prompt: str) -> str:
        """Return raw assistant response string or raise transport exception."""
        ...


@dataclass(frozen=True)
class OrchestratorConfig:
    """Configuration options for OrchestratorV2."""

    retry_on_invalid_json: bool = True
    max_rationale_chars_for_log: int = 200
    include_prompt_text_in_event: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_rationale_chars_for_log, bool)
            or not isinstance(self.max_rationale_chars_for_log, int)
            or self.max_rationale_chars_for_log < 1
        ):
            raise ValueError(
                f"max_rationale_chars_for_log must be an int >= 1, got {self.max_rationale_chars_for_log!r}"
            )


class OrchestratorOutcome(str, Enum):
    """Possible outcomes of an OrchestratorV2 decision."""

    SUCCESS = "success"
    BLOCKED_BY_COOLDOWN = "blocked_by_cooldown"
    LLM_TRANSPORT_ERROR = "llm_transport_error"
    INVALID_JSON = "invalid_json"
    VOCAB_REJECTED = "vocab_rejected"
    PROMPT_BUILD_ERROR = "prompt_build_error"
    DISABLED = "disabled"


@dataclass(frozen=True)
class OrchestratorResult:
    """Result of an OrchestratorV2 strategy decision cycle."""

    outcome: OrchestratorOutcome
    strategy: ValidatedStrategy | None
    prompt_hash: str
    attempts: int
    latency_ms: float
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, OrchestratorOutcome):
            raise ValueError(  # noqa: TRY004
                f"outcome must be an OrchestratorOutcome, got {self.outcome!r}"
            )

        if self.outcome == OrchestratorOutcome.SUCCESS:
            if self.strategy is None:
                raise ValueError("outcome == SUCCESS implies strategy is not None")
            if self.reason != "":
                raise ValueError("outcome == SUCCESS implies reason == ''")
        else:
            if self.strategy is not None:
                raise ValueError(
                    f"outcome != SUCCESS implies strategy is None, got {self.strategy!r}"
                )
            if self.reason == "":
                raise ValueError("outcome != SUCCESS implies reason != ''")

        if (
            isinstance(self.attempts, bool)
            or not isinstance(self.attempts, int)
            or self.attempts < 0
        ):
            raise ValueError(
                f"attempts must be an int >= 0, got {self.attempts!r}"
            )

        if (
            isinstance(self.latency_ms, bool)
            or not isinstance(self.latency_ms, (int, float))
            or math.isnan(self.latency_ms)
            or float(self.latency_ms) < 0.0
        ):
            raise ValueError(
                f"latency_ms must be a float >= 0.0, got {self.latency_ms!r}"
            )

        if not isinstance(self.prompt_hash, str):
            raise ValueError("prompt_hash must be a string")  # noqa: TRY004

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the OrchestratorResult."""
        return {
            "outcome": self.outcome.value,
            "strategy": (
                {
                    "goal": self.strategy.goal,
                    "target": self.strategy.target,
                    "rationale": self.strategy.rationale,
                }
                if self.strategy is not None
                else None
            ),
            "prompt_hash": self.prompt_hash,
            "attempts": self.attempts,
            "latency_ms": self.latency_ms,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class JsonStrategy:
    """Raw parsed JSON strategy candidate matching expected schema keys."""

    goal: str
    target: str | None
    rationale: str


def parse_strategy_json(raw: str) -> JsonStrategy | None:
    """Attempt to parse `raw` as a JSON object matching strategy response schema.

    Returns JsonStrategy on valid object shape, or None on any error or invalid structure.
    Never raises.
    """
    if not isinstance(raw, str):
        return None

    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None

        # Disallow unknown top-level keys
        if set(data.keys()) - {"goal", "target", "rationale"}:
            return None

        # Check required string keys
        if "goal" not in data or "rationale" not in data:
            return None

        goal = data["goal"]
        rationale = data["rationale"]

        if not isinstance(goal, str) or not isinstance(rationale, str):
            return None

        target = data.get("target")
        if target is not None and not isinstance(target, str):
            return None

        return JsonStrategy(goal=goal, target=target, rationale=rationale)
    except Exception:  # noqa: BLE001
        return None


class OrchestratorV2:
    """Ties prompt building, cooldown gating, local LLM call, parsing, and vocab validation into one decision."""

    def __init__(
        self,
        *,
        llm: LlmClient,
        cooldown: CooldownGate,
        guard: VocabularyGuard,
        session: Session | None = None,
        config: OrchestratorConfig | None = None,
        enabled: bool = True,
        timer: Callable[[], float] | None = None,
    ) -> None:
        self._llm: LlmClient = llm
        self._cooldown: CooldownGate = cooldown
        self._guard: VocabularyGuard = guard
        self._session: Session | None = session
        self._config: OrchestratorConfig = (
            config if config is not None else OrchestratorConfig()
        )
        self._enabled: bool = enabled
        self._timer: Callable[[], float] = timer if timer is not None else time.perf_counter
        self._last_result: OrchestratorResult | None = None

    def last_result(self) -> OrchestratorResult | None:
        """Return the most recent OrchestratorResult, or None if decide() has not been called."""
        return self._last_result

    def reset(self) -> None:
        """Clear last_result without resetting cooldown or guard state."""
        self._last_result = None

    def decide(
        self,
        *,
        meta: MetaStateLike,
        world: WorldSummary,
        state: GameStateView,
        now: float,
    ) -> OrchestratorResult:
        """Evaluate strategist decision cycle for current state and world context."""
        # 1. Check enabled flag
        if not self._enabled:
            result = OrchestratorResult(
                outcome=OrchestratorOutcome.DISABLED,
                strategy=None,
                prompt_hash="",
                attempts=0,
                latency_ms=0.0,
                reason="disabled",
            )
            self._last_result = result
            return result

        # 2. Cooldown check
        check = self._cooldown.check(state.fsm_state, now=now)
        if check.decision not in (CooldownDecision.ALLOWED, CooldownDecision.FORCED_ALLOWED):
            result = OrchestratorResult(
                outcome=OrchestratorOutcome.BLOCKED_BY_COOLDOWN,
                strategy=None,
                prompt_hash="",
                attempts=0,
                latency_ms=0.0,
                reason=check.decision.value,
            )
            self._last_result = result
            return result

        # 3. Prompt build
        try:
            bundle = build_prompt(meta, world, state)
        except Exception as exc:  # noqa: BLE001
            if self._session is not None:
                self._session.write_event({
                    "event": "strategist_prompt_error",
                    "error": type(exc).__name__,
                })
            result = OrchestratorResult(
                outcome=OrchestratorOutcome.PROMPT_BUILD_ERROR,
                strategy=None,
                prompt_hash="",
                attempts=0,
                latency_ms=0.0,
                reason=type(exc).__name__,
            )
            self._last_result = result
            return result

        # 4. LLM call (first attempt)
        start = self._timer()
        try:
            raw = self._llm.complete(bundle.text)
        except Exception as exc:  # noqa: BLE001
            latency_ms = (self._timer() - start) * 1000.0
            if self._session is not None:
                self._session.write_event({
                    "event": "strategist_llm_error",
                    "prompt_hash": bundle.prompt_hash,
                    "error": type(exc).__name__,
                })
            result = OrchestratorResult(
                outcome=OrchestratorOutcome.LLM_TRANSPORT_ERROR,
                strategy=None,
                prompt_hash=bundle.prompt_hash,
                attempts=1,
                latency_ms=latency_ms,
                reason=type(exc).__name__,
            )
            self._last_result = result
            return result

        # 5. Parse first response
        parsed = parse_strategy_json(raw)
        attempts = 1

        # 6. Retry once if configured and parsing failed
        if parsed is None and self._config.retry_on_invalid_json:
            if self._session is not None:
                self._session.write_event({
                    "event": "strategist_retry",
                    "prompt_hash": bundle.prompt_hash,
                })
            try:
                raw = self._llm.complete(bundle.text)
            except Exception as exc:  # noqa: BLE001
                latency_ms = (self._timer() - start) * 1000.0
                if self._session is not None:
                    self._session.write_event({
                        "event": "strategist_llm_error",
                        "prompt_hash": bundle.prompt_hash,
                        "error": type(exc).__name__,
                    })
                result = OrchestratorResult(
                    outcome=OrchestratorOutcome.LLM_TRANSPORT_ERROR,
                    strategy=None,
                    prompt_hash=bundle.prompt_hash,
                    attempts=2,
                    latency_ms=latency_ms,
                    reason=type(exc).__name__,
                )
                self._last_result = result
                return result
            parsed = parse_strategy_json(raw)
            attempts = 2

        # 7. If parsed is still None
        if parsed is None:
            latency_ms = (self._timer() - start) * 1000.0
            if self._session is not None:
                self._session.write_event({
                    "event": "strategist_invalid_json",
                    "prompt_hash": bundle.prompt_hash,
                    "attempts": attempts,
                })
            result = OrchestratorResult(
                outcome=OrchestratorOutcome.INVALID_JSON,
                strategy=None,
                prompt_hash=bundle.prompt_hash,
                attempts=attempts,
                latency_ms=latency_ms,
                reason="parse_failed",
            )
            self._last_result = result
            return result

        # 8. Vocabulary guard
        candidate = StrategyCandidate(
            goal=parsed.goal,
            target=parsed.target,
            rationale=parsed.rationale,
        )
        decision = self._guard.validate(candidate)
        latency_ms = (self._timer() - start) * 1000.0
        if not decision.accepted:
            reason_str = (
                decision.reason.value
                if decision.reason is not None
                else "unknown_rejection"
            )
            result = OrchestratorResult(
                outcome=OrchestratorOutcome.VOCAB_REJECTED,
                strategy=None,
                prompt_hash=bundle.prompt_hash,
                attempts=attempts,
                latency_ms=latency_ms,
                reason=reason_str,
            )
            self._last_result = result
            return result

        # 9. Success
        assert decision.strategy is not None
        if self._session is not None:
            payload: dict[str, Any] = {
                "event": "strategist_success",
                "prompt_hash": bundle.prompt_hash,
                "goal": decision.strategy.goal,
                "has_target": decision.strategy.target is not None,
                "attempts": attempts,
                "latency_ms": latency_ms,
            }
            if self._config.include_prompt_text_in_event:
                payload["prompt_text"] = bundle.text
            self._session.write_event(payload)

        result = OrchestratorResult(
            outcome=OrchestratorOutcome.SUCCESS,
            strategy=decision.strategy,
            prompt_hash=bundle.prompt_hash,
            attempts=attempts,
            latency_ms=latency_ms,
            reason="",
        )
        self._last_result = result
        return result
