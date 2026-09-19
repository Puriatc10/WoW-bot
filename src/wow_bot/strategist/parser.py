"""Response parser and validator for LLM Strategy output (Task 4.3).

Strictly parses raw text output from local LLM strategy generation and converts it
into a validated domain ``Strategy`` object.

Contract & Security Invariants:
- Stateless, pure parsing with zero side effects (no network, disk, clock, or RNG).
- ``valid_until`` is supplied strictly by the caller; no wall clock is read.
- Top-level schema is strictly enforced: extra or missing fields cause immediate rejection.
- ``reasoning`` is validated as a non-empty string LLM diagnostic field, but NOT added to
  the ``Strategy`` dataclass.
- Raw response string is preserved exactly in ``Strategy.raw_llm_output``.
- Non-standard JSON constants (NaN, Infinity, -Infinity) and non-JSON syntax are rejected.
- All parse/validation errors raise ``StrategyParseError``.
"""

from __future__ import annotations

import json
import math
from typing import Final, NoReturn

from wow_bot.shared.interfaces import STRATEGY_GOALS, Strategy

EXPECTED_TOP_LEVEL_KEYS: Final[frozenset[str]] = frozenset(
    {"reasoning", "goal", "region", "risk_tolerance", "priority", "constraints"}
)

EXPECTED_CONSTRAINT_KEYS: Final[frozenset[str]] = frozenset(
    {"max_deaths_per_hour", "max_session_minutes", "avoid_pvp"}
)


class StrategyParseError(ValueError):
    """Raised when an LLM strategy response violates the response contract."""


def _reject_non_finite_constant(val: str) -> NoReturn:
    """Callback for json.loads to reject non-standard float constants like NaN or Infinity."""
    raise ValueError(f"Non-standard JSON numeric constant rejected: {val}")


def _unwrap_optional_code_fence(text: str) -> str:
    """Unwrap a single full-string Markdown code fence if present.

    Rejects arbitrary surrounding prose or multiple fences.
    """
    if not text.startswith("```"):
        return text

    if not text.endswith("```"):
        raise StrategyParseError("Unclosed code fence in strategy response")

    lines = text.splitlines()
    if len(lines) < 2:
        raise StrategyParseError("Malformed code fence in strategy response")

    first_line = lines[0].rstrip()
    if first_line not in ("```", "```json", "```JSON"):
        raise StrategyParseError(f"Unsupported code fence language tag: {first_line}")

    inner_lines = lines[1:-1]
    for line in inner_lines:
        if "```" in line:
            raise StrategyParseError("Multiple code fences are not allowed")

    return "\n".join(inner_lines).strip()


def parse_strategy_response(raw: str, valid_until: float) -> Strategy:
    """Parse and validate raw LLM output into a domain Strategy object.

    Args:
        raw: The raw response string returned by the LLM.
        valid_until: Floating-point Unix timestamp indicating strategy expiration,
            supplied by the caller.

    Returns:
        A validated domain ``Strategy`` instance.

    Raises:
        StrategyParseError: If ``valid_until`` or the LLM payload violates the Task 4.2 contract.
    """
    if (
        isinstance(valid_until, bool)
        or not isinstance(valid_until, (int, float))
        or not math.isfinite(valid_until)
    ):
        raise StrategyParseError(f"valid_until must be a finite number, got {valid_until!r}")

    if not isinstance(raw, str):
        raise StrategyParseError(f"Raw response must be a string, got {type(raw).__name__}")

    trimmed = raw.strip()
    if not trimmed:
        raise StrategyParseError("LLM strategy response is empty")

    cleaned_text = _unwrap_optional_code_fence(trimmed)

    try:
        payload = json.loads(cleaned_text, parse_constant=_reject_non_finite_constant)
    except json.JSONDecodeError as exc:
        raise StrategyParseError("LLM strategy response is not valid JSON") from exc
    except ValueError as exc:
        raise StrategyParseError(str(exc)) from exc

    if not isinstance(payload, dict):
        raise StrategyParseError("LLM strategy payload must be a JSON object")

    payload_keys = set(payload.keys())
    missing_keys = EXPECTED_TOP_LEVEL_KEYS - payload_keys
    if missing_keys:
        raise StrategyParseError(f"Missing required strategy field: {min(missing_keys)}")

    extra_keys = payload_keys - EXPECTED_TOP_LEVEL_KEYS
    if extra_keys:
        raise StrategyParseError(f"Unexpected strategy field: {min(extra_keys)}")

    # 1. reasoning validation
    reasoning = payload["reasoning"]
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise StrategyParseError("reasoning must be a non-empty string")

    # 2. goal validation
    goal = payload["goal"]
    if not isinstance(goal, str) or goal not in STRATEGY_GOALS:
        raise StrategyParseError(f"Unsupported strategy goal: {goal!r}")

    # 3. region validation
    region = payload["region"]
    if not isinstance(region, str) or not region.strip():
        raise StrategyParseError("region must be a non-empty string")

    # 4. risk_tolerance validation
    risk_tolerance = payload["risk_tolerance"]
    if (
        isinstance(risk_tolerance, bool)
        or not isinstance(risk_tolerance, (int, float))
        or not math.isfinite(risk_tolerance)
        or not (0.0 <= float(risk_tolerance) <= 1.0)
    ):
        raise StrategyParseError("risk_tolerance must be a finite number in [0, 1]")

    # 5. priority validation
    priority = payload["priority"]
    if not isinstance(priority, list):
        raise StrategyParseError("priority must be a list of non-empty strings")
    for item in priority:
        if not isinstance(item, str) or not item.strip():
            raise StrategyParseError("priority entries must be non-empty strings")

    # 6. constraints validation
    constraints = payload["constraints"]
    if not isinstance(constraints, dict):
        raise StrategyParseError("constraints must be a JSON object")

    constraint_keys = set(constraints.keys())
    missing_c = EXPECTED_CONSTRAINT_KEYS - constraint_keys
    if missing_c:
        raise StrategyParseError(f"Missing required constraint field: {min(missing_c)}")

    extra_c = constraint_keys - EXPECTED_CONSTRAINT_KEYS
    if extra_c:
        raise StrategyParseError(f"Unexpected constraint field: {min(extra_c)}")

    max_deaths = constraints["max_deaths_per_hour"]
    if isinstance(max_deaths, bool) or type(max_deaths) is not int or max_deaths < 0:
        raise StrategyParseError("max_deaths_per_hour must be a non-negative integer")

    max_session = constraints["max_session_minutes"]
    if isinstance(max_session, bool) or type(max_session) is not int or max_session <= 0:
        raise StrategyParseError("max_session_minutes must be a positive integer")

    avoid_pvp = constraints["avoid_pvp"]
    if not isinstance(avoid_pvp, bool):
        raise StrategyParseError("avoid_pvp must be a boolean")

    try:
        return Strategy(
            goal=goal,
            region=region,
            risk_tolerance=float(risk_tolerance),
            priority=list(priority),
            constraints=dict(constraints),
            valid_until=float(valid_until),
            raw_llm_output=raw,
        )
    except Exception as exc:
        raise StrategyParseError(f"Failed to construct Strategy domain object: {exc}") from exc
