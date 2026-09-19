"""Strategist prompt templates and dynamic context construction.

This module is a pure string-building layer separating:
  1. persistent player profile (static, in system_prompt.txt on disk);
  2. dynamic runtime context (time, session, availability via DynamicContext);
  3. current internal state (MetaState);
  4. recent events;
  5. previous strategy;
  6. required output contract.

Invariants:
  - Pure string generation: no LLM calls, network I/O, or database reads.
  - No wall-clock reads: time arrives exclusively via DynamicContext.
  - Deterministic output given identical inputs.
"""

from __future__ import annotations

import functools
import json
import zoneinfo
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from wow_bot.shared.interfaces import DRIVE_NAMES, Event, MetaState, Strategy
from wow_bot.shared.logger import get_logger

log = get_logger("PROMPTS")

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.txt"


@dataclass(frozen=True)
class DynamicContext:
    """Runtime context supplied by the caller.

    All time fields are Unix timestamps (float seconds).
    The prompt-building logic never reads the clock directly.
    """

    now: float
    session_start: float
    available_regions: tuple[str, ...]
    previous_strategy: Strategy | None = None
    timezone: str = "America/New_York"
    is_weekend: bool | None = None
    sleep_window: tuple[int, int] = (1, 7)


@functools.lru_cache(maxsize=1)
def load_system_prompt() -> str:
    """Return the persistent persona system prompt loaded from disk.

    The file is read once per process and cached using lru_cache.
    """
    text = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    log.info(f"Loaded system prompt from disk ({len(text)} characters)")
    return text


def _clear_system_prompt_cache() -> None:
    """Clear the module-level system prompt cache (for test isolation only)."""
    load_system_prompt.cache_clear()


def _get_day_part(hour: int) -> str:
    """Map local hour (0..23) to day part according to project standards.

    05:00–11:59 -> morning
    12:00–16:59 -> afternoon
    17:00–21:59 -> evening
    22:00–04:59 -> late_night
    """
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 22:
        return "evening"
    return "late_night"


def _estimate_session_remaining_minutes(session_minutes: int) -> int:
    """Estimate remaining session time in minutes.

    Heuristic: assumes a standard target session length of 90 minutes,
    bounded between 5 and 60 minutes remaining.
    """
    return max(5, min(60, 90 - session_minutes))


def _format_time_context(dc: DynamicContext) -> str:
    """Render the time and session context block."""
    tz = zoneinfo.ZoneInfo(dc.timezone)
    now_dt = datetime.fromtimestamp(dc.now, tz=tz)
    start_dt = datetime.fromtimestamp(dc.session_start, tz=tz)

    weekday_name = now_dt.strftime("%A")
    day_part = _get_day_part(now_dt.hour)

    if dc.is_weekend is not None:
        weekend_flag = dc.is_weekend
    else:
        weekend_flag = now_dt.weekday() in (5, 6)

    session_minutes = int(max(0.0, (dc.now - dc.session_start) / 60.0))
    remaining_minutes = _estimate_session_remaining_minutes(session_minutes)

    lines = [
        f"Current time: {now_dt.isoformat()}",
        f"Timezone: {dc.timezone}",
        f"Weekday: {weekday_name}",
        f"Day part: {day_part}",
        f"Is weekend: {weekend_flag}",
        f"Session start: {start_dt.isoformat()}",
        f"Session duration: {session_minutes} minutes",
        f"Estimated session remaining: {remaining_minutes} minutes",
    ]
    return "\n".join(lines)


def _format_internal_state(meta_state: MetaState) -> str:
    """Render the drive levels in MetaState.vector in canonical order."""
    drives = meta_state.drives()
    lines = [f"{name}: {drives.get(name, 0.0):.2f}" for name in DRIVE_NAMES]
    return "\n".join(lines)


def _format_events(events: list[Event], now: float) -> str:
    """Render recent events sorted newest first with relative timestamps."""
    if not events:
        return "- (no recent events)"

    sorted_events = sorted(events, key=lambda e: e.timestamp, reverse=True)
    lines: list[str] = []
    for ev in sorted_events:
        rel_min = max(0, int((now - ev.timestamp) / 60.0))
        if ev.data:
            data_items = [f"{k}: {v}" for k, v in ev.data.items()]
            data_str = f" ({', '.join(data_items)})"
        else:
            data_str = ""
        lines.append(f"- {rel_min} min ago: {ev.type}{data_str}")

    return "\n".join(lines)


def _format_previous_strategy(previous: Strategy | None) -> str:
    """Render previous strategy details or fallback indicator if absent."""
    if previous is None:
        return "(none — this is the first strategy)"

    payload = {
        "goal": previous.goal,
        "region": previous.region,
        "risk_tolerance": previous.risk_tolerance,
        "priority": previous.priority,
        "constraints": previous.constraints,
    }
    return json.dumps(payload, indent=2)


def _format_available_regions(regions: tuple[str, ...]) -> str:
    """Render available region names or fallback indicator if empty."""
    if not regions:
        return "(none supplied)"
    return ", ".join(regions)


def build_user_prompt(
    meta_state: MetaState,
    dynamic_context: DynamicContext,
) -> str:
    """Render the user prompt for the next strategy generation call.

    Combines runtime context, internal state, events, previous strategy,
    and available regions in a strictly ordered, sectioned template.
    """
    ctx_str = _format_time_context(dynamic_context)
    state_str = _format_internal_state(meta_state)
    events_str = _format_events(meta_state.recent_events, dynamic_context.now)
    prev_str = _format_previous_strategy(dynamic_context.previous_strategy)
    regions_str = _format_available_regions(dynamic_context.available_regions)

    prompt = (
        f"<context>\n{ctx_str}\n</context>\n\n"
        f"<internal_state>\n{state_str}\n</internal_state>\n\n"
        f"<recent_events>\n{events_str}\n</recent_events>\n\n"
        f"<previous_strategy>\n{prev_str}\n</previous_strategy>\n\n"
        f"<available_regions>\n{regions_str}\n</available_regions>\n\n"
        "Generate a new strategy for the next 20-40 minutes.\n"
    )

    has_prev = dynamic_context.previous_strategy is not None
    log.debug(
        f"Built user prompt (len={len(prompt)} chars, events={len(meta_state.recent_events)}, prev_strategy={has_prev})"
    )
    return prompt
