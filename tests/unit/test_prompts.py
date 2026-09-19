"""Unit tests for Strategist prompt construction (Task 4.2).

Covers all required behavioral test cases (A through S):
  - A: System prompt loads with persona identifiers and schema requirements.
  - B: System prompt loading is cached after first read.
  - C: User prompt contains required XML tags in strict order and ends with terminal action.
  - D: Internal state drives vector formatted correctly in canonical order.
  - E: Recent events formatted newest-first with relative minutes and data.
  - F: Empty recent events list renders fallback string.
  - G: Previous strategy renders goal, region, and risk_tolerance.
  - H: Missing previous strategy renders explicit fallback string.
  - I: Time and session context renders weekday, day part, duration, and remaining time.
  - J: Weekend detection auto-detects or respects explicit override.
  - K: Available regions pass through comma-separated.
  - L: build_user_prompt is byte-identical deterministic given identical inputs.
  - M: Literal 'None' does not appear in rendered user prompt.
  - N: Formatting discipline (no trailing whitespace per line, no consecutive blank lines, single trailing newline).
  - O: Importing prompts module does not import LLM/HTTP client modules.
  - P: system_prompt.txt is valid UTF-8.
  - Q: Research framing phrase present in system_prompt.txt.
  - R: Anti-cheat / evasion language absent from system prompt.
  - S: Rigid anti-repetition rules absent from system prompt.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

from wow_bot.shared.interfaces import Event, MetaState, Strategy
from wow_bot.strategist.prompts import (
    DynamicContext,
    _clear_system_prompt_cache,
    build_user_prompt,
    load_system_prompt,
)


def setup_function() -> None:
    """Clear cache before each test for test isolation."""
    _clear_system_prompt_cache()


def test_a_system_prompt_loads_and_has_required_content() -> None:
    text = load_system_prompt()
    assert len(text) > 0
    assert "Alex" in text
    assert "America/New_York" in text or "Eastern Time" in text
    assert "Boston" in text
    assert "<example_1>" in text
    assert "<example_2>" in text
    assert "<example_3>" in text
    assert text.count("<example_") == 3
    assert "<forbidden_behaviors>" in text
    for field_name in ("reasoning", "goal", "region", "risk_tolerance", "priority", "constraints"):
        assert field_name in text


def test_b_system_prompt_caching() -> None:
    _clear_system_prompt_cache()
    prompt1 = load_system_prompt()

    # Patch read_text to verify disk is not accessed on second call
    with patch("pathlib.Path.read_text", side_effect=AssertionError("Disk read on cached call")):
        prompt2 = load_system_prompt()

    assert prompt1 == prompt2


def test_c_user_prompt_structure_and_order() -> None:
    now = 1700000000.0
    dc = DynamicContext(
        now=now,
        session_start=now - 600.0,
        available_regions=("Elwynn Forest",),
    )
    meta = MetaState(vector=[0.5, 0.5, 0.5, 0.5, 0.5], recent_events=[], timestamp=now)

    prompt = build_user_prompt(meta, dc)

    idx_ctx = prompt.index("<context>")
    idx_state = prompt.index("<internal_state>")
    idx_events = prompt.index("<recent_events>")
    idx_prev = prompt.index("<previous_strategy>")
    idx_regions = prompt.index("<available_regions>")

    assert idx_ctx < idx_state < idx_events < idx_prev < idx_regions
    assert prompt.endswith("Generate a new strategy for the next 20-40 minutes.\n")


def test_d_internal_state_formatting() -> None:
    now = 1700000000.0
    dc = DynamicContext(now=now, session_start=now - 600.0, available_regions=())
    vector = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
    meta = MetaState(vector=vector, recent_events=[], timestamp=now)

    prompt = build_user_prompt(meta, dc)

    expected_lines = [
        "hunger: 0.10",
        "fatigue: 0.20",
        "curiosity: 0.30",
        "aggression: 0.40",
        "social: 0.50",
    ]
    for line in expected_lines:
        assert line in prompt


def test_e_events_formatting() -> None:
    now = 1700000000.0
    ev1 = Event(type="death", timestamp=now - 180.0, data={"cause": "pull"})
    ev2 = Event(type="rare_loot", timestamp=now - 1200.0, data={"item": "Green Item"})
    dc = DynamicContext(now=now, session_start=now - 1500.0, available_regions=())
    meta = MetaState(vector=[0.5] * 5, recent_events=[ev2, ev1], timestamp=now)

    prompt = build_user_prompt(meta, dc)

    assert "- 3 min ago: death (cause: pull)" in prompt
    assert "- 20 min ago: rare_loot (item: Green Item)" in prompt

    idx_death = prompt.index("death")
    idx_loot = prompt.index("rare_loot")
    assert idx_death < idx_loot  # newest first


def test_f_no_events_formatting() -> None:
    now = 1700000000.0
    dc = DynamicContext(now=now, session_start=now - 600.0, available_regions=())
    meta = MetaState(vector=[0.5] * 5, recent_events=[], timestamp=now)

    prompt = build_user_prompt(meta, dc)

    assert "- (no recent events)" in prompt


def test_g_previous_strategy_present() -> None:
    now = 1700000000.0
    prev_strat = Strategy(
        goal="farm_herbs",
        region="Ashenvale",
        risk_tolerance=0.35,
        priority=["gather", "avoid_mobs"],
        constraints={"max_deaths_per_hour": 2},
        valid_until=now + 1800.0,
    )
    dc = DynamicContext(
        now=now,
        session_start=now - 600.0,
        available_regions=(),
        previous_strategy=prev_strat,
    )
    meta = MetaState(vector=[0.5] * 5, recent_events=[], timestamp=now)

    prompt = build_user_prompt(meta, dc)

    assert "farm_herbs" in prompt
    assert "Ashenvale" in prompt
    assert "0.35" in prompt


def test_h_previous_strategy_absent() -> None:
    now = 1700000000.0
    dc = DynamicContext(
        now=now,
        session_start=now - 600.0,
        available_regions=(),
        previous_strategy=None,
    )
    meta = MetaState(vector=[0.5] * 5, recent_events=[], timestamp=now)

    prompt = build_user_prompt(meta, dc)

    assert "(none — this is the first strategy)" in prompt


def test_i_time_and_session_context() -> None:
    # Fixed timestamp: Tuesday June 3 2025 at 21:30 Eastern Time
    fixed_dt = datetime(2025, 6, 3, 21, 30, tzinfo=ZoneInfo("America/New_York"))
    now = fixed_dt.timestamp()
    session_start = now - 900.0  # 15 minutes ago

    dc = DynamicContext(now=now, session_start=session_start, available_regions=())
    meta = MetaState(vector=[0.5] * 5, recent_events=[], timestamp=now)

    prompt = build_user_prompt(meta, dc)

    assert "Weekday: Tuesday" in prompt
    assert "Day part: evening" in prompt
    assert "Session duration: 15 minutes" in prompt
    assert "Estimated session remaining: 60 minutes" in prompt  # max(5, min(60, 90 - 15)) = 60


def test_j_weekend_detection() -> None:
    # Saturday June 7 2025 at 14:00 ET
    sat_dt = datetime(2025, 6, 7, 14, 0, tzinfo=ZoneInfo("America/New_York"))
    now = sat_dt.timestamp()
    meta = MetaState(vector=[0.5] * 5, recent_events=[], timestamp=now)

    # Auto-detection (is_weekend=None)
    dc_auto = DynamicContext(now=now, session_start=now - 300, available_regions=())
    prompt_auto = build_user_prompt(meta, dc_auto)
    assert "Is weekend: True" in prompt_auto

    # Explicit override (is_weekend=False)
    dc_override = DynamicContext(now=now, session_start=now - 300, available_regions=(), is_weekend=False)
    prompt_override = build_user_prompt(meta, dc_override)
    assert "Is weekend: False" in prompt_override


def test_k_available_regions() -> None:
    now = 1700000000.0
    regions = ("Elwynn Forest", "Westfall", "Darkshore")
    dc = DynamicContext(now=now, session_start=now - 600.0, available_regions=regions)
    meta = MetaState(vector=[0.5] * 5, recent_events=[], timestamp=now)

    prompt = build_user_prompt(meta, dc)

    assert "Elwynn Forest, Westfall, Darkshore" in prompt


def test_l_determinism() -> None:
    now = 1700000000.0
    dc = DynamicContext(now=now, session_start=now - 600.0, available_regions=("Durotar",))
    meta = MetaState(vector=[0.2, 0.4, 0.6, 0.8, 1.0], recent_events=[], timestamp=now)

    prompt1 = build_user_prompt(meta, dc)
    prompt2 = build_user_prompt(meta, dc)

    assert prompt1 == prompt2


def test_m_no_none_literal_leak() -> None:
    now = 1700000000.0
    dc = DynamicContext(
        now=now,
        session_start=now - 600.0,
        available_regions=(),
        previous_strategy=None,
    )
    meta = MetaState(vector=[0.5] * 5, recent_events=[], timestamp=now)

    prompt = build_user_prompt(meta, dc)

    assert "None" not in prompt


def test_n_formatting_discipline() -> None:
    now = 1700000000.0
    dc = DynamicContext(now=now, session_start=now - 600.0, available_regions=("Ashenvale",))
    meta = MetaState(vector=[0.5] * 5, recent_events=[], timestamp=now)

    prompt = build_user_prompt(meta, dc)

    lines = prompt.splitlines()
    for line in lines:
        assert line == line.rstrip(), f"Trailing whitespace on line: {line!r}"

    assert "\n\n\n" not in prompt, "Consecutive blank lines found in prompt"
    assert prompt.endswith("\n"), "Prompt must end with a single newline"
    assert not prompt.endswith("\n\n"), "Prompt must not end with multiple newlines"


def test_o_no_llm_imports_on_prompts_import() -> None:
    # Run in a clean python subprocess with PYTHONPATH set to src
    code = (
        "import sys\n"
        "import wow_bot.strategist.prompts\n"
        "assert 'openai' not in sys.modules, 'openai was imported'\n"
        "assert 'httpx' not in sys.modules, 'httpx was imported'\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f"src:{env.get('PYTHONPATH', '')}"
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"Subprocess failed: {result.stderr}"


def test_p_system_prompt_file_utf8() -> None:
    from wow_bot.strategist.prompts import _SYSTEM_PROMPT_PATH

    raw_bytes = _SYSTEM_PROMPT_PATH.read_bytes()
    decoded_text = raw_bytes.decode("utf-8")
    assert len(decoded_text) > 0


def test_q_persona_research_framing() -> None:
    text = load_system_prompt()
    assert "synthetic player persona used in a research simulation" in text


def test_r_no_anti_cheat_language() -> None:
    text = load_system_prompt().lower()
    forbidden_terms = [
        "anti-cheat",
        "undetectable",
        "evade detection",
        "avoid detection",
        "warden",
        "hide automation",
    ]
    for term in forbidden_terms:
        assert term not in text, f"Forbidden anti-cheat term found: {term!r}"


def test_s_no_rigid_anti_repetition_rules() -> None:
    text = load_system_prompt().lower()
    forbidden_rules = [
        "never use the same goal",
        "never use the same region",
        "always change",
        "must alternate",
    ]
    for rule in forbidden_rules:
        assert rule not in text, f"Forbidden rigid anti-repetition rule found: {rule!r}"
