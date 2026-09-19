"""Unit tests for the Strategy response parser and validator (Task 4.3).

Covers required tests A through AL from Task 4.3 specification.
All tests are offline, fast, and deterministic.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import pytest

from wow_bot.shared.interfaces import STRATEGY_GOALS, Strategy
from wow_bot.strategist.parser import StrategyParseError, parse_strategy_response


def valid_payload() -> dict[str, Any]:
    """Helper fixture providing a canonical valid Task 4.2 response payload."""
    return {
        "reasoning": "Feeling energetic and ready to farm herbs in a safe zone.",
        "goal": "farm_herbs",
        "region": "Arathi Highlands",
        "risk_tolerance": 0.35,
        "priority": ["gather_herbs", "avoid_aggro"],
        "constraints": {
            "max_deaths_per_hour": 3,
            "max_session_minutes": 30,
            "avoid_pvp": False,
        },
    }


# Test A — canonical valid payload
def test_parse_canonical_valid_payload() -> None:
    payload = valid_payload()
    raw = json.dumps(payload)
    valid_until = 1700000000.0

    strategy = parse_strategy_response(raw, valid_until=valid_until)

    assert isinstance(strategy, Strategy)
    assert strategy.goal == "farm_herbs"
    assert strategy.region == "Arathi Highlands"
    assert strategy.risk_tolerance == 0.35
    assert strategy.priority == ["gather_herbs", "avoid_aggro"]
    assert strategy.constraints == {
        "max_deaths_per_hour": 3,
        "max_session_minutes": 30,
        "avoid_pvp": False,
    }
    assert strategy.valid_until == valid_until
    assert strategy.raw_llm_output == raw


# Test B — reasoning is not a Strategy field
def test_reasoning_not_added_to_strategy_fields() -> None:
    payload = valid_payload()
    payload["reasoning"] = "Reasoning diagnostic string"
    raw = json.dumps(payload)

    strategy = parse_strategy_response(raw, valid_until=100.0)

    assert not hasattr(strategy, "reasoning")
    assert "reasoning" not in strategy.to_dict()
    assert strategy.raw_llm_output == raw
    assert "Reasoning diagnostic string" in strategy.raw_llm_output


# Test C — supplied valid_until is used exactly
def test_valid_until_used_exactly() -> None:
    raw = json.dumps(valid_payload())
    ts = 12345.5

    strategy = parse_strategy_response(raw, valid_until=ts)
    assert strategy.valid_until == ts


# Test D — raw response preserved exactly (unstripped)
def test_raw_response_preserved_exactly() -> None:
    inner = json.dumps(valid_payload())
    raw = f"  \n\n{inner}\n\n  "

    strategy = parse_strategy_response(raw, valid_until=100.0)
    assert strategy.raw_llm_output == raw


# Test E — leading/trailing whitespace
def test_leading_trailing_whitespace() -> None:
    raw = f"\n\t  {json.dumps(valid_payload())}  \t\n"
    strategy = parse_strategy_response(raw, valid_until=100.0)
    assert strategy.goal == "farm_herbs"


# Test F — complete JSON Markdown fence
@pytest.mark.parametrize("tag", ["json", "JSON", ""])
def test_markdown_code_fence(tag: str) -> None:
    inner = json.dumps(valid_payload(), indent=2)
    raw = f"```{tag}\n{inner}\n```"

    strategy = parse_strategy_response(raw, valid_until=100.0)
    assert strategy.goal == "farm_herbs"
    assert strategy.raw_llm_output == raw


# Test G — prose before JSON
def test_prose_before_json_rejected() -> None:
    raw = f"Here is the strategy:\n{json.dumps(valid_payload())}"
    with pytest.raises(StrategyParseError, match="not valid JSON|Unclosed code fence"):
        parse_strategy_response(raw, valid_until=100.0)


# Test H — prose after JSON
def test_prose_after_json_rejected() -> None:
    raw = f"{json.dumps(valid_payload())}\nDone."
    with pytest.raises(StrategyParseError, match="not valid JSON"):
        parse_strategy_response(raw, valid_until=100.0)


# Test I — malformed JSON
def test_malformed_json_rejected() -> None:
    raw = '{"goal": "farm_herbs"'
    with pytest.raises(StrategyParseError, match="not valid JSON") as exc_info:
        parse_strategy_response(raw, valid_until=100.0)
    assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)


# Test J — Python dict syntax
def test_python_dict_syntax_rejected() -> None:
    raw = "{'goal': 'farm_herbs', 'region': 'Arathi Highlands'}"
    with pytest.raises(StrategyParseError, match="not valid JSON"):
        parse_strategy_response(raw, valid_until=100.0)


# Test K — empty response
@pytest.mark.parametrize("raw", ["", "   ", "\n\t"])
def test_empty_response_rejected(raw: str) -> None:
    with pytest.raises(StrategyParseError, match="empty"):
        parse_strategy_response(raw, valid_until=100.0)


# Test L — multiple objects
def test_multiple_objects_rejected() -> None:
    p1 = json.dumps(valid_payload())
    p2 = json.dumps(valid_payload())
    raw = f"{p1}\n{p2}"
    with pytest.raises(StrategyParseError, match="not valid JSON"):
        parse_strategy_response(raw, valid_until=100.0)


# Test M — non-object top-level values
@pytest.mark.parametrize("val", [[], None, "hello", 42, True])
def test_non_object_toplevel_rejected(val: Any) -> None:
    raw = json.dumps(val)
    with pytest.raises(StrategyParseError, match="JSON object"):
        parse_strategy_response(raw, valid_until=100.0)


# Test N — missing reasoning
def test_missing_reasoning_rejected() -> None:
    payload = valid_payload()
    del payload["reasoning"]
    with pytest.raises(StrategyParseError, match="Missing required strategy field: reasoning"):
        parse_strategy_response(json.dumps(payload), valid_until=100.0)


# Test O — empty reasoning
@pytest.mark.parametrize("reasoning", ["", "   ", "\n"])
def test_empty_reasoning_rejected(reasoning: str) -> None:
    payload = valid_payload()
    payload["reasoning"] = reasoning
    with pytest.raises(StrategyParseError, match="reasoning must be a non-empty string"):
        parse_strategy_response(json.dumps(payload), valid_until=100.0)


# Test P — missing Strategy-mapped field
@pytest.mark.parametrize("field_name", ["goal", "region", "risk_tolerance", "priority", "constraints"])
def test_missing_strategy_field_rejected(field_name: str) -> None:
    payload = valid_payload()
    del payload[field_name]
    with pytest.raises(StrategyParseError, match=f"Missing required strategy field: {field_name}"):
        parse_strategy_response(json.dumps(payload), valid_until=100.0)


# Test Q — extra top-level field
def test_extra_toplevel_field_rejected() -> None:
    payload = valid_payload()
    payload["confidence"] = 0.9
    with pytest.raises(StrategyParseError, match="Unexpected strategy field: confidence"):
        parse_strategy_response(json.dumps(payload), valid_until=100.0)


# Test R — all canonical goals accepted
@pytest.mark.parametrize("goal", sorted(STRATEGY_GOALS))
def test_all_canonical_goals_accepted(goal: str) -> None:
    payload = valid_payload()
    payload["goal"] = goal
    strategy = parse_strategy_response(json.dumps(payload), valid_until=100.0)
    assert strategy.goal == goal


# Test S — unsupported goal rejected
def test_unsupported_goal_rejected() -> None:
    payload = valid_payload()
    payload["goal"] = "farm_gold"
    with pytest.raises(StrategyParseError, match="Unsupported strategy goal: 'farm_gold'"):
        parse_strategy_response(json.dumps(payload), valid_until=100.0)


# Test T — region validation
@pytest.mark.parametrize("region", ["", "   ", 123, None, []])
def test_invalid_region_rejected(region: Any) -> None:
    payload = valid_payload()
    payload["region"] = region
    with pytest.raises(StrategyParseError, match="region must be a non-empty string"):
        parse_strategy_response(json.dumps(payload), valid_until=100.0)


# Test U — risk lower bound
def test_risk_lower_bound() -> None:
    payload = valid_payload()
    payload["risk_tolerance"] = -0.01
    with pytest.raises(StrategyParseError, match="risk_tolerance must be a finite number in \\[0, 1\\]"):
        parse_strategy_response(json.dumps(payload), valid_until=100.0)


# Test V — risk upper bound
def test_risk_upper_bound() -> None:
    payload = valid_payload()
    payload["risk_tolerance"] = 1.01
    with pytest.raises(StrategyParseError, match="risk_tolerance must be a finite number in \\[0, 1\\]"):
        parse_strategy_response(json.dumps(payload), valid_until=100.0)


# Test W — valid risk boundaries
@pytest.mark.parametrize("boundary", [0.0, 1.0])
def test_valid_risk_boundaries(boundary: float) -> None:
    payload = valid_payload()
    payload["risk_tolerance"] = boundary
    strategy = parse_strategy_response(json.dumps(payload), valid_until=100.0)
    assert strategy.risk_tolerance == boundary


# Test X — boolean risk rejected
def test_boolean_risk_rejected() -> None:
    payload = valid_payload()
    payload["risk_tolerance"] = True
    with pytest.raises(StrategyParseError, match="risk_tolerance must be a finite number in \\[0, 1\\]"):
        parse_strategy_response(json.dumps(payload), valid_until=100.0)


# Test Y — string risk rejected
def test_string_risk_rejected() -> None:
    payload = valid_payload()
    payload["risk_tolerance"] = "0.5"
    with pytest.raises(StrategyParseError, match="risk_tolerance must be a finite number in \\[0, 1\\]"):
        parse_strategy_response(json.dumps(payload), valid_until=100.0)


# Test Z — non-finite JSON constants
@pytest.mark.parametrize("constant_str", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_json_constants_rejected(constant_str: str) -> None:
    raw = json.dumps(valid_payload()).replace("0.35", constant_str)
    with pytest.raises(StrategyParseError, match="Non-standard JSON numeric constant rejected"):
        parse_strategy_response(raw, valid_until=100.0)


# Test AA — priority must be array
def test_priority_must_be_array() -> None:
    payload = valid_payload()
    payload["priority"] = "safe"
    with pytest.raises(StrategyParseError, match="priority must be a list of non-empty strings"):
        parse_strategy_response(json.dumps(payload), valid_until=100.0)


# Test AB — invalid priority member
@pytest.mark.parametrize("invalid_item", [42, "", "   ", None])
def test_invalid_priority_member(invalid_item: Any) -> None:
    payload = valid_payload()
    payload["priority"] = ["safe", invalid_item]
    with pytest.raises(StrategyParseError, match="priority entries must be non-empty strings"):
        parse_strategy_response(json.dumps(payload), valid_until=100.0)


# Test AC — valid constraints
def test_valid_constraints_mapped() -> None:
    payload = valid_payload()
    strategy = parse_strategy_response(json.dumps(payload), valid_until=100.0)
    assert strategy.constraints == {
        "max_deaths_per_hour": 3,
        "max_session_minutes": 30,
        "avoid_pvp": False,
    }


# Test AD — missing constraint key
@pytest.mark.parametrize("c_key", ["max_deaths_per_hour", "max_session_minutes", "avoid_pvp"])
def test_missing_constraint_key_rejected(c_key: str) -> None:
    payload = valid_payload()
    del payload["constraints"][c_key]
    with pytest.raises(StrategyParseError, match=f"Missing required constraint field: {c_key}"):
        parse_strategy_response(json.dumps(payload), valid_until=100.0)


# Test AE — unexpected constraint key
def test_unexpected_constraint_key_rejected() -> None:
    payload = valid_payload()
    payload["constraints"]["unknown"] = True
    with pytest.raises(StrategyParseError, match="Unexpected constraint field: unknown"):
        parse_strategy_response(json.dumps(payload), valid_until=100.0)


# Test AF — max deaths type/range
@pytest.mark.parametrize("val", [-1, 2.5, "3", True, False])
def test_max_deaths_type_range_rejected(val: Any) -> None:
    payload = valid_payload()
    payload["constraints"]["max_deaths_per_hour"] = val
    with pytest.raises(StrategyParseError, match="max_deaths_per_hour must be a non-negative integer"):
        parse_strategy_response(json.dumps(payload), valid_until=100.0)


def test_max_deaths_zero_accepted() -> None:
    payload = valid_payload()
    payload["constraints"]["max_deaths_per_hour"] = 0
    strategy = parse_strategy_response(json.dumps(payload), valid_until=100.0)
    assert strategy.constraints["max_deaths_per_hour"] == 0


# Test AG — session minutes type/range
@pytest.mark.parametrize("val", [0, -10, 15.5, "30", True, False])
def test_max_session_minutes_type_range_rejected(val: Any) -> None:
    payload = valid_payload()
    payload["constraints"]["max_session_minutes"] = val
    with pytest.raises(StrategyParseError, match="max_session_minutes must be a positive integer"):
        parse_strategy_response(json.dumps(payload), valid_until=100.0)


# Test AH — avoid_pvp strict boolean
@pytest.mark.parametrize("val", [True, False])
def test_avoid_pvp_boolean_accepted(val: bool) -> None:
    payload = valid_payload()
    payload["constraints"]["avoid_pvp"] = val
    strategy = parse_strategy_response(json.dumps(payload), valid_until=100.0)
    assert strategy.constraints["avoid_pvp"] is val


@pytest.mark.parametrize("val", [0, 1, "true", "false", None])
def test_avoid_pvp_non_boolean_rejected(val: Any) -> None:
    payload = valid_payload()
    payload["constraints"]["avoid_pvp"] = val
    with pytest.raises(StrategyParseError, match="avoid_pvp must be a boolean"):
        parse_strategy_response(json.dumps(payload), valid_until=100.0)


# Test AI — invalid valid_until
@pytest.mark.parametrize("val", [float("nan"), float("inf"), float("-inf"), True, False, "100"])
def test_invalid_valid_until_rejected(val: Any) -> None:
    raw = json.dumps(valid_payload())
    with pytest.raises(StrategyParseError, match="valid_until must be a finite number"):
        parse_strategy_response(raw, valid_until=val)


# Test AJ — parser deterministic
def test_parser_deterministic() -> None:
    raw = json.dumps(valid_payload())
    valid_until = 5000.0

    res1 = parse_strategy_response(raw, valid_until=valid_until)
    res2 = parse_strategy_response(raw, valid_until=valid_until)

    assert res1 == res2


# Test AK — no clock usage
def test_no_clock_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    def _clock_trap(*args: Any, **kwargs: Any) -> float:
        raise AssertionError("Clock call detected inside parser!")

    monkeypatch.setattr("time.time", _clock_trap)
    monkeypatch.setattr("datetime.datetime", _clock_trap)

    raw = json.dumps(valid_payload())
    strategy = parse_strategy_response(raw, valid_until=100.0)
    assert strategy.goal == "farm_herbs"


# Test AL — no network coupling
def test_no_network_coupling() -> None:
    assert "wow_bot.strategist.parser" in sys.modules
    parser_mod = sys.modules["wow_bot.strategist.parser"]
    assert not hasattr(parser_mod, "LLMClient")
    assert not hasattr(parser_mod, "AsyncOpenAI")
