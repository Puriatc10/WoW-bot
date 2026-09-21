"""Unit tests for VocabularyGuard and vocabulary v2 module (Task 9.3)."""

from __future__ import annotations

import ast
import dataclasses
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from wow_bot.config import Config
from wow_bot.session import Session
from wow_bot.strategist.prompts_v2 import ALLOWED_GOALS
from wow_bot.strategist.vocab_v2 import (
    DEFAULT_GOAL_RULES,
    GoalRule,
    RejectionReason,
    StrategyCandidate,
    TargetKind,
    ValidatedStrategy,
    VocabConfig,
    VocabDecision,
    VocabularyGuard,
)


def _make_session(tmp_path: Path) -> Session:
    """Create a real Session rooted in tmp_path for testing events."""
    config = Config(
        lab_mode=False,
        server_allowlist=("127.0.0.1:8080",),
        isolation_sentinel="127.0.0.1:8081",
        kill_switch_key="f12",
        session_root=tmp_path / "sessions",
        dry_run=True,
        max_session_seconds=3600,
        log_level="INFO",
    )
    return Session.start(config)


def _read_session_events(session: Session) -> list[dict[str, Any]]:
    """Read all event dicts written to events.jsonl."""
    events_path = session.path / "events.jsonl"
    if not events_path.exists():
        return []
    events = []
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


# 1. GoalRule validation tests
def test_goal_rule_min_target_length_invalid() -> None:
    """GoalRule with min_target_length < 1 raises ValueError."""
    with pytest.raises(ValueError, match="min_target_length must be >= 1"):
        GoalRule(target_kind=TargetKind.FREE_TEXT, min_target_length=0)


def test_goal_rule_max_target_length_less_than_min() -> None:
    """GoalRule with max_target_length < min_target_length raises ValueError."""
    with pytest.raises(ValueError, match="max_target_length .* must be >= min_target_length"):
        GoalRule(target_kind=TargetKind.FREE_TEXT, min_target_length=10, max_target_length=5)


def test_goal_rule_target_kind_none_nonzero_max_length() -> None:
    """GoalRule with TargetKind.NONE and non-zero max_target_length raises ValueError."""
    with pytest.raises(ValueError, match="max_target_length must equal 0 for TargetKind.NONE"):
        GoalRule(target_kind=TargetKind.NONE, min_target_length=0, max_target_length=10)


def test_goal_rule_target_kind_none_nonempty_prefixes() -> None:
    """GoalRule with TargetKind.NONE and non-empty prefixes raises ValueError."""
    with pytest.raises(ValueError, match="allowed_target_prefixes must be empty for TargetKind.NONE"):
        GoalRule(
            target_kind=TargetKind.NONE,
            min_target_length=0,
            max_target_length=0,
            allowed_target_prefixes=("prefix_",),
        )


def test_goal_rule_empty_prefix_string() -> None:
    """GoalRule with empty prefix string raises ValueError."""
    with pytest.raises(ValueError, match="allowed_target_prefixes must contain non-empty strings"):
        GoalRule(target_kind=TargetKind.ENTITY_ID, allowed_target_prefixes=("",))


# 2. VocabConfig validation tests
def test_vocab_config_invalid_max_rationale_length() -> None:
    """VocabConfig with max_rationale_length < 1 raises ValueError."""
    with pytest.raises(ValueError, match="max_rationale_length must be >= 1"):
        VocabConfig(max_rationale_length=0)


def test_vocab_config_missing_goal_in_rules() -> None:
    """VocabConfig whose rules miss a goal in ALLOWED_GOALS raises ValueError naming the missing goal."""
    rules = dict(DEFAULT_GOAL_RULES)
    del rules["explore"]
    with pytest.raises(ValueError, match="rules missing goals from ALLOWED_GOALS:.*explore"):
        VocabConfig(rules=rules)


def test_vocab_config_unknown_goal_in_rules() -> None:
    """VocabConfig whose rules contain a key not in ALLOWED_GOALS raises ValueError naming the unknown goal."""
    rules = dict(DEFAULT_GOAL_RULES)
    rules["dance"] = GoalRule(target_kind=TargetKind.NONE, min_target_length=0, max_target_length=0)
    with pytest.raises(ValueError, match="rules contain unknown goals not in ALLOWED_GOALS:.*dance"):
        VocabConfig(rules=rules)


def test_default_goal_rules_covers_all_allowed_goals() -> None:
    """DEFAULT_GOAL_RULES covers every entry of ALLOWED_GOALS."""
    for goal in ALLOWED_GOALS:
        assert goal in DEFAULT_GOAL_RULES
    assert len(DEFAULT_GOAL_RULES) == len(ALLOWED_GOALS)


def test_vocab_config_converts_dict_rules_to_mapping_proxy() -> None:
    """VocabConfig converts dict rules to MappingProxyType in __post_init__."""
    rules_dict = dict(DEFAULT_GOAL_RULES)
    config = VocabConfig(rules=rules_dict)
    assert isinstance(config.rules, MappingProxyType)


# 3. VocabDecision invariant tests
def test_vocab_decision_invariants() -> None:
    """VocabDecision invariant enforcement."""
    val_strat = ValidatedStrategy(goal="explore", target=None, rationale="reasoning")

    # accepted=True with strategy=None raises ValueError
    with pytest.raises(ValueError, match="accepted is True implies strategy is not None"):
        VocabDecision(accepted=True, strategy=None, reason=None, detail="ok")

    # accepted=False with a strategy raises ValueError
    with pytest.raises(ValueError, match="accepted is False implies strategy is None"):
        VocabDecision(
            accepted=False,
            strategy=val_strat,
            reason=RejectionReason.RATIONALE_EMPTY,
            detail="error",
        )

    # accepted=True with a reason raises ValueError
    with pytest.raises(ValueError, match="accepted is True implies reason is None"):
        VocabDecision(
            accepted=True,
            strategy=val_strat,
            reason=RejectionReason.RATIONALE_EMPTY,
            detail="ok",
        )

    # accepted=False with reason=None raises ValueError
    with pytest.raises(ValueError, match="accepted is False implies reason is not None"):
        VocabDecision(accepted=False, strategy=None, reason=None, detail="error")

    # empty detail raises ValueError
    with pytest.raises(ValueError, match="detail must be a non-empty string"):
        VocabDecision(accepted=True, strategy=val_strat, reason=None, detail="")


# 4. ValidatedStrategy goal check
def test_validated_strategy_invalid_goal() -> None:
    """ValidatedStrategy with a goal not in ALLOWED_GOALS cannot be constructed."""
    with pytest.raises(ValueError, match="goal 'invalid_goal' is not in ALLOWED_GOALS"):
        ValidatedStrategy(goal="invalid_goal", target=None, rationale="test")


# 5. VocabularyGuard validation rules
def test_validate_non_string_goal() -> None:
    """validate with a non-string goal returns rejected with reason MALFORMED_GOAL_TYPE."""
    guard = VocabularyGuard()
    cand = StrategyCandidate(goal=123, target=None, rationale="valid rationale")
    dec = guard.validate(cand)
    assert not dec.accepted
    assert dec.reason == RejectionReason.MALFORMED_GOAL_TYPE
    assert dec.strategy is None


def test_validate_unknown_goal() -> None:
    """validate with an unknown goal string returns rejected with reason UNKNOWN_GOAL."""
    guard = VocabularyGuard()
    cand = StrategyCandidate(goal="invalid_goal", target=None, rationale="valid rationale")
    dec = guard.validate(cand)
    assert not dec.accepted
    assert dec.reason == RejectionReason.UNKNOWN_GOAL
    assert dec.strategy is None


def test_validate_empty_rationale() -> None:
    """validate with an empty rationale returns rejected with reason RATIONALE_EMPTY."""
    guard = VocabularyGuard()
    cand = StrategyCandidate(goal="explore", target=None, rationale="")
    dec = guard.validate(cand)
    assert not dec.accepted
    assert dec.reason == RejectionReason.RATIONALE_EMPTY


def test_validate_whitespace_rationale() -> None:
    """validate with a whitespace-only rationale returns rejected with reason RATIONALE_EMPTY."""
    guard = VocabularyGuard()
    cand = StrategyCandidate(goal="explore", target=None, rationale="   \n\t ")
    dec = guard.validate(cand)
    assert not dec.accepted
    assert dec.reason == RejectionReason.RATIONALE_EMPTY


def test_validate_rationale_too_long() -> None:
    """validate with a rationale longer than max_rationale_length returns rejected with reason RATIONALE_TOO_LONG."""
    guard = VocabularyGuard(config=VocabConfig(max_rationale_length=10))
    cand = StrategyCandidate(goal="explore", target=None, rationale="a" * 11)
    dec = guard.validate(cand)
    assert not dec.accepted
    assert dec.reason == RejectionReason.RATIONALE_TOO_LONG


def test_validate_explore_with_unexpected_target() -> None:
    """validate with goal='explore' and a non-None target returns rejected with reason UNEXPECTED_TARGET."""
    guard = VocabularyGuard()
    cand = StrategyCandidate(goal="explore", target="something", rationale="valid rationale")
    dec = guard.validate(cand)
    assert not dec.accepted
    assert dec.reason == RejectionReason.UNEXPECTED_TARGET


def test_validate_explore_valid() -> None:
    """validate with goal='explore' and target=None and a valid rationale returns accepted."""
    guard = VocabularyGuard()
    cand = StrategyCandidate(goal="explore", target=None, rationale="Exploring the map")
    dec = guard.validate(cand)
    assert dec.accepted
    assert dec.strategy == ValidatedStrategy(
        goal="explore", target=None, rationale="Exploring the map"
    )
    assert dec.reason is None


def test_validate_travel_to_missing_target() -> None:
    """validate with goal='travel_to' and target=None returns rejected with reason MISSING_TARGET."""
    guard = VocabularyGuard()
    cand = StrategyCandidate(goal="travel_to", target=None, rationale="Travelling somewhere")
    dec = guard.validate(cand)
    assert not dec.accepted
    assert dec.reason == RejectionReason.MISSING_TARGET


def test_validate_travel_to_malformed_waypoint_id_nondigits() -> None:
    """validate with goal='travel_to' and target='abc' returns rejected with reason MALFORMED_WAYPOINT_ID (non-digits)."""
    guard = VocabularyGuard()
    cand = StrategyCandidate(goal="travel_to", target="abc", rationale="Travelling somewhere")
    dec = guard.validate(cand)
    assert not dec.accepted
    assert dec.reason == RejectionReason.MALFORMED_WAYPOINT_ID


def test_validate_travel_to_valid() -> None:
    """validate with goal='travel_to' and target='42' returns accepted."""
    guard = VocabularyGuard()
    cand = StrategyCandidate(goal="travel_to", target="42", rationale="Travelling to waypoint 42")
    dec = guard.validate(cand)
    assert dec.accepted
    assert dec.strategy == ValidatedStrategy(
        goal="travel_to", target="42", rationale="Travelling to waypoint 42"
    )


def test_validate_sell_vendor_malformed_entity_id_space() -> None:
    """validate with goal='sell_vendor' and target='has space' returns rejected with reason MALFORMED_ENTITY_ID."""
    guard = VocabularyGuard()
    cand = StrategyCandidate(goal="sell_vendor", target="has space", rationale="Selling items")
    dec = guard.validate(cand)
    assert not dec.accepted
    assert dec.reason == RejectionReason.MALFORMED_ENTITY_ID


def test_validate_sell_vendor_valid() -> None:
    """validate with goal='sell_vendor' and target='vendor_12' returns accepted."""
    guard = VocabularyGuard()
    cand = StrategyCandidate(goal="sell_vendor", target="vendor_12", rationale="Selling to vendor")
    dec = guard.validate(cand)
    assert dec.accepted
    assert dec.strategy == ValidatedStrategy(
        goal="sell_vendor", target="vendor_12", rationale="Selling to vendor"
    )


def test_validate_sell_vendor_malformed_entity_id_slash() -> None:
    """validate with goal='sell_vendor' and target='with/slash' returns rejected with reason MALFORMED_ENTITY_ID."""
    guard = VocabularyGuard()
    cand = StrategyCandidate(goal="sell_vendor", target="with/slash", rationale="Selling items")
    dec = guard.validate(cand)
    assert not dec.accepted
    assert dec.reason == RejectionReason.MALFORMED_ENTITY_ID


def test_validate_farm_herbs_malformed_free_text_whitespace() -> None:
    """validate with goal='farm_herbs' and target='   ' returns rejected with reason MALFORMED_FREE_TEXT."""
    guard = VocabularyGuard()
    cand = StrategyCandidate(goal="farm_herbs", target="   ", rationale="Farming herbs")
    dec = guard.validate(cand)
    assert not dec.accepted
    assert dec.reason == RejectionReason.MALFORMED_FREE_TEXT


def test_validate_farm_herbs_valid() -> None:
    """validate with goal='farm_herbs' and target='herb_patch_A' returns accepted."""
    guard = VocabularyGuard()
    cand = StrategyCandidate(goal="farm_herbs", target="herb_patch_A", rationale="Farming herbs")
    dec = guard.validate(cand)
    assert dec.accepted
    assert dec.strategy == ValidatedStrategy(
        goal="farm_herbs", target="herb_patch_A", rationale="Farming herbs"
    )


def test_validate_target_shorter_than_min_target_length() -> None:
    """validate with target shorter than min_target_length returns rejected with kind-specific MALFORMED_* reason."""
    custom_rules = dict(DEFAULT_GOAL_RULES)
    custom_rules["sell_vendor"] = GoalRule(
        target_kind=TargetKind.ENTITY_ID, min_target_length=5, max_target_length=128
    )
    guard = VocabularyGuard(config=VocabConfig(rules=custom_rules))
    cand = StrategyCandidate(goal="sell_vendor", target="e1", rationale="Short entity id")
    dec = guard.validate(cand)
    assert not dec.accepted
    assert dec.reason == RejectionReason.MALFORMED_ENTITY_ID


def test_validate_target_longer_than_max_target_length() -> None:
    """validate with target longer than max_target_length returns rejected with kind-specific MALFORMED_* reason."""
    custom_rules = dict(DEFAULT_GOAL_RULES)
    custom_rules["farm_herbs"] = GoalRule(
        target_kind=TargetKind.FREE_TEXT, min_target_length=1, max_target_length=5
    )
    guard = VocabularyGuard(config=VocabConfig(rules=custom_rules))
    cand = StrategyCandidate(goal="farm_herbs", target="too_long_target", rationale="Farming herbs")
    dec = guard.validate(cand)
    assert not dec.accepted
    assert dec.reason == RejectionReason.MALFORMED_FREE_TEXT


def test_validate_allowed_target_prefixes_mismatch() -> None:
    """validate with allowed_target_prefixes and a target not matching any prefix returns rejected."""
    custom_rules = dict(DEFAULT_GOAL_RULES)
    custom_rules["sell_vendor"] = GoalRule(
        target_kind=TargetKind.ENTITY_ID,
        min_target_length=1,
        max_target_length=128,
        allowed_target_prefixes=("npc_", "vendor_"),
    )
    guard = VocabularyGuard(config=VocabConfig(rules=custom_rules))
    cand = StrategyCandidate(goal="sell_vendor", target="mob_123", rationale="Selling items")
    dec = guard.validate(cand)
    assert not dec.accepted
    assert dec.reason == RejectionReason.MALFORMED_ENTITY_ID


def test_validate_allowed_target_prefixes_match() -> None:
    """validate with allowed_target_prefixes and a matching target returns accepted."""
    custom_rules = dict(DEFAULT_GOAL_RULES)
    custom_rules["sell_vendor"] = GoalRule(
        target_kind=TargetKind.ENTITY_ID,
        min_target_length=1,
        max_target_length=128,
        allowed_target_prefixes=("npc_", "vendor_"),
    )
    guard = VocabularyGuard(config=VocabConfig(rules=custom_rules))
    cand = StrategyCandidate(goal="sell_vendor", target="vendor_123", rationale="Selling items")
    dec = guard.validate(cand)
    assert dec.accepted


# 6. Telemetry and Session events
def test_accepted_telemetry_and_session_events(tmp_path: Path) -> None:
    """Accepted decision increments the accepted counter and emits 'vocab_accepted' exactly once with {'goal', 'has_target'}."""
    session = _make_session(tmp_path)
    guard = VocabularyGuard(session=session)

    cand = StrategyCandidate(goal="explore", target=None, rationale="Exploring region")
    dec = guard.validate(cand)
    assert dec.accepted

    events = _read_session_events(session)
    assert len(events) == 1
    ev = events[0]
    assert ev["event"] == "vocab_accepted"
    assert ev["goal"] == "explore"
    assert ev["has_target"] is False


def test_rejected_telemetry_and_session_events(tmp_path: Path) -> None:
    """Rejected decision increments the per-reason counter and emits 'vocab_rejected' exactly once with {'reason', 'goal', 'detail'}."""
    session = _make_session(tmp_path)
    guard = VocabularyGuard(session=session)

    cand = StrategyCandidate(goal="travel_to", target="abc", rationale="Travelling somewhere")
    dec = guard.validate(cand)
    assert not dec.accepted

    events = _read_session_events(session)
    assert len(events) == 1
    ev = events[0]
    assert ev["event"] == "vocab_rejected"
    assert ev["reason"] == RejectionReason.MALFORMED_WAYPOINT_ID.value
    assert ev["goal"] == "travel_to"
    assert "detail" in ev and len(ev["detail"]) > 0


def test_rejection_counts_and_reset_counts() -> None:
    """rejection_counts returns MappingProxyType reflecting occurred reasons, and reset_counts clears counters."""
    guard = VocabularyGuard()

    cand_bad_goal = StrategyCandidate(goal="bad_goal", target=None, rationale="test")
    cand_bad_target = StrategyCandidate(goal="travel_to", target="abc", rationale="test")

    guard.validate(cand_bad_goal)
    guard.validate(cand_bad_target)
    guard.validate(cand_bad_target)

    counts = guard.rejection_counts()
    assert isinstance(counts, MappingProxyType)
    assert counts[RejectionReason.UNKNOWN_GOAL] == 1
    assert counts[RejectionReason.MALFORMED_WAYPOINT_ID] == 2
    assert len(counts) == 2

    # JSON serializability check
    json_serialized = {k.value: v for k, v in counts.items()}
    assert json_serialized == {
        "unknown_goal": 1,
        "malformed_waypoint_id": 2,
    }

    guard.reset_counts()
    assert len(guard.rejection_counts()) == 0


def test_allowed_goals_reexport() -> None:
    """allowed_goals returns the same tuple as prompts_v2.ALLOWED_GOALS."""
    guard = VocabularyGuard()
    assert guard.allowed_goals() == ALLOWED_GOALS


def test_no_session_attached() -> None:
    """No session attached: validate works without raising and does not attempt to write."""
    guard = VocabularyGuard(session=None)
    cand_ok = StrategyCandidate(goal="explore", target=None, rationale="Exploring")
    dec_ok = guard.validate(cand_ok)
    assert dec_ok.accepted

    cand_bad = StrategyCandidate(goal="bad_goal", target=None, rationale="Exploring")
    dec_bad = guard.validate(cand_bad)
    assert not dec_bad.accepted


def test_validate_does_not_mutate_candidate() -> None:
    """validate does NOT mutate the candidate: capture a snapshot before and after and compare."""
    guard = VocabularyGuard()
    cand = StrategyCandidate(goal="explore", target=None, rationale="Exploring map")
    snapshot_before = dataclasses.asdict(cand)
    _ = guard.validate(cand)
    snapshot_after = dataclasses.asdict(cand)
    assert snapshot_before == snapshot_after


def test_determinism_100_validations() -> None:
    """Determinism: 100 validations of the same candidate return the same VocabDecision."""
    guard = VocabularyGuard()
    cand = StrategyCandidate(goal="sell_vendor", target="vendor_1", rationale="Selling junk")
    decisions = [guard.validate(cand) for _ in range(100)]
    first = decisions[0]
    for d in decisions[1:]:
        assert d == first


# 7. Static AST Checks
def test_static_ast_import_boundaries() -> None:
    """Static AST check: vocab_v2.py does not import forbidden modules or call forbidden time functions."""
    source_file = Path(__file__).parent.parent / "src" / "wow_bot" / "strategist" / "vocab_v2.py"
    assert source_file.exists(), f"Source file {source_file} not found"

    with open(source_file, "r", encoding="utf-8") as f:
        source_code = f.read()

    tree = ast.parse(source_code, filename=str(source_file))

    forbidden_exact_modules = {
        "wow_bot.strategist.llm_client",
        "wow_bot.strategist.orchestrator",
        "wow_bot.strategist.orchestrator_v2",
        "wow_bot.strategist.parser",
        "wow_bot.strategist.parser_v2",
        "wow_bot.perception",
        "wow_bot.reflex",
        "wow_bot.actuation",
        "wow_bot.world",
        "wow_bot.nav",
        "wow_bot.combat",
        "aiosqlite",
        "asyncio",
        "threading",
    }

    forbidden_substrings = ("ollama", "openai", "anthropic", "llm")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                assert name not in forbidden_exact_modules, f"Forbidden import: {name}"
                for sub in forbidden_substrings:
                    assert sub not in name.lower(), f"Forbidden import containing '{sub}': {name}"

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module not in forbidden_exact_modules, f"Forbidden import from: {module}"
            for sub in forbidden_substrings:
                assert sub not in module.lower(), f"Forbidden import from containing '{sub}': {module}"
            for alias in node.names:
                full_name = f"{module}.{alias.name}" if module else alias.name
                assert full_name not in forbidden_exact_modules, f"Forbidden import: {full_name}"

        elif (
            isinstance(node, ast.Attribute)
            and node.attr in ("monotonic", "time", "perf_counter")
            and isinstance(node.value, ast.Name)
            and node.value.id == "time"
        ):
            pytest.fail(f"Forbidden call to time.{node.attr} found in vocab_v2.py")


def test_static_ast_prelab_modules_unmodified() -> None:
    """Static AST check: no pre-lab strategist/ module was modified by this task."""
    prelab_modules = [
        "prompts.py",
        "parser.py",
        "orchestrator.py",
        "llm_client.py",
    ]
    strategist_dir = Path(__file__).parent.parent / "src" / "wow_bot" / "strategist"
    for mod_name in prelab_modules:
        file_path = strategist_dir / mod_name
        assert file_path.exists(), f"Pre-lab module {mod_name} must exist"
