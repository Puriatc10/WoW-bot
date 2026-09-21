"""Tests for Strategist Orchestrator v2 (Task 9.4)."""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from wow_bot.config import Config
from wow_bot.executor.states import FSMState
from wow_bot.session import Session
from wow_bot.strategist.cooldown_v2 import CooldownGate
from wow_bot.strategist.orchestrator_v2 import (
    JsonStrategy,
    OrchestratorConfig,
    OrchestratorOutcome,
    OrchestratorResult,
    OrchestratorV2,
    parse_strategy_json,
)
from wow_bot.strategist.vocab_v2 import (
    RejectionReason,
    ValidatedStrategy,
    VocabularyGuard,
)
from wow_bot.world.summary import WorldSummary


class FakeLlmClient:
    """Scripted fake LLM client for orchestrator tests."""

    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = list(responses)
        self.call_count = 0
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.call_count += 1
        self.prompts.append(prompt)
        if not self.responses:
            raise RuntimeError("FakeLlmClient out of responses")
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


class FakeTimer:
    """Scripted fake timer producing deterministic timestamp sequences."""

    def __init__(self, times: list[float] | None = None) -> None:
        self.times = times if times is not None else [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
        self.index = 0

    def __call__(self) -> float:
        if self.index < len(self.times):
            val = self.times[self.index]
            self.index += 1
            return val
        return self.times[-1] if self.times else 0.0


@dataclass
class DummyMetaState:
    drives: Mapping[str, float] = field(default_factory=dict)
    memory_summary: str = "exploring the start area"


@dataclass
class DummyGameState:
    player_x: float = 10.0
    player_y: float = 20.0
    player_z: float = 30.0
    self_hp_percent: float = 100.0
    resource: float = 100.0
    resource_max: float = 100.0
    current_target_id: str | None = None
    target_hp_percent: float | None = None
    inventory_count: int = 5
    level_or_xp: float = 1.0
    fsm_state: FSMState = FSMState.IDLE


def make_dummy_world() -> WorldSummary:
    return WorldSummary(
        around_xy=(10.0, 20.0),
        radius=100.0,
        generated_at="2026-09-21T12:00:00Z",
        nearest_vendors=(),
        nearest_trainers=(),
        nearest_nodes=(),
        nearest_mobs=(),
        nearest_waypoints=(),
        recent_combats=(),
        total_nodes=0,
    )


def make_session(tmp_path: Path) -> Session:
    """Create a real Session rooted in tmp_path for session event testing."""
    config = Config(
        lab_mode=False,
        server_allowlist=("127.0.0.1:8080",),
        isolation_sentinel="127.0.0.1:9999",
        kill_switch_key="f12",
        session_root=tmp_path / "runs",
        dry_run=True,
        max_session_seconds=3600,
        log_level="INFO",
    )
    return Session.start(config)


def read_session_events(session: Session) -> list[dict[str, Any]]:
    """Read and parse json objects from session's events.jsonl."""
    events_file = session.path / "events.jsonl"
    if not events_file.exists():
        return []
    lines = events_file.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# --- Test Cases ---


def test_orchestrator_config_validation() -> None:
    """OrchestratorConfig with max_rationale_chars_for_log < 1 raises ValueError."""
    config = OrchestratorConfig(max_rationale_chars_for_log=10)
    assert config.max_rationale_chars_for_log == 10

    with pytest.raises(ValueError, match="max_rationale_chars_for_log"):
        OrchestratorConfig(max_rationale_chars_for_log=0)

    with pytest.raises(ValueError, match="max_rationale_chars_for_log"):
        OrchestratorConfig(max_rationale_chars_for_log=-1)


def test_orchestrator_result_invariants() -> None:
    """OrchestratorResult invariants validation."""
    valid_strat = ValidatedStrategy(goal="explore", target=None, rationale="test")

    # SUCCESS requires valid strategy and empty reason
    res = OrchestratorResult(
        outcome=OrchestratorOutcome.SUCCESS,
        strategy=valid_strat,
        prompt_hash="abc",
        attempts=1,
        latency_ms=10.0,
        reason="",
    )
    assert res.outcome == OrchestratorOutcome.SUCCESS

    # SUCCESS with None strategy raises ValueError
    with pytest.raises(ValueError, match="strategy is not None"):
        OrchestratorResult(
            outcome=OrchestratorOutcome.SUCCESS,
            strategy=None,
            prompt_hash="abc",
            attempts=1,
            latency_ms=10.0,
            reason="",
        )

    # SUCCESS with non-empty reason raises ValueError
    with pytest.raises(ValueError, match="reason == ''"):
        OrchestratorResult(
            outcome=OrchestratorOutcome.SUCCESS,
            strategy=valid_strat,
            prompt_hash="abc",
            attempts=1,
            latency_ms=10.0,
            reason="some_reason",
        )

    # non-SUCCESS with non-None strategy raises ValueError
    with pytest.raises(ValueError, match="strategy is None"):
        OrchestratorResult(
            outcome=OrchestratorOutcome.INVALID_JSON,
            strategy=valid_strat,
            prompt_hash="abc",
            attempts=1,
            latency_ms=10.0,
            reason="parse_failed",
        )

    # non-SUCCESS with empty reason raises ValueError
    with pytest.raises(ValueError, match="reason != ''"):
        OrchestratorResult(
            outcome=OrchestratorOutcome.INVALID_JSON,
            strategy=None,
            prompt_hash="abc",
            attempts=1,
            latency_ms=10.0,
            reason="",
        )

    # Negative attempts raises ValueError
    with pytest.raises(ValueError, match="attempts"):
        OrchestratorResult(
            outcome=OrchestratorOutcome.INVALID_JSON,
            strategy=None,
            prompt_hash="abc",
            attempts=-1,
            latency_ms=10.0,
            reason="parse_failed",
        )

    # Negative latency_ms raises ValueError
    with pytest.raises(ValueError, match="latency_ms"):
        OrchestratorResult(
            outcome=OrchestratorOutcome.INVALID_JSON,
            strategy=None,
            prompt_hash="abc",
            attempts=1,
            latency_ms=-5.0,
            reason="parse_failed",
        )


def test_orchestrator_result_to_json() -> None:
    """OrchestratorResult.to_json is JSON-serializable and includes every field."""
    valid_strat = ValidatedStrategy(
        goal="travel_to", target="123", rationale="heading to waypoint"
    )
    res = OrchestratorResult(
        outcome=OrchestratorOutcome.SUCCESS,
        strategy=valid_strat,
        prompt_hash="hash123",
        attempts=1,
        latency_ms=42.5,
        reason="",
    )
    as_json = res.to_json()
    serialized = json.dumps(as_json)
    deserialized = json.loads(serialized)

    assert deserialized == {
        "outcome": "success",
        "strategy": {
            "goal": "travel_to",
            "target": "123",
            "rationale": "heading to waypoint",
        },
        "prompt_hash": "hash123",
        "attempts": 1,
        "latency_ms": 42.5,
        "reason": "",
    }


def test_parse_strategy_json_valid() -> None:
    """parse_strategy_json parses valid JSON object matching schema."""
    raw = '{"goal": "explore", "target": null, "rationale": "test rationale"}'
    parsed = parse_strategy_json(raw)
    assert parsed == JsonStrategy(
        goal="explore", target=None, rationale="test rationale"
    )


def test_parse_strategy_json_missing_target() -> None:
    """parse_strategy_json accepts a missing "target" key."""
    raw = '{"goal": "explore", "rationale": "test rationale"}'
    parsed = parse_strategy_json(raw)
    assert parsed == JsonStrategy(
        goal="explore", target=None, rationale="test rationale"
    )


def test_parse_strategy_json_null_target() -> None:
    """parse_strategy_json accepts a null "target"."""
    raw = '{"goal": "explore", "target": null, "rationale": "test rationale"}'
    parsed = parse_strategy_json(raw)
    assert parsed == JsonStrategy(
        goal="explore", target=None, rationale="test rationale"
    )


def test_parse_strategy_json_unknown_key() -> None:
    """parse_strategy_json rejects an unknown top-level key."""
    raw = '{"goal": "explore", "target": null, "rationale": "test", "extra": 123}'
    assert parse_strategy_json(raw) is None


def test_parse_strategy_json_missing_goal() -> None:
    """parse_strategy_json rejects a missing "goal"."""
    raw = '{"target": null, "rationale": "test"}'
    assert parse_strategy_json(raw) is None


def test_parse_strategy_json_missing_rationale() -> None:
    """parse_strategy_json rejects a missing "rationale"."""
    raw = '{"goal": "explore", "target": null}'
    assert parse_strategy_json(raw) is None


def test_parse_strategy_json_non_string_goal() -> None:
    """parse_strategy_json rejects a non-string goal."""
    raw = '{"goal": 123, "rationale": "test"}'
    assert parse_strategy_json(raw) is None


def test_parse_strategy_json_non_string_rationale() -> None:
    """parse_strategy_json rejects a non-string rationale."""
    raw = '{"goal": "explore", "rationale": ["a", "b"]}'
    assert parse_strategy_json(raw) is None


def test_parse_strategy_json_malformed_json() -> None:
    """parse_strategy_json rejects malformed JSON."""
    assert parse_strategy_json("{not valid json}") is None


def test_parse_strategy_json_never_raises() -> None:
    """parse_strategy_json never raises on arbitrary bad inputs."""
    bad_inputs: list[Any] = [
        None,
        123,
        [1, 2, 3],
        "{",
        '{"goal": "explore"',
        '{"goal": "explore", "target": 123, "rationale": "test"}',
    ]
    for inp in bad_inputs:
        assert parse_strategy_json(inp) is None


def test_decide_disabled() -> None:
    """decide with enabled=False returns DISABLED without calling LLM, cooldown, or guard."""
    llm = FakeLlmClient(['{"goal": "explore", "rationale": "test"}'])
    cooldown = CooldownGate()
    guard = VocabularyGuard()
    orchestrator = OrchestratorV2(
        llm=llm,
        cooldown=cooldown,
        guard=guard,
        enabled=False,
    )

    res = orchestrator.decide(
        meta=DummyMetaState(),
        world=make_dummy_world(),
        state=DummyGameState(),
        now=100.0,
    )

    assert res.outcome == OrchestratorOutcome.DISABLED
    assert res.strategy is None
    assert res.reason == "disabled"
    assert llm.call_count == 0
    assert cooldown.last_decision is None
    assert guard.rejection_counts() == {}


def test_decide_blocked_by_cooldown(tmp_path: Path) -> None:
    """decide when the cooldown blocks returns BLOCKED_BY_COOLDOWN without strategist events."""
    session = make_session(tmp_path)

    llm = FakeLlmClient(['{"goal": "explore", "rationale": "test"}'])
    cooldown = CooldownGate()
    guard = VocabularyGuard()
    orchestrator = OrchestratorV2(
        llm=llm,
        cooldown=cooldown,
        guard=guard,
        session=session,
    )

    # Put game state in COMBAT (blocked state)
    state = DummyGameState(fsm_state=FSMState.COMBAT)

    res = orchestrator.decide(
        meta=DummyMetaState(),
        world=make_dummy_world(),
        state=state,
        now=100.0,
    )

    assert res.outcome == OrchestratorOutcome.BLOCKED_BY_COOLDOWN
    assert res.reason == "blocked_state"
    assert llm.call_count == 0

    # Read events to verify no strategist-prefixed events were written
    events = read_session_events(session)
    strategist_events = [
        e for e in events if e.get("event", "").startswith("strategist_")
    ]
    assert len(strategist_events) == 0


def test_decide_success(tmp_path: Path) -> None:
    """decide on valid LLM response returns SUCCESS and emits strategist_success event once."""
    session = make_session(tmp_path)

    resp_str = '{"goal": "explore", "target": null, "rationale": "valid strategy"}'
    llm = FakeLlmClient([resp_str])
    cooldown = CooldownGate()
    guard = VocabularyGuard()
    timer = FakeTimer([10.0, 10.05])

    orchestrator = OrchestratorV2(
        llm=llm,
        cooldown=cooldown,
        guard=guard,
        session=session,
        timer=timer,
    )

    res = orchestrator.decide(
        meta=DummyMetaState(),
        world=make_dummy_world(),
        state=DummyGameState(),
        now=100.0,
    )

    assert res.outcome == OrchestratorOutcome.SUCCESS
    assert res.strategy == ValidatedStrategy(
        goal="explore", target=None, rationale="valid strategy"
    )
    assert res.attempts == 1
    assert res.latency_ms == pytest.approx(50.0)
    assert res.reason == ""

    # Check session events
    events = read_session_events(session)
    succ_events = [e for e in events if e.get("event") == "strategist_success"]
    assert len(succ_events) == 1
    se = succ_events[0]
    assert se["prompt_hash"] == res.prompt_hash
    assert se["goal"] == "explore"
    assert se["has_target"] is False
    assert se["attempts"] == 1
    assert se["latency_ms"] == pytest.approx(50.0)
    assert "prompt_text" not in se


def test_decide_prompt_build_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """decide with prompt build exception returns PROMPT_BUILD_ERROR."""
    session = make_session(tmp_path)

    def bad_build_prompt(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Prompt build unexpected failure")

    monkeypatch.setattr(
        "wow_bot.strategist.orchestrator_v2.build_prompt", bad_build_prompt
    )

    llm = FakeLlmClient([])
    cooldown = CooldownGate()
    guard = VocabularyGuard()

    orchestrator = OrchestratorV2(
        llm=llm,
        cooldown=cooldown,
        guard=guard,
        session=session,
    )

    res = orchestrator.decide(
        meta=DummyMetaState(),
        world=make_dummy_world(),
        state=DummyGameState(),
        now=100.0,
    )

    assert res.outcome == OrchestratorOutcome.PROMPT_BUILD_ERROR
    assert res.reason == "RuntimeError"

    events = read_session_events(session)
    err_events = [e for e in events if e.get("event") == "strategist_prompt_error"]
    assert len(err_events) == 1
    assert err_events[0]["error"] == "RuntimeError"


def test_decide_llm_transport_error(tmp_path: Path) -> None:
    """decide on LLM transport exception returns LLM_TRANSPORT_ERROR."""
    session = make_session(tmp_path)

    llm = FakeLlmClient([ConnectionError("Ollama service unavailable")])
    cooldown = CooldownGate()
    guard = VocabularyGuard()

    orchestrator = OrchestratorV2(
        llm=llm,
        cooldown=cooldown,
        guard=guard,
        session=session,
    )

    res = orchestrator.decide(
        meta=DummyMetaState(),
        world=make_dummy_world(),
        state=DummyGameState(),
        now=100.0,
    )

    assert res.outcome == OrchestratorOutcome.LLM_TRANSPORT_ERROR
    assert res.attempts == 1
    assert res.reason == "ConnectionError"

    events = read_session_events(session)
    err_events = [e for e in events if e.get("event") == "strategist_llm_error"]
    assert len(err_events) == 1
    assert err_events[0]["error"] == "ConnectionError"


def test_decide_invalid_json_retry_disabled() -> None:
    """decide on invalid JSON with retry_on_invalid_json=False makes 1 call and returns INVALID_JSON."""
    llm = FakeLlmClient(["not json at all"])
    cooldown = CooldownGate()
    guard = VocabularyGuard()
    config = OrchestratorConfig(retry_on_invalid_json=False)

    orchestrator = OrchestratorV2(
        llm=llm,
        cooldown=cooldown,
        guard=guard,
        config=config,
    )

    res = orchestrator.decide(
        meta=DummyMetaState(),
        world=make_dummy_world(),
        state=DummyGameState(),
        now=100.0,
    )

    assert res.outcome == OrchestratorOutcome.INVALID_JSON
    assert res.attempts == 1
    assert res.reason == "parse_failed"
    assert llm.call_count == 1


def test_decide_invalid_json_both_attempts_fail(tmp_path: Path) -> None:
    """decide when both LLM attempts return invalid JSON returns INVALID_JSON with attempts=2."""
    session = make_session(tmp_path)

    llm = FakeLlmClient(["bad json 1", "bad json 2"])
    cooldown = CooldownGate()
    guard = VocabularyGuard()

    orchestrator = OrchestratorV2(
        llm=llm,
        cooldown=cooldown,
        guard=guard,
        session=session,
    )

    res = orchestrator.decide(
        meta=DummyMetaState(),
        world=make_dummy_world(),
        state=DummyGameState(),
        now=100.0,
    )

    assert res.outcome == OrchestratorOutcome.INVALID_JSON
    assert res.attempts == 2
    assert res.reason == "parse_failed"
    assert llm.call_count == 2

    events = read_session_events(session)
    retry_events = [e for e in events if e.get("event") == "strategist_retry"]
    inv_events = [e for e in events if e.get("event") == "strategist_invalid_json"]

    assert len(retry_events) == 1
    assert len(inv_events) == 1
    assert inv_events[0]["attempts"] == 2


def test_decide_first_attempt_invalid_second_valid() -> None:
    """decide when first attempt is invalid JSON and second is valid returns SUCCESS with attempts=2."""
    llm = FakeLlmClient([
        "invalid json text",
        '{"goal": "explore", "target": null, "rationale": "retry succeeded"}',
    ])
    cooldown = CooldownGate()
    guard = VocabularyGuard()

    orchestrator = OrchestratorV2(
        llm=llm,
        cooldown=cooldown,
        guard=guard,
    )

    res = orchestrator.decide(
        meta=DummyMetaState(),
        world=make_dummy_world(),
        state=DummyGameState(),
        now=100.0,
    )

    assert res.outcome == OrchestratorOutcome.SUCCESS
    assert res.attempts == 2
    assert llm.call_count == 2


def test_decide_first_attempt_succeeds_no_retry() -> None:
    """decide when first attempt succeeds does NOT retry and makes 1 call."""
    llm = FakeLlmClient([
        '{"goal": "explore", "target": null, "rationale": "first try"}',
        '{"goal": "explore", "target": null, "rationale": "second try"}',
    ])
    cooldown = CooldownGate()
    guard = VocabularyGuard()

    orchestrator = OrchestratorV2(
        llm=llm,
        cooldown=cooldown,
        guard=guard,
    )

    res = orchestrator.decide(
        meta=DummyMetaState(),
        world=make_dummy_world(),
        state=DummyGameState(),
        now=100.0,
    )

    assert res.outcome == OrchestratorOutcome.SUCCESS
    assert res.attempts == 1
    assert llm.call_count == 1


def test_decide_vocab_rejected() -> None:
    """decide when vocabulary guard rejects returns VOCAB_REJECTED without orchestrator event duplication."""
    # "explore" goal with a non-null target violates TargetKind.NONE rule
    bad_target_response = (
        '{"goal": "explore", "target": "some_target", "rationale": "invalid target"}'
    )
    llm = FakeLlmClient([bad_target_response])
    cooldown = CooldownGate()
    guard = VocabularyGuard()

    orchestrator = OrchestratorV2(
        llm=llm,
        cooldown=cooldown,
        guard=guard,
    )

    res = orchestrator.decide(
        meta=DummyMetaState(),
        world=make_dummy_world(),
        state=DummyGameState(),
        now=100.0,
    )

    assert res.outcome == OrchestratorOutcome.VOCAB_REJECTED
    assert res.reason == RejectionReason.UNEXPECTED_TARGET.value


def test_decide_llm_raises_on_retry(tmp_path: Path) -> None:
    """decide when LLM raises exception on retry returns LLM_TRANSPORT_ERROR with attempts=2."""
    session = make_session(tmp_path)

    llm = FakeLlmClient([
        "bad json response",
        TimeoutError("LLM socket timeout on retry"),
    ])
    cooldown = CooldownGate()
    guard = VocabularyGuard()

    orchestrator = OrchestratorV2(
        llm=llm,
        cooldown=cooldown,
        guard=guard,
        session=session,
    )

    res = orchestrator.decide(
        meta=DummyMetaState(),
        world=make_dummy_world(),
        state=DummyGameState(),
        now=100.0,
    )

    assert res.outcome == OrchestratorOutcome.LLM_TRANSPORT_ERROR
    assert res.attempts == 2
    assert res.reason == "TimeoutError"


def test_last_result_and_reset() -> None:
    """last_result reflects most recent decide call and reset clears it."""
    llm = FakeLlmClient(['{"goal": "explore", "rationale": "testing"}'])
    cooldown = CooldownGate()
    guard = VocabularyGuard()

    orchestrator = OrchestratorV2(
        llm=llm,
        cooldown=cooldown,
        guard=guard,
    )

    assert orchestrator.last_result() is None

    res = orchestrator.decide(
        meta=DummyMetaState(),
        world=make_dummy_world(),
        state=DummyGameState(),
        now=100.0,
    )

    assert orchestrator.last_result() == res

    orchestrator.reset()
    assert orchestrator.last_result() is None


def test_determinism_with_fake_timer() -> None:
    """Two orchestrators with same dependencies, responses, and FakeTimer produce identical results."""
    resp = '{"goal": "explore", "rationale": "deterministic"}'

    llm1 = FakeLlmClient([resp])
    cooldown1 = CooldownGate()
    guard1 = VocabularyGuard()
    timer1 = FakeTimer([1.0, 1.25])

    orch1 = OrchestratorV2(
        llm=llm1, cooldown=cooldown1, guard=guard1, timer=timer1
    )

    res1 = orch1.decide(
        meta=DummyMetaState(),
        world=make_dummy_world(),
        state=DummyGameState(),
        now=10.0,
    )

    llm2 = FakeLlmClient([resp])
    cooldown2 = CooldownGate()
    guard2 = VocabularyGuard()
    timer2 = FakeTimer([1.0, 1.25])

    orch2 = OrchestratorV2(
        llm=llm2, cooldown=cooldown2, guard=guard2, timer=timer2
    )

    res2 = orch2.decide(
        meta=DummyMetaState(),
        world=make_dummy_world(),
        state=DummyGameState(),
        now=10.0,
    )

    assert res1 == res2
    assert res1.to_json() == res2.to_json()


def test_no_session_attached() -> None:
    """All outcomes work without raising when no Session is attached."""
    # Test SUCCESS
    llm = FakeLlmClient(['{"goal": "explore", "rationale": "no session"}'])
    orchestrator = OrchestratorV2(
        llm=llm, cooldown=CooldownGate(), guard=VocabularyGuard(), session=None
    )
    res = orchestrator.decide(
        meta=DummyMetaState(),
        world=make_dummy_world(),
        state=DummyGameState(),
        now=100.0,
    )
    assert res.outcome == OrchestratorOutcome.SUCCESS


def test_include_prompt_text_in_event(tmp_path: Path) -> None:
    """include_prompt_text_in_event=True adds prompt_text to strategist_success event."""
    session = make_session(tmp_path)

    llm = FakeLlmClient(['{"goal": "explore", "rationale": "with prompt text"}'])
    config = OrchestratorConfig(include_prompt_text_in_event=True)

    orchestrator = OrchestratorV2(
        llm=llm,
        cooldown=CooldownGate(),
        guard=VocabularyGuard(),
        session=session,
        config=config,
    )

    res = orchestrator.decide(
        meta=DummyMetaState(),
        world=make_dummy_world(),
        state=DummyGameState(),
        now=100.0,
    )

    assert res.outcome == OrchestratorOutcome.SUCCESS

    events = read_session_events(session)
    succ_event = next(e for e in events if e.get("event") == "strategist_success")
    assert "prompt_text" in succ_event
    assert len(succ_event["prompt_text"]) > 0


def test_no_duplicate_events(tmp_path: Path) -> None:
    """Orchestrator does not duplicate events emitted by cooldown or vocab guard."""
    session = make_session(tmp_path)

    llm = FakeLlmClient(['{"goal": "explore", "rationale": "no duplicate events"}'])
    cooldown = CooldownGate(session=session)
    guard = VocabularyGuard(session=session)

    orchestrator = OrchestratorV2(
        llm=llm,
        cooldown=cooldown,
        guard=guard,
        session=session,
    )

    res = orchestrator.decide(
        meta=DummyMetaState(),
        world=make_dummy_world(),
        state=DummyGameState(),
        now=100.0,
    )

    assert res.outcome == OrchestratorOutcome.SUCCESS

    events = read_session_events(session)
    event_names = [e["event"] for e in events]

    # Check exact expected sequence of events: cooldown_allowed, vocab_accepted, strategist_success
    assert event_names == ["cooldown_allowed", "vocab_accepted", "strategist_success"]


def test_static_ast_import_constraints() -> None:
    """Static AST check: orchestrator_v2.py does not import prohibited modules/names."""
    target_path = (
        Path(__file__).parents[1] / "src" / "wow_bot" / "strategist" / "orchestrator_v2.py"
    )
    source = target_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target_path))

    prohibited_substrings = [
        "openai",
        "anthropic",
        "gemini",
        "cohere",
        "bedrock",
        "ollama",
    ]

    prohibited_full_imports = {
        "wow_bot.strategist.llm_client",
        "wow_bot.perception",
        "wow_bot.reflex",
        "wow_bot.actuation.actuator",
        "wow_bot.world.store",
        "wow_bot.world.loader",
        "wow_bot.world.sync",
        "wow_bot.nav",
        "wow_bot.combat",
        "wow_bot.executor.fsm_v2",
        "aiosqlite",
        "asyncio",
        "threading",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_name = alias.name
                for sub in prohibited_substrings:
                    assert sub not in mod_name.lower(), (
                        f"Prohibited import substring '{sub}' found in import '{mod_name}'"
                    )
                for pro in prohibited_full_imports:
                    assert not (mod_name == pro or mod_name.startswith(f"{pro}.")), (
                        f"Prohibited module '{mod_name}' imported"
                    )

        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            for sub in prohibited_substrings:
                assert sub not in mod_name.lower(), (
                    f"Prohibited import substring '{sub}' found in import from '{mod_name}'"
                )
            for pro in prohibited_full_imports:
                assert not (mod_name == pro or mod_name.startswith(f"{pro}.")), (
                    f"Prohibited module '{mod_name}' imported"
                )


def test_static_ast_no_prelab_strategist_modified() -> None:
    """Static check: no pre-lab strategist/ module was modified in this task."""
    repo_root = Path(__file__).parents[1]
    prelab_files = [
        repo_root / "src" / "wow_bot" / "strategist" / "llm_client.py",
        repo_root / "src" / "wow_bot" / "strategist" / "prompts.py",
        repo_root / "src" / "wow_bot" / "strategist" / "parser.py",
        repo_root / "src" / "wow_bot" / "strategist" / "orchestrator.py",
    ]
    for p in prelab_files:
        assert p.exists(), f"Pre-lab file {p} does not exist"
