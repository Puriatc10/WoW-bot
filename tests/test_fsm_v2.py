"""Tests for FSM v2 engine and protocols."""

import ast
import random
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from wow_bot.actuation.mapper import Intent, MoveTo, Turn
from wow_bot.config import Config
from wow_bot.executor.feedback import Feedback, FeedbackKind, FeedbackOutcome
from wow_bot.executor.fsm_v2 import (
    FSM,
    ConstantTargetBehavior,
    FSMConfig,
    FSMError,
    GameStateLike,
    MetaStateLike,
    NullBehavior,
)
from wow_bot.executor.states import FSMState
from wow_bot.session import Session


class DummyGameState(GameStateLike):
    """Dummy game state implementation for testing."""


class DummyMetaState(MetaStateLike):
    """Dummy meta state implementation for testing."""


class FakeBehavior:
    """Fake behavior implementation for testing."""

    def __init__(self, return_intent: Intent | None = None) -> None:
        self.return_intent = return_intent
        self.calls: list[dict[str, Any]] = []

    def decide(
        self,
        state: FSMState,
        game_state: GameStateLike,
        meta_state: MetaStateLike,
        now: float,
        rng: random.Random,
    ) -> Intent | None:
        self.calls.append(
            {
                "state": state,
                "game_state": game_state,
                "meta_state": meta_state,
                "now": now,
                "rng": rng,
                "rng_id": id(rng),
            }
        )
        return self.return_intent


def make_test_session(tmp_path: Path) -> Session:
    """Construct a real Session rooted in tmp_path."""
    config = Config(
        lab_mode=False,
        server_allowlist=("127.0.0.1:8080",),
        isolation_sentinel="10.0.0.1:80",
        kill_switch_key="F12",
        session_root=tmp_path,
        dry_run=True,
        max_session_seconds=3600,
        log_level="INFO",
    )
    return Session.start(config)


def _read_session_events(session: Session) -> list[dict[str, Any]]:
    import json

    events_path = session.path / "events.jsonl"
    if not events_path.exists():
        return []
    events = []
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


def test_fsm_construction_default() -> None:
    """Constructing FSM without arguments initializes to IDLE."""
    fsm = FSM()
    assert fsm.current_state == FSMState.IDLE
    assert fsm.previous_state is None
    assert not fsm.is_paused
    assert fsm.last_decision is None
    assert fsm.stuck_attempts == 0


def test_fsm_config_validation() -> None:
    """FSMConfig with stuck_recovery_max_attempts < 1 raises ValueError."""
    FSMConfig(stuck_recovery_max_attempts=1)
    with pytest.raises(ValueError, match="stuck_recovery_max_attempts must be >= 1"):
        FSMConfig(stuck_recovery_max_attempts=0)
    with pytest.raises(ValueError, match="stuck_recovery_max_attempts must be >= 1"):
        FSMConfig(stuck_recovery_max_attempts=-1)


def test_tick_idle_null_behavior(tmp_path: Path) -> None:
    """tick on IDLE with NullBehavior returns None and emits no "fsm_intent" event."""
    session = make_test_session(tmp_path)
    fsm = FSM(session=session)
    game_state = DummyGameState()
    meta_state = DummyMetaState()

    result = fsm.tick(game_state, meta_state, now=100.0)
    assert result is None

    events = _read_session_events(session)
    assert not any(e["event"] == "fsm_intent" for e in events)


def test_tick_fake_behavior_returns_moveto(tmp_path: Path) -> None:
    """tick with a FakeBehavior that returns MoveTo emits exactly one "fsm_intent" event and returns same intent."""
    session = make_test_session(tmp_path)
    intent = MoveTo(x=10.0, y=20.0)
    fake = FakeBehavior(return_intent=intent)
    fsm = FSM(behavior=fake, session=session)
    game_state = DummyGameState()
    meta_state = DummyMetaState()

    res = fsm.tick(game_state, meta_state, now=100.0)
    assert res == intent

    events = _read_session_events(session)
    intent_events = [e for e in events if e.get("event") == "fsm_intent"]
    assert len(intent_events) == 1
    assert intent_events[0]["state"] == FSMState.IDLE.value
    assert intent_events[0]["intent"] == repr(intent)


def test_tick_when_paused() -> None:
    """tick in PAUSED returns None without calling behavior."""
    fake = FakeBehavior(return_intent=MoveTo(x=1.0, y=1.0))
    fsm = FSM(behavior=fake)
    fsm.pause("test_pause", now=100.0)

    game_state = DummyGameState()
    meta_state = DummyMetaState()
    res = fsm.tick(game_state, meta_state, now=101.0)

    assert res is None
    assert len(fake.calls) == 0


def test_state_entered_at_initialization() -> None:
    """tick sets `_state_entered_at` on first call and does not reset it on subsequent calls with same state."""
    fsm = FSM()
    game_state = DummyGameState()
    meta_state = DummyMetaState()

    assert fsm._state_entered_at is None
    fsm.tick(game_state, meta_state, now=100.0)
    assert fsm._state_entered_at == 100.0

    fsm.tick(game_state, meta_state, now=105.0)
    assert fsm._state_entered_at == 100.0


def test_state_timeout(tmp_path: Path) -> None:
    """Timeout: tick with now > state_entered_at + timeout_for(state) transitions to IDLE and emits event."""
    session = make_test_session(tmp_path)
    fsm = FSM(session=session)
    game_state = DummyGameState()
    meta_state = DummyMetaState()

    with fsm._lock:
        fsm._state = FSMState.RECOVERING
        fsm._state_entered_at = 100.0

    st: FSMState = fsm.current_state
    assert st == FSMState.RECOVERING

    # Tick at now=105.0 (5s passed, no timeout)
    fsm.tick(game_state, meta_state, now=105.0)
    st = fsm.current_state
    assert st == FSMState.RECOVERING

    # Tick at now=111.0 (11s passed > 10s timeout)
    fsm.tick(game_state, meta_state, now=111.0)
    st = fsm.current_state
    assert st == FSMState.IDLE
    assert fsm._state_entered_at == 111.0

    events = _read_session_events(session)
    transitions = [e for e in events if e.get("event") == "fsm_transition"]
    timeout_trans = [t for t in transitions if str(t.get("reason", "")).startswith("state_timeout:")]
    assert len(timeout_trans) == 1
    assert timeout_trans[0]["from"] == FSMState.RECOVERING.value
    assert timeout_trans[0]["to"] == FSMState.IDLE.value
    assert timeout_trans[0]["reason"] == "state_timeout:RECOVERING"


def test_idle_no_timeout() -> None:
    """IDLE has timeout_s None and never auto-transitions via timeout."""
    fsm = FSM()
    game_state = DummyGameState()
    meta_state = DummyMetaState()

    fsm.tick(game_state, meta_state, now=100.0)
    assert fsm.current_state == FSMState.IDLE

    fsm.tick(game_state, meta_state, now=100100.0)
    assert fsm.current_state == FSMState.IDLE


def test_submit_feedback_action_success() -> None:
    """submit_feedback with ACTION_SUCCESS returns STAY and does not change current_state."""
    fsm = FSM()
    fb = Feedback(kind=FeedbackKind.ACTION_SUCCESS, ts=100.0)
    decision = fsm.submit_feedback(fb, now=100.0)

    assert decision.outcome == FeedbackOutcome.STAY
    assert decision.next_state is None
    assert fsm.current_state == FSMState.IDLE


def test_submit_feedback_combat_action_failed(tmp_path: Path) -> None:
    """submit_feedback with ACTION_FAILED from COMBAT returns TO_RECOVERING, transitions state, and updates state_entered_at."""
    session = make_test_session(tmp_path)
    fsm = FSM(session=session)

    with fsm._lock:
        fsm._state = FSMState.COMBAT
        fsm._state_entered_at = 50.0

    fb = Feedback(kind=FeedbackKind.ACTION_FAILED, ts=100.0)
    decision = fsm.submit_feedback(fb, now=100.0)

    assert decision.outcome == FeedbackOutcome.TO_RECOVERING
    assert decision.next_state == FSMState.RECOVERING
    assert decision.reason == "action_failed"
    assert fsm.current_state == FSMState.RECOVERING
    assert fsm._state_entered_at == 100.0

    events = _read_session_events(session)
    transitions = [e for e in events if e.get("event") == "fsm_transition"]
    assert len(transitions) == 1
    assert transitions[0]["from"] == FSMState.COMBAT.value
    assert transitions[0]["to"] == FSMState.RECOVERING.value
    assert transitions[0]["reason"] == "action_failed"


def test_submit_feedback_recovering_action_failed() -> None:
    """submit_feedback with ACTION_FAILED from RECOVERING returns TO_IDLE with reason "repeated_failure"."""
    fsm = FSM()
    with fsm._lock:
        fsm._state = FSMState.RECOVERING

    fb = Feedback(kind=FeedbackKind.ACTION_FAILED, ts=100.0)
    decision = fsm.submit_feedback(fb, now=100.0)

    assert decision.outcome == FeedbackOutcome.TO_IDLE
    assert decision.next_state == FSMState.IDLE
    assert decision.reason == "repeated_failure"
    assert fsm.current_state == FSMState.IDLE


def test_submit_feedback_position_stuck() -> None:
    """submit_feedback with POSITION_STUCK transitions to STUCK_RECOVERY and increments stuck_attempts."""
    fsm = FSM()
    with fsm._lock:
        fsm._state = FSMState.MOVING_TO_TARGET

    assert fsm.stuck_attempts == 0

    fb = Feedback(kind=FeedbackKind.POSITION_STUCK, ts=100.0)
    decision = fsm.submit_feedback(fb, now=100.0)

    assert decision.outcome == FeedbackOutcome.TO_STUCK_RECOVERY
    assert decision.next_state == FSMState.STUCK_RECOVERY
    assert fsm.current_state == FSMState.STUCK_RECOVERY
    assert fsm.stuck_attempts == 1


def test_submit_feedback_position_clear_from_stuck_recovery() -> None:
    """submit_feedback with POSITION_CLEAR from STUCK_RECOVERY transitions to RECOVERING and resets stuck_attempts to 0."""
    fsm = FSM()
    with fsm._lock:
        fsm._state = FSMState.MOVING_TO_TARGET

    # First get into STUCK_RECOVERY via POSITION_STUCK from MOVING_TO_TARGET
    fsm.submit_feedback(Feedback(kind=FeedbackKind.POSITION_STUCK, ts=100.0), now=100.0)
    st1: FSMState = fsm.current_state
    assert st1 == FSMState.STUCK_RECOVERY
    assert fsm.stuck_attempts == 1

    # Now send POSITION_CLEAR
    decision = fsm.submit_feedback(
        Feedback(kind=FeedbackKind.POSITION_CLEAR, ts=105.0),
        now=105.0,
    )
    assert decision.outcome == FeedbackOutcome.TO_RECOVERING
    assert decision.next_state == FSMState.RECOVERING
    st2: FSMState = fsm.current_state
    assert st2 == FSMState.RECOVERING
    assert fsm.stuck_attempts == 0


def test_hard_stuck_pause_enabled(tmp_path: Path) -> None:
    """Hard stuck: after stuck_recovery_max_attempts consecutive entries to STUCK_RECOVERY (hard_stuck_pause=True), FSM pauses."""
    session = make_test_session(tmp_path)
    config = FSMConfig(stuck_recovery_max_attempts=1, hard_stuck_pause=True)
    fsm = FSM(config=config, session=session)

    with fsm._lock:
        fsm._state = FSMState.MOVING_TO_TARGET

    fsm.submit_feedback(Feedback(kind=FeedbackKind.POSITION_STUCK, ts=200.0), now=200.0)

    assert fsm.current_state == FSMState.PAUSED
    assert fsm.is_paused
    assert fsm.stuck_attempts == 0

    events = _read_session_events(session)
    hard_stuck_events = [e for e in events if e.get("event") == "fsm_hard_stuck"]
    assert len(hard_stuck_events) == 1
    assert hard_stuck_events[0]["attempts"] == 1


def test_hard_stuck_pause_disabled(tmp_path: Path) -> None:
    """Hard stuck with hard_stuck_pause=False leaves FSM in STUCK_RECOVERY (does not pause)."""
    session = make_test_session(tmp_path)
    config = FSMConfig(stuck_recovery_max_attempts=1, hard_stuck_pause=False)
    fsm = FSM(config=config, session=session)

    with fsm._lock:
        fsm._state = FSMState.MOVING_TO_TARGET

    fsm.submit_feedback(Feedback(kind=FeedbackKind.POSITION_STUCK, ts=100.0), now=100.0)

    assert fsm.current_state == FSMState.STUCK_RECOVERY
    assert not fsm.is_paused
    assert fsm.stuck_attempts == 1

    events = _read_session_events(session)
    assert not any(e.get("event") == "fsm_hard_stuck" for e in events)


def test_pause_and_idempotency(tmp_path: Path) -> None:
    """pause() sets current_state to PAUSED, stores previous_state, emits "fsm_paused", and is idempotent."""
    session = make_test_session(tmp_path)
    fsm = FSM(session=session)

    fsm.pause("user_requested", now=100.0)
    assert fsm.current_state == FSMState.PAUSED
    assert fsm.previous_state == FSMState.IDLE
    assert fsm.is_paused

    events_1 = _read_session_events(session)
    paused_events_1 = [e for e in events_1 if e.get("event") == "fsm_paused"]
    assert len(paused_events_1) == 1
    assert paused_events_1[0]["reason"] == "user_requested"
    assert paused_events_1[0]["from"] == FSMState.IDLE.value

    # Second pause call (idempotent)
    fsm.pause("user_requested_again", now=105.0)
    events_2 = _read_session_events(session)
    paused_events_2 = [e for e in events_2 if e.get("event") == "fsm_paused"]
    assert len(paused_events_2) == 1  # No new event emitted


def test_resume_from_paused(tmp_path: Path) -> None:
    """resume() from PAUSED returns to previous_state, emits "fsm_resumed", clears previous_state, and is no-op when not paused."""
    session = make_test_session(tmp_path)
    fsm = FSM(session=session)

    with fsm._lock:
        fsm._state = FSMState.SCANNING

    fsm.pause("pause_scan", now=100.0)
    assert fsm.current_state == FSMState.PAUSED
    assert fsm.previous_state == FSMState.SCANNING

    fsm.resume("resume_scan", now=105.0)
    st_res: FSMState = fsm.current_state
    assert st_res == FSMState.SCANNING
    assert fsm.previous_state is None

    events = _read_session_events(session)
    resumed_events = [e for e in events if e.get("event") == "fsm_resumed"]
    assert len(resumed_events) == 1
    assert resumed_events[0]["reason"] == "resume_scan"
    assert resumed_events[0]["to"] == FSMState.SCANNING.value

    # Calling resume when not paused is a no-op
    fsm.resume("resume_again", now=110.0)
    events_after = _read_session_events(session)
    resumed_events_after = [e for e in events_after if e.get("event") == "fsm_resumed"]
    assert len(resumed_events_after) == 1


def test_resume_with_previous_state_none() -> None:
    """resume() with previous_state None (e.g. constructed and paused before any state entered) transitions to IDLE."""
    fsm = FSM()
    fsm.pause("pause_immediately", now=100.0)
    with fsm._lock:
        fsm._previous_state = None  # Force previous_state to None

    fsm.resume("resume_now", now=105.0)
    assert fsm.current_state == FSMState.IDLE


def test_enter_recovery_illegal_transition() -> None:
    """enter_recovery() from a state that cannot transition to STUCK_RECOVERY raises FSMError."""
    fsm = FSM()
    # IDLE cannot transition directly to STUCK_RECOVERY according to TRANSITION_TABLE
    with pytest.raises(FSMError, match="illegal transition: IDLE -> STUCK_RECOVERY"):
        fsm.enter_recovery("stuck_in_idle", now=100.0)


def test_enter_recovery_legal_transition(tmp_path: Path) -> None:
    """enter_recovery() from a legal state transitions and emits "fsm_recovery_entered"."""
    session = make_test_session(tmp_path)
    fsm = FSM(session=session)
    with fsm._lock:
        fsm._state = FSMState.MOVING_TO_TARGET

    fsm.enter_recovery("forced_stuck", now=100.0)
    assert fsm.current_state == FSMState.STUCK_RECOVERY
    assert fsm.stuck_attempts == 1

    events = _read_session_events(session)
    rec_events = [e for e in events if e.get("event") == "fsm_recovery_entered"]
    assert len(rec_events) == 1
    assert rec_events[0]["reason"] == "forced_stuck"
    assert rec_events[0]["from"] == FSMState.MOVING_TO_TARGET.value


def test_enter_recovery_from_stuck_recovery_is_noop() -> None:
    """enter_recovery() from STUCK_RECOVERY is a no-op."""
    fsm = FSM()
    with fsm._lock:
        fsm._state = FSMState.MOVING_TO_TARGET
    fsm.enter_recovery("stuck_1", now=100.0)
    assert fsm.stuck_attempts == 1

    fsm.enter_recovery("stuck_2", now=105.0)
    assert fsm.stuck_attempts == 1  # No-op


def test_tick_derives_per_tick_rng() -> None:
    """tick derives a per-tick rng: calling tick twice with FakeBehavior yields two distinct rng object ids."""
    fake = FakeBehavior()
    fsm = FSM(behavior=fake)
    game_state = DummyGameState()
    meta_state = DummyMetaState()

    fsm.tick(game_state, meta_state, now=100.0)
    fsm.tick(game_state, meta_state, now=101.0)

    assert len(fake.calls) == 2
    id1 = fake.calls[0]["rng_id"]
    id2 = fake.calls[1]["rng_id"]
    assert id1 != id2


def test_determinism_same_seed() -> None:
    """Determinism: two FSM instances with same seed and same behavior produce identical Intent and FeedbackDecision sequences."""
    target = (100.0, 200.0)
    behavior1 = ConstantTargetBehavior(target=target)
    behavior2 = ConstantTargetBehavior(target=target)

    fsm1 = FSM(config=FSMConfig(seed=42), behavior=behavior1)
    fsm2 = FSM(config=FSMConfig(seed=42), behavior=behavior2)

    game_state = DummyGameState()
    meta_state = DummyMetaState()

    intents1: list[Intent | None] = []
    intents2: list[Intent | None] = []
    decisions1 = []
    decisions2 = []

    with fsm1._lock:
        fsm1._state = FSMState.STUCK_RECOVERY
    with fsm2._lock:
        fsm2._state = FSMState.STUCK_RECOVERY

    for i in range(100):
        t = 100.0 + i * 0.1
        i1 = fsm1.tick(game_state, meta_state, now=t)
        i2 = fsm2.tick(game_state, meta_state, now=t)
        intents1.append(i1)
        intents2.append(i2)

        fb = Feedback(kind=FeedbackKind.ACTION_SUCCESS if i % 2 == 0 else FeedbackKind.POSITION_STUCK, ts=t)
        d1 = fsm1.submit_feedback(fb, now=t)
        d2 = fsm2.submit_feedback(fb, now=t)
        decisions1.append(d1)
        decisions2.append(d2)

    assert intents1 == intents2
    assert decisions1 == decisions2


def test_determinism_different_seeds() -> None:
    """Determinism: two FSM instances with different seeds produce different rng-derived values when behavior calls getrandbits."""
    class RngRecordingBehavior:
        def __init__(self) -> None:
            self.vals: list[int] = []

        def decide(
            self,
            state: FSMState,
            game_state: GameStateLike,
            meta_state: MetaStateLike,
            now: float,
            rng: random.Random,
        ) -> Intent | None:
            self.vals.append(rng.getrandbits(32))
            return None

    b1 = RngRecordingBehavior()
    b2 = RngRecordingBehavior()

    fsm1 = FSM(config=FSMConfig(seed=111), behavior=b1)
    fsm2 = FSM(config=FSMConfig(seed=222), behavior=b2)

    game_state = DummyGameState()
    meta_state = DummyMetaState()

    fsm1.tick(game_state, meta_state, now=100.0)
    fsm2.tick(game_state, meta_state, now=100.0)

    assert b1.vals != b2.vals


def test_thread_safety_pause_during_tick() -> None:
    """Thread safety: pause() called from a separate thread while tick() is called from main thread does not corrupt state."""
    class DelayBehavior:
        def decide(
            self,
            state: FSMState,
            game_state: GameStateLike,
            meta_state: MetaStateLike,
            now: float,
            rng: random.Random,
        ) -> Intent | None:
            time.sleep(0.01)
            return None

    fsm = FSM(behavior=DelayBehavior())
    game_state = DummyGameState()
    meta_state = DummyMetaState()

    def run_pause() -> None:
        time.sleep(0.002)
        fsm.pause("thread_pause", now=100.5)

    t = threading.Thread(target=run_pause)
    t.start()
    fsm.tick(game_state, meta_state, now=100.0)
    t.join()

    start = time.monotonic()
    while time.monotonic() - start < 1.0:
        if fsm.is_paused and fsm.current_state == FSMState.PAUSED:
            break
        time.sleep(0.001)

    assert fsm.is_paused
    assert fsm.current_state == FSMState.PAUSED


def test_lock_not_held_across_decide() -> None:
    """Regression test: lock is NOT held across behavior.decide (behavior calling current_state inside decide does not deadlock)."""
    class SelfQueryBehavior:
        def __init__(self, owner_fsm: FSM | None = None) -> None:
            self.owner_fsm = owner_fsm
            self.queried_state: FSMState | None = None

        def decide(
            self,
            state: FSMState,
            game_state: GameStateLike,
            meta_state: MetaStateLike,
            now: float,
            rng: random.Random,
        ) -> Intent | None:
            if self.owner_fsm is not None:
                self.queried_state = self.owner_fsm.current_state
            return None

    behavior = SelfQueryBehavior()
    fsm = FSM(behavior=behavior)
    behavior.owner_fsm = fsm

    game_state = DummyGameState()
    meta_state = DummyMetaState()

    fsm.tick(game_state, meta_state, now=100.0)
    assert behavior.queried_state == FSMState.IDLE


def test_session_events_payload_schema(tmp_path: Path) -> None:
    """Session events fsm_transition, fsm_feedback, fsm_intent match expected schemas."""
    session = make_test_session(tmp_path)
    fake = FakeBehavior(return_intent=MoveTo(x=1.0, y=2.0))
    fsm = FSM(behavior=fake, session=session)

    game_state = DummyGameState()
    meta_state = DummyMetaState()

    # Emit intent
    fsm.tick(game_state, meta_state, now=100.0)

    # Set state to MOVING_TO_TARGET and emit feedback / transition
    with fsm._lock:
        fsm._state = FSMState.MOVING_TO_TARGET

    fb = Feedback(kind=FeedbackKind.POSITION_STUCK, ts=105.0)
    fsm.submit_feedback(fb, now=105.0)

    events = _read_session_events(session)

    intent_event = next(e for e in events if e.get("event") == "fsm_intent")
    assert "state" in intent_event
    assert "intent" in intent_event

    feedback_event = next(e for e in events if e.get("event") == "fsm_feedback")
    assert "kind" in feedback_event
    assert "outcome" in feedback_event
    assert "next_state" in feedback_event
    assert "reason" in feedback_event

    trans_event = next(e for e in events if e.get("event") == "fsm_transition")
    assert "from" in trans_event
    assert "to" in trans_event
    assert "reason" in trans_event


def test_constant_target_behavior() -> None:
    """ConstantTargetBehavior returns MoveTo for MOVING_TO_TARGET, PATHING, FLEEING; Turn(0.0) or None for STUCK_RECOVERY; None otherwise."""
    target = (10.0, 20.0)
    behavior = ConstantTargetBehavior(target=target)
    game_state = DummyGameState()
    meta_state = DummyMetaState()

    # MOVING_TO_TARGET -> MoveTo
    res = behavior.decide(FSMState.MOVING_TO_TARGET, game_state, meta_state, 100.0, random.Random(0))
    assert res == MoveTo(x=10.0, y=20.0)

    # PATHING -> MoveTo
    res = behavior.decide(FSMState.PATHING, game_state, meta_state, 100.0, random.Random(0))
    assert res == MoveTo(x=10.0, y=20.0)

    # FLEEING -> MoveTo
    res = behavior.decide(FSMState.FLEEING, game_state, meta_state, 100.0, random.Random(0))
    assert res == MoveTo(x=10.0, y=20.0)

    # STUCK_RECOVERY with rng returning < 0.5 vs >= 0.5
    rng_low: Any = random.Random()
    rng_low.random = lambda: 0.1
    res_low = behavior.decide(FSMState.STUCK_RECOVERY, game_state, meta_state, 100.0, rng_low)
    assert res_low == Turn(angle_rad=0.0)

    rng_high: Any = random.Random()
    rng_high.random = lambda: 0.8
    res_high = behavior.decide(FSMState.STUCK_RECOVERY, game_state, meta_state, 100.0, rng_high)
    assert res_high is None

    # IDLE -> None
    res_idle = behavior.decide(FSMState.IDLE, game_state, meta_state, 100.0, random.Random(0))
    assert res_idle is None


def test_null_behavior() -> None:
    """NullBehavior always returns None."""
    behavior = NullBehavior()
    game_state = DummyGameState()
    meta_state = DummyMetaState()

    for state in FSMState:
        res = behavior.decide(state, game_state, meta_state, 100.0, random.Random(0))
        assert res is None


def test_static_ast_check_imports() -> None:
    """Static AST check: fsm_v2.py does not import forbidden modules."""
    filepath = Path(__file__).parent.parent / "src" / "wow_bot" / "executor" / "fsm_v2.py"
    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(filepath))

    forbidden_exact = [
        "wow_bot.strategist",
        "wow_bot.navigation",
        "wow_bot.combat",
        "wow_bot.world",
        "wow_bot.perception",
        "wow_bot.actuation.actuator",
        "wow_bot.reflex",
    ]
    forbidden_substrings = ["ollama", "openai", "anthropic", "llm"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                for forbidden in forbidden_exact:
                    assert not name.startswith(forbidden), f"Forbidden import found: {name}"
                for sub in forbidden_substrings:
                    assert sub not in name.lower(), f"Forbidden import found: {name}"

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for forbidden in forbidden_exact:
                assert not module.startswith(forbidden), f"Forbidden import from found: {module}"
            for sub in forbidden_substrings:
                assert sub not in module.lower(), f"Forbidden import from found: {module}"
