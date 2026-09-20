"""Unit tests for HumanizedActuator and HumanizerConfig."""

import ast
import json
import random
import time
from pathlib import Path
from typing import Any

import pytest

from wow_bot.actuation.actuator import NullActuator, RealActuator
from wow_bot.actuation.drivers.null import NullDriver
from wow_bot.actuation.focus import FocusManager
from wow_bot.actuation.humanized import (
    HumanizedActuator,
    HumanizedActuatorError,
    HumanizerConfig,
    make_humanized_actuator,
)
from wow_bot.actuation.mapper import (
    ActionMapper,
    ActionResult,
    ActionStatus,
    Intent,
    MoveTo,
    NullDelay,
    Turn,
)
from wow_bot.config import Config
from wow_bot.humanize.cursor import TrajectoryPoint
from wow_bot.humanize.imperfections import ImperfectionConfig
from wow_bot.humanize.intervals import IntervalConfig
from wow_bot.safety import SafetyLayer
from wow_bot.session import Session


class FakeWrappedActuator:
    """Fake Actuator implementation for testing HumanizedActuator delegation."""

    def __init__(self, result: ActionResult | None = None) -> None:
        self._aborted = False
        self.execute_calls: list[tuple[Intent, tuple[float, float], float]] = []
        self.abort_calls: list[str] = []
        self.close_calls: int = 0
        self.custom_result = (
            result
            if result is not None
            else ActionResult(
                status=ActionStatus.SUCCESS,
                latency_ms=12.5,
                notes="fake_result",
            )
        )
        self.should_raise: Exception | None = None

    def execute(
        self,
        intent: Intent,
        *,
        position: tuple[float, float],
    ) -> ActionResult:
        if self.should_raise is not None:
            raise self.should_raise
        self.execute_calls.append((intent, position, time.perf_counter()))
        return self.custom_result

    def abort(self, reason: str) -> None:
        self.abort_calls.append(reason)
        self._aborted = True

    def is_aborted(self) -> bool:
        return self._aborted

    def close(self) -> None:
        self.close_calls += 1


class FakeFocus(FocusManager):
    """Fake focus manager for testing RealActuator instantiation."""

    def __init__(self) -> None:
        pass

    def assert_focused(self) -> None:
        pass


def _make_test_config(session_root: Path) -> Config:
    """Helper to create a valid Config for tests."""
    return Config(
        lab_mode=False,
        server_allowlist=("127.0.0.1:8080",),
        isolation_sentinel="10.0.0.1:80",
        kill_switch_key="F12",
        session_root=session_root,
        dry_run=True,
        max_session_seconds=3600,
        log_level="INFO",
    )


def create_test_session(tmp_path: Path) -> Session:
    """Helper to create a test Session in tmp_path."""
    cfg = _make_test_config(tmp_path / "sessions")
    return Session.start(cfg)


def read_session_events(session_path: Path) -> list[dict[str, Any]]:
    """Helper to read events from session events.jsonl."""
    events_file = session_path / "events.jsonl"
    if not events_file.exists():
        return []
    events = []
    with open(events_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


# 1. HumanizerConfig validation: min_delay_s < 0 raises ValueError
def test_config_negative_min_delay_raises_value_error() -> None:
    with pytest.raises(ValueError, match="min_delay_s"):
        HumanizerConfig(min_delay_s=-0.1)


# 2. HumanizerConfig validation: cursor_enabled=True raises ValueError
def test_config_cursor_enabled_true_raises_value_error() -> None:
    with pytest.raises(ValueError, match="cursor_enabled=True is not available"):
        HumanizerConfig(cursor_enabled=True)


# 3. HumanizerConfig defaults
def test_config_defaults() -> None:
    cfg = HumanizerConfig()
    assert cfg.enabled is True
    assert cfg.cursor_enabled is False
    assert cfg.log_to_session is True
    assert cfg.min_delay_s == 0.0


# 4 & 5. HumanizedActuator with enabled=False
def test_humanized_actuator_disabled_delegates_without_delay_sampling_events(tmp_path: Path) -> None:
    fake = FakeWrappedActuator()
    delay = NullDelay()
    session = create_test_session(tmp_path)
    cfg = HumanizerConfig(enabled=False)
    actuator = HumanizedActuator(fake, config=cfg, delay=delay, session=session)

    intent = MoveTo(x=10.0, y=20.0)
    result = actuator.execute(intent, position=(0.0, 0.0))

    assert result == fake.custom_result
    assert len(fake.execute_calls) == 1
    assert len(delay.waits()) == 0
    assert actuator.imperfection_log_size == 0

    events = read_session_events(session.path)
    humanizer_action_events = [e for e in events if e.get("event") == "humanizer_action"]
    assert len(humanizer_action_events) == 0


# 6 & 7. With enabled=True, execute calls delay.wait at least once (interval)
def test_execute_calls_delay_wait_within_clip_range_and_min_delay() -> None:
    fake = FakeWrappedActuator()
    delay = NullDelay()
    interval_cfg = IntervalConfig(clip_low=0.10, clip_high=0.50)
    cfg = HumanizerConfig(
        interval_config=interval_cfg,
        imperfection_config=ImperfectionConfig(pause_probability=0.0),
        min_delay_s=0.15,
    )
    actuator = HumanizedActuator(fake, config=cfg, delay=delay, rng=random.Random(42))

    actuator.execute(MoveTo(x=1.0, y=1.0), position=(0.0, 0.0))

    waits = delay.waits()
    assert len(waits) == 1
    assert waits[0] >= 0.15
    assert waits[0] <= 0.50


# 8, 9, 10. Delay ordering: interval, then optional pause, then delegation
def test_delay_ordering_and_pause_occurrence() -> None:
    fake = FakeWrappedActuator()
    delay = NullDelay()
    # Force pause with probability 1.0
    cfg_pause = HumanizerConfig(
        imperfection_config=ImperfectionConfig(
            pause_probability=1.0,
            pause_duration_clip_low=0.2,
            pause_duration_clip_high=0.5,
        ),
    )
    actuator_pause = HumanizedActuator(fake, config=cfg_pause, delay=delay, rng=random.Random(123))
    actuator_pause.execute(Turn(angle_rad=0.5), position=(0.0, 0.0))

    waits = delay.waits()
    assert len(waits) == 2  # interval, then pause
    assert len(fake.execute_calls) == 1

    # Force no pause with probability 0.0
    delay_no_pause = NullDelay()
    fake_no_pause = FakeWrappedActuator()
    cfg_no_pause = HumanizerConfig(
        imperfection_config=ImperfectionConfig(pause_probability=0.0),
    )
    actuator_no_pause = HumanizedActuator(fake_no_pause, config=cfg_no_pause, delay=delay_no_pause, rng=random.Random(123))
    actuator_no_pause.execute(Turn(angle_rad=0.5), position=(0.0, 0.0))

    waits_no_pause = delay_no_pause.waits()
    assert len(waits_no_pause) == 1  # only interval


# 11. execute returns same ActionResult as wrapped actuator
def test_execute_returns_same_action_result() -> None:
    expected_result = ActionResult(status=ActionStatus.TIMEOUT, latency_ms=45.6, notes="custom_timeout")
    fake = FakeWrappedActuator(result=expected_result)
    delay = NullDelay()
    actuator = HumanizedActuator(fake, delay=delay)

    res = actuator.execute(MoveTo(x=5.0, y=5.0), position=(0.0, 0.0))
    assert res.status == ActionStatus.TIMEOUT
    assert res.latency_ms == 45.6
    assert res.notes == "custom_timeout"


# 12, 13, 14, 15, 16. Session event logging for humanizer_action
def test_session_events_humanizer_action(tmp_path: Path) -> None:
    fake = FakeWrappedActuator()
    delay = NullDelay()
    session = create_test_session(tmp_path)

    cfg = HumanizerConfig(
        imperfection_config=ImperfectionConfig(
            pause_probability=1.0,
            pause_duration_clip_low=0.3,
            pause_duration_clip_high=0.4,
        ),
        log_to_session=True,
    )
    actuator = HumanizedActuator(fake, config=cfg, delay=delay, session=session, rng=random.Random(10))
    actuator.execute(MoveTo(x=2.0, y=3.0), position=(1.0, 1.0))

    events = read_session_events(session.path)
    action_events = [e for e in events if e.get("event") == "humanizer_action"]
    assert len(action_events) == 1
    ev = action_events[0]

    assert "MoveTo" in ev["intent"]
    assert isinstance(ev["interval_s"], float)
    assert ev["pause_occurred"] is True
    assert ev["pause_duration_s"] > 0.0
    assert ev["status"] == "success"
    assert ev["latency_ms"] == 12.5

    # log_to_session=False emits no event
    session2 = create_test_session(tmp_path)
    cfg_no_log = HumanizerConfig(log_to_session=False)
    actuator_no_log = HumanizedActuator(fake, config=cfg_no_log, delay=delay, session=session2)
    actuator_no_log.execute(MoveTo(x=2.0, y=3.0), position=(1.0, 1.0))
    events2 = read_session_events(session2.path)
    assert len([e for e in events2 if e.get("event") == "humanizer_action"]) == 0

    # session is None emits no event
    actuator_no_session = HumanizedActuator(fake, config=cfg, delay=delay, session=None)
    actuator_no_session.execute(MoveTo(x=2.0, y=3.0), position=(1.0, 1.0))


# 17. Exceptions from wrapped actuator propagate unchanged
def test_exceptions_from_wrapped_actuator_propagate_unchanged() -> None:
    fake = FakeWrappedActuator()
    fake.should_raise = RuntimeError("wrapped error")
    delay = NullDelay()
    actuator = HumanizedActuator(fake, delay=delay)

    with pytest.raises(RuntimeError, match="wrapped error"):
        actuator.execute(MoveTo(x=1.0, y=1.0), position=(0.0, 0.0))


# 18 & 19. _apply_cursor_trajectory_if_applicable returns None for MoveTo and Turn
def test_apply_cursor_trajectory_returns_none() -> None:
    fake = FakeWrappedActuator()
    actuator = HumanizedActuator(fake, delay=NullDelay())
    rng = random.Random(0)

    assert actuator._apply_cursor_trajectory_if_applicable(MoveTo(x=1.0, y=1.0), rng) is None
    assert actuator._apply_cursor_trajectory_if_applicable(Turn(angle_rad=1.0), rng) is None


# 20. Guard in step 5 raises HumanizedActuatorError if non-None trajectory returned
def test_cursor_trajectory_non_none_raises_humanized_actuator_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeWrappedActuator()
    actuator = HumanizedActuator(fake, delay=NullDelay())

    dummy_point = TrajectoryPoint(x=1.0, y=2.0, t_offset=0.1)
    monkeypatch.setattr(
        actuator,
        "_apply_cursor_trajectory_if_applicable",
        lambda intent, rng: [dummy_point],
    )

    with pytest.raises(HumanizedActuatorError, match="Cursor trajectory returned non-None value"):
        actuator.execute(MoveTo(x=1.0, y=1.0), position=(0.0, 0.0))


# 21 & 22. Determinism across seeds
def test_determinism_same_and_different_seeds() -> None:
    fake1 = FakeWrappedActuator()
    delay1 = NullDelay()
    act1 = HumanizedActuator(fake1, delay=delay1, rng=random.Random(999))

    fake2 = FakeWrappedActuator()
    delay2 = NullDelay()
    act2 = HumanizedActuator(fake2, delay=delay2, rng=random.Random(999))

    fake3 = FakeWrappedActuator()
    delay3 = NullDelay()
    act3 = HumanizedActuator(fake3, delay=delay3, rng=random.Random(888))

    for _ in range(100):
        act1.execute(MoveTo(x=1.0, y=1.0), position=(0.0, 0.0))
        act2.execute(MoveTo(x=1.0, y=1.0), position=(0.0, 0.0))
        act3.execute(MoveTo(x=1.0, y=1.0), position=(0.0, 0.0))

    assert delay1.waits() == delay2.waits()
    assert delay1.waits() != delay3.waits()


# 23, 24, 25, 26. abort idempotency, event emission, timing, and no delay wait
def test_abort_semantics(tmp_path: Path) -> None:
    fake = FakeWrappedActuator()
    delay = NullDelay()
    session = create_test_session(tmp_path)
    actuator = HumanizedActuator(fake, delay=delay, session=session)

    start = time.perf_counter()
    actuator.abort("test_reason")
    duration_ms = (time.perf_counter() - start) * 1000.0

    assert duration_ms < 120.0
    assert len(fake.abort_calls) == 1
    assert fake.abort_calls[0] == "test_reason"
    assert len(delay.waits()) == 0

    # Second abort call is a no-op
    actuator.abort("second_reason")
    assert len(fake.abort_calls) == 1

    events = read_session_events(session.path)
    abort_events = [e for e in events if e.get("event") == "humanizer_abort"]
    assert len(abort_events) == 1
    assert abort_events[0]["reason"] == "test_reason"


# 27 & 28. is_aborted returns True if wrapper or wrapped is aborted
def test_is_aborted_behavior() -> None:
    fake = FakeWrappedActuator()
    actuator = HumanizedActuator(fake, delay=NullDelay())

    assert actuator.is_aborted() is False

    # Wrapped is aborted externally
    fake.abort("external")
    assert actuator.is_aborted() is True

    # Reset fake, then abort wrapper
    fake2 = FakeWrappedActuator()
    actuator2 = HumanizedActuator(fake2, delay=NullDelay())
    actuator2.abort("wrapper")
    assert actuator2.is_aborted() is True


# 29 & 30. close is idempotent, calls abort("close"), and does not close wrapped actuator
def test_close_semantics() -> None:
    fake = FakeWrappedActuator()
    actuator = HumanizedActuator(fake, delay=NullDelay())

    actuator.close()
    assert len(fake.abort_calls) == 1
    assert fake.abort_calls[0] == "close"
    assert fake.close_calls == 0

    # Idempotent second call
    actuator.close()
    assert len(fake.abort_calls) == 1


# 31 & 32. Imperfection log entries and size
def test_imperfection_log_tracking() -> None:
    fake = FakeWrappedActuator()
    delay = NullDelay()
    # High pause probability (1.0)
    cfg_pause = HumanizerConfig(
        imperfection_config=ImperfectionConfig(pause_probability=1.0)
    )
    actuator = HumanizedActuator(fake, config=cfg_pause, delay=delay)

    actuator.execute(MoveTo(x=1.0, y=1.0), position=(0.0, 0.0))
    assert actuator.imperfection_log_size == 1

    entries = actuator.last_imperfection_entries()
    assert len(entries) == 1
    assert entries[0]["kind"] == "pause"
    assert "duration_s" in entries[0]

    # Non-pause action does not increment log
    cfg_no_pause = HumanizerConfig(
        imperfection_config=ImperfectionConfig(pause_probability=0.0)
    )
    actuator_no_pause = HumanizedActuator(fake, config=cfg_no_pause, delay=delay)
    actuator_no_pause.execute(MoveTo(x=1.0, y=1.0), position=(0.0, 0.0))
    assert actuator_no_pause.imperfection_log_size == 0


# 33 & 34. describe_humanizer output
def test_describe_humanizer_output() -> None:
    fake = FakeWrappedActuator()
    actuator = HumanizedActuator(fake, delay=NullDelay())

    desc = actuator.describe_humanizer()
    # Confirm JSON-serializable
    json_str = json.dumps(desc)
    assert json_str is not None

    assert desc["enabled"] is True
    assert desc["min_delay_s"] == 0.0
    assert desc["log_to_session"] is True
    assert desc["cursor_enabled"] is False
    assert desc["rng_initialized"] is True
    assert "interval_config" in desc
    assert "imperfection_config" in desc
    assert "cursor_config" in desc

    # Raw RNG state must not be present
    assert "state" not in desc
    assert "rng" not in desc


# 35-40. make_humanized_actuator factory tests
def test_make_humanized_actuator_factory(tmp_path: Path) -> None:
    # Mode MOCK returns NullActuator
    mock_act = make_humanized_actuator(mode="MOCK")
    assert isinstance(mock_act, NullActuator)

    # Mode LAB missing wrapped
    with pytest.raises(HumanizedActuatorError, match="LAB mode requires a valid RealActuator"):
        make_humanized_actuator(mode="LAB", delay=NullDelay())

    # Mode LAB missing delay
    driver = NullDriver()
    focus = FakeFocus()
    mapper = ActionMapper(driver)
    sys_config = _make_test_config(tmp_path / "sys")
    safety = SafetyLayer(sys_config)
    session = create_test_session(tmp_path)
    real_wrapped = RealActuator(driver=driver, focus=focus, mapper=mapper, safety=safety, session=session)

    with pytest.raises(HumanizedActuatorError, match="LAB mode requires a DelayProvider"):
        make_humanized_actuator(mode="LAB", wrapped=real_wrapped)

    # Mode LAB with non-RealActuator wrapped
    fake = FakeWrappedActuator()
    with pytest.raises(HumanizedActuatorError, match="LAB mode requires a valid RealActuator"):
        make_humanized_actuator(mode="LAB", wrapped=fake, delay=NullDelay())

    # Mode LAB valid return
    lab_act = make_humanized_actuator(
        mode="LAB",
        wrapped=real_wrapped,
        delay=NullDelay(),
        session=session,
    )
    assert isinstance(lab_act, HumanizedActuator)

    # Mode bogus
    with pytest.raises(HumanizedActuatorError, match="Invalid mode"):
        make_humanized_actuator(mode="bogus")


# 41. Static AST checks
def test_static_ast_checks() -> None:
    module_path = Path("src/wow_bot/actuation/humanized.py")
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module_path))

    forbidden_modules = {
        "wow_bot.strategist",
        "wow_bot.perception",
        "wow_bot.reflex",
        "wow_bot.executor",
        "wow_bot.world",
        "wow_bot.nav",
        "wow_bot.combat",
        "aiosqlite",
        "asyncio",
        "threading",
    }
    forbidden_substrings = ["ollama", "openai", "anthropic", "llm"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                for forbidden in forbidden_modules:
                    assert not mod.startswith(forbidden), f"Forbidden import: {mod}"
                for sub in forbidden_substrings:
                    assert sub not in mod.lower(), f"Forbidden import substring '{sub}': {mod}"

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod = node.module
                for forbidden in forbidden_modules:
                    assert not mod.startswith(forbidden), f"Forbidden import: {mod}"
                for sub in forbidden_substrings:
                    assert sub not in mod.lower(), f"Forbidden import substring '{sub}': {mod}"

        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "time":
            assert node.attr not in {
                "monotonic",
                "time",
                "sleep",
                "perf_counter",
            }, f"Forbidden time attribute call: time.{node.attr}"
