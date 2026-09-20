"""Tests for RealActuator facade and NullActuator."""

import ast
import time
from pathlib import Path
from typing import Any

import pytest

from wow_bot.actuation.actuator import (
    ActuatorError,
    NullActuator,
    RealActuator,
    make_actuator,
)
from wow_bot.actuation.backends.focus_null import NullFocusBackend
from wow_bot.actuation.drivers.null import NullDriver
from wow_bot.actuation.focus import FocusLost, FocusManager
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
from wow_bot.safety import SafetyLayer


class FakeSession:
    """Fake session that records written events in memory for test verification."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def write_event(self, event: dict[str, Any]) -> None:
        self.events.append(dict(event))


class CountingFocusManager(FocusManager):
    """FocusManager subclass that counts assert_focused calls and allows toggling focus loss."""

    def __init__(self, window_title: str = "MockWindow") -> None:
        super().__init__(window_title=window_title, backend=NullFocusBackend())
        self.assert_count = 0
        self.should_fail = False

    def assert_focused(self) -> None:
        self.assert_count += 1
        if self.should_fail:
            raise FocusLost("Focus lost in test")


class CountingActionMapper:
    """Fake ActionMapper that tracks execute calls and returns canned ActionResult or raises."""

    def __init__(self, canned_result: ActionResult | None = None) -> None:
        self.execute_count = 0
        self.last_intent: Intent | None = None
        self.last_position: tuple[float, float] | None = None
        self.canned_result = canned_result or ActionResult(
            status=ActionStatus.SUCCESS,
            latency_ms=10.0,
            notes="canned",
        )
        self.raise_exc: Exception | None = None

    def execute(self, intent: Intent, *, position: tuple[float, float]) -> ActionResult:
        self.execute_count += 1
        self.last_intent = intent
        self.last_position = position
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.canned_result


class SlowDriver(NullDriver):
    """Driver subclass with release_all counter, optional delay, and exception injection."""

    def __init__(self, delay_s: float = 0.0, raise_on_release: Exception | None = None) -> None:
        super().__init__()
        self.release_all_count = 0
        self.delay_s = delay_s
        self.raise_on_release = raise_on_release

    def release_all(self) -> None:
        self.release_all_count += 1
        super().release_all()
        if self.delay_s > 0:
            time.sleep(self.delay_s)
        if self.raise_on_release is not None:
            raise self.raise_on_release


def _make_config(tmp_path: Path) -> Config:
    return Config(
        lab_mode=False,
        server_allowlist=("127.0.0.1:8080",),
        isolation_sentinel="127.0.0.1:9999",
        kill_switch_key="f12",
        session_root=tmp_path,
        dry_run=True,
        max_session_seconds=3600,
        log_level="INFO",
    )


# Acceptance Item 1 & 2
def test_null_actuator_execute_and_abort() -> None:
    null_act = NullActuator()
    intent = MoveTo(x=10.0, y=20.0)
    pos = (1.0, 2.0)

    res = null_act.execute(intent, position=pos)
    assert res.status == ActionStatus.SUCCESS
    assert res.notes == "null"
    assert null_act.recorded() == [(intent, pos)]

    assert not null_act.is_aborted()
    null_act.abort("test")
    assert null_act.is_aborted()
    null_act.abort("test_again")  # idempotent
    assert null_act.is_aborted()


# Acceptance Item 3
def test_real_actuator_rejected_when_safety_aborted(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    session = FakeSession()
    safety = SafetyLayer(config, session)  # type: ignore[arg-type]
    driver = SlowDriver()
    focus = CountingFocusManager()
    mapper = CountingActionMapper()

    actuator = RealActuator(
        driver=driver,
        focus=focus,
        mapper=mapper,  # type: ignore[arg-type]
        safety=safety,
        session=session,  # type: ignore[arg-type]
    )

    safety.abort("test_safety_abort")
    intent = Turn(angle_rad=0.5)

    res = actuator.execute(intent, position=(0.0, 0.0))

    assert res.status == ActionStatus.FAILED
    assert res.notes == "safety_aborted"
    assert focus.assert_count == 0
    assert mapper.execute_count == 0

    rejections = [e for e in session.events if e.get("event") == "actuator_rejected"]
    assert len(rejections) == 1
    assert rejections[0]["reason"] == "safety_aborted"
    assert rejections[0]["intent"] == repr(intent)


# Acceptance Item 4
def test_real_actuator_rejected_when_actuator_aborted(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    session = FakeSession()
    safety = SafetyLayer(config, session)  # type: ignore[arg-type]
    driver = SlowDriver()
    focus = CountingFocusManager()
    mapper = CountingActionMapper()

    actuator = RealActuator(
        driver=driver,
        focus=focus,
        mapper=mapper,  # type: ignore[arg-type]
        safety=safety,
        session=session,  # type: ignore[arg-type]
    )

    actuator.abort("test_actuator_abort")
    intent = MoveTo(x=5.0, y=5.0)

    res = actuator.execute(intent, position=(0.0, 0.0))

    assert res.status == ActionStatus.FAILED
    assert res.notes == "actuator_aborted"
    assert mapper.execute_count == 0

    rejections = [e for e in session.events if e.get("event") == "actuator_rejected"]
    assert len(rejections) == 1
    assert rejections[0]["reason"] == "actuator_aborted"
    assert rejections[0]["intent"] == repr(intent)


# Acceptance Item 5
def test_real_actuator_rejected_when_focus_lost(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    session = FakeSession()
    safety = SafetyLayer(config, session)  # type: ignore[arg-type]
    driver = SlowDriver()
    focus = CountingFocusManager()
    focus.should_fail = True
    mapper = CountingActionMapper()

    actuator = RealActuator(
        driver=driver,
        focus=focus,
        mapper=mapper,  # type: ignore[arg-type]
        safety=safety,
        session=session,  # type: ignore[arg-type]
    )

    intent = MoveTo(x=1.0, y=1.0)
    res = actuator.execute(intent, position=(0.0, 0.0))

    assert res.status == ActionStatus.FAILED
    assert res.notes == "focus_lost"
    assert focus.assert_count == 1
    assert mapper.execute_count == 0

    rejections = [e for e in session.events if e.get("event") == "actuator_rejected"]
    assert len(rejections) == 1
    assert rejections[0]["reason"] == "focus_lost"


# Acceptance Item 6, 7 & 8
def test_real_actuator_execute_success_and_events(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    session = FakeSession()
    safety = SafetyLayer(config, session)  # type: ignore[arg-type]
    driver = SlowDriver()
    focus = CountingFocusManager()
    expected_res = ActionResult(status=ActionStatus.SUCCESS, latency_ms=12.5, notes="ok")
    mapper = CountingActionMapper(canned_result=expected_res)

    actuator = RealActuator(
        driver=driver,
        focus=focus,
        mapper=mapper,  # type: ignore[arg-type]
        safety=safety,
        session=session,  # type: ignore[arg-type]
    )

    intent = MoveTo(x=3.0, y=4.0)
    pos = (1.0, 2.0)
    res = actuator.execute(intent, position=pos)

    assert res == expected_res
    assert mapper.execute_count == 1
    assert mapper.last_intent == intent
    assert mapper.last_position == pos

    events = session.events
    assert len(events) == 2
    assert events[0]["event"] == "actuator_intent"
    assert events[0]["intent"] == repr(intent)
    assert events[0]["position"] == [1.0, 2.0]

    assert events[1]["event"] == "actuator_result"
    assert events[1]["intent"] == repr(intent)
    assert events[1]["status"] == "success"
    assert events[1]["latency_ms"] == 12.5
    assert events[1]["notes"] == "ok"


# Acceptance Item 9
def test_real_actuator_mapper_exception_propagates_no_result_event(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    session = FakeSession()
    safety = SafetyLayer(config, session)  # type: ignore[arg-type]
    driver = SlowDriver()
    focus = CountingFocusManager()
    mapper = CountingActionMapper()
    mapper.raise_exc = ValueError("Driver failure in mapper")

    actuator = RealActuator(
        driver=driver,
        focus=focus,
        mapper=mapper,  # type: ignore[arg-type]
        safety=safety,
        session=session,  # type: ignore[arg-type]
    )

    intent = Turn(angle_rad=1.0)
    with pytest.raises(ValueError, match="Driver failure in mapper"):
        actuator.execute(intent, position=(0.0, 0.0))

    events = session.events
    assert len(events) == 1
    assert events[0]["event"] == "actuator_intent"
    assert not any(e.get("event") == "actuator_result" for e in events)


# Acceptance Item 10, 11, 12, 13
def test_real_actuator_abort_semantics_and_timing(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    session = FakeSession()
    safety = SafetyLayer(config, session)  # type: ignore[arg-type]
    driver = SlowDriver(delay_s=0.010)  # 10 ms sleep
    focus = CountingFocusManager()
    mapper = CountingActionMapper()

    actuator = RealActuator(
        driver=driver,
        focus=focus,
        mapper=mapper,  # type: ignore[arg-type]
        safety=safety,
        session=session,  # type: ignore[arg-type]
    )

    t0 = time.monotonic()
    actuator.abort("user_requested")
    elapsed_ms = (time.monotonic() - t0) * 1000.0

    assert elapsed_ms < 100.0
    assert actuator.is_aborted()
    assert driver.release_all_count == 1

    abort_events = [e for e in session.events if e.get("event") == "actuator_abort"]
    assert len(abort_events) == 1
    assert abort_events[0]["reason"] == "user_requested"

    # Second call (idempotent)
    actuator.abort("user_requested_again")
    assert driver.release_all_count == 1
    assert len([e for e in session.events if e.get("event") == "actuator_abort"]) == 1


# Acceptance Item 14
def test_real_actuator_abort_driver_raises_trapped(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    session = FakeSession()
    safety = SafetyLayer(config, session)  # type: ignore[arg-type]
    driver_error = RuntimeError("Hardware failure during release")
    driver = SlowDriver(raise_on_release=driver_error)
    focus = CountingFocusManager()
    mapper = CountingActionMapper()

    actuator = RealActuator(
        driver=driver,
        focus=focus,
        mapper=mapper,  # type: ignore[arg-type]
        safety=safety,
        session=session,  # type: ignore[arg-type]
    )

    # Should not raise
    actuator.abort("hardware_fault")

    assert actuator.is_aborted()
    assert actuator.last_abort_error() is driver_error


# Acceptance Item 15
def test_real_actuator_close_and_context_manager(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    session = FakeSession()
    safety = SafetyLayer(config, session)  # type: ignore[arg-type]
    driver = SlowDriver()
    focus = CountingFocusManager()
    mapper = CountingActionMapper()

    actuator = RealActuator(
        driver=driver,
        focus=focus,
        mapper=mapper,  # type: ignore[arg-type]
        safety=safety,
        session=session,  # type: ignore[arg-type]
    )

    with actuator:
        pass

    assert actuator.is_aborted()
    assert driver.release_all_count == 1
    abort_events = [e for e in session.events if e.get("event") == "actuator_abort"]
    assert len(abort_events) == 1
    assert abort_events[0]["reason"] == "close"

    # Explicit close again is no-op
    actuator.close()
    assert driver.release_all_count == 1


# Acceptance Item 16, 17, 18, 19
def test_make_actuator_factory(tmp_path: Path) -> None:
    mock_act = make_actuator(mode="MOCK")
    assert isinstance(mock_act, NullActuator)

    # LAB mode with missing dependencies
    with pytest.raises(ActuatorError, match="Missing required LAB mode dependencies") as exc_info:
        make_actuator(mode="LAB")
    assert "driver" in str(exc_info.value)
    assert "focus" in str(exc_info.value)
    assert "mapper" in str(exc_info.value)
    assert "safety" in str(exc_info.value)
    assert "session" in str(exc_info.value)

    # LAB mode with all 5 dependencies
    config = _make_config(tmp_path)
    session = FakeSession()
    safety = SafetyLayer(config, session)  # type: ignore[arg-type]
    driver = NullDriver()
    focus = FocusManager(window_title="MockWindow", backend=NullFocusBackend())
    mapper = ActionMapper(driver=driver, delay=NullDelay())

    lab_act = make_actuator(
        mode="LAB",
        driver=driver,
        focus=focus,
        mapper=mapper,
        safety=safety,
        session=session,  # type: ignore[arg-type]
    )
    assert isinstance(lab_act, RealActuator)

    # Bogus mode
    with pytest.raises(ActuatorError, match="Invalid mode: 'bogus'"):
        make_actuator(mode="bogus")


# Acceptance Item 20
def test_static_ast_forbidden_imports() -> None:
    actuator_file = Path(__file__).parent.parent / "src" / "wow_bot" / "actuation" / "actuator.py"
    source = actuator_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(actuator_file))

    forbidden_modules = {
        "wow_bot.strategist",
        "wow_bot.executor",
        "wow_bot.reflex",
        "wow_bot.navigation",
        "wow_bot.combat",
        "wow_bot.world",
        "wow_bot.perception",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for forbidden in forbidden_modules:
                    assert not alias.name.startswith(forbidden), (
                        f"Forbidden import '{alias.name}' in actuator.py"
                    )
        elif isinstance(node, ast.ImportFrom) and node.module:
            for forbidden in forbidden_modules:
                assert not node.module.startswith(forbidden), (
                    f"Forbidden import from '{node.module}' in actuator.py"
                )
