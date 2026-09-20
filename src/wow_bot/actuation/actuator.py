"""RealActuator facade for system action execution and safety/focus gating."""

from types import TracebackType
from typing import Protocol, Self, runtime_checkable

from wow_bot.actuation.driver import InputDriver
from wow_bot.actuation.focus import FocusLost, FocusManager
from wow_bot.actuation.mapper import ActionMapper, ActionResult, ActionStatus, Intent
from wow_bot.safety import SafetyLayer
from wow_bot.session import Session


class ActuatorError(Exception):
    """Raised when actuation configuration or execution encounters an unrecoverable error."""


@runtime_checkable
class Actuator(Protocol):
    """Protocol for system actuation facades."""

    def execute(
        self,
        intent: Intent,
        *,
        position: tuple[float, float],
    ) -> ActionResult:
        """Execute a single action intent from the given starting position."""
        ...

    def abort(self, reason: str) -> None:
        """Abort actuation immediately, releasing all driver inputs."""
        ...

    def is_aborted(self) -> bool:
        """Return True if actuation is currently aborted, False otherwise."""
        ...

    def close(self) -> None:
        """Close the actuator and release driver resources."""
        ...


class NullActuator:
    """Mock actuator implementation for test and MOCK_MODE environments."""

    def __init__(self) -> None:
        self._aborted = False
        self._recorded: list[tuple[Intent, tuple[float, float]]] = []

    def execute(
        self,
        intent: Intent,
        *,
        position: tuple[float, float],
    ) -> ActionResult:
        """Record execution request and return synthetic success result."""
        self._recorded.append((intent, position))
        return ActionResult(
            status=ActionStatus.SUCCESS,
            latency_ms=0.0,
            notes="null",
        )

    def abort(self, reason: str) -> None:
        """Set internal abort flag idempotently."""
        self._aborted = True

    def is_aborted(self) -> bool:
        """Return internal abort state."""
        return self._aborted

    def close(self) -> None:
        """No-op close for NullActuator."""

    def recorded(self) -> list[tuple[Intent, tuple[float, float]]]:
        """Return a copy of all recorded (intent, position) execution pairs."""
        return list(self._recorded)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()


class RealActuator:
    """Real actuation facade enforcing safety, focus assertions, logging, and abort semantics."""

    def __init__(
        self,
        driver: InputDriver,
        focus: FocusManager,
        mapper: ActionMapper,
        safety: SafetyLayer,
        session: Session,
    ) -> None:
        if driver is None or focus is None or mapper is None or safety is None or session is None:
            raise ActuatorError("All dependencies (driver, focus, mapper, safety, session) are required")

        self._driver = driver
        self._focus = focus
        self._mapper = mapper
        self._safety = safety
        self._session = session
        self._aborted = False
        self._last_abort_error: BaseException | None = None

    def execute(
        self,
        intent: Intent,
        *,
        position: tuple[float, float],
    ) -> ActionResult:
        """Execute a single intent step following strict safety, focus, and logging sequence."""
        # 1. Safety check
        if self._safety.is_aborted():
            self._session.write_event({
                "event": "actuator_rejected",
                "intent": repr(intent),
                "reason": "safety_aborted",
            })
            return ActionResult(
                status=ActionStatus.FAILED,
                latency_ms=0.0,
                notes="safety_aborted",
            )

        # 2. Actuator-level abort check
        if self._aborted:
            self._session.write_event({
                "event": "actuator_rejected",
                "intent": repr(intent),
                "reason": "actuator_aborted",
            })
            return ActionResult(
                status=ActionStatus.FAILED,
                latency_ms=0.0,
                notes="actuator_aborted",
            )

        # 3. Focus check
        try:
            self._focus.assert_focused()
        except FocusLost:
            self._session.write_event({
                "event": "actuator_rejected",
                "intent": repr(intent),
                "reason": "focus_lost",
            })
            return ActionResult(
                status=ActionStatus.FAILED,
                latency_ms=0.0,
                notes="focus_lost",
            )

        # 4. Emit intent event
        self._session.write_event({
            "event": "actuator_intent",
            "intent": repr(intent),
            "position": [position[0], position[1]],
        })

        # 5. Execute action step via mapper
        result = self._mapper.execute(intent, position=position)

        # 6. Emit result event
        self._session.write_event({
            "event": "actuator_result",
            "intent": repr(intent),
            "status": result.status.value,
            "latency_ms": result.latency_ms,
            "notes": result.notes,
        })

        # 7. Return result
        return result

    def abort(self, reason: str) -> None:
        """Abort actuation idempotently, setting flag first and releasing all driver inputs."""
        if self._aborted:
            return

        self._aborted = True

        try:
            self._driver.release_all()
        except BaseException as exc:  # noqa: BLE001
            self._last_abort_error = exc

        try:
            self._session.write_event({
                "event": "actuator_abort",
                "reason": reason,
            })
        except Exception:  # noqa: BLE001, S110
            pass

    def is_aborted(self) -> bool:
        """Return True if actuator-level abort has been triggered, False otherwise."""
        return self._aborted

    def close(self) -> None:
        """Close actuator idempotently, aborting if not already aborted."""
        if not self._aborted:
            self.abort("close")

    def last_abort_error(self) -> BaseException | None:
        """Return last exception raised by driver.release_all() during abort, or None."""
        return self._last_abort_error

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()


def make_actuator(
    *,
    mode: str,
    driver: InputDriver | None = None,
    focus: FocusManager | None = None,
    mapper: ActionMapper | None = None,
    safety: SafetyLayer | None = None,
    session: Session | None = None,
) -> Actuator:
    """Factory function for instantiating Actuator instances by mode."""
    if mode == "MOCK":
        return NullActuator()
    elif mode == "LAB":
        missing: list[str] = []
        if driver is None:
            missing.append("driver")
        if focus is None:
            missing.append("focus")
        if mapper is None:
            missing.append("mapper")
        if safety is None:
            missing.append("safety")
        if session is None:
            missing.append("session")

        if missing or driver is None or focus is None or mapper is None or safety is None or session is None:
            raise ActuatorError(f"Missing required LAB mode dependencies: {', '.join(missing)}")

        return RealActuator(
            driver=driver,
            focus=focus,
            mapper=mapper,
            safety=safety,
            session=session,
        )
    else:
        raise ActuatorError(f"Invalid mode: '{mode}'")
