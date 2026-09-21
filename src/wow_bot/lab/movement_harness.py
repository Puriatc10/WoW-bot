# Pre-lab (MOCK_MODE)
"""Movement harness for driving actuators through scenario waypoints."""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from wow_bot.actuation.actuator import Actuator
from wow_bot.actuation.mapper import ActionStatus, MoveTo
from wow_bot.lab.scenario import Scenario
from wow_bot.safety import SafetyLayer
from wow_bot.session import Session


@dataclass(frozen=True)
class HarnessResult:
    """Immutable result snapshot of a scenario movement harness run."""

    scenario_name: str
    completed: bool
    waypoints_reached: int
    total_waypoints: int
    duration_s: float
    exit_reason: str
    per_waypoint_steps: tuple[int, ...]
    total_failed_steps: int

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation of the result."""
        return {
            "scenario_name": self.scenario_name,
            "completed": self.completed,
            "waypoints_reached": self.waypoints_reached,
            "total_waypoints": self.total_waypoints,
            "duration_s": self.duration_s,
            "exit_reason": self.exit_reason,
            "per_waypoint_steps": list(self.per_waypoint_steps),
            "total_failed_steps": self.total_failed_steps,
        }


class MovementHarness:
    """Outer scenario harness driving actuator waypoint by waypoint."""

    def __init__(
        self,
        actuator: Actuator,
        position_source: Callable[[], tuple[float, float]],
        scenario: Scenario,
        session: Session,
        safety: SafetyLayer,
        *,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._actuator = actuator
        self._position_source = position_source
        self._scenario = scenario
        self._session = session
        self._safety = safety
        self._clock = clock if clock is not None else time.monotonic
        self._sleep = sleep if sleep is not None else time.sleep

    def run(self) -> HarnessResult:
        """Run the movement scenario through all waypoints until arrival, timeout, or abort.

        Returns HarnessResult. Emits session events on waypoint reached/timeout and finished.
        Does not catch exceptions raised by actuator.execute or position_source.
        """
        start_time = self._clock()
        deadline = start_time + self._scenario.max_session_seconds

        waypoints_reached = 0
        per_waypoint_steps: list[int] = []
        total_failed_steps = 0

        def _make_result(exit_reason: str) -> HarnessResult:
            duration_s = self._clock() - start_time
            res = HarnessResult(
                scenario_name=self._scenario.name,
                completed=(exit_reason == "completed"),
                waypoints_reached=waypoints_reached,
                total_waypoints=len(self._scenario.waypoints),
                duration_s=duration_s,
                exit_reason=exit_reason,
                per_waypoint_steps=tuple(per_waypoint_steps),
                total_failed_steps=total_failed_steps,
            )
            self._session.write_event({
                "event": "harness_finished",
                "payload": res.to_json(),
            })
            return res

        for i, wp in enumerate(self._scenario.waypoints):
            steps = 0
            while steps < self._scenario.max_steps_per_waypoint:
                # 1. Deadline check
                if self._clock() >= deadline:
                    return _make_result("timeout")

                # 2. Kill switch check
                if self._safety.is_aborted():
                    return _make_result("kill_switch")

                # 3. Actuator abort check
                if self._actuator.is_aborted():
                    return _make_result("aborted")

                # 4. Read position
                pos = self._position_source()

                # 5. Arrival check
                d = math.sqrt((pos[0] - wp[0]) ** 2 + (pos[1] - wp[1]) ** 2)
                if d <= self._scenario.arrival_tolerance:
                    break

                # 6. One actuator step
                intent = MoveTo(x=wp[0], y=wp[1])
                result = self._actuator.execute(intent, position=pos)
                if result.status == ActionStatus.FAILED:
                    total_failed_steps += 1
                steps += 1

                # 7. Sleep
                if self._scenario.step_interval_s > 0:
                    self._sleep(self._scenario.step_interval_s)
            else:
                # while loop exhausted without break -> max_steps
                per_waypoint_steps.append(steps)
                self._session.write_event({
                    "event": "harness_waypoint_timeout",
                    "payload": {"index": i, "steps": steps},
                })
                return _make_result("max_steps")

            # Break reached: waypoint arrived
            waypoints_reached += 1
            per_waypoint_steps.append(steps)
            self._session.write_event({
                "event": "harness_waypoint_reached",
                "payload": {"index": i, "steps": steps},
            })

        return _make_result("completed")
