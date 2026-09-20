"""CLI script for running movement scenario demo runs in MOCK or LAB mode."""

import argparse
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path

# Ensure 'src' is in sys.path when script is executed directly
_src_dir = str(Path(__file__).resolve().parent.parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from wow_bot.actuation.actuator import make_actuator
from wow_bot.actuation.backends.focus_null import NullFocusBackend
from wow_bot.actuation.backends.focus_win32 import Win32FocusBackend
from wow_bot.actuation.driver import InputDriver, make_driver
from wow_bot.actuation.drivers.null import NullDriver
from wow_bot.actuation.focus import FocusManager
from wow_bot.actuation.mapper import ActionMapper, NullDelay, RealDelay
from wow_bot.config import Config, ConfigError
from wow_bot.lab.movement_harness import MovementHarness
from wow_bot.lab.scenario import ScenarioError, load_scenario
from wow_bot.mode import ModeError, enter_mode
from wow_bot.session import Session


class ScriptedApproachSource:
    """Simulated position source for MOCK mode demo execution."""

    def __init__(
        self,
        waypoints: tuple[tuple[float, float], ...],
        step_speed: float = 1.0,
        arrival_tolerance: float = 0.5,
    ) -> None:
        self._waypoints = waypoints
        self._step_speed = step_speed
        self._arrival_tolerance = arrival_tolerance
        self._target_idx = 0
        if waypoints:
            self._current_pos = [waypoints[0][0], waypoints[0][1]]
        else:
            self._current_pos = [0.0, 0.0]

    def __call__(self) -> tuple[float, float]:
        pos = (float(self._current_pos[0]), float(self._current_pos[1]))

        if self._target_idx < len(self._waypoints):
            target = self._waypoints[self._target_idx]
            dx = target[0] - self._current_pos[0]
            dy = target[1] - self._current_pos[1]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist <= self._arrival_tolerance:
                self._target_idx += 1
                if self._target_idx < len(self._waypoints):
                    target = self._waypoints[self._target_idx]
                    dx = target[0] - self._current_pos[0]
                    dy = target[1] - self._current_pos[1]
                    dist = math.sqrt(dx * dx + dy * dy)

            if dist > 0 and self._target_idx < len(self._waypoints):
                step = min(self._step_speed, dist)
                self._current_pos[0] += (dx / dist) * step
                self._current_pos[1] += (dy / dist) * step

        return pos


def main() -> int:
    """Parse CLI arguments and run movement scenario harness."""
    parser = argparse.ArgumentParser(description="Run movement scenario demo in MOCK or LAB mode.")
    parser.add_argument("--scenario", required=True, type=Path, help="Path to scenario JSON file")
    parser.add_argument(
        "--mode",
        choices=["MOCK", "LAB"],
        default="MOCK",
        help="Execution mode (default: MOCK)",
    )
    parser.add_argument(
        "--session-root",
        type=Path,
        default=Path("runs/lab"),
        help="Session directory root (default: runs/lab)",
    )
    parser.add_argument(
        "--driver",
        choices=["null", "pynput", "interception"],
        default="null",
        help="Input driver name (default: null)",
    )
    parser.add_argument(
        "--window-title",
        type=str,
        default=None,
        help="Target window title (required in LAB mode)",
    )
    parser.add_argument(
        "--position-bridge",
        type=str,
        default=None,
        help="Position bridge specifier for LAB mode",
    )

    args = parser.parse_args()

    if args.mode == "LAB" and not args.window_title:
        sys.stderr.write("Error: --window-title is required when --mode LAB\n")
        return 2

    try:
        scenario = load_scenario(args.scenario)
    except ScenarioError as e:
        sys.stderr.write(f"Scenario error: {e}\n")
        return 2

    lab_mode = (args.mode == "LAB")
    config = Config(
        lab_mode=lab_mode,
        server_allowlist=("127.0.0.1:8080",),
        isolation_sentinel="127.0.0.1:9999",
        kill_switch_key="f12",
        session_root=args.session_root,
        dry_run=(not lab_mode),
        max_session_seconds=math.ceil(scenario.max_session_seconds),
        log_level="INFO",
    )

    try:
        session = Session.start(config)
    except ConfigError as e:
        sys.stderr.write(f"Session startup failed: {e}\n")
        return 2

    with session:
        try:
            mode_ctx = enter_mode(config, session)
        except (ModeError, Exception) as e:  # noqa: BLE001
            sys.stderr.write(f"Mode entry failed: {e}\n")
            return 2

        position_source: Callable[[], tuple[float, float]]
        driver: InputDriver

        if args.mode == "MOCK":
            driver = NullDriver()
            focus = FocusManager("unused", backend=NullFocusBackend())
            mapper = ActionMapper(driver, delay=NullDelay())
            actuator = make_actuator(mode="MOCK")
            position_source = ScriptedApproachSource(
                waypoints=scenario.waypoints,
                step_speed=1.0,
                arrival_tolerance=scenario.arrival_tolerance,
            )
        else:
            if not args.position_bridge:
                sys.stderr.write(
                    "Error: LAB mode position bridge is not implemented. Pass --position-bridge when available.\n"
                )
                return 2

            try:
                driver = make_driver(args.driver, lab_mode=True)
                focus = FocusManager(args.window_title, backend=Win32FocusBackend())
                mapper = ActionMapper(driver, delay=RealDelay())
                actuator = make_actuator(
                    mode="LAB",
                    driver=driver,
                    focus=focus,
                    mapper=mapper,
                    safety=mode_ctx.safety,
                    session=session,
                )
            except Exception as e:  # noqa: BLE001
                sys.stderr.write(f"LAB mode initialization failed: {e}\n")
                return 2

            raise NotImplementedError("LAB mode position bridge integration is not available in this phase.")

        harness = MovementHarness(
            actuator=actuator,
            position_source=position_source,
            scenario=scenario,
            session=session,
            safety=mode_ctx.safety,
        )

        result = harness.run()
        print(json.dumps(result.to_json(), indent=2))

        return 0 if result.completed else 1


if __name__ == "__main__":
    sys.exit(main())
