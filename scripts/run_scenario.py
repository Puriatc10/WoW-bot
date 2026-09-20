#!/usr/bin/env python3
"""Scenario Runner CLI Script (Task 7.2).

Orchestrates scenario execution using the Task 7.1 async pipeline runtime and
collects structured experiment metrics into a research report JSON file.

Usage:
    python scripts/run_scenario.py --scenario combat_light --duration 600
    python scripts/run_scenario.py --scenario peaceful_farm --duration 60 --seed 42
    python scripts/run_scenario.py --scenario death_loop --duration 120 --output reports/custom.json
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path

# Ensure 'src' is in sys.path when script is executed directly
_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from wow_bot.main import build_runtime, run_pipeline
from wow_bot.mocks.mock_perception import SCENARIO_NAMES
from wow_bot.reporting.scenario import ScenarioReportCollector, write_report_atomically
from wow_bot.shared.logger import get_logger

logger = get_logger("SCENARIO_RUNNER")


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate command line arguments.

    Args:
        args: Command line arguments (defaults to sys.argv[1:]).

    Returns:
        Parsed Namespace object.
    """
    parser = argparse.ArgumentParser(
        description="Run a WoW-Bot simulation scenario and produce a research report."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        help=f"Canonical scenario name. Supported: {', '.join(SCENARIO_NAMES)}",
    )
    parser.add_argument(
        "--duration",
        type=float,
        required=True,
        help="Requested real runtime duration in seconds (must be finite float > 0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for destination JSON report (default: reports/{scenario}_seed-{seed}_duration-{duration}.json).",
    )

    parsed = parser.parse_args(args)

    if parsed.scenario not in SCENARIO_NAMES:
        parser.error(
            f"Unknown scenario '{parsed.scenario}'. Supported canonical scenarios are: {list(SCENARIO_NAMES)}"
        )

    if (
        isinstance(parsed.duration, bool)
        or not isinstance(parsed.duration, (int, float))
        or not math.isfinite(parsed.duration)
        or parsed.duration <= 0.0
    ):
        parser.error(
            f"Requested duration must be a finite positive number (> 0), got {parsed.duration!r}"
        )

    return parsed


def get_default_output_path(scenario: str, seed: int, duration: float) -> Path:
    """Construct the standard output path for a scenario report."""
    dur_str = f"{duration:.0f}" if duration.is_integer() else f"{duration}"
    filename = f"{scenario}_seed-{seed}_duration-{dur_str}.json"
    return Path("reports") / filename


async def execute_scenario(
    scenario: str,
    duration: float,
    seed: int,
    output_path: Path,
) -> int:
    """Build and execute the scenario pipeline, collecting metrics and writing report.

    Args:
        scenario: Validated scenario name.
        duration: Validated requested duration in seconds.
        seed: Experiment seed integer.
        output_path: Destination path for report.

    Returns:
        CLI exit code (0 for success, non-zero for error).
    """
    logger.info(
        f"Starting scenario '{scenario}' (duration={duration}s, seed={seed}, output='{output_path}')"
    )

    collector = ScenarioReportCollector(
        scenario=scenario,
        seed=seed,
        requested_duration_seconds=duration,
    )

    components = None

    start_mono = time.monotonic()
    completed_normally = False
    error_type: str | None = None
    exit_code = 0

    try:
        components = await build_runtime(scenario=scenario, observer=collector, seed=seed)
        await run_pipeline(components, run_duration_seconds=duration)
        completed_normally = True
        logger.info(f"Scenario '{scenario}' completed normally.")
    except asyncio.CancelledError:
        logger.warning(f"Scenario '{scenario}' was cancelled (e.g. SIGINT/Ctrl+C).")
        completed_normally = False
        error_type = "CancelledError"
        exit_code = 130
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Scenario '{scenario}' failed with error: {type(exc).__name__}")
        completed_normally = False
        error_type = type(exc).__name__
        exit_code = 1

    if components is not None:
        collector.shutdown_requested = components.watchdog.shutdown_event.is_set()
    completed_duration = time.monotonic() - start_mono

    try:
        report = collector.build_report(
            completed_duration_seconds=completed_duration,
            completed_normally=completed_normally,
            error_type=error_type,
        )
        write_report_atomically(report, output_path)
        logger.info(f"Scenario report successfully written to '{output_path}'.")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to write scenario report to '{output_path}': {exc}")
        if exit_code == 0:
            exit_code = 1

    return exit_code


def main(args: Sequence[str] | None = None) -> None:
    """CLI script entry point."""
    parsed = parse_args(args)

    output_path = (
        Path(parsed.output)
        if parsed.output is not None
        else get_default_output_path(parsed.scenario, parsed.seed, parsed.duration)
    )

    try:
        exit_code = asyncio.run(
            execute_scenario(
                scenario=parsed.scenario,
                duration=parsed.duration,
                seed=parsed.seed,
                output_path=output_path,
            )
        )
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("Scenario runner interrupted by KeyboardInterrupt.")
        sys.exit(130)


if __name__ == "__main__":
    main()
