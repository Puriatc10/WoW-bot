#!/usr/bin/env python3
"""24-Hour Long-Running Soak Test Harness CLI (Task 8.3).

Executes the WoW-Bot research prototype pipeline for long durations while
concurrently sampling operational process-tree CPU, RSS memory, log size,
Watchdog supervisor status, and pipeline progress.

Usage:
    python scripts/run_soak_test.py --scenario peaceful_farm --duration 86400 --sample-interval 30 --seed 42 --output reports/soak_24h.json
    python scripts/run_soak_test.py --scenario peaceful_farm --duration 300 --sample-interval 5 --output reports/soak_smoke_5m.json
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

from wow_bot.analysis.soak import (
    ProcessResourceSampler,
    ResourceSampler,
    SoakObserver,
    SoakSample,
    build_soak_report,
    calculate_log_size,
    resolve_log_path,
    write_soak_report_atomically,
)
from wow_bot.main import build_runtime, run_pipeline
from wow_bot.mocks.mock_perception import SCENARIO_NAMES
from wow_bot.shared.logger import get_logger

logger = get_logger("SOAK_RUNNER")


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate command line arguments for soak testing."""
    parser = argparse.ArgumentParser(
        description="Run a WoW-Bot long-running soak test and record resource stability evidence."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="peaceful_farm",
        help=f"Canonical scenario name (default: peaceful_farm). Supported: {', '.join(SCENARIO_NAMES)}",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=86400.0,
        help="Requested real runtime duration in seconds (default: 86400 / 24h). Must be finite float > 0.",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=30.0,
        help="Operational metrics sampling interval in seconds (default: 30.0). Must be finite float > 0.",
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
        default="reports/soak_24h.json",
        help="Destination path for soak report JSON (default: reports/soak_24h.json).",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default=None,
        help="Optional explicit path to runtime log file or directory for log-size tracking.",
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

    if (
        isinstance(parsed.sample_interval, bool)
        or not isinstance(parsed.sample_interval, (int, float))
        or not math.isfinite(parsed.sample_interval)
        or parsed.sample_interval <= 0.0
    ):
        parser.error(
            f"Sampling interval must be a finite positive number (> 0), got {parsed.sample_interval!r}"
        )

    return parsed


async def run_soak_test_async(
    scenario: str,
    duration: float,
    sample_interval: float,
    seed: int,
    output_path: Path,
    log_path_arg: str | None = None,
    resource_sampler: ResourceSampler | None = None,
) -> int:
    """Execute the long-running pipeline while periodically collecting resource metrics.

    Args:
        scenario: Validated scenario name.
        duration: Validated real runtime duration in seconds.
        sample_interval: Validated sampling interval in seconds.
        seed: Experiment seed.
        output_path: Destination JSON output path.
        log_path_arg: Optional path argument for log file/directory.
        resource_sampler: Optional custom ResourceSampler for testing.

    Returns:
        CLI exit code (0 for success, non-zero for error/interruption).
    """
    if output_path.exists():
        logger.error(
            f"Output report file already exists at '{output_path}'. Overwrite refused to prevent evidence loss."
        )
        return 1

    resolved_log_path, log_status = resolve_log_path(log_path_arg)
    sampler = (
        resource_sampler
        if resource_sampler is not None
        else ProcessResourceSampler()
    )

    soak_observer = SoakObserver()

    components = await build_runtime(
        scenario=scenario,
        observer=soak_observer,
        seed=seed,
    )

    samples: list[SoakSample] = []
    start_mono = time.monotonic()
    completed_normally = False
    termination_reason = "unknown"
    failure_type: str | None = None
    failure_message: str | None = None
    exit_code = 0

    shutdown_event = asyncio.Event()

    async def _metrics_sampling_loop() -> None:
        """Periodically sample operational metrics and checkpoint report."""
        while not shutdown_event.is_set():
            now_mono = time.monotonic()
            elapsed = max(0.0, now_mono - start_mono)

            res = sampler.sample()
            cur_log_size = calculate_log_size(resolved_log_path)
            watchdog_alive = components.watchdog.is_alive
            watchdog_shutdown = components.watchdog.shutdown_event.is_set()

            sample = SoakSample(
                elapsed_seconds=round(elapsed, 2),
                cpu_percent=res.cpu_percent,
                memory_rss_mb=res.memory_rss_mb,
                log_size_bytes=cur_log_size,
                progress_token=soak_observer.progress_token,
                watchdog_alive=watchdog_alive,
                shutdown_requested=watchdog_shutdown,
                fsm_state=soak_observer.current_fsm_state,
            )
            samples.append(sample)

            # Atomic checkpoint write after sample
            try:
                report_dict = build_soak_report(
                    scenario=scenario,
                    seed=seed,
                    requested_duration_seconds=duration,
                    sample_interval_seconds=sample_interval,
                    completed_duration_seconds=elapsed,
                    samples=samples,
                    completed_normally=False,
                    termination_reason="running_checkpoint",
                    log_path=resolved_log_path,
                    log_size_status=log_status,
                )
                write_soak_report_atomically(report_dict, output_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Error updating report checkpoint at '{output_path}': {exc}")

            try:
                await asyncio.sleep(sample_interval)
            except asyncio.CancelledError:
                break

    sampler_task = asyncio.create_task(_metrics_sampling_loop(), name="soak_sampling_loop")

    try:
        logger.info(
            f"Starting soak test: scenario='{scenario}' duration={duration}s "
            f"sample_interval={sample_interval}s seed={seed} output='{output_path}'"
        )
        await run_pipeline(components, run_duration_seconds=duration)

        completed_normally = True
        termination_reason = "duration_completed"
        logger.info(f"Soak test completed normally after requested duration ({duration}s).")

    except asyncio.CancelledError:
        logger.warning("Soak run interrupted by asyncio cancellation (e.g. SIGINT/Ctrl+C).")
        completed_normally = False
        termination_reason = "keyboard_interrupt"
        failure_type = "CancelledError"
        exit_code = 130

    except KeyboardInterrupt:
        logger.warning("Soak run interrupted by KeyboardInterrupt.")
        completed_normally = False
        termination_reason = "keyboard_interrupt"
        failure_type = "KeyboardInterrupt"
        exit_code = 130

    except Exception as exc:  # noqa: BLE001
        if components.watchdog.shutdown_event.is_set():
            logger.error("Soak test stopped due to Watchdog supervisor emergency shutdown.")
            completed_normally = False
            termination_reason = "watchdog_shutdown"
            failure_type = type(exc).__name__
            failure_message = str(exc)
            exit_code = 1
        else:
            logger.error(f"Soak test failed due to pipeline exception: {type(exc).__name__}: {exc}")
            completed_normally = False
            termination_reason = "pipeline_failure"
            failure_type = type(exc).__name__
            failure_message = str(exc)
            exit_code = 1

    finally:
        shutdown_event.set()
        if not sampler_task.done():
            sampler_task.cancel()
            await asyncio.gather(sampler_task, return_exceptions=True)

        final_elapsed = max(0.0, time.monotonic() - start_mono)

        # Record final sample before exiting
        try:
            res = sampler.sample()
            final_sample = SoakSample(
                elapsed_seconds=round(final_elapsed, 2),
                cpu_percent=res.cpu_percent,
                memory_rss_mb=res.memory_rss_mb,
                log_size_bytes=calculate_log_size(resolved_log_path),
                progress_token=soak_observer.progress_token,
                watchdog_alive=components.watchdog.is_alive,
                shutdown_requested=components.watchdog.shutdown_event.is_set(),
                fsm_state=soak_observer.current_fsm_state,
            )
            samples.append(final_sample)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Error taking final soak resource sample: {exc}")

        # Final atomic report write
        try:
            final_report = build_soak_report(
                scenario=scenario,
                seed=seed,
                requested_duration_seconds=duration,
                sample_interval_seconds=sample_interval,
                completed_duration_seconds=final_elapsed,
                samples=samples,
                completed_normally=completed_normally,
                termination_reason=termination_reason,
                failure_type=failure_type,
                failure_message=failure_message,
                log_path=resolved_log_path,
                log_size_status=log_status,
            )
            write_soak_report_atomically(final_report, output_path)
            logger.info(
                f"Final soak evidence report successfully written to '{output_path}' "
                f"(samples={len(samples)}, completed_normally={completed_normally})."
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to write final soak report to '{output_path}': {exc}")
            if exit_code == 0:
                exit_code = 1

    return exit_code


def main(args: Sequence[str] | None = None) -> None:
    """CLI entry point for run_soak_test.py."""
    parsed = parse_args(args)
    output_path = Path(parsed.output).resolve()

    try:
        code = asyncio.run(
            run_soak_test_async(
                scenario=parsed.scenario,
                duration=parsed.duration,
                sample_interval=parsed.sample_interval,
                seed=parsed.seed,
                output_path=output_path,
                log_path_arg=parsed.log_path,
            )
        )
        sys.exit(code)
    except KeyboardInterrupt:
        logger.warning("Soak test process killed by KeyboardInterrupt.")
        sys.exit(130)


if __name__ == "__main__":
    main()
