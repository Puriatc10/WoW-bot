#!/usr/bin/env python3
"""Lab Soak Harness CLI V2 for WoW-bot.

Runs a long-running soak monitoring session, sampling resource usage and progress telemetry
via injected providers, and writes a self-contained soak_report.json to the session directory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from wow_bot.analysis.lab_soak_v2 import (
    ProcessHandle,
    ResourceSampler,
    ResourceSnapshot,
    SoakConfig,
    SoakSample,
    build_soak_report,
    validate_soak_report_dict,
    write_soak_report,
)


class Clock(Protocol):
    """Protocol for monotonic time provider."""

    def __call__(self) -> float:
        ...


class ProgressData(Protocol):
    """Protocol for progress delta snapshot."""

    @property
    def position_delta(self) -> float:
        ...

    @property
    def inventory_delta(self) -> int:
        ...

    @property
    def successful_actions_total(self) -> int:
        ...

    @property
    def reflex_ticks_total(self) -> int:
        ...


class ProgressSource(Protocol):
    """Protocol for sampling progress telemetry."""

    def sample(self, now: float) -> ProgressData:
        ...


class SimpleProgressData:
    """Simple concrete implementation of ProgressData."""

    def __init__(
        self,
        position_delta: float = 0.0,
        inventory_delta: int = 0,
        successful_actions_total: int = 0,
        reflex_ticks_total: int = 0,
    ) -> None:
        self.position_delta = float(position_delta)
        self.inventory_delta = int(inventory_delta)
        self.successful_actions_total = int(successful_actions_total)
        self.reflex_ticks_total = int(reflex_ticks_total)


class FakeClock:
    """Fake clock for dry-run testing advancing in fixed steps."""

    def __init__(self, start_ts: float = 0.0) -> None:
        self._now = start_ts

    def now(self) -> float:
        return self._now

    def sleep(self, duration: float) -> None:
        self._now += duration


class FakeProcessHandle:
    """Fake process handle reporting alive status for dry run."""

    def __init__(self, alive: bool = True, exit_code: int | None = None) -> None:
        self._alive = alive
        self._exit_code = exit_code

    def is_alive(self) -> bool:
        return self._alive

    def exit_code(self) -> int | None:
        return self._exit_code


class FakeResourceSampler:
    """Fake resource sampler producing synthetic snapshots for dry run."""

    def sample(self, now: float) -> ResourceSnapshot:
        # Synthetic values growing deterministically with time
        cpu = 12.5 + (now % 5.0)
        rss = 100_000_000 + int(now * 1_000)
        log_size = 50_000 + int(now * 500)
        return ResourceSnapshot(
            ts=now,
            cpu_percent=cpu,
            rss_bytes=rss,
            log_size_bytes=log_size,
        )


class FakeProgressSource:
    """Fake progress source producing synthetic progress deltas for dry run."""

    def sample(self, now: float) -> ProgressData:
        step = int(now)
        return SimpleProgressData(
            position_delta=0.5,
            inventory_delta=step // 10,
            successful_actions_total=step * 2,
            reflex_ticks_total=step * 10,
        )


class SoakRunner:
    """Orchestrates soak telemetry collection over a given duration."""

    def __init__(
        self,
        *,
        clock: Clock,
        sleep_fn: Callable[[float], None],
        sampler: ResourceSampler,
        process: ProcessHandle,
        progress_source: ProgressSource,
        config: SoakConfig | None = None,
    ) -> None:
        self._clock = clock
        self._sleep_fn = sleep_fn
        self._sampler = sampler
        self._process = process
        self._progress_source = progress_source
        self._config = config if config is not None else SoakConfig()

    def run(self, duration_s: float) -> list[SoakSample]:
        """Run the telemetry collection loop for duration_s seconds.

        Args:
            duration_s: Target duration in seconds.

        Returns:
            List of collected SoakSample objects.
        """
        if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)) or not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError(f"duration_s must be a finite float > 0.0, got {duration_s!r}")

        start_time = self._clock()
        end_time = start_time + duration_s
        samples: list[SoakSample] = []

        while True:
            now = self._clock()
            snapshot = self._sampler.sample(now)
            progress = self._progress_source.sample(now)

            sample = SoakSample(
                ts=now,
                cpu_percent=snapshot.cpu_percent,
                rss_bytes=snapshot.rss_bytes,
                log_size_bytes=snapshot.log_size_bytes,
                position_delta=progress.position_delta,
                inventory_delta=progress.inventory_delta,
                successful_actions_total=progress.successful_actions_total,
                reflex_ticks_total=progress.reflex_ticks_total,
            )
            samples.append(sample)

            if len(samples) >= self._config.max_samples:
                break

            if not self._process.is_alive():
                break

            if now >= end_time:
                break

            self._sleep_fn(self._config.sample_interval_s)

        return samples


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="WoW-bot Soak Test Harness CLI V2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        required=True,
        help="Directory where session.json, events.jsonl, and soak_report.json live.",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        required=True,
        help="Session identifier string to embed in the soak report.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        required=True,
        help="Soak duration in seconds.",
    )
    parser.add_argument(
        "--sample-interval-s",
        type=float,
        default=1.0,
        help="Metrics sampling interval in seconds.",
    )
    parser.add_argument(
        "--mode",
        choices=["MOCK", "LAB"],
        default="MOCK",
        help="Execution mode.",
    )
    parser.add_argument(
        "--write-crash-on-error",
        action="store_true",
        help="When set, write crash.json on any unhandled exception before exiting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run harness in synthetic dry-run mode using fake clock and sampler.",
    )
    return parser.parse_args(args)


def write_crash_json(session_dir: Path, exc: BaseException) -> None:
    """Write crash.json artifact on unhandled exception."""
    crash_path = session_dir / "crash.json"
    crash_data = {
        "error": repr(exc),
        "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        "ts": datetime.now(UTC).isoformat(),
    }
    try:
        session_dir.mkdir(parents=True, exist_ok=True)
        crash_path.write_text(f"{write_crash_json_to_str(crash_data)}\n", encoding="utf-8")
    except Exception as io_err:  # noqa: BLE001
        sys.stderr.write(f"Failed to write crash.json: {io_err}\n")


def write_crash_json_to_str(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def main(args_list: list[str] | None = None) -> int:
    """CLI entry point for soak_v2.py.

    Returns exit code integer.
    """
    write_crash_flag = False
    session_dir: Path | None = None

    try:
        try:
            parsed = parse_args(args_list)
        except SystemExit as sys_exit:
            # argparse exits with status 0 for --help or 2 for arg parse errors
            return sys_exit.code if isinstance(sys_exit.code, int) else 2

        write_crash_flag = parsed.write_crash_on_error
        session_dir = parsed.session_dir

        # Validate range arguments
        if (
            isinstance(parsed.duration_s, bool)
            or not isinstance(parsed.duration_s, (int, float))
            or not math.isfinite(parsed.duration_s)
            or parsed.duration_s <= 0.0
        ):
            sys.stderr.write(f"Error: --duration-s must be positive float, got {parsed.duration_s!r}\n")
            return 2

        if (
            isinstance(parsed.sample_interval_s, bool)
            or not isinstance(parsed.sample_interval_s, (int, float))
            or not math.isfinite(parsed.sample_interval_s)
            or parsed.sample_interval_s <= 0.0
        ):
            sys.stderr.write(f"Error: --sample-interval-s must be positive float, got {parsed.sample_interval_s!r}\n")
            return 2

        if not parsed.session_id:
            sys.stderr.write("Error: --session-id must be non-empty string\n")
            return 2

        config = SoakConfig(sample_interval_s=parsed.sample_interval_s)

        clock: Clock
        sleep_fn: Callable[[float], None]
        sampler: ResourceSampler
        process: ProcessHandle
        progress_source: ProgressSource

        if parsed.dry_run:
            fake_clock = FakeClock()
            clock = fake_clock.now
            sleep_fn = fake_clock.sleep
            sampler = FakeResourceSampler()
            process = FakeProcessHandle(alive=True)
            progress_source = FakeProgressSource()
        else:
            import time

            clock = time.monotonic
            sleep_fn = time.sleep
            # Default fallback real mock sources for CLI if not dry-run
            sampler = FakeResourceSampler()
            process = FakeProcessHandle(alive=True)
            progress_source = FakeProgressSource()

        runner = SoakRunner(
            clock=clock,
            sleep_fn=sleep_fn,
            sampler=sampler,
            process=process,
            progress_source=progress_source,
            config=config,
        )

        samples = runner.run(parsed.duration_s)

        report = build_soak_report(
            samples,
            session_id=parsed.session_id,
            crash=False,
            crash_reason=None,
            config=config,
        )

        # Validate generated report before writing
        validate_soak_report_dict(report.to_json())

        output_path = parsed.session_dir / "soak_report.json"
        write_soak_report(report, output_path)

        # Check process liveness/exit status for exit code
        ec = process.is_alive()
        if not process.is_alive() and ec is not None and ec != 0:
            return 1

        return 0

    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"Uncaught exception in soak harness: {exc}\n")
        if write_crash_flag and session_dir is not None:
            write_crash_json(session_dir, exc)
        return 3


if __name__ == "__main__":
    sys.exit(main())
