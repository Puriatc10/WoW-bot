#!/usr/bin/env python3
"""MOCK Soak Harness CLI and library function for Phase 12 (Task 12.1).

Runs the lab runner in MOCK mode for a bounded wall-clock duration, samples
process resource usage periodically, and produces soak_report.json via lab_soak_v2.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:
    resource = None  # type: ignore[assignment]

from wow_bot.actuation.backends.focus_null import NullFocusBackend
from wow_bot.analysis.lab_soak_v2 import (
    LabSoakError,
    ResourceSampler,
    ResourceSnapshot,
    SoakConfig,
    SoakReport,
    SoakSample,
    build_soak_report,
    write_soak_report,
)
from wow_bot.combat.rotation import load_rotation_from_dict
from wow_bot.config import ConfigError, load_config
from wow_bot.executor.states import FSMState
from wow_bot.farm.profile import load_profile
from wow_bot.lab.runner_v2 import (
    LabRunnerConfig,
    LabRunResult,
    LabRunStatus,
    build_lab_runtime_async,
    run_lab_loop_async,
)
from wow_bot.session import Session, SessionInfo, _make_config_snapshot, _utc_now_iso
from wow_bot.watchdog.metrics import ProgressSample


class ProcessResourceSampler:
    """ResourceSampler implementation capturing current process RSS, CPU, and log size."""

    def __init__(self, *, log_path: Path | None = None) -> None:
        if resource is None:
            raise RuntimeError("resource module is unavailable on this platform")
        self._log_path = log_path
        self._prev_cpu_time: float | None = None
        self._prev_ts: float | None = None

    def sample(self, now: float) -> ResourceSnapshot:
        """Sample current process resource usage at given timestamp `now`."""
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform == "darwin":
            rss_bytes = int(rusage.ru_maxrss)
        else:
            rss_bytes = int(rusage.ru_maxrss * 1024)

        t = os.times()
        cpu_time = float(t.user + t.system)

        if self._prev_cpu_time is None or self._prev_ts is None:
            cpu_percent = 0.0
        else:
            dt = now - self._prev_ts
            if dt > 0.0:
                dcpu = cpu_time - self._prev_cpu_time
                cpu_percent = float(max(0.0, (dcpu / dt) * 100.0))
            else:
                cpu_percent = 0.0

        self._prev_cpu_time = cpu_time
        self._prev_ts = now

        log_size_bytes = 0
        if self._log_path is not None and self._log_path.exists():
            log_size_bytes = self._log_path.stat().st_size

        return ResourceSnapshot(
            ts=now,
            cpu_percent=cpu_percent,
            rss_bytes=rss_bytes,
            log_size_bytes=log_size_bytes,
        )


def make_soak_sleep(
    base_sleep: Callable[[float], None],
    sampler: ResourceSampler,
    clock: Callable[[], float],
    *,
    deadline: float,
    sample_interval_s: float,
    stop_event: asyncio.Event,
    samples: list[SoakSample],
    sample_builder: Callable[[float, ResourceSnapshot], SoakSample],
) -> Callable[[float], None]:
    """Construct a sampling sleep wrapper for run_lab_loop_async.

    Calls base_sleep, checks clock against deadline, sets stop_event when expired,
    and samples resources at sample_interval_s cadence when running before deadline.
    """
    last_sample_ts: float | None = None

    def soak_sleep(seconds: float) -> None:
        nonlocal last_sample_ts
        base_sleep(seconds)
        now = clock()

        if now >= deadline:
            if not stop_event.is_set():
                stop_event.set()
            return

        if last_sample_ts is None or (now - last_sample_ts) >= sample_interval_s:
            snap = sampler.sample(now)
            sample = sample_builder(now, snap)
            samples.append(sample)
            last_sample_ts = now

    return soak_sleep


class _FakeLlmClient:
    """Private LLM client implementation for CLI mock soak runs."""

    def complete(self, prompt: str) -> str:
        """Return valid JSON strategy response for mock strategist calls."""
        return '{"goal": "farm", "target": null, "rationale": "mock soak run"}'


@dataclass
class _SyntheticGameState:
    """Private synthetic GameState satisfying GameStateView, LootStateView, VendorStateView, CombatStateView."""

    player_x: float = 100.0
    player_y: float = 100.0
    player_z: float = 0.0
    player_heading: float = 0.0
    self_x: float = 100.0
    self_y: float = 100.0
    self_hp_percent: float = 100.0
    self_in_combat: bool = False
    fsm_state: FSMState = FSMState.IDLE
    current_target_id: str | None = None
    target_entity_id: str | None = None
    target_in_range: bool = False
    target_is_alive: bool = False
    target_is_lootable: bool = False
    target_distance: float | None = None
    target_x: float | None = None
    target_y: float | None = None
    target_hp_percent: float | None = None
    resource: float = 100.0
    resource_max: float = 100.0
    adds_count: int = 0
    inventory_count: int = 5
    inventory_max: int = 10_000_000
    durability_fraction: float | None = 1.0
    level: float = 10.0
    xp: float = 1000.0
    level_or_xp: float = 1000.0
    incoming_casts: tuple[Any, ...] = ()
    entities: tuple[Any, ...] = ()

    def spell_cooldown_ready(self, spell_id: str) -> bool:
        """Return True indicating all spell cooldowns are ready."""
        return True


@dataclass
class _SyntheticMetaState:
    """Private synthetic MetaState view."""

    drives: dict[str, float] = field(default_factory=dict)
    memory_summary: str = ""
    drive_hunger: float = 0.0
    drive_fatigue: float = 0.0
    chaos_level: float = 0.0


async def run_soak_async(
    *,
    session_dir: Path,
    session_id: str,
    config_path: Path,
    world_db_path: Path,
    profile_path: Path,
    rotation_path: Path,
    duration_s: float = 60.0,
    sample_interval_s: float = 1.0,
    max_cycles: int = 10_000_000,
    clock: Callable[[], float],
    base_sleep: Callable[[float], None],
    sampler: ResourceSampler,
    game_state_source: Callable[[], object],
    meta_state_source: Callable[[], object],
    llm_client: object,
    progress_source: Callable[[], ProgressSample] | None = None,
) -> tuple[LabRunResult, SoakReport]:
    """Asynchronously execute a MOCK mode soak run for duration_s and produce SoakReport."""
    if (
        isinstance(duration_s, bool)
        or not isinstance(duration_s, (int, float))
        or duration_s <= 0.0
        or duration_s > 7200.0
    ):
        raise LabSoakError(f"duration_s must be a float in (0, 7200], got {duration_s!r}")

    if (
        isinstance(sample_interval_s, bool)
        or not isinstance(sample_interval_s, (int, float))
        or sample_interval_s <= 0.0
    ):
        raise LabSoakError(f"sample_interval_s must be a float > 0, got {sample_interval_s!r}")

    if isinstance(max_cycles, bool) or not isinstance(max_cycles, int) or max_cycles < 1:
        raise LabSoakError(f"max_cycles must be an integer >= 1, got {max_cycles!r}")

    config = load_config(config_path)
    profile = load_profile(profile_path)

    if not rotation_path.exists():
        raise LabSoakError(f"Rotation file does not exist: '{rotation_path}'")
    try:
        with open(rotation_path, "r", encoding="utf-8") as f:  # noqa: ASYNC230
            rot_raw = json.load(f)
        rotation = load_rotation_from_dict(rot_raw)
    except Exception as exc:
        raise LabSoakError(f"Failed to load rotation file '{rotation_path}': {exc}") from exc

    started_at = clock()
    deadline = started_at + duration_s
    samples: list[SoakSample] = []
    stop_event = asyncio.Event()

    def sample_builder(ts: float, snap: ResourceSnapshot) -> SoakSample:
        pos_delta = 0.0
        inv_delta = 0
        succ_actions = 0
        refl_ticks = 0
        if progress_source is not None:
            prog = progress_source()
            pos_delta = float(getattr(prog, "position_delta", 0.0))
            inv_delta = int(getattr(prog, "inventory_delta", 0))
            succ_actions = int(getattr(prog, "successful_actions_total", 0))
            refl_ticks = int(getattr(prog, "reflex_ticks_total", 0))

        return SoakSample(
            ts=ts,
            cpu_percent=snap.cpu_percent,
            rss_bytes=snap.rss_bytes,
            log_size_bytes=snap.log_size_bytes,
            position_delta=pos_delta,
            inventory_delta=inv_delta,
            successful_actions_total=succ_actions,
            reflex_ticks_total=refl_ticks,
        )

    soak_sleep = make_soak_sleep(
        base_sleep=base_sleep,
        sampler=sampler,
        clock=clock,
        deadline=deadline,
        sample_interval_s=sample_interval_s,
        stop_event=stop_event,
        samples=samples,
        sample_builder=sample_builder,
    )

    # Prepare session directory and initial files
    session_dir.mkdir(parents=True, exist_ok=True)
    events_path = session_dir / "events.jsonl"
    if not events_path.exists():
        try:
            fd = os.open(events_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
            os.close(fd)
        except OSError:
            pass

    session_json_path = session_dir / "session.json"
    s_iso = _utc_now_iso()
    cfg_snap = _make_config_snapshot(config)
    initial_data = {
        "session_id": session_id,
        "started_at": s_iso,
        "mode": "MOCK" if not config.lab_mode else "LAB",
        "config_snapshot": cfg_snap,
    }
    if not session_json_path.exists():
        with open(session_json_path, "w", encoding="utf-8") as f:  # noqa: ASYNC230
            json.dump(initial_data, f, indent=2, ensure_ascii=False)
            f.write("\n")

    info = SessionInfo(
        session_id=session_id,
        path=session_dir,
        started_at=s_iso,
        mode="MOCK" if not config.lab_mode else "LAB",
        config_snapshot=cfg_snap,
    )
    session = Session(info)
    runtime = None

    try:
        # Override lab_mode to True on config so build_lab_runtime_async constructs runtime
        lab_config = dataclasses.replace(config, lab_mode=True)
        runtime = await build_lab_runtime_async(
            config=lab_config,
            session=session,
            game_state_source=game_state_source,
            meta_state_source=meta_state_source,
            rotation_config=rotation,
            farm_profile=profile,
            world_db_path=str(world_db_path),
            driver_name="null",
            runner_config=LabRunnerConfig(max_cycles_per_run=max_cycles),
            clock=clock,
            sleep=soak_sleep,
            focus_backend=NullFocusBackend(),
            llm_client=llm_client,
            include_reflex_loop=False,
            include_watchdog=False,
        )

        result = await run_lab_loop_async(
            runtime,
            max_cycles=max_cycles,
            stop_event=stop_event,
        )

        crash = result.status in {
            LabRunStatus.HEALTH_CRITICAL,
            LabRunStatus.LOOP_DETECTED,
            LabRunStatus.MAX_FAILURES_REACHED,
            LabRunStatus.RUNTIME_ERROR,
            LabRunStatus.BUILD_ERROR,
        }
        crash_reason = result.reason if crash else None

        if not samples:
            if crash:
                report = build_soak_report(
                    [],
                    session_id=session_id,
                    crash=True,
                    crash_reason=crash_reason,
                    config=SoakConfig(),
                )
            else:
                raise LabSoakError("Non-crash soak run produced no samples")
        else:
            report = build_soak_report(
                samples,
                session_id=session_id,
                crash=crash,
                crash_reason=crash_reason,
                config=SoakConfig(),
            )

        write_soak_report(report, session_dir / "soak_report.json")
        return result, report
    except Exception as exc:
        session.write_crash(exc)
        raise
    finally:
        session.close("soak_complete")
        if runtime is not None and hasattr(runtime.world, "close"):
            await runtime.world.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for full_soak.py script."""
    parser = argparse.ArgumentParser(
        description="Run WoW-bot MOCK mode soak test harness."
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        required=True,
        help="Path to output session directory.",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        required=True,
        help="Unique session ID string.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=60.0,
        help="Duration of soak in wall-clock seconds (default: 60.0, max: 7200.0).",
    )
    parser.add_argument(
        "--sample-interval-s",
        type=float,
        default=1.0,
        help="Telemetry sampling interval in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--mode",
        choices=["MOCK"],
        default="MOCK",
        help="Execution mode (MOCK required, LAB rejected).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to lab config TOML file.",
    )
    parser.add_argument(
        "--world-db",
        type=Path,
        required=True,
        help="Path to World Model DB file.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        required=True,
        help="Path to farm profile TOML file.",
    )
    parser.add_argument(
        "--rotation",
        type=Path,
        required=True,
        help="Path to rotation TOML file.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=10_000_000,
        help="Upper bound cycle count to prevent runaway (default: 10,000,000).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for MOCK soak harness."""
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 0
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"Argument parse error: {exc}\n")
        return 2

    if args.duration_s <= 0.0 or args.duration_s > 7200.0:
        sys.stderr.write(
            f"Invalid --duration-s {args.duration_s}: must be > 0 and <= 7200\n"
        )
        return 2

    if args.sample_interval_s <= 0.0:
        sys.stderr.write(
            f"Invalid --sample-interval-s {args.sample_interval_s}: must be > 0\n"
        )
        return 2

    if args.max_cycles < 1:
        sys.stderr.write(
            f"Invalid --max-cycles {args.max_cycles}: must be >= 1\n"
        )
        return 2

    if args.mode != "MOCK":
        sys.stderr.write(
            f"Invalid --mode {args.mode}: LAB mode is rejected in Phase 12\n"
        )
        return 2

    try:
        sampler = ProcessResourceSampler(
            log_path=args.session_dir / "app.log"
        )
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"Resource sampler construction error: {exc}\n")
        return 2

    state_inst = _SyntheticGameState()
    meta_inst = _SyntheticMetaState()

    def game_state_source() -> _SyntheticGameState:
        state_inst.player_x += 1.0
        state_inst.player_y += 1.0
        state_inst.self_x += 1.0
        state_inst.self_y += 1.0
        state_inst.level += 1.0
        state_inst.xp += 10.0
        state_inst.level_or_xp += 10.0
        state_inst.inventory_count += 1
        return state_inst

    def meta_state_source() -> _SyntheticMetaState:
        return meta_inst

    try:
        result, report = asyncio.run(
            run_soak_async(
                session_dir=args.session_dir,
                session_id=args.session_id,
                config_path=args.config,
                world_db_path=args.world_db,
                profile_path=args.profile,
                rotation_path=args.rotation,
                duration_s=args.duration_s,
                sample_interval_s=args.sample_interval_s,
                max_cycles=args.max_cycles,
                clock=time.monotonic,
                base_sleep=time.sleep,
                sampler=sampler,
                game_state_source=game_state_source,
                meta_state_source=meta_state_source,
                llm_client=_FakeLlmClient(),
            )
        )
    except ConfigError as exc:
        sys.stderr.write(f"Config error: {exc}\n")
        return 2
    except LabSoakError as exc:
        sys.stderr.write(f"Lab soak error: {exc}\n")
        return 2
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"Unhandled exception during soak: {exc}\n")
        return 3

    sys.stdout.write(json.dumps(report.to_json(), ensure_ascii=False, indent=2) + "\n")

    if result.status in (LabRunStatus.STOP_EVENT_SET, LabRunStatus.MAX_CYCLES_REACHED):
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
