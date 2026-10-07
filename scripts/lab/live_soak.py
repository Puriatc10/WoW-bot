#!/usr/bin/env python3
"""Phase 13 Real Perception Live Soak Harness (Task 13.1).

Runs the full agent stack in LAB mode against a live game client:
- RealPerceptionBackend (ScreenCapture via mss, Tesseract OCR, template matching)
- RealActuator (pynput or interception)
- Fast reflex loop at 10-20 Hz
- FSM, World Model, Navigation, Combat Engine, Strategist
- Telemetry sampling and soak_report.json generation
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import tomllib
from pathlib import Path

from wow_bot.analysis.lab_soak_v2 import (
    LabSoakError,
    SoakConfig,
    SoakReport,
    SoakSample,
    build_soak_report,
    make_default_resource_sampler,
    write_soak_report,
)
from wow_bot.combat.rotation import RotationConfig, load_rotation_from_dict
from wow_bot.config import load_config
from wow_bot.farm.profile import load_profile
from wow_bot.lab.runner_v2 import (
    LabRunnerConfig,
    LabRunResult,
    LabRunStatus,
    build_lab_runtime_async,
    run_lab_loop_async,
)
from wow_bot.logging_setup import setup_logging
from wow_bot.perception.capture import ScreenCapture
from wow_bot.perception.perception_config import load_perception_config
from wow_bot.perception.real_backend import RealPerceptionBackend
from wow_bot.session import Session


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for Phase 13 live soak harness."""
    parser = argparse.ArgumentParser(
        description="Run WoW-bot Phase 13 Real Perception Live Soak Harness."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to lab config TOML file.",
    )
    parser.add_argument(
        "--perception-config",
        type=Path,
        default=None,
        help="Path to perception configuration TOML file.",
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
        help="Path to combat rotation TOML file.",
    )
    parser.add_argument(
        "--world-db",
        type=Path,
        default=Path("./runs/lab/world_live.db"),
        help="Path to World Model DB file.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=3600.0,
        help="Duration of soak in wall-clock seconds (default: 3600.0, max: 72000.0).",
    )
    parser.add_argument(
        "--sample-interval-s",
        type=float,
        default=2.0,
        help="Telemetry sampling interval in seconds (default: 2.0).",
    )
    parser.add_argument(
        "--driver",
        choices=["pynput", "interception", "null"],
        default="pynput",
        help="Actuation input driver backend (default: pynput).",
    )
    parser.add_argument(
        "--window-title",
        type=str,
        default="WoW",
        help="Target game window title (default: WoW).",
    )
    parser.add_argument(
        "--capture-region",
        type=str,
        default=None,
        help="Optional screen capture ROI as 'left,top,width,height' for windowed mode.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=10_000,
        help="Maximum cycles before stopping (default: 10,000).",
    )
    return parser.parse_args(argv)


def _load_rotation(path: Path) -> RotationConfig:
    """Load rotation configuration."""
    if not path.exists():
        raise LabSoakError(f"Rotation file does not exist: {path}")
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return load_rotation_from_dict(data)


async def run_live_soak_async(
    *,
    config_path: Path,
    perception_config_path: Path | None,
    profile_path: Path,
    rotation_path: Path,
    world_db_path: Path,
    duration_s: float,
    sample_interval_s: float,
    driver_name: str,
    window_title: str,
    capture_region: tuple[int, int, int, int] | None,
    max_cycles: int,
) -> tuple[LabRunResult, SoakReport]:
    """Execute live soak asynchronously."""
    cfg = load_config(config_path)
    if not cfg.lab_mode:
        raise LabSoakError("Configuration lab_mode must be True for Phase 13 live soak.")
    if cfg.dry_run:
        raise LabSoakError("dry_run must be False for Phase 13 live soak.")

    # Determine perception configuration path
    p_path = perception_config_path
    if p_path is None:
        if Path("config/perception.toml").exists():
            p_path = Path("config/perception.toml")
        else:
            p_path = Path("config/perception.example.toml")
    if not p_path.exists():
        raise LabSoakError(f"Perception config file not found: {p_path}")

    p_cfg = load_perception_config(p_path)

    # Initialize ScreenCapture
    capture = ScreenCapture(
        idle_fps=p_cfg.idle_fps,
        combat_fps=p_cfg.combat_fps,
        monitor_idx=1,
        region=capture_region,
    )

    # Build RealPerceptionBackend
    base_dir = p_path.parent if (p_path.parent / "models").exists() else p_path.parent.parent
    backend = RealPerceptionBackend.from_config(
        p_cfg,
        capture=capture,
        base_dir=base_dir,
    )

    # Load profile and rotation
    profile = load_profile(profile_path)
    rotation = _load_rotation(rotation_path)

    # Start Session
    session = Session.start(cfg)
    session_dir = session.path
    session_id = session.session_id

    setup_logging(session, cfg.log_level)
    sampler = make_default_resource_sampler(log_path=session_dir / "app.log")

    deadline = time.monotonic() + duration_s
    stop_event = asyncio.Event()
    samples: list[SoakSample] = []
    last_sample_time = 0.0

    def sampling_sleep(seconds: float) -> None:
        nonlocal last_sample_time
        time.sleep(seconds)
        now = time.monotonic()
        if now >= deadline:
            stop_event.set()
        elif (now - last_sample_time) >= sample_interval_s:
            snap = sampler.sample(now)
            samples.append(
                SoakSample(
                    ts=now,
                    cpu_percent=snap.cpu_percent,
                    rss_bytes=snap.rss_bytes,
                    log_size_bytes=snap.log_size_bytes,
                    position_delta=0.0,
                    inventory_delta=0,
                    successful_actions_total=0,
                    reflex_ticks_total=0,
                )
            )
            last_sample_time = now

    runtime = await build_lab_runtime_async(
        config=cfg,
        session=session,
        perception_backend=backend,
        meta_state_source=lambda: None,
        rotation_config=rotation,
        farm_profile=profile,
        world_db_path=str(world_db_path),
        driver_name=driver_name,
        window_title=window_title,
        runner_config=LabRunnerConfig(
            max_cycles_per_run=max_cycles,
        ),
        sleep=sampling_sleep,
    )

    try:
        run_res = await run_lab_loop_async(
            runtime,
            max_cycles=max_cycles,
            stop_event=stop_event,
        )
    finally:
        session.close("soak_complete")
        await runtime.close()

    # Build and write soak report
    crash = run_res.status in {
        LabRunStatus.HEALTH_CRITICAL,
        LabRunStatus.LOOP_DETECTED,
        LabRunStatus.MAX_FAILURES_REACHED,
        LabRunStatus.RUNTIME_ERROR,
        LabRunStatus.BUILD_ERROR,
    }
    crash_reason = run_res.reason if crash else None
    soak_report = build_soak_report(
        samples,
        session_id=session_id,
        crash=crash,
        crash_reason=crash_reason,
        config=SoakConfig(),
    )
    report_path = session_dir / "soak_report.json"
    write_soak_report(soak_report, report_path)

    return run_res, soak_report


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for Phase 13 live soak."""
    args = parse_args(argv)

    cap_region = None
    if args.capture_region:
        parts = [int(p.strip()) for p in args.capture_region.split(",")]
        if len(parts) != 4:
            sys.stderr.write("Invalid --capture-region: must be 'left,top,width,height'\n")
            return 2
        cap_region = (parts[0], parts[1], parts[2], parts[3])

    try:
        result, _report = asyncio.run(
            run_live_soak_async(
                config_path=args.config,
                perception_config_path=args.perception_config,
                profile_path=args.profile,
                rotation_path=args.rotation,
                world_db_path=args.world_db,
                duration_s=args.duration_s,
                sample_interval_s=args.sample_interval_s,
                driver_name=args.driver,
                window_title=args.window_title,
                capture_region=cap_region,
                max_cycles=args.max_cycles,
            )
        )
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"Live soak run failed: {exc}\n")
        return 1

    print("\n" + "=" * 60)
    print("  Phase 13 Live Soak Execution Completed")
    print("=" * 60)
    print(f"Status:            {result.status}")
    print(f"Cycles completed:  {result.cycles_completed}")
    print(f"Duration:          {result.duration_s:.1f} s")
    print(f"Success actions:   {result.successful_actions_total}")
    print("=" * 60)

    return 0 if result.status in (LabRunStatus.COMPLETED, LabRunStatus.MAX_CYCLES_REACHED) else 1


if __name__ == "__main__":
    sys.exit(main())
