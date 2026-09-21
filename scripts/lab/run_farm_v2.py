#!/usr/bin/env python3
"""CLI runner script for WoW-bot LAB mode farm cycle loop (Task 11.4)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wow_bot.combat.rotation import RotationConfig, load_rotation_from_dict
from wow_bot.config import Config
from wow_bot.executor.states import FSMState
from wow_bot.farm.profile import (
    CycleSpec,
    FarmProfile,
    NodeReference,
    RoutePreferences,
    VendorReference,
    load_profile,
)
from wow_bot.lab.runner_v2 import (
    LabRunnerConfig,
    LabRunStatus,
    build_lab_runtime,
    run_lab_loop,
)
from wow_bot.session import Session


@dataclass
class SyntheticGameState:
    """Synthetic GameState for CLI mock run mode."""

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
    adds_count: int = 0
    inventory_count: int = 5
    inventory_max: int = 30
    durability_fraction: float | None = 1.0
    level: float = 10.0
    xp: float = 1000.0
    incoming_casts: tuple[Any, ...] = ()
    entities: tuple[Any, ...] = ()

    def spell_cooldown_ready(self, spell_id: str) -> bool:
        """Return True indicating all spell cooldowns are ready."""
        return True


@dataclass
class SyntheticMetaState:
    """Synthetic MetaState for CLI mock run mode."""

    drive_hunger: float = 0.0
    drive_fatigue: float = 0.0
    chaos_level: float = 0.0


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for run_farm_v2."""
    parser = argparse.ArgumentParser(
        description="Run WoW-bot farm loop in MOCK or LAB mode."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to lab config TOML.",
    )
    parser.add_argument(
        "--world-db",
        type=str,
        default=":memory:",
        help="Path to World Model DB file or :memory:.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        help="Path to farm profile TOML file.",
    )
    parser.add_argument(
        "--rotation",
        type=Path,
        help="Path to rotation TOML file.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=100,
        help="Maximum cycles to run (default: 100).",
    )
    parser.add_argument(
        "--driver",
        choices=["null", "pynput", "interception"],
        default="null",
        help="Input driver backend (default: null).",
    )
    parser.add_argument(
        "--window-title",
        type=str,
        default="WoW",
        help="Target game window title (default: WoW).",
    )
    parser.add_argument(
        "--mode",
        choices=["MOCK", "LAB"],
        default="MOCK",
        help="Execution mode (default: MOCK).",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    """CLI entry point for running farm loop."""
    parsed = parse_args(args)

    sess_root = Path("/tmp/runs/lab")
    sess_root.mkdir(parents=True, exist_ok=True)

    config_obj = Config(
        lab_mode=True,
        server_allowlist=["127.0.0.1:8080"],
        isolation_sentinel="127.0.0.1:9999",
        kill_switch_key="F12",
        session_root=sess_root,
        dry_run=(parsed.mode == "MOCK"),
        max_session_seconds=3600,
        log_level="INFO",
    )

    if parsed.profile is not None and parsed.profile.exists():
        profile_obj = load_profile(parsed.profile)
    else:
        profile_obj = FarmProfile(
            schema_version=1,
            name="cli_default",
            description="CLI default profile",
            cycle=CycleSpec(
                nodes=(NodeReference(kind="mob", name="mob_1"),),
                vendor=VendorReference(kind="vendor", name="vendor_1"),
                repair=VendorReference(kind="vendor", name="vendor_1"),
                stop_when_inventory_full=True,
                stop_after_cycles=0,
            ),
            route_preferences=RoutePreferences(),
            metadata={},
        )

    if parsed.rotation is not None and parsed.rotation.exists():
        with open(parsed.rotation, "r", encoding="utf-8") as f:
            rot_data = json.load(f)
        rotation_cfg = load_rotation_from_dict(rot_data)
    else:
        rotation_cfg = RotationConfig(rules=())

    session_obj = Session.start(config_obj)

    state_inst = SyntheticGameState()
    meta_inst = SyntheticMetaState()

    def moving_state_source() -> SyntheticGameState:
        state_inst.player_x += 2.0
        state_inst.player_y += 2.0
        state_inst.self_x += 2.0
        state_inst.self_y += 2.0
        state_inst.xp += 10.0
        state_inst.inventory_count += 1
        return state_inst

    runner_cfg = LabRunnerConfig(max_cycles_per_run=parsed.max_cycles)

    runtime = build_lab_runtime(
        config=config_obj,
        session=session_obj,
        game_state_source=moving_state_source,
        meta_state_source=lambda: meta_inst,
        rotation_config=rotation_cfg,
        farm_profile=profile_obj,
        world_db_path=parsed.world_db,
        driver_name=parsed.driver,
        window_title=parsed.window_title,
        runner_config=runner_cfg,
        include_watchdog=False,
    )

    result = run_lab_loop(runtime, max_cycles=parsed.max_cycles)
    print(json.dumps(result.to_json(), indent=2))

    if result.status in (LabRunStatus.COMPLETED, LabRunStatus.MAX_CYCLES_REACHED):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
