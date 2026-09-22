"""Grounded prompt builder for Strategist v2."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol

from wow_bot.executor.states import FSMState
from wow_bot.world.summary import WorldSummary


class PromptError(Exception):
    """Exception raised when prompt construction fails."""


class MetaStateLike(Protocol):
    """Opaque view of MetaState consumed by the prompt builder.

    Implementations MUST provide:
    """

    drives: Mapping[str, float]
    memory_summary: str


class GameStateView(Protocol):
    """Opaque view of GameState consumed by the prompt builder.

    Implementations MUST provide:
    """

    player_x: float
    player_y: float
    player_z: float
    self_hp_percent: float
    resource: float
    resource_max: float
    current_target_id: str | None
    target_hp_percent: float | None
    inventory_count: int
    level_or_xp: float
    fsm_state: FSMState


@dataclass(frozen=True)
class PromptConfig:
    schema_version: int = 1
    max_memory_chars: int = 400
    max_vendors: int = 3
    max_trainers: int = 3
    max_nodes: int = 3
    max_mobs: int = 3
    max_waypoints: int = 5
    max_recent_combats: int = 3
    include_raw_state_json: bool = False

    def __post_init__(self) -> None:
        if self.schema_version < 1:
            raise ValueError(
                f"schema_version must be >= 1, got {self.schema_version}"
            )
        if self.max_memory_chars < 0:
            raise ValueError(
                f"max_memory_chars must be >= 0, got {self.max_memory_chars}"
            )
        limits = {
            "max_vendors": self.max_vendors,
            "max_trainers": self.max_trainers,
            "max_nodes": self.max_nodes,
            "max_mobs": self.max_mobs,
            "max_waypoints": self.max_waypoints,
            "max_recent_combats": self.max_recent_combats,
        }
        for name, val in limits.items():
            if val < 0:
                raise ValueError(f"{name} must be >= 0, got {val}")


@dataclass(frozen=True)
class PromptBundle:
    text: str
    prompt_hash: str
    schema_version: int
    token_estimate: int
    section_lengths: Mapping[str, int]

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of the PromptBundle."""
        return {
            "text": self.text,
            "prompt_hash": self.prompt_hash,
            "schema_version": self.schema_version,
            "token_estimate": self.token_estimate,
            "section_lengths": dict(self.section_lengths),
        }


ALLOWED_GOALS: tuple[str, ...] = (
    "farm_herbs",
    "grind_humans",
    "explore",
    "flee",
    "sell_vendor",
    "repair",
    "travel_to",
)

GOAL_DESCRIPTIONS: dict[str, str] = {
    "farm_herbs": "Gather herbalism nodes in the nearby area.",
    "grind_humans": "Engage humanoid mobs for XP and loot.",
    "explore": "Move to unexplored or high-priority map areas.",
    "flee": "Disengage and retreat to a safe location or waypoint.",
    "sell_vendor": "Travel to a vendor to clear inventory.",
    "repair": "Visit a trainer or vendor to repair gear.",
    "travel_to": "Move to a specific destination node or waypoint.",
}


def render_schema(config: PromptConfig | None = None) -> str:
    """Render the JSON schema block for response validation.

    Returns ONLY the JSON schema block that appears in the "response_schema"
    section, as a pretty-printed JSON string with sort_keys=True, indent=2,
    and ensure_ascii=False.
    """
    _ = config
    schema_obj = {
        "goal": "<one of the allowed goals>",
        "rationale": "<short string, max 200 chars>",
        "target": "<optional string, e.g. entity id or waypoint id>",
    }
    return json.dumps(schema_obj, indent=2, sort_keys=True, ensure_ascii=False)


def build_prompt(
    meta: MetaStateLike,
    world: WorldSummary,
    state: GameStateView,
    *,
    config: PromptConfig | None = None,
) -> PromptBundle:
    """Build a deterministic, grounded prompt string from meta state, world summary, and game state.

    The text is composed of 8 sections in exact order separated by blank lines ("\\n\\n"):
    1. system
    2. allowed_goals
    3. drives
    4. memory
    5. world
    6. state
    7. response_schema
    8. instructions

    section_lengths mapping rule:
    Split `text` on the double-newline blank line separator ("\\n\\n"). For each section,
    extract the first word of its first line. If that first word is unique across all
    sections, use it as the section key. If not unique, use the section's full first line
    up to the first colon ":" (or full line if no colon).
    """
    cfg = config if config is not None else PromptConfig()

    # 1. system
    system_text = (
        "You are a planning agent for a research prototype operating in an isolated lab environment. "
        "You are NOT a live game assistant. "
        "Your job is to choose the next high-level goal from the allowed list, given the current state and world context. "
        "You MUST reply with a single JSON object and nothing else. "
        f"Prompt schema version: {cfg.schema_version}."
    )

    # 2. allowed_goals
    goal_lines = [
        f"- {goal}: {GOAL_DESCRIPTIONS.get(goal, 'Execute goal behavior.')}"
        for goal in ALLOWED_GOALS
    ]
    allowed_goals_text = "\n".join(goal_lines)

    # 3. drives
    if meta.drives:
        sorted_drives = sorted(meta.drives.items())
        drive_lines = [f"  {k}: {v:.3f}" for k, v in sorted_drives]
        drives_text = "\n".join(drive_lines)
    else:
        drives_text = "  (no drives)"

    # 4. memory
    mem_summary = meta.memory_summary or ""
    if not mem_summary:
        memory_text = "(none)"
    else:
        if len(mem_summary) > cfg.max_memory_chars:
            memory_text = mem_summary[: cfg.max_memory_chars] + "…"
        else:
            memory_text = mem_summary

    # 5. world
    world_sub_blocks: list[str] = []

    if cfg.max_vendors > 0 and world.nearest_vendors:
        vendors = world.nearest_vendors[: cfg.max_vendors]
        if vendors:
            lines = ["vendors:"] + [
                f"  id={n.id} kind={n.kind} d={n.distance:.1f} xy=({n.x:.1f},{n.y:.1f})"
                for n in vendors
            ]
            world_sub_blocks.append("\n".join(lines))

    if cfg.max_trainers > 0 and world.nearest_trainers:
        trainers = world.nearest_trainers[: cfg.max_trainers]
        if trainers:
            lines = ["trainers:"] + [
                f"  id={n.id} kind={n.kind} d={n.distance:.1f} xy=({n.x:.1f},{n.y:.1f})"
                for n in trainers
            ]
            world_sub_blocks.append("\n".join(lines))

    if cfg.max_nodes > 0 and world.nearest_nodes:
        nodes = world.nearest_nodes[: cfg.max_nodes]
        if nodes:
            lines = ["nodes:"] + [
                f"  id={n.id} kind={n.kind} d={n.distance:.1f} xy=({n.x:.1f},{n.y:.1f})"
                for n in nodes
            ]
            world_sub_blocks.append("\n".join(lines))

    if cfg.max_mobs > 0 and world.nearest_mobs:
        mobs = world.nearest_mobs[: cfg.max_mobs]
        if mobs:
            lines = ["mobs:"] + [
                f"  id={n.id} kind={n.kind} d={n.distance:.1f} xy=({n.x:.1f},{n.y:.1f})"
                for n in mobs
            ]
            world_sub_blocks.append("\n".join(lines))

    if cfg.max_waypoints > 0 and world.nearest_waypoints:
        waypoints = world.nearest_waypoints[: cfg.max_waypoints]
        if waypoints:
            lines = ["waypoints:"] + [
                f"  id={n.id} kind={n.kind} d={n.distance:.1f} xy=({n.x:.1f},{n.y:.1f})"
                for n in waypoints
            ]
            world_sub_blocks.append("\n".join(lines))

    if cfg.max_recent_combats > 0 and world.recent_combats:
        combats = world.recent_combats[: cfg.max_recent_combats]
        if combats:
            lines = ["combat:"] + [
                f"  combat_id={c.combat_id} target={c.target_entity_id} outcome={c.outcome} ended={c.ended_at}"
                for c in combats
            ]
            world_sub_blocks.append("\n".join(lines))

    if world_sub_blocks:
        world_text = "\n".join(world_sub_blocks)
    else:
        world_text = "(no world data)"

    # 6. state
    target_str = (
        state.current_target_id if state.current_target_id is not None else "none"
    )
    if state.target_hp_percent is not None:
        target_hp_str = f"{state.target_hp_percent:.1f}%"
    else:
        target_hp_str = "unknown"

    state_lines = [
        f"  fsm_state: {state.fsm_state.value}",
        f"  position: ({state.player_x:.2f},{state.player_y:.2f},{state.player_z:.2f})",
        f"  self_hp: {state.self_hp_percent:.1f}%",
        f"  resource: {state.resource:.1f}/{state.resource_max:.1f}",
        f"  target: {target_str}",
        f"  target_hp: {target_hp_str}",
        f"  inventory: {state.inventory_count}",
        f"  level_or_xp: {state.level_or_xp:.1f}",
    ]

    if cfg.include_raw_state_json:
        raw_dict = {
            "current_target_id": state.current_target_id,
            "fsm_state": state.fsm_state.value,
            "inventory_count": state.inventory_count,
            "level_or_xp": state.level_or_xp,
            "player_x": state.player_x,
            "player_y": state.player_y,
            "player_z": state.player_z,
            "resource": state.resource,
            "resource_max": state.resource_max,
            "self_hp_percent": state.self_hp_percent,
            "target_hp_percent": state.target_hp_percent,
        }
        json_block = json.dumps(raw_dict, indent=2, sort_keys=True, ensure_ascii=False)
        state_lines.append(f"```json\n{json_block}\n```")

    state_text = "\n".join(state_lines)

    # 7. response_schema
    response_schema_text = render_schema(cfg)

    # 8. instructions
    instructions_text = (
        "Reply with a single JSON object matching the response_schema. "
        "Do NOT include any prose outside the JSON. "
        "Do NOT invent goals outside the allowed_goals list. "
        'If no goal is appropriate, reply with {"goal": "explore", "rationale": "fallback"}.'
    )

    sections = [
        system_text,
        allowed_goals_text,
        drives_text,
        memory_text,
        world_text,
        state_text,
        response_schema_text,
        instructions_text,
    ]

    text = "\n\n".join(sections)

    prompt_hash = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
    token_estimate = len(text) // 4

    # Calculate section_lengths
    section_list = text.split("\n\n")
    first_words = [sec.split("\n")[0].strip().split()[0] for sec in section_list]

    section_lengths: dict[str, int] = {}
    for i, sec in enumerate(section_list):
        fw = first_words[i]
        if first_words.count(fw) == 1:
            key = fw
        else:
            first_line = sec.split("\n")[0].strip()
            if ":" in first_line:
                key = first_line.split(":")[0].strip()
            else:
                key = first_line
        section_lengths[key] = len(sec)

    return PromptBundle(
        text=text,
        prompt_hash=prompt_hash,
        schema_version=cfg.schema_version,
        token_estimate=token_estimate,
        section_lengths=MappingProxyType(section_lengths),
    )
