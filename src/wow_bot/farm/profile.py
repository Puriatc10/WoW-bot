"""Farm profile loader and validator.

Provides a deterministic, validated parser for TOML farm profiles describing
where and how the farm loop operates.
"""

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any


class FarmProfileError(Exception):
    """Raised when a farm profile file or dictionary fails validation or parsing."""


PROFILE_SCHEMA_VERSION: int = 1

ALLOWED_TOP_LEVEL_KEYS: frozenset[str] = frozenset({
    "profile",
    "cycle",
    "route_preferences",
    "metadata",
})

ALLOWED_PROFILE_KEYS: frozenset[str] = frozenset({
    "name",
    "schema_version",
    "description",
})

ALLOWED_CYCLE_KEYS: frozenset[str] = frozenset({
    "nodes",
    "vendor",
    "repair",
    "stop_when_inventory_full",
    "stop_after_cycles",
})

ALLOWED_NODE_REFERENCE_KEYS: frozenset[str] = frozenset({
    "kind",
    "name",
    "behavior_hints",
})

ALLOWED_VENDOR_REFERENCE_KEYS: frozenset[str] = frozenset({
    "kind",
    "name",
})

ALLOWED_ROUTE_PREFERENCES_KEYS: frozenset[str] = frozenset({
    "avoid_kinds",
    "prefer_kinds",
    "max_detour_factor",
})

VALID_NODE_KINDS: frozenset[str] = frozenset({
    "vendor",
    "trainer",
    "node",
    "mob",
    "waypoint",
    "unknown",
})

VALID_VENDOR_KINDS: frozenset[str] = frozenset({
    "vendor",
    "trainer",
})


def _is_valid_name(name: Any) -> bool:
    """Check if name is a non-empty string without edge whitespace or internal newlines."""
    if not isinstance(name, str) or not name:
        return False
    if name.strip() != name:
        return False
    return not ("\n" in name or "\r" in name)


@dataclass(frozen=True)
class NodeReference:
    """Reference to a World Model node kind + name pair with optional behavior hints."""

    kind: str
    name: str
    behavior_hints: Mapping[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if self.kind not in VALID_NODE_KINDS:
            raise ValueError(
                f"Invalid node reference kind '{self.kind}'. Must be one of {sorted(VALID_NODE_KINDS)}"
            )
        if not _is_valid_name(self.name):
            raise ValueError(
                f"Invalid node reference name '{self.name}'. Must be non-empty, stripped, without newlines."
            )
        if not isinstance(self.behavior_hints, Mapping):
            raise TypeError("behavior_hints must be a Mapping")
        for k, v in self.behavior_hints.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise TypeError("behavior_hints keys and values must be strings")


@dataclass(frozen=True)
class VendorReference:
    """Reference to a vendor or trainer for selling, buying, or repairing."""

    kind: str
    name: str

    def __post_init__(self) -> None:
        if self.kind not in VALID_VENDOR_KINDS:
            raise ValueError(
                f"Invalid vendor reference kind '{self.kind}'. Must be one of {sorted(VALID_VENDOR_KINDS)}"
            )
        if not _is_valid_name(self.name):
            raise ValueError(
                f"Invalid vendor reference name '{self.name}'. Must be non-empty, stripped, without newlines."
            )


@dataclass(frozen=True)
class RoutePreferences:
    """Navigation preferences and constraints for farming route pathfinding."""

    avoid_kinds: tuple[str, ...] = ()
    prefer_kinds: tuple[str, ...] = ()
    max_detour_factor: float = 1.5

    def __post_init__(self) -> None:
        if not isinstance(self.avoid_kinds, tuple):
            raise TypeError("avoid_kinds must be a tuple")
        if not isinstance(self.prefer_kinds, tuple):
            raise TypeError("prefer_kinds must be a tuple")
        for kind in self.avoid_kinds:
            if kind not in VALID_NODE_KINDS:
                raise ValueError(f"Invalid avoid_kind '{kind}'")
        for kind in self.prefer_kinds:
            if kind not in VALID_NODE_KINDS:
                raise ValueError(f"Invalid prefer_kind '{kind}'")
        if set(self.avoid_kinds) & set(self.prefer_kinds):
            raise ValueError("avoid_kinds and prefer_kinds must be disjoint")
        if (
            isinstance(self.max_detour_factor, bool)
            or not isinstance(self.max_detour_factor, (int, float))
        ):
            raise TypeError("max_detour_factor must be a float")
        detour = float(self.max_detour_factor)
        if not math.isfinite(detour) or detour < 1.0:
            raise ValueError("max_detour_factor must be finite and >= 1.0")


@dataclass(frozen=True)
class CycleSpec:
    """High-level farming loop cycle specification."""

    nodes: tuple[NodeReference, ...]
    vendor: VendorReference
    repair: VendorReference
    stop_when_inventory_full: bool
    stop_after_cycles: int

    def __post_init__(self) -> None:
        if not isinstance(self.nodes, tuple) or len(self.nodes) == 0:
            raise TypeError("nodes must be a non-empty tuple")
        for node in self.nodes:
            if not isinstance(node, NodeReference):
                raise TypeError("All elements in nodes must be NodeReference instances")
        if not isinstance(self.vendor, VendorReference):
            raise TypeError("vendor must be a VendorReference instance")
        if not isinstance(self.repair, VendorReference):
            raise TypeError("repair must be a VendorReference instance")
        if not isinstance(self.stop_when_inventory_full, bool):
            raise TypeError("stop_when_inventory_full must be a bool")
        if (
            isinstance(self.stop_after_cycles, bool)
            or not isinstance(self.stop_after_cycles, int)
            or self.stop_after_cycles < 0
        ):
            raise TypeError("stop_after_cycles must be an integer >= 0")


@dataclass(frozen=True)
class FarmProfile:
    """Complete static farm profile specification."""

    schema_version: int
    name: str
    description: str
    cycle: CycleSpec
    route_preferences: RoutePreferences
    metadata: Mapping[str, str]

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != PROFILE_SCHEMA_VERSION
        ):
            raise TypeError(f"schema_version must equal {PROFILE_SCHEMA_VERSION}")
        if not _is_valid_name(self.name):
            raise ValueError(
                f"Invalid profile name '{self.name}'. Must be non-empty, stripped, without newlines."
            )
        if not isinstance(self.description, str):
            raise TypeError("description must be a string")
        if not isinstance(self.cycle, CycleSpec):
            raise TypeError("cycle must be a CycleSpec instance")
        if not isinstance(self.route_preferences, RoutePreferences):
            raise TypeError("route_preferences must be a RoutePreferences instance")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a Mapping")
        for k, v in self.metadata.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise TypeError("metadata keys and values must be strings")


def _parse_and_validate_dict(data: dict[str, Any]) -> FarmProfile:
    """Internal helper parsing and validating a profile dict into FarmProfile."""
    if not isinstance(data, dict):
        raise FarmProfileError("Root of farm profile must be a table (dict)")

    # 1. Top-level keys validation
    for key in data:
        if key not in ALLOWED_TOP_LEVEL_KEYS:
            raise FarmProfileError(f"Unknown top-level key: '{key}'")

    if "profile" not in data:
        raise FarmProfileError("Missing required top-level section: 'profile'")
    if "cycle" not in data:
        raise FarmProfileError("Missing required top-level section: 'cycle'")

    # 2. Validate [profile] section
    profile_sec = data["profile"]
    if not isinstance(profile_sec, dict):
        raise FarmProfileError("[profile] section must be a table (dict)")

    for key in profile_sec:
        if key not in ALLOWED_PROFILE_KEYS:
            raise FarmProfileError(f"Unknown key in [profile]: '{key}'")

    if "schema_version" not in profile_sec:
        raise FarmProfileError("Missing required key 'schema_version' in [profile]")
    schema_ver = profile_sec["schema_version"]
    if isinstance(schema_ver, bool) or not isinstance(schema_ver, int):
        raise FarmProfileError("'schema_version' in [profile] must be an integer")
    if schema_ver != PROFILE_SCHEMA_VERSION:
        raise FarmProfileError(
            f"Unsupported schema_version {schema_ver}. Expected {PROFILE_SCHEMA_VERSION}"
        )

    if "name" not in profile_sec:
        raise FarmProfileError("Missing required key 'name' in [profile]")
    name_val = profile_sec["name"]
    if not _is_valid_name(name_val):
        raise FarmProfileError(
            f"Invalid 'name' in [profile]: {name_val!r}. Must be non-empty, stripped, without newlines."
        )

    description_val = profile_sec.get("description", "")
    if not isinstance(description_val, str):
        raise FarmProfileError("'description' in [profile] must be a string")

    # 3. Validate [cycle] section
    cycle_sec = data["cycle"]
    if not isinstance(cycle_sec, dict):
        raise FarmProfileError("[cycle] section must be a table (dict)")

    for key in cycle_sec:
        if key not in ALLOWED_CYCLE_KEYS:
            raise FarmProfileError(f"Unknown key in [cycle]: '{key}'")

    if "nodes" not in cycle_sec:
        raise FarmProfileError("Missing required key 'nodes' in [cycle]")
    raw_nodes = cycle_sec["nodes"]
    if not isinstance(raw_nodes, list) or len(raw_nodes) == 0:
        raise FarmProfileError("'nodes' in [cycle] must be a non-empty list")

    parsed_nodes: list[NodeReference] = []
    for idx, raw_node in enumerate(raw_nodes):
        if not isinstance(raw_node, dict):
            raise FarmProfileError(f"Node entry at index {idx} in 'nodes' must be a table (dict)")
        for nkey in raw_node:
            if nkey not in ALLOWED_NODE_REFERENCE_KEYS:
                raise FarmProfileError(
                    f"Unknown key in node reference at index {idx}: '{nkey}'"
                )
        if "kind" not in raw_node:
            raise FarmProfileError(f"Missing required key 'kind' in node reference at index {idx}")
        if "name" not in raw_node:
            raise FarmProfileError(f"Missing required key 'name' in node reference at index {idx}")

        raw_hints = raw_node.get("behavior_hints")
        if raw_hints is not None:
            if not isinstance(raw_hints, dict):
                raise FarmProfileError(
                    f"'behavior_hints' in node reference at index {idx} must be a table (dict)"
                )
            for hk, hv in raw_hints.items():
                if not isinstance(hk, str) or not isinstance(hv, str):
                    raise FarmProfileError(
                        f"'behavior_hints' keys and values at index {idx} must be strings"
                    )
            behavior_hints: Mapping[str, str] = MappingProxyType(raw_hints)
        else:
            behavior_hints = MappingProxyType({})

        try:
            node_ref = NodeReference(
                kind=raw_node["kind"],
                name=raw_node["name"],
                behavior_hints=behavior_hints,
            )
        except (ValueError, TypeError) as exc:
            raise FarmProfileError(f"Invalid node reference at index {idx}: {exc}") from exc
        parsed_nodes.append(node_ref)

    if "vendor" not in cycle_sec:
        raise FarmProfileError("Missing required key 'vendor' in [cycle]")
    raw_vendor = cycle_sec["vendor"]
    if not isinstance(raw_vendor, dict):
        raise FarmProfileError("'vendor' in [cycle] must be a table (dict)")
    for vkey in raw_vendor:
        if vkey not in ALLOWED_VENDOR_REFERENCE_KEYS:
            raise FarmProfileError(f"Unknown key in 'vendor' reference: '{vkey}'")
    if "kind" not in raw_vendor:
        raise FarmProfileError("Missing required key 'kind' in 'vendor' reference")
    if "name" not in raw_vendor:
        raise FarmProfileError("Missing required key 'name' in 'vendor' reference")

    try:
        vendor_ref = VendorReference(kind=raw_vendor["kind"], name=raw_vendor["name"])
    except (ValueError, TypeError) as exc:
        raise FarmProfileError(f"Invalid 'vendor' reference: {exc}") from exc

    if "repair" in cycle_sec:
        raw_repair = cycle_sec["repair"]
        if not isinstance(raw_repair, dict):
            raise FarmProfileError("'repair' in [cycle] must be a table (dict)")
        for rkey in raw_repair:
            if rkey not in ALLOWED_VENDOR_REFERENCE_KEYS:
                raise FarmProfileError(f"Unknown key in 'repair' reference: '{rkey}'")
        if "kind" not in raw_repair:
            raise FarmProfileError("Missing required key 'kind' in 'repair' reference")
        if "name" not in raw_repair:
            raise FarmProfileError("Missing required key 'name' in 'repair' reference")

        try:
            repair_ref = VendorReference(kind=raw_repair["kind"], name=raw_repair["name"])
        except (ValueError, TypeError) as exc:
            raise FarmProfileError(f"Invalid 'repair' reference: {exc}") from exc
    else:
        repair_ref = vendor_ref

    stop_inv_full = cycle_sec.get("stop_when_inventory_full", True)
    if not isinstance(stop_inv_full, bool):
        raise FarmProfileError("'stop_when_inventory_full' in [cycle] must be a boolean")

    stop_cycles = cycle_sec.get("stop_after_cycles", 0)
    if isinstance(stop_cycles, bool) or not isinstance(stop_cycles, int) or stop_cycles < 0:
        raise FarmProfileError("'stop_after_cycles' in [cycle] must be an integer >= 0")

    try:
        cycle_spec = CycleSpec(
            nodes=tuple(parsed_nodes),
            vendor=vendor_ref,
            repair=repair_ref,
            stop_when_inventory_full=stop_inv_full,
            stop_after_cycles=stop_cycles,
        )
    except (ValueError, TypeError) as exc:
        raise FarmProfileError(f"Invalid cycle specification: {exc}") from exc

    # 4. Validate optional [route_preferences] section
    if "route_preferences" in data:
        route_sec = data["route_preferences"]
        if not isinstance(route_sec, dict):
            raise FarmProfileError("[route_preferences] section must be a table (dict)")

        for rkey in route_sec:
            if rkey not in ALLOWED_ROUTE_PREFERENCES_KEYS:
                raise FarmProfileError(f"Unknown key in [route_preferences]: '{rkey}'")

        avoid_kinds_raw = route_sec.get("avoid_kinds", [])
        if not isinstance(avoid_kinds_raw, list) or not all(
            isinstance(x, str) for x in avoid_kinds_raw
        ):
            raise FarmProfileError("'avoid_kinds' in [route_preferences] must be a list of strings")

        prefer_kinds_raw = route_sec.get("prefer_kinds", [])
        if not isinstance(prefer_kinds_raw, list) or not all(
            isinstance(x, str) for x in prefer_kinds_raw
        ):
            raise FarmProfileError(
                "'prefer_kinds' in [route_preferences] must be a list of strings"
            )

        detour_raw = route_sec.get("max_detour_factor", 1.5)
        if isinstance(detour_raw, bool) or not isinstance(detour_raw, (int, float)):
            raise FarmProfileError(
                "'max_detour_factor' in [route_preferences] must be a float"
            )

        try:
            route_prefs = RoutePreferences(
                avoid_kinds=tuple(avoid_kinds_raw),
                prefer_kinds=tuple(prefer_kinds_raw),
                max_detour_factor=float(detour_raw),
            )
        except (ValueError, TypeError) as exc:
            raise FarmProfileError(f"Invalid [route_preferences]: {exc}") from exc
    else:
        route_prefs = RoutePreferences()

    # 5. Validate optional [metadata] section
    if "metadata" in data:
        meta_sec = data["metadata"]
        if not isinstance(meta_sec, dict):
            raise FarmProfileError("[metadata] section must be a table (dict)")

        for mk, mv in meta_sec.items():
            if not isinstance(mk, str) or not isinstance(mv, str):
                raise FarmProfileError("[metadata] keys and values must all be strings")
        meta_dict: Mapping[str, str] = MappingProxyType(meta_sec)
    else:
        meta_dict = MappingProxyType({})

    try:
        return FarmProfile(
            schema_version=schema_ver,
            name=name_val,
            description=description_val,
            cycle=cycle_spec,
            route_preferences=route_prefs,
            metadata=meta_dict,
        )
    except (ValueError, TypeError) as exc:
        raise FarmProfileError(f"Invalid profile construction: {exc}") from exc


def validate_profile_dict(data: dict[str, Any]) -> None:
    """Validate that `data` represents a valid farm profile without mutating `data`.

    Raises FarmProfileError if any validation check fails.
    """
    _parse_and_validate_dict(data)


def load_profile(path: Path) -> FarmProfile:
    """Load and validate a farm profile TOML file at `path`.

    Raises FarmProfileError if the file does not exist, is invalid TOML,
    or fails profile schema validation.
    """
    if not isinstance(path, Path):
        path = Path(path)

    if not path.is_file():
        raise FarmProfileError(f"Farm profile file not found at path: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        raise FarmProfileError(f"Failed to read farm profile file at {path}: {exc}") from exc

    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise FarmProfileError(f"TOML parse error in profile at {path}: {exc}") from exc

    return _parse_and_validate_dict(data)


def profile_summary(profile: FarmProfile) -> dict[str, Any]:
    """Return a JSON-serializable dict summarizing `profile` for logging."""
    return {
        "schema_version": profile.schema_version,
        "name": profile.name,
        "node_count": len(profile.cycle.nodes),
        "vendor": {
            "kind": profile.cycle.vendor.kind,
            "name": profile.cycle.vendor.name,
        },
        "repair": {
            "kind": profile.cycle.repair.kind,
            "name": profile.cycle.repair.name,
        },
        "stop_when_inventory_full": profile.cycle.stop_when_inventory_full,
        "stop_after_cycles": profile.cycle.stop_after_cycles,
        "route_preferences": {
            "avoid_kinds": list(profile.route_preferences.avoid_kinds),
            "prefer_kinds": list(profile.route_preferences.prefer_kinds),
            "max_detour_factor": profile.route_preferences.max_detour_factor,
        },
    }
