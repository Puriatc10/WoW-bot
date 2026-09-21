# Pre-lab (MOCK_MODE)
"""Scenario data structures and JSON loader for lab movement runs."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALLOWED_TOP_LEVEL_KEYS = {
    "name",
    "waypoints",
    "arrival_tolerance",
    "max_steps_per_waypoint",
    "step_interval_s",
    "max_session_seconds",
}


class ScenarioError(Exception):
    """Raised when scenario loading, parsing, or validation fails."""


@dataclass(frozen=True)
class Scenario:
    """Immutable scenario specification for movement runs."""

    name: str
    waypoints: tuple[tuple[float, float], ...]
    arrival_tolerance: float = 0.5
    max_steps_per_waypoint: int = 200
    step_interval_s: float = 0.1
    max_session_seconds: float = 60.0


def validate_waypoints(
    waypoints: tuple[tuple[float, float], ...],
) -> None:
    """Validate that waypoints tuple is non-empty and contains valid 2D coordinates.

    Raises ScenarioError if waypoints is empty, not a sequence of 2-element tuples,
    or contains non-finite numbers (NaN or Inf).
    """
    if not isinstance(waypoints, tuple):
        raise ScenarioError("Waypoints must be a tuple")

    if not waypoints:
        raise ScenarioError("Waypoints cannot be empty")

    for i, wp in enumerate(waypoints):
        if not isinstance(wp, tuple) or len(wp) != 2:
            raise ScenarioError(f"Waypoint at index {i} must be a 2-element tuple, got {wp!r}")

        for j, coord in enumerate(wp):
            if isinstance(coord, bool) or not isinstance(coord, (int, float)):
                raise ScenarioError(f"Waypoint at index {i} coordinate {j} must be a number, got {coord!r}")
            if not math.isfinite(coord):
                raise ScenarioError(f"Waypoint at index {i} coordinate {j} must be finite, got {coord!r}")


def load_scenario(path: Path) -> Scenario:
    """Read and validate a scenario specification from a JSON file.

    Raises ScenarioError on missing file, invalid JSON, unknown keys,
    missing required fields, or invalid numeric values/types.
    """
    if not isinstance(path, Path):
        path = Path(path)

    if not path.exists() or not path.is_file():
        raise ScenarioError(f"Scenario file does not exist: '{path}'")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ScenarioError(f"Invalid JSON in scenario file '{path}': {e}") from e
    except OSError as e:
        raise ScenarioError(f"Error reading scenario file '{path}': {e}") from e

    if not isinstance(data, dict):
        raise ScenarioError(f"Scenario root must be a JSON object, got {type(data).__name__}")

    unknown_keys = set(data.keys()) - ALLOWED_TOP_LEVEL_KEYS
    if unknown_keys:
        raise ScenarioError(f"Unknown top-level key(s) in scenario: {sorted(unknown_keys)}")

    if "name" not in data:
        raise ScenarioError("Missing required key 'name' in scenario")
    if "waypoints" not in data:
        raise ScenarioError("Missing required key 'waypoints' in scenario")

    name = data["name"]
    if not isinstance(name, str) or not name.strip():
        raise ScenarioError(f"'name' must be a non-empty string, got {name!r}")

    waypoints_raw = data["waypoints"]
    if not isinstance(waypoints_raw, (list, tuple)):
        raise ScenarioError(f"'waypoints' must be a list/tuple, got {type(waypoints_raw).__name__}")

    parsed_waypoints: list[tuple[float, float]] = []
    for i, wp in enumerate(waypoints_raw):
        if not isinstance(wp, (list, tuple)):
            raise ScenarioError(f"Waypoint at index {i} must be a 2-element list/tuple, got {type(wp).__name__}")
        if len(wp) != 2:
            raise ScenarioError(f"Waypoint at index {i} must be exactly 2 elements, got length {len(wp)}")

        c1, c2 = wp[0], wp[1]
        if isinstance(c1, bool) or not isinstance(c1, (int, float)):
            raise ScenarioError(f"Waypoint at index {i} coordinate 0 must be a number, got {c1!r}")
        if isinstance(c2, bool) or not isinstance(c2, (int, float)):
            raise ScenarioError(f"Waypoint at index {i} coordinate 1 must be a number, got {c2!r}")

        f1, f2 = float(c1), float(c2)
        if not math.isfinite(f1) or not math.isfinite(f2):
            raise ScenarioError(f"Waypoint at index {i} coordinates must be finite numbers, got ({f1}, {f2})")

        parsed_waypoints.append((f1, f2))

    waypoints_tuple = tuple(parsed_waypoints)
    validate_waypoints(waypoints_tuple)

    # Optional fields with default values
    kwargs: dict[str, Any] = {
        "name": name,
        "waypoints": waypoints_tuple,
    }

    if "arrival_tolerance" in data:
        val = data["arrival_tolerance"]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ScenarioError(f"'arrival_tolerance' must be a number, got {val!r}")
        f_val = float(val)
        if not math.isfinite(f_val) or f_val <= 0:
            raise ScenarioError(f"'arrival_tolerance' must be > 0, got {f_val!r}")
        kwargs["arrival_tolerance"] = f_val

    if "max_steps_per_waypoint" in data:
        val = data["max_steps_per_waypoint"]
        if isinstance(val, bool) or not isinstance(val, int):
            raise ScenarioError(f"'max_steps_per_waypoint' must be an integer, got {val!r}")
        if val < 1:
            raise ScenarioError(f"'max_steps_per_waypoint' must be >= 1, got {val!r}")
        kwargs["max_steps_per_waypoint"] = val

    if "step_interval_s" in data:
        val = data["step_interval_s"]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ScenarioError(f"'step_interval_s' must be a number, got {val!r}")
        f_val = float(val)
        if not math.isfinite(f_val) or f_val < 0:
            raise ScenarioError(f"'step_interval_s' must be >= 0, got {f_val!r}")
        kwargs["step_interval_s"] = f_val

    if "max_session_seconds" in data:
        val = data["max_session_seconds"]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ScenarioError(f"'max_session_seconds' must be a number, got {val!r}")
        f_val = float(val)
        if not math.isfinite(f_val) or f_val <= 0:
            raise ScenarioError(f"'max_session_seconds' must be > 0, got {f_val!r}")
        kwargs["max_session_seconds"] = f_val

    return Scenario(**kwargs)
