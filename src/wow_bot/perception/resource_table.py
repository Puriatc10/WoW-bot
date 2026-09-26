"""Class-and-level to maximum-resource lookup for the derived ``resource_max``.

``resource_max`` is a game-data constant, not a per-frame observation, so no
vision channel can produce it: `docs/PERCEPTION.md` specifies only normalized
``[0,1]`` bar output, which cannot yield a maximum. ADR-002 classifies it
**derived**, computes it here, and never stores it on ``GameState``.

Provenance status: **the real values are not in this repository.** ADR-002
records transcribing them from lab-server game data as an unresolved question,
and no source in the repo contains a class/level resource table. Rather than
invent numbers, the default table is empty and every lookup returns ``None``.
The mechanism is implemented and tested with an injected table; supplying real
entries is a lab-server data task.

No file reads: the table is constructed in memory by its caller.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

__all__ = ["EMPTY_RESOURCE_TABLE", "ResourceTable"]


@dataclass(frozen=True)
class ResourceTable:
    """Lookup from ``(character class, level)`` to maximum resource points."""

    entries: Mapping[tuple[str, int], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key, value in self.entries.items():
            char_class, level = key
            if not char_class:
                raise ValueError("resource table keys require a non-empty class name")
            if level <= 0:
                raise ValueError(f"resource table levels must be positive, got {level}")
            if value <= 0.0:
                raise ValueError(f"resource table values must be positive, got {value}")

    def max_resource(self, char_class: str | None, level: int | None) -> float | None:
        """Return the maximum resource for the pair, or ``None`` when unknown.

        ``None`` means "not determinable": the class or level was not observed,
        or the table holds no entry for the pair. A miss is never turned into a
        guessed or default value, because a wrong maximum silently rescales
        every resource-dependent consumer decision.
        """
        if char_class is None or level is None:
            return None
        return self.entries.get((char_class, level))


EMPTY_RESOURCE_TABLE = ResourceTable()
