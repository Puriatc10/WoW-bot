"""Runtime context for controller-owned quantities (ADR-002 "internal" class).

`gcd_ready`, `spell_cooldown_ready` and `target_has_debuff` are state owned by
the cooldown subsystem, not by perception: vision cannot observe the Python
controller's own cooldowns. ADR-002 classifies them **internal** and routes
them through a context object supplied at projection time.

This module provides the seam only. The context the lab pipeline uses must be
backed by the real cooldown tracker; ``StaticRuntimeContext`` is a
deterministic test double, never a production default.

Constraints (enforced by tests on the perception package): no file reads, no
global RNG, no LLM, no consumer Protocol imported or inherited.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = ["RuntimeContext", "StaticRuntimeContext"]


@runtime_checkable
class RuntimeContext(Protocol):
    """Structural contract for the controller-owned quantities a view needs."""

    def gcd_ready(self) -> bool:
        """Return True when the global cooldown has elapsed."""
        ...

    def spell_cooldown_ready(self, spell_id: str) -> bool:
        """Return True when the named spell is off cooldown."""
        ...

    def target_has_debuff(self, debuff_id: str) -> bool:
        """Return True when the current target carries the named debuff."""
        ...


@dataclass(frozen=True)
class StaticRuntimeContext:
    """Deterministic context for tests and MOCK_MODE wiring.

    Answers from fixed sets instead of a live cooldown tracker, so it is only
    meaningful for deterministic tests and mock runs. A LAB_MODE run must
    supply a context backed by the real cooldown subsystem before these
    queries can influence live combat decisions.
    """

    gcd_is_ready: bool = True
    ready_spells: frozenset[str] = field(default_factory=frozenset)
    target_debuffs: frozenset[str] = field(default_factory=frozenset)

    def gcd_ready(self) -> bool:
        return self.gcd_is_ready

    def spell_cooldown_ready(self, spell_id: str) -> bool:
        return spell_id in self.ready_spells

    def target_has_debuff(self, debuff_id: str) -> bool:
        return debuff_id in self.target_debuffs
