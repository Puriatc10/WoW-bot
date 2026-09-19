"""MetaState Generator module (Task 3.5).

Coordinates the Internal Dynamics subsystem components for each simulation step:
  1. LorenzAttractor (chaos)
  2. OscillatorBank (coupled slow oscillations)
  3. Drives (five physiological/psychological drive levels)
  4. MemoryStore (SQLite persistence of events and drive state vectors)

Conceptually:
  GameState → LorenzAttractor → OscillatorBank → Drives → events + decay + MemoryStore → MetaState

Also provides the adaptive decision rule ``should_trigger_llm`` for
determining whether the LLM Strategist should be triggered based on vector Euclidean change
relative to a slow deterministic adaptive threshold.

Invariants & Behavior:
  - Injected dependencies: MetaStateGenerator coordinates collaborators but does NOT own
    their lifecycles or instantiate them.
  - dt < 0 raises ValueError before mutating any collaborators or internal time state.
  - Deterministic execution sequence:
      validate dt → advance simulated time → chaos step → chaos normalization →
      oscillator step → drives step → apply events → decay → persist events → construct MetaState
  - MetaState timestamp strictly uses game_state.timestamp (no time.time()).
  - MetaState.vector strictly matches drives.vector (canonical 5-drive vector).
  - Trigger threshold oscillates slowly according to threshold(t) = base + 0.1 * sin(2π * t / 7200).
  - trigger rule compares Euclidean distance of drive vectors against trigger_threshold using strict (>).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from wow_bot.internal_dynamics.chaos import LorenzAttractor
from wow_bot.internal_dynamics.drives import Drives
from wow_bot.internal_dynamics.memory import MemoryStore
from wow_bot.internal_dynamics.oscillators import OscillatorBank
from wow_bot.shared.config import InternalDynamicsConfig, Settings
from wow_bot.shared.interfaces import GameState, MetaState


class MetaStateGenerator:
    """Coordinator for Internal Dynamics simulation steps and LLM trigger decisions."""

    def __init__(
        self,
        config: InternalDynamicsConfig | Settings | dict[str, Any] | float | None,
        drives: Drives,
        oscillators: OscillatorBank,
        chaos: LorenzAttractor,
        memory: MemoryStore,
    ) -> None:
        """Initialize the MetaStateGenerator with injected dependencies and config.

        Args:
            config: Internal dynamics configuration, Settings object, dict, threshold float, or None.
            drives: Drives instance.
            oscillators: OscillatorBank instance.
            chaos: LorenzAttractor instance.
            memory: MemoryStore instance.

        Raises:
            ValueError: If trigger_threshold_base is not positive and finite.
        """
        self._drives = drives
        self._oscillators = oscillators
        self._chaos = chaos
        self._memory = memory
        self._last_oscillator_value: float = 0.0

        # Extract trigger_threshold_base from config
        threshold: float = 0.3
        if isinstance(config, (float, int)):
            threshold = float(config)
        elif isinstance(config, Settings):
            threshold = float(config.internal_dynamics.trigger_threshold_base)
        elif isinstance(config, InternalDynamicsConfig):
            threshold = float(config.trigger_threshold_base)
        elif isinstance(config, dict):
            threshold = float(config.get("trigger_threshold_base", 0.3))

        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError(
                f"trigger_threshold_base must be a positive finite number, got {threshold}"
            )

        self._trigger_threshold_base: float = threshold
        self._elapsed_seconds: float = 0.0

    @property
    def trigger_threshold_base(self) -> float:
        """Return the base trigger threshold value."""
        return self._trigger_threshold_base

    @property
    def trigger_threshold(self) -> float:
        """Return the current calculated adaptive trigger threshold.

        Threshold formula: base + 0.1 * sin(2 * pi * (elapsed_seconds % 7200.0) / 7200.0).
        Pure read-only accessor that does not alter state.
        """
        phase = 2.0 * math.pi * (self._elapsed_seconds % 7200.0) / 7200.0
        return self._trigger_threshold_base + 0.1 * math.sin(phase)

    async def step(
        self,
        dt: float,
        game_state: GameState,
    ) -> MetaState:
        """Advance internal dynamics for one simulation step and return MetaState.

        Args:
            dt: Simulation time step in seconds (must be >= 0.0).
            game_state: Authoritative GameState snapshot for the step.

        Returns:
            MetaState snapshot capturing current drive vector, recent events, and timestamp.

        Raises:
            ValueError: If dt < 0.0.
        """
        # Step 1 — validate dt BEFORE mutating any collaborator or time state
        if dt < 0.0:
            raise ValueError(f"dt must be non-negative, got {dt}")

        # Advance internal simulated time
        self._elapsed_seconds += dt

        # Step 2 — advance chaos
        self._chaos.step()
        chaos_value = self._chaos.normalized()

        # Step 3 — advance oscillators
        oscillator_value = self._oscillators.step(dt)
        self._last_oscillator_value = oscillator_value

        # Step 4 — advance Drives
        self._drives.step(dt, chaos_component=chaos_value)

        # Step 5 — apply GameState events in order
        for event in game_state.events:
            self._drives.apply_event(event)

        # Step 6 — decay Drives towards baseline
        self._drives.decay(dt)

        # Step 7 — persist events to memory with resulting 5D drive vector
        current_vector = self._drives.vector
        for event in game_state.events:
            await self._memory.add(event, current_vector)

        # Step 8 — create MetaState snapshot with isolated vector and events list
        return MetaState(
            vector=current_vector.copy(),
            recent_events=list(game_state.events),
            timestamp=float(game_state.timestamp),
        )

    def should_trigger_llm(
        self,
        current: MetaState,
        last: MetaState,
    ) -> bool:
        """Evaluate adaptive decision rule for triggering the LLM Strategist.

        Triggers when Euclidean norm of (current.vector - last.vector) > current adaptive trigger_threshold.

        Args:
            current: Current step's MetaState.
            last: Previous step's MetaState.

        Returns:
            True if Euclidean distance > trigger_threshold, False otherwise.
        """
        curr_vec = np.asarray(current.vector, dtype=np.float64)
        last_vec = np.asarray(last.vector, dtype=np.float64)

        if curr_vec.shape != (5,) or last_vec.shape != (5,):
            raise ValueError(
                f"MetaState vectors must have shape (5,), got {curr_vec.shape} and {last_vec.shape}"
            )

        if not (np.all(np.isfinite(curr_vec)) and np.all(np.isfinite(last_vec))):
            raise ValueError("MetaState vectors must contain finite values")

        delta = float(np.linalg.norm(curr_vec - last_vec))
        return bool(delta > self.trigger_threshold)
