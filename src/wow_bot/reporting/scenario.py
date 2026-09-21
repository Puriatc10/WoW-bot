"""# Pre-lab (MOCK_MODE)
Scenario execution reporting and passive instrumentation (Task 7.2).

Provides structured metrics collection, report schema validation, and atomic JSON
report serialization for research scenario execution.

Public API:
    - :data:`SCENARIO_REPORT_SCHEMA_VERSION`
    - :data:`META_STATE_DIMENSIONS`
    - :class:`PipelineObserver` (DEPRECATED: see REPORTING_RECONCILIATION.md)
    - :class:`ScenarioReportCollector` (DEPRECATED: see REPORTING_RECONCILIATION.md)
    - :func:`validate_report_dict`
    - :func:`write_report_atomically`
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from wow_bot.executor.humanize import human_delay
from wow_bot.shared.interfaces import GameState, MetaState, Strategy

SCENARIO_REPORT_SCHEMA_VERSION: int = 1
META_STATE_DIMENSIONS: tuple[str, ...] = (
    "hunger",
    "fatigue",
    "curiosity",
    "aggression",
    "social",
)


@runtime_checkable
class PipelineObserver(Protocol):
    """Protocol for passive runtime pipeline instrumentation.

    DEPRECATED: Pre-lab scenario observer protocol. See REPORTING_RECONCILIATION.md.
    """

    def on_game_state(self, game_state: GameState) -> None: ...
    def on_meta_state(self, meta_state: MetaState) -> None: ...
    def on_strategy_attempt(self, sim_ts: float) -> None: ...
    def on_strategy_accepted(self, strategy: Strategy, sim_ts: float) -> None: ...
    def on_strategy_fallback(self, sim_ts: float) -> None: ...
    def on_fsm_transition(
        self, sim_ts: float, from_state: str, to_state: str, reason: str
    ) -> None: ...
    def on_fsm_tick(self, sim_ts: float, fatigue: float) -> None: ...
    def on_idle_intent(self, sim_ts: float, behavior_name: str) -> None: ...
    def on_death_event(self, sim_ts: float) -> None: ...
    def on_progress_step(self) -> None: ...


@dataclass
class ScenarioReportCollector:
    """Passively collects pipeline execution events and constructs scenario research reports.

    DEPRECATED: Pre-lab in-memory event collector. See REPORTING_RECONCILIATION.md.
    """

    scenario: str
    seed: int
    requested_duration_seconds: float
    base_ms: int = 200

    game_state_count: int = 0
    meta_state_count: int = 0
    progress_steps: int = 0

    generation_attempts: int = 0
    new_strategies: int = 0
    fallbacks: int = 0
    goals: dict[str, int] = field(default_factory=dict)
    regions: dict[str, int] = field(default_factory=dict)

    fsm_transition_count: int = 0
    state_entry_counts: dict[str, int] = field(default_factory=dict)
    fsm_transitions: list[dict[str, Any]] = field(default_factory=list)

    idle_intent_count: int = 0
    idle_intent_counts_by_type: dict[str, int] = field(default_factory=dict)

    death_event_count: int = 0
    death_timestamps: list[float] = field(default_factory=list)

    meta_state_samples: list[dict[str, Any]] = field(default_factory=list)
    timing_samples_ms: list[float] = field(default_factory=list)
    timing_samples: list[dict[str, Any]] = field(default_factory=list)

    health_transition_count: int = 0
    shutdown_requested: bool = False

    def __post_init__(self) -> None:
        # Derived deterministic timing RNG seed
        timing_seed = (self.seed + 1000) & 0x7FFFFFFF
        self._timing_rng = np.random.default_rng(timing_seed)
        self._last_meta_ts: float | None = None

    def on_game_state(self, game_state: GameState) -> None:
        """Record game state observation."""
        self.game_state_count += 1

    def on_meta_state(self, meta_state: MetaState) -> None:
        """Record MetaState time series sample."""
        self.meta_state_count += 1
        ts = float(meta_state.timestamp)
        vec = [float(v) for v in meta_state.vector]
        self.meta_state_samples.append({
            "simulation_timestamp": ts,
            "vector": vec,
        })

    def on_strategy_attempt(self, sim_ts: float) -> None:
        """Record Strategist query generation attempt."""
        self.generation_attempts += 1

    def on_strategy_accepted(self, strategy: Strategy, sim_ts: float) -> None:
        """Record genuinely new accepted Strategy."""
        self.new_strategies += 1
        goal = str(strategy.goal)
        region = str(strategy.region)
        self.goals[goal] = self.goals.get(goal, 0) + 1
        self.regions[region] = self.regions.get(region, 0) + 1

    def on_strategy_fallback(self, sim_ts: float) -> None:
        """Record Strategist generation fallback."""
        self.fallbacks += 1

    def on_fsm_transition(
        self, sim_ts: float, from_state: str, to_state: str, reason: str
    ) -> None:
        """Record FSM state transition."""
        self.fsm_transition_count += 1
        self.state_entry_counts[to_state] = self.state_entry_counts.get(to_state, 0) + 1
        self.fsm_transitions.append({
            "simulation_timestamp": float(sim_ts),
            "from_state": str(from_state),
            "to_state": str(to_state),
            "reason": str(reason),
        })

    def on_fsm_tick(
        self,
        sim_ts: float,
        fatigue: float,
        chaos_component: float = 0.0,
        base_ms: int | None = None,
    ) -> None:
        """Sample synthetic human timing delay for an FSM tick."""
        used_base_ms = self.base_ms if base_ms is None else base_ms
        sample = human_delay(
            base_ms=used_base_ms,
            fatigue=fatigue,
            chaos_component=chaos_component,
            rng=self._timing_rng,
        )
        delay_float = float(sample)
        self.timing_samples_ms.append(delay_float)
        self.timing_samples.append({
            "simulation_timestamp": float(sim_ts),
            "delay_ms": delay_float,
            "base_ms": int(used_base_ms),
            "fatigue": float(fatigue),
            "chaos_component": float(chaos_component),
        })

    def on_idle_intent(self, sim_ts: float, behavior_name: str) -> None:
        """Record symbolic idle behavior intent."""
        self.idle_intent_count += 1
        self.idle_intent_counts_by_type[behavior_name] = (
            self.idle_intent_counts_by_type.get(behavior_name, 0) + 1
        )

    def on_death_event(self, sim_ts: float) -> None:
        """Record explicit canonical death event."""
        self.death_event_count += 1
        self.death_timestamps.append(float(sim_ts))

    def on_progress_step(self) -> None:
        """Record pipeline progress token advancement."""
        self.progress_steps += 1

    def build_report(
        self,
        *,
        completed_duration_seconds: float,
        completed_normally: bool,
        error_type: str | None = None,
    ) -> dict[str, Any]:
        """Construct the complete structured report dictionary."""
        # Calculate timing summary
        timing_summary: dict[str, Any] = {
            "unit": "ms",
            "model": "lognormal_simulation",
            "samples_ms": self.timing_samples_ms,
            "samples": self.timing_samples,
            "count": len(self.timing_samples_ms),
            "mean_ms": float(np.mean(self.timing_samples_ms)) if self.timing_samples_ms else 0.0,
            "std_ms": float(np.std(self.timing_samples_ms)) if self.timing_samples_ms else 0.0,
            "min_ms": float(np.min(self.timing_samples_ms)) if self.timing_samples_ms else 0.0,
            "max_ms": float(np.max(self.timing_samples_ms)) if self.timing_samples_ms else 0.0,
        }

        report = {
            "schema_version": SCENARIO_REPORT_SCHEMA_VERSION,
            "run": {
                "scenario": self.scenario,
                "seed": self.seed,
                "requested_duration_seconds": float(self.requested_duration_seconds),
                "completed_duration_seconds": float(completed_duration_seconds),
                "completed_normally": bool(completed_normally),
                "error_type": error_type,
            },
            "summary": {
                "game_state_count": self.game_state_count,
                "meta_state_count": self.meta_state_count,
                "progress_steps": self.progress_steps,
                "strategy_count": self.new_strategies,
                "llm_query_count": self.generation_attempts,
                "fsm_transition_count": self.fsm_transition_count,
                "idle_intent_count": self.idle_intent_count,
                "death_event_count": self.death_event_count,
            },
            "strategist": {
                "generation_attempts": self.generation_attempts,
                "new_strategies": self.new_strategies,
                "fallbacks": self.fallbacks,
                "goals": self.goals,
                "regions": self.regions,
            },
            "fsm": {
                "transition_count": self.fsm_transition_count,
                "state_entry_counts": self.state_entry_counts,
                "transitions": self.fsm_transitions,
            },
            "meta_state": {
                "dimensions": list(META_STATE_DIMENSIONS),
                "samples": self.meta_state_samples,
            },
            "timing": timing_summary,
            "watchdog": {
                "health_transition_count": self.health_transition_count,
                "shutdown_requested": self.shutdown_requested,
            },
            "deaths": {
                "count": self.death_event_count,
                "timestamps": self.death_timestamps,
            },
            "idle_intents": {
                "count": self.idle_intent_count,
                "counts_by_type": self.idle_intent_counts_by_type,
            },
        }

        validate_report_dict(report)
        return report


def validate_report_dict(report: dict[str, Any]) -> None:
    """Validate report structural and numeric invariants.

    Raises:
        ValueError: If any report invariant is violated.
    """
    if report.get("schema_version") != SCENARIO_REPORT_SCHEMA_VERSION:
        raise ValueError(
            f"Expected schema_version {SCENARIO_REPORT_SCHEMA_VERSION}, got {report.get('schema_version')}"
        )

    # Validate MetaState samples
    meta_state = report.get("meta_state", {})
    dimensions = meta_state.get("dimensions", [])
    if tuple(dimensions) != META_STATE_DIMENSIONS:
        raise ValueError(f"Invalid MetaState dimensions: {dimensions!r}")

    samples = meta_state.get("samples", [])
    last_ts: float | None = None
    for idx, sample in enumerate(samples):
        ts = sample.get("simulation_timestamp")
        if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts):
            raise ValueError(f"Sample #{idx} has invalid timestamp: {ts!r}")
        if last_ts is not None and ts < last_ts:
            raise ValueError(f"Sample #{idx} timestamp decreased: {ts} < {last_ts}")
        last_ts = float(ts)

        vec = sample.get("vector")
        if not isinstance(vec, list) or len(vec) != 5:
            raise ValueError(f"Sample #{idx} vector must be length 5 list, got {vec!r}")
        for elem in vec:
            if isinstance(elem, bool) or not isinstance(elem, (int, float)) or not math.isfinite(elem):
                raise ValueError(f"Sample #{idx} vector contains non-finite element: {elem!r}")

    # Validate timing samples
    timing = report.get("timing", {})
    samples_ms = timing.get("samples_ms", [])
    for idx, sample in enumerate(samples_ms):
        if isinstance(sample, bool) or not isinstance(sample, (int, float)) or not math.isfinite(sample):
            raise ValueError(f"Timing sample #{idx} is non-finite: {sample!r}")

    timing_samples = timing.get("samples", [])
    if timing_samples is not None:
        if not isinstance(timing_samples, list):
            raise ValueError(f"timing.samples must be a list, got {type(timing_samples).__name__}")
        for idx, ts_sample in enumerate(timing_samples):
            if not isinstance(ts_sample, dict):
                raise ValueError(f"Timing detailed sample #{idx} must be dict, got {ts_sample!r}")  # noqa: TRY004 - report validation uses ValueError

            sim_ts = ts_sample.get("simulation_timestamp")
            if isinstance(sim_ts, bool) or not isinstance(sim_ts, (int, float)) or not math.isfinite(sim_ts):
                raise ValueError(f"Timing detailed sample #{idx} invalid simulation_timestamp: {sim_ts!r}")

            delay = ts_sample.get("delay_ms")
            if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not math.isfinite(delay) or delay <= 0:
                raise ValueError(f"Timing detailed sample #{idx} invalid delay_ms: {delay!r}")

            b_ms = ts_sample.get("base_ms")
            if isinstance(b_ms, bool) or not isinstance(b_ms, int) or b_ms <= 0:
                raise ValueError(f"Timing detailed sample #{idx} invalid base_ms: {b_ms!r}")

            fat = ts_sample.get("fatigue")
            if isinstance(fat, bool) or not isinstance(fat, (int, float)) or not math.isfinite(fat) or fat < 0.0 or fat > 1.0:
                raise ValueError(f"Timing detailed sample #{idx} invalid fatigue: {fat!r}")

            chaos = ts_sample.get("chaos_component")
            if isinstance(chaos, bool) or not isinstance(chaos, (int, float)) or not math.isfinite(chaos) or chaos < -1.0 or chaos > 1.0:
                raise ValueError(f"Timing detailed sample #{idx} invalid chaos_component: {chaos!r}")


def write_report_atomically(report: dict[str, Any], output_path: str | Path) -> None:
    """Validate and atomically write a report dictionary to output_path as formatted JSON.

    Args:
        report: Report dictionary.
        output_path: Target path for the final JSON report file.

    Raises:
        ValueError: If report validation fails or JSON contains NaN/Inf.
    """
    validate_report_dict(report)

    target_path = Path(output_path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    json_text = json.dumps(
        report,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )

    # Temporary file write + atomic rename
    temp_fd, temp_path_str = tempfile.mkstemp(
        dir=target_path.parent,
        prefix=f".tmp_{target_path.name}_",
    )
    temp_path = Path(temp_path_str)

    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            f.write(json_text)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_path, target_path)
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise
