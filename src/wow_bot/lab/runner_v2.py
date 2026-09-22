"""LAB mode asynchronous runtime builder and single-threaded farm cycle orchestrator.

Provides build_lab_runtime, build_lab_runtime_async, run_lab_loop, run_lab_loop_async,
and associated configuration/result frozen dataclasses for Phase 11.

Sync Policy for Async Operations:
  Inside the synchronous runner entry point `run_lab_loop`:
    - Async operations (`WorldSync.sync_once`, `summarize`, `resolve_vendor_node`) are run via
      `asyncio.run(...)` per cycle if no event loop is currently running.
    - If a running event loop is detected when `run_lab_loop` or `build_lab_runtime` is called,
      it raises `LabRunnerError` instructing the caller to use `run_lab_loop_async` or
      `build_lab_runtime_async` instead.
    - If a sync helper is invoked while a loop is running, it skips the async operation and emits
      a `"sync_skipped"` session event.
  Inside the asynchronous runner entry point `run_lab_loop_async`:
    - All three operations (`WorldSync.sync_once`, `summarize`, `resolve_vendor_node`) are
      directly awaited in the coroutine loop.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from wow_bot.actuation.actuator import make_actuator
from wow_bot.actuation.backends.focus_null import NullFocusBackend
from wow_bot.actuation.driver import make_driver
from wow_bot.actuation.focus import FocusManager
from wow_bot.actuation.humanized import HumanizedActuator, HumanizerConfig
from wow_bot.actuation.mapper import ActionMapper, ActionStatus, RealDelay
from wow_bot.combat.flee import FleeConfig, FleeController
from wow_bot.combat.loop import CombatLoop, CombatLoopConfig
from wow_bot.combat.reactive import (
    ReactiveCombat,
    ReactiveCombatConfig,
    ReactiveCombatSource,
)
from wow_bot.combat.rotation import RotationConfig, RotationTable
from wow_bot.combat.targeting import TargetConfig, TargetSelector
from wow_bot.executor.feedback import Feedback, FeedbackKind, classify_action_result
from wow_bot.executor.fsm_v2 import FSM, Behavior, FSMConfig, GameStateLike, MetaStateLike
from wow_bot.executor.recovery import RecoveryBehavior, RecoveryConfig
from wow_bot.executor.reflex_bridge import FSMReflexBridge
from wow_bot.executor.states import FSMState
from wow_bot.farm.loot import InventoryTracker, LootConfig, LootController
from wow_bot.farm.profile import FarmProfile
from wow_bot.farm.vendor import (
    VendorConfig,
    VendorController,
    VendorLocation,
    VendorStatus,
    resolve_vendor_node,
)
from wow_bot.nav.graph import GraphConfig, NavGraph, build_graph
from wow_bot.nav.navigator import NavConfig, Navigator, NavStatus, SimpleReplanner
from wow_bot.nav.telemetry import RealCpuSource, TelemetryCollector, TelemetryConfig
from wow_bot.reflex.loop import NullClock, RealClock, ReflexLoop
from wow_bot.reflex.rules import default_rules
from wow_bot.reflex.sinks import ActuatorAbortSink, FSMSink, RecoverySink
from wow_bot.reflex.sources import (
    FocusSignalSource,
    SafetySignalSource,
    SessionTimeoutSignalSource,
)
from wow_bot.reflex.stuck import PositionStuckSignalSource
from wow_bot.safety import SafetyLayer
from wow_bot.session import Session
from wow_bot.strategist.cooldown_v2 import CooldownConfig, CooldownGate
from wow_bot.strategist.orchestrator_v2 import (
    OrchestratorConfig,
    OrchestratorOutcome,
    OrchestratorV2,
)
from wow_bot.strategist.vocab_v2 import VocabConfig, VocabularyGuard
from wow_bot.watchdog.health import HealthConfig, HealthState, HealthStateMachine
from wow_bot.watchdog.loops import ActionObservation, LoopConfig, LoopDetector
from wow_bot.watchdog.metrics import MetricsConfig, ProgressSample, ProgressTracker
from wow_bot.watchdog.shutdown import GracefulShutdown, ShutdownConfig, ShutdownReason
from wow_bot.watchdog.watchdog import WatchdogProcess
from wow_bot.world.loader import LoaderConfig, LoaderError, load_world
from wow_bot.world.store import WorldModel
from wow_bot.world.summary import SummaryConfig, WorldSummary, summarize
from wow_bot.world.sync import SyncConfig, WorldSync

REFLEX_INTERVAL_S: float = 0.05


class LabRunnerError(Exception):
    """Raised when runtime assembly or loop execution encounters an invalid state."""


class LabRunStatus(str, Enum):
    """Status outcomes for LAB mode farm cycle runs."""

    COMPLETED = "completed"
    MAX_CYCLES_REACHED = "max_cycles_reached"
    HEALTH_CRITICAL = "health_critical"
    LOOP_DETECTED = "loop_detected"
    MAX_FAILURES_REACHED = "max_failures_reached"
    STOP_EVENT_SET = "stop_event_set"
    BUILD_ERROR = "build_error"
    RUNTIME_ERROR = "runtime_error"


@dataclass(frozen=True)
class LabRunnerConfig:
    """Configuration parameters for LAB mode execution runs."""

    max_cycles_per_run: int = 1000
    max_consecutive_failures: int = 5
    world_sync_interval_cycles: int = 1
    summarize_every_cycles: int = 5
    vendor_repair_threshold: float = 0.5
    fail_on_async_context: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameter bounds."""
        if (
            isinstance(self.max_cycles_per_run, bool)
            or not isinstance(self.max_cycles_per_run, int)
            or self.max_cycles_per_run < 1
        ):
            raise ValueError(
                f"max_cycles_per_run must be an integer >= 1, got {self.max_cycles_per_run!r}"
            )
        if (
            isinstance(self.max_consecutive_failures, bool)
            or not isinstance(self.max_consecutive_failures, int)
            or self.max_consecutive_failures < 1
        ):
            raise ValueError(
                f"max_consecutive_failures must be an integer >= 1, got {self.max_consecutive_failures!r}"
            )
        if (
            isinstance(self.world_sync_interval_cycles, bool)
            or not isinstance(self.world_sync_interval_cycles, int)
            or self.world_sync_interval_cycles < 1
        ):
            raise ValueError(
                f"world_sync_interval_cycles must be an integer >= 1, got {self.world_sync_interval_cycles!r}"
            )
        if (
            isinstance(self.summarize_every_cycles, bool)
            or not isinstance(self.summarize_every_cycles, int)
            or self.summarize_every_cycles < 1
        ):
            raise ValueError(
                f"summarize_every_cycles must be an integer >= 1, got {self.summarize_every_cycles!r}"
            )
        if (
            isinstance(self.vendor_repair_threshold, bool)
            or not isinstance(self.vendor_repair_threshold, (int, float))
            or not math.isfinite(float(self.vendor_repair_threshold))
            or not (0.0 <= float(self.vendor_repair_threshold) <= 1.0)
        ):
            raise ValueError(
                f"vendor_repair_threshold must be a float in [0.0, 1.0], got {self.vendor_repair_threshold!r}"
            )
        if not isinstance(self.fail_on_async_context, bool):
            raise TypeError("fail_on_async_context must be a boolean")


@dataclass(frozen=True)
class LabRuntime:
    """Immutable container holding all instantiated components for LAB mode execution."""

    mode: str
    config_snapshot: dict[str, Any]
    safety: SafetyLayer
    session: Session
    clock: Callable[[], float]
    sleep: Callable[[float], None]
    game_state_source: Callable[[], object]
    meta_state_source: Callable[[], object]
    driver: Any
    focus: FocusManager
    mapper: ActionMapper
    actuator: HumanizedActuator
    world: WorldModel
    world_sync: WorldSync
    graph: NavGraph
    navigator: Navigator
    rotation: RotationTable
    targeting: TargetSelector
    combat_loop: CombatLoop
    reactive_combat: ReactiveCombat
    flee: FleeController
    fsm: FSM
    reflex_loop: ReflexLoop | None
    orchestrator: OrchestratorV2
    cooldown: CooldownGate
    guard: VocabularyGuard
    watchdog: WatchdogProcess | None
    health: HealthStateMachine
    progress: ProgressTracker
    loops: LoopDetector
    shutdown: GracefulShutdown
    profile: FarmProfile
    loot: LootController
    inventory: InventoryTracker
    vendor: VendorController
    telemetry: TelemetryCollector
    telemetry_config: TelemetryConfig
    session_events_total: int


@dataclass(frozen=True)
class LabRunResult:
    """Result snapshot of a LAB mode execution run."""

    status: LabRunStatus
    cycles_completed: int
    failures: int
    duration_s: float
    shutdown_report: Any | None
    reason: str

    def to_json(self) -> dict[str, Any]:
        """Convert result snapshot into a JSON-serializable dictionary representation."""
        report_json = None
        if self.shutdown_report is not None and hasattr(self.shutdown_report, "to_json"):
            report_json = self.shutdown_report.to_json()

        return {
            "status": self.status.value,
            "cycles_completed": self.cycles_completed,
            "failures": self.failures,
            "duration_s": float(self.duration_s),
            "shutdown_report": report_json,
            "reason": self.reason,
        }


class _NullLlmClient:
    """Null LLM client raising LabRunnerError when completion is requested."""

    def complete(self, prompt: str) -> str:
        """Raise LabRunnerError indicating no real or fake LLM client was injected."""
        raise LabRunnerError("No LlmClient provided to runner")


class _CompositeBehavior(Behavior):
    """Behavior implementation dispatching decisions to recovery, combat, or loot behaviors."""

    def __init__(
        self,
        *,
        recovery_behavior: RecoveryBehavior,
        combat_loop: CombatLoop,
        loot_controller: LootController,
    ) -> None:
        self.recovery_behavior = recovery_behavior
        self.combat_loop = combat_loop
        self.loot_controller = loot_controller

    def decide(
        self,
        state: FSMState,
        game_state: GameStateLike,
        meta_state: MetaStateLike,
        now: float,
        rng: random.Random,
    ) -> Any | None:
        """Dispatch decision based on current FSMState."""
        if state == FSMState.STUCK_RECOVERY:
            return self.recovery_behavior.decide(state, game_state, meta_state, now, rng)
        if state == FSMState.COMBAT:
            return self.combat_loop.decide(state, game_state, meta_state, now, rng)  # type: ignore[arg-type]
        if state == FSMState.LOOTING:
            return self.loot_controller.decide(state, game_state, meta_state, now, rng)
        return None


class _TelemetryClockAdapter:
    """Small clock adapter satisfying TelemetryCollector Clock protocol."""

    def __init__(self, clock_fn: Callable[[], float]) -> None:
        self._clock_fn = clock_fn

    def now(self) -> float:
        return self._clock_fn()


def _extract_position(state: object) -> tuple[float, float]:
    """Extract (x, y) coordinates from a game state object."""
    # Attribute names are static; existence is probed with hasattr and accessed
    # directly rather than via getattr (B009).
    if hasattr(state, "position"):
        pos = state.position
        if isinstance(pos, (tuple, list)) and len(pos) >= 2:
            return (float(pos[0]), float(pos[1]))
    if hasattr(state, "player_x") and hasattr(state, "player_y"):
        return (float(state.player_x), float(state.player_y))
    if hasattr(state, "self_x") and hasattr(state, "self_y"):
        return (float(state.self_x), float(state.self_y))
    if hasattr(state, "x") and hasattr(state, "y"):
        return (float(state.x), float(state.y))
    return (0.0, 0.0)


def _extract_heading(state: object) -> float:
    """Extract player heading from a game state object."""
    # Attribute names are static; existence is probed with hasattr and accessed
    # directly rather than via getattr (B009).
    if hasattr(state, "player_heading"):
        return float(state.player_heading)
    if hasattr(state, "heading"):
        return float(state.heading)
    if hasattr(state, "self_heading"):
        return float(state.self_heading)
    return 0.0


def _sync_world_sync(
    world_sync: WorldSync,
    state: Any,
    session: Session | None,
) -> None:
    """Synchronously execute world_sync.sync_once using asyncio.run if no event loop exists.

    If an event loop is already running, skip the sync and emit a 'sync_skipped' event.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(world_sync.sync_once(state))
    else:
        if session is not None:
            session.write_event({
                "event": "sync_skipped",
                "reason": "running_event_loop_in_sync_runner",
                "operation": "world_sync",
            })


def _sync_summarize(
    world: WorldModel,
    pos: tuple[float, float],
    session: Session | None,
) -> WorldSummary | None:
    """Synchronously execute summarize using asyncio.run if no event loop exists.

    If an event loop is already running, skip the summary and emit a 'sync_skipped' event.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(summarize(world, pos, config=SummaryConfig()))
    else:
        if session is not None:
            session.write_event({
                "event": "sync_skipped",
                "reason": "running_event_loop_in_sync_runner",
                "operation": "summarize",
            })
        return None


def _sync_resolve_vendor_node(
    world: WorldModel,
    pos: tuple[float, float],
    session: Session | None,
) -> VendorLocation | None:
    """Synchronously execute resolve_vendor_node using asyncio.run if no event loop exists.

    If an event loop is already running, skip resolution and emit a 'sync_skipped' event.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            resolve_vendor_node(
                world=world,
                from_xy=pos,
                search_radius_units=50.0,
            )
        )
    else:
        if session is not None:
            session.write_event({
                "event": "sync_skipped",
                "reason": "running_event_loop_in_sync_runner",
                "operation": "resolve_vendor_node",
            })
        return None


async def build_lab_runtime_async(
    *,
    config: object,
    session: Session,
    game_state_source: Callable[[], object],
    meta_state_source: Callable[[], object],
    rotation_config: RotationConfig,
    farm_profile: FarmProfile,
    world_db_path: str,
    driver_name: str = "null",
    window_title: str = "WoW",
    runner_config: LabRunnerConfig | None = None,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
    rng_seed: int = 0,
    include_reflex_loop: bool = True,
    include_watchdog: bool = True,
    focus_backend: object | None = None,
    llm_client: object | None = None,
) -> LabRuntime:
    """Asynchronously construct and wire all LAB mode runtime infrastructure components."""
    if not getattr(config, "lab_mode", False):
        raise LabRunnerError("config.lab_mode must be True for LAB execution mode")

    rcfg = runner_config if runner_config is not None else LabRunnerConfig()
    clk = clock if clock is not None else time.monotonic
    slp = sleep if sleep is not None else time.sleep

    try:
        world, _report = await load_world(
            LoaderConfig(
                db_path=world_db_path,
                require_existing=False,
            ),
            session=session,
        )
    except (LoaderError, Exception) as exc:
        raise LabRunnerError(f"Failed to load World Model: {exc}") from exc

    try:
        graph = await build_graph(world, config=GraphConfig())
    except Exception as exc:
        raise LabRunnerError(f"Failed to build NavGraph: {exc}") from exc

    safety = SafetyLayer(config=config, session=session)  # type: ignore[arg-type]
    driver = make_driver(driver_name, lab_mode=True)
    backend = focus_backend if focus_backend is not None else NullFocusBackend()
    focus = FocusManager(window_title, backend=backend)  # type: ignore[arg-type]
    mapper = ActionMapper(driver, delay=RealDelay())
    base_actuator = make_actuator(
        mode="LAB",
        driver=driver,
        focus=focus,
        mapper=mapper,
        safety=safety,
        session=session,
    )
    actuator = HumanizedActuator(
        base_actuator,
        config=HumanizerConfig(),
        delay=RealDelay(),
        session=session,
        rng=random.Random(rng_seed),
    )

    def pos_source() -> tuple[float, float]:
        return _extract_position(game_state_source())

    def head_source() -> float:
        return _extract_heading(game_state_source())

    navigator = Navigator(
        graph=graph,
        actuator=actuator,
        position_source=pos_source,
        config=NavConfig(),
        replanner=SimpleReplanner(),
        session=session,
        clock=clk,
        sleep=slp,
    )

    rotation = RotationTable(rotation_config)
    targeting = TargetSelector(config=TargetConfig())
    combat_loop = CombatLoop(
        config=CombatLoopConfig(),
        rotation=rotation,
        targeting=targeting,
    )
    reactive_combat = ReactiveCombat(config=ReactiveCombatConfig())
    flee = FleeController(
        navigator=navigator,
        world=world,
        graph=graph,
        config=FleeConfig(),
        session=session,
        clock=clk,
    )

    recovery_behavior = RecoveryBehavior(
        config=RecoveryConfig(),
        position_source=pos_source,
        heading_source=head_source,
    )
    loot_controller = LootController(
        config=LootConfig(),
        session=session,
    )
    composite_behavior = _CompositeBehavior(
        recovery_behavior=recovery_behavior,
        combat_loop=combat_loop,
        loot_controller=loot_controller,
    )
    fsm = FSM(
        config=FSMConfig(seed=rng_seed),
        behavior=composite_behavior,
        session=session,
    )

    cooldown = CooldownGate(config=CooldownConfig(), session=session)
    guard = VocabularyGuard(config=VocabConfig(), session=session)
    llm_inst = llm_client if llm_client is not None else _NullLlmClient()
    orchestrator = OrchestratorV2(
        llm=llm_inst,  # type: ignore[arg-type]
        cooldown=cooldown,
        guard=guard,
        session=session,
        config=OrchestratorConfig(),
    )

    world_sync = WorldSync(
        world=world,
        config=SyncConfig(),
        session=session,
    )

    health = HealthStateMachine(
        config=HealthConfig.default(),
        session=session,
    )
    progress = ProgressTracker(config=MetricsConfig())
    loops = LoopDetector(
        config=LoopConfig(),
        session=session,
    )
    shutdown = GracefulShutdown(
        safety=safety,
        actuator=actuator,
        session=session,
        health=health,
        config=ShutdownConfig(),
        clock=NullClock() if clk is not time.monotonic else RealClock(),
    )

    inventory = InventoryTracker(session=session)
    vendor = VendorController(
        navigator=navigator,
        actuator=actuator,
        config=VendorConfig(repair_durability_threshold=rcfg.vendor_repair_threshold),
        session=session,
        clock=clk,
    )

    telemetry_config = TelemetryConfig()
    sess_id = getattr(session, "session_id", "lab_session")
    sess_id_str = sess_id() if callable(sess_id) else str(sess_id)
    telemetry = TelemetryCollector(
        session_id=sess_id_str,
        target_xy=(0.0, 0.0),
        clock=_TelemetryClockAdapter(clk),
        cpu_source=RealCpuSource(),
        config=telemetry_config,
    )

    reflex_loop: ReflexLoop | None = None
    if include_reflex_loop:
        start_t = clk()
        max_sess_s = float(getattr(config, "max_session_seconds", 3600.0))
        focus_src = FocusSignalSource(focus)
        safety_src = SafetySignalSource(safety)
        timeout_src = SessionTimeoutSignalSource(
            deadline_s=start_t + max_sess_s, start_time_s=start_t
        )
        stuck_src = PositionStuckSignalSource(pos_source)
        reactive_src = ReactiveCombatSource(game_state_source)  # type: ignore[arg-type]

        bridge = FSMReflexBridge(fsm, clock=clk, session=session)
        abort_sink = ActuatorAbortSink(actuator)
        fsm_sink = FSMSink(bridge)
        recovery_sink = RecoverySink(bridge)

        reflex_clk = RealClock() if clk is time.monotonic else NullClock()
        reflex_loop = ReflexLoop(
            rate_hz=20.0,
            sources=(focus_src, safety_src, timeout_src, stuck_src, reactive_src),
            sinks=(abort_sink, fsm_sink, recovery_sink),
            rules=default_rules,
            clock=reflex_clk,
            seed=rng_seed,
            session=session,
        )

    watchdog: WatchdogProcess | None = None
    if include_watchdog:
        watchdog = WatchdogProcess()

    config_snapshot = {
        "mode": "LAB",
        "driver_name": driver_name,
        "window_title": window_title,
        "world_db_path": world_db_path,
        "runner_config": {
            "max_cycles_per_run": rcfg.max_cycles_per_run,
            "max_consecutive_failures": rcfg.max_consecutive_failures,
            "world_sync_interval_cycles": rcfg.world_sync_interval_cycles,
            "summarize_every_cycles": rcfg.summarize_every_cycles,
            "vendor_repair_threshold": rcfg.vendor_repair_threshold,
            "fail_on_async_context": rcfg.fail_on_async_context,
        },
    }

    return LabRuntime(
        mode="LAB",
        config_snapshot=config_snapshot,
        safety=safety,
        session=session,
        clock=clk,
        sleep=slp,
        game_state_source=game_state_source,
        meta_state_source=meta_state_source,
        driver=driver,
        focus=focus,
        mapper=mapper,
        actuator=actuator,
        world=world,
        world_sync=world_sync,
        graph=graph,
        navigator=navigator,
        rotation=rotation,
        targeting=targeting,
        combat_loop=combat_loop,
        reactive_combat=reactive_combat,
        flee=flee,
        fsm=fsm,
        reflex_loop=reflex_loop,
        orchestrator=orchestrator,
        cooldown=cooldown,
        guard=guard,
        watchdog=watchdog,
        health=health,
        progress=progress,
        loops=loops,
        shutdown=shutdown,
        profile=farm_profile,
        loot=loot_controller,
        inventory=inventory,
        vendor=vendor,
        telemetry=telemetry,
        telemetry_config=telemetry_config,
        session_events_total=0,
    )


def build_lab_runtime(
    *,
    config: object,
    session: Session,
    game_state_source: Callable[[], object],
    meta_state_source: Callable[[], object],
    rotation_config: RotationConfig,
    farm_profile: FarmProfile,
    world_db_path: str,
    driver_name: str = "null",
    window_title: str = "WoW",
    runner_config: LabRunnerConfig | None = None,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
    rng_seed: int = 0,
    include_reflex_loop: bool = True,
    include_watchdog: bool = True,
    focus_backend: object | None = None,
    llm_client: object | None = None,
) -> LabRuntime:
    """Synchronous builder wrapping build_lab_runtime_async via asyncio.run."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        running_loop = False
    else:
        running_loop = True

    if running_loop:
        raise LabRunnerError(
            "build_lab_runtime called inside a running event loop; use build_lab_runtime_async instead"
        )

    return asyncio.run(
        build_lab_runtime_async(
            config=config,
            session=session,
            game_state_source=game_state_source,
            meta_state_source=meta_state_source,
            rotation_config=rotation_config,
            farm_profile=farm_profile,
            world_db_path=world_db_path,
            driver_name=driver_name,
            window_title=window_title,
            runner_config=runner_config,
            clock=clock,
            sleep=sleep,
            rng_seed=rng_seed,
            include_reflex_loop=include_reflex_loop,
            include_watchdog=include_watchdog,
            focus_backend=focus_backend,
            llm_client=llm_client,
        )
    )


async def _run_cycle(
    runtime: LabRuntime,
    prev_goal: str,
    cached_vendor: VendorLocation | None,
    cycle_index: int,
    last_summary: WorldSummary | None,
) -> tuple[str, VendorLocation | None, bool, WorldSummary | None]:
    """Execute a single farm loop cycle step."""
    rcfg = runtime.config_snapshot.get("runner_config", {})
    sync_interval = int(rcfg.get("world_sync_interval_cycles", 1))
    sum_interval = int(rcfg.get("summarize_every_cycles", 5))

    state = runtime.game_state_source()
    pos = _extract_position(state)

    if cycle_index % sync_interval == 0:
        await runtime.world_sync.sync_once(state)  # type: ignore[arg-type]

    meta = runtime.meta_state_source()

    world_summary = last_summary
    if last_summary is None or cycle_index % sum_interval == 0:
        world_summary = await summarize(
            runtime.world,
            pos,
            config=SummaryConfig(),
        )

    failed = False
    new_goal = prev_goal

    if prev_goal == "farm":
        if world_summary is None:
            world_summary = await summarize(
                runtime.world,
                pos,
                config=SummaryConfig(),
            )
        outcome = runtime.orchestrator.decide(
            meta=meta,  # type: ignore[arg-type]
            world=world_summary,
            state=state,  # type: ignore[arg-type]
            now=runtime.clock(),
        )
        if outcome.outcome == OrchestratorOutcome.SUCCESS and outcome.strategy is not None:
            new_goal = outcome.strategy.goal

    # Subsystem dispatch
    curr_fsm_state = runtime.fsm.current_state

    if (
        curr_fsm_state in (FSMState.LOOTING, FSMState.COMBAT)
        or new_goal in ("loot", "grind_humans", "farm_herbs", "combat")
    ):
        intent = runtime.fsm.tick(state, meta, now=runtime.clock())
        if intent is not None:
            act_res = runtime.actuator.execute(intent, position=pos)
            fb = classify_action_result(act_res, ts=runtime.clock())
            runtime.fsm.submit_feedback(fb, now=runtime.clock())
            if act_res.status == ActionStatus.FAILED:
                failed = True

    elif new_goal == "flee" or curr_fsm_state == FSMState.FLEEING:
        flee_res = runtime.flee.execute(state)  # type: ignore[arg-type]
        if flee_res.target_xy is not None:
            nav_res = runtime.navigator.go_to(flee_res.target_xy)
            if nav_res.status in (NavStatus.FAILED, NavStatus.HARD_FAILURE, NavStatus.TIMEOUT):
                failed = True
                fb = Feedback(
                    kind=FeedbackKind.ACTION_FAILED,
                    ts=runtime.clock(),
                    reason=nav_res.reason,
                )
                runtime.fsm.submit_feedback(fb, now=runtime.clock())

    elif new_goal in ("sell_vendor", "repair", "go_to_vendor"):
        if cached_vendor is None:
            cached_vendor = await resolve_vendor_node(
                world=runtime.world,
                from_xy=pos,
                search_radius_units=50.0,
            )
        v_res = runtime.vendor.run(
            cached_vendor,
            state,  # type: ignore[arg-type]
            need_repair=(new_goal == "repair"),
        )
        if v_res.status in (VendorStatus.SUCCESS, VendorStatus.SKIPPED_NO_ACTION):
            runtime.inventory.reset()
            new_goal = "farm"
        else:
            failed = True

    elif new_goal == "travel_to":
        target_xy = (0.0, 0.0)
        nav_res = runtime.navigator.go_to(target_xy)
        if nav_res.status == NavStatus.SUCCESS:
            new_goal = "farm"
        elif nav_res.status in (NavStatus.FAILED, NavStatus.HARD_FAILURE, NavStatus.TIMEOUT):
            failed = True
            fb = Feedback(
                kind=FeedbackKind.ACTION_FAILED,
                ts=runtime.clock(),
                reason=nav_res.reason,
            )
            runtime.fsm.submit_feedback(fb, now=runtime.clock())

    else:
        # IDLE / SCANNING / tick
        intent = runtime.fsm.tick(state, meta, now=runtime.clock())
        if intent is not None:
            act_res = runtime.actuator.execute(intent, position=pos)
            fb = classify_action_result(act_res, ts=runtime.clock())
            runtime.fsm.submit_feedback(fb, now=runtime.clock())
            if act_res.status == ActionStatus.FAILED:
                failed = True

    # Inventory observation
    runtime.inventory.observe(
        inventory_count=getattr(state, "inventory_count", 0),
        inventory_max=getattr(state, "inventory_max", None),
        now=runtime.clock(),
    )
    if runtime.inventory.is_full():
        new_goal = "go_to_vendor"

    return new_goal, cached_vendor, failed, world_summary


def _finish_run(
    status: LabRunStatus,
    cycles_completed: int,
    failures: int,
    start_time: float,
    runtime: LabRuntime,
    reason: str,
    shutdown_report: Any | None = None,
) -> LabRunResult:
    """Construct LabRunResult and emit session event if attached."""
    duration_s = max(0.0, runtime.clock() - start_time)
    res = LabRunResult(
        status=status,
        cycles_completed=cycles_completed,
        failures=failures,
        duration_s=duration_s,
        shutdown_report=shutdown_report,
        reason=reason,
    )
    if runtime.session is not None:
        runtime.session.write_event({
            "event": "lab_run_finished",
            "status": status.value,
            "cycles_completed": cycles_completed,
            "failures": failures,
            "duration_s": duration_s,
            "reason": reason,
        })
    return res


async def run_lab_loop_async(
    runtime: LabRuntime,
    *,
    max_cycles: int | None = None,
    stop_event: asyncio.Event | None = None,
) -> LabRunResult:
    """Asynchronously run the farm cycle loop using injected clock and sleeps."""
    rcfg = runtime.config_snapshot.get("runner_config", {})
    max_c = (
        max_cycles
        if max_cycles is not None
        else int(rcfg.get("max_cycles_per_run", 1000))
    )
    max_fails = int(rcfg.get("max_consecutive_failures", 5))

    if max_c < 1:
        raise LabRunnerError(f"max_cycles must be >= 1, got {max_c}")

    start_time = runtime.clock()
    cycles_completed = 0
    failures = 0
    consecutive_failures = 0
    prev_goal = "farm"
    cached_vendor: VendorLocation | None = None
    last_summary: WorldSummary | None = None

    while True:
        if stop_event is not None and stop_event.is_set():
            return _finish_run(
                LabRunStatus.STOP_EVENT_SET,
                cycles_completed,
                failures,
                start_time,
                runtime,
                reason="stop_event_set",
            )

        if cycles_completed >= max_c:
            return _finish_run(
                LabRunStatus.MAX_CYCLES_REACHED,
                cycles_completed,
                failures,
                start_time,
                runtime,
                reason="max_cycles_reached",
            )

        if consecutive_failures >= max_fails:
            return _finish_run(
                LabRunStatus.MAX_FAILURES_REACHED,
                cycles_completed,
                failures,
                start_time,
                runtime,
                reason="max_consecutive_failures_reached",
            )

        try:
            prev_goal, cached_vendor, failed, last_summary = await _run_cycle(
                runtime, prev_goal, cached_vendor, cycles_completed, last_summary
            )
        except Exception as exc:  # noqa: BLE001
            return _finish_run(
                LabRunStatus.RUNTIME_ERROR,
                cycles_completed,
                failures,
                start_time,
                runtime,
                reason=f"{type(exc).__name__}:{exc}",
            )

        cycles_completed += 1
        if failed:
            failures += 1
            consecutive_failures += 1
        else:
            consecutive_failures = 0

        # Health check
        state = runtime.game_state_source()
        pos = _extract_position(state)
        sample = ProgressSample(
            ts=runtime.clock(),
            position=pos,
            inventory_count=getattr(state, "inventory_count", 0),
            level_or_xp=float(getattr(state, "level", getattr(state, "xp", 0.0))),
            successful_actions_total=cycles_completed * 10,
            reflex_ticks_total=cycles_completed * 20,
        )
        runtime.progress.update(sample)
        snapshot = runtime.progress.snapshot()
        runtime.health.observe(snapshot, now=runtime.clock())

        if runtime.health.current_state == HealthState.CRITICAL:
            report = runtime.shutdown.run(ShutdownReason.HEALTH_CRITICAL)
            return _finish_run(
                LabRunStatus.HEALTH_CRITICAL,
                cycles_completed,
                failures,
                start_time,
                runtime,
                reason="health_critical",
                shutdown_report=report,
            )

        # Loop detector check
        observation = ActionObservation(
            ts=runtime.clock(),
            signature=f"cycle_{cycles_completed}:{prev_goal}",
            level_or_xp=float(getattr(state, "level", getattr(state, "xp", 0.0))),
        )
        loop_res = runtime.loops.observe(observation)
        if loop_res is not None and loop_res.detected:
            report = runtime.shutdown.run(ShutdownReason.LOOP_DETECTED)
            return _finish_run(
                LabRunStatus.LOOP_DETECTED,
                cycles_completed,
                failures,
                start_time,
                runtime,
                reason="loop_detected",
                shutdown_report=report,
            )

        if runtime.sleep is not None:
            runtime.sleep(REFLEX_INTERVAL_S)


def run_lab_loop(
    runtime: LabRuntime,
    *,
    max_cycles: int | None = None,
    stop_event: object | None = None,
) -> LabRunResult:
    """Synchronous runner wrapping run_lab_loop_async via asyncio.run."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        running_loop = False
    else:
        running_loop = True

    if running_loop:
        raise LabRunnerError(
            "run_lab_loop called inside a running event loop; use run_lab_loop_async instead"
        )

    async_event: asyncio.Event | None = None
    if stop_event is not None and hasattr(stop_event, "is_set"):
        async_event = asyncio.Event()
        if stop_event.is_set():
            async_event.set()

    return asyncio.run(
        run_lab_loop_async(
            runtime,
            max_cycles=max_cycles,
            stop_event=async_event,
        )
    )
