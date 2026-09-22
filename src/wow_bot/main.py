# Pre-lab (MOCK_MODE)
"""Async Application Pipeline entry point (Task 7.1).

Integrates Perception (MockPerception), Internal Dynamics (MetaStateGenerator),
Strategist (LLM-driven Strategist), Executor (ExecutorFSM and Controller),
and Watchdog supervisor into an asynchronous data flow.

Pipeline Data Flow:
    MockPerception
         ↓
     GameState
         ↓
  Internal Dynamics
         ↓
MetaState / PipelineSnapshot
         ↓
    Strategist
         ↓
     Strategy
         ↓
    ExecutorFSM

Watchdog Side-Channel:
    Runtime health metadata / heartbeats / death events
         ↓
    Watchdog process
         ↓
    shutdown request
         ↓
    graceful application shutdown

Clock Domains:
    - SIMULATION TIME: GameState.timestamp, dt, strategy expiry, strategist cooldown.
    - MONOTONIC OPERATIONAL TIME: Watchdog liveness metadata only.
    - REAL ASYNC SCHEDULING: sleep, polling cadence, bounded run duration limit.
"""

from __future__ import annotations

import asyncio
import math
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from wow_bot.executor.controller import Controller
from wow_bot.executor.fsm import ExecutorFSM
from wow_bot.executor.idle_behaviors import IdleBehaviorEngine
from wow_bot.internal_dynamics.chaos import LorenzAttractor
from wow_bot.internal_dynamics.drives import Drives
from wow_bot.internal_dynamics.memory import MemoryStore
from wow_bot.internal_dynamics.meta_state import MetaStateGenerator
from wow_bot.internal_dynamics.oscillators import OscillatorBank
from wow_bot.mocks.mock_perception import MockPerception
from wow_bot.reporting.scenario import PipelineObserver
from wow_bot.shared.config import Settings, get_settings
from wow_bot.shared.events import DEATH
from wow_bot.shared.interfaces import GameState, MetaState, Strategy
from wow_bot.shared.logger import get_logger
from wow_bot.strategist.llm_client import LLMClient
from wow_bot.strategist.orchestrator import Strategist
from wow_bot.strategist.prompts import DynamicContext
from wow_bot.watchdog.shutdown import GracefulShutdown, ShutdownReason, ShutdownReport
from wow_bot.watchdog.watchdog import DeathEventMessage, HeartbeatMessage, WatchdogProcess

logger = get_logger("MAIN")

#: Cooldown in simulation seconds after a failed/fallback Strategist refresh.
STRATEGIST_RETRY_COOLDOWN_SECONDS: float = 10.0

#: Watchdog heartbeat interval in real wall-clock seconds.
WATCHDOG_HEARTBEAT_INTERVAL_SECONDS: float = 1.0


@dataclass(frozen=True)
class PipelineSnapshot:
    """Immutable integration snapshot coupling GameState and MetaState."""

    game_state: GameState
    meta_state: MetaState


@dataclass
class RuntimeComponents:
    """Container holding instantiated pipeline runtime components."""

    config: Settings
    memory: MemoryStore
    drives: Drives
    oscillators: OscillatorBank
    chaos: LorenzAttractor
    meta_state_generator: MetaStateGenerator
    llm_client: LLMClient
    strategist: Strategist
    controller: Controller
    perception: MockPerception
    watchdog: WatchdogProcess
    idle_engine: IdleBehaviorEngine
    observer: PipelineObserver | None = None
    _watchdog_shutdown_wire: _WatchdogShutdownWire | None = None


def _put_latest[T](queue: asyncio.Queue[T], item: T) -> None:
    """Put item into a latest-value queue (capacity 1), discarding stale item if full."""
    if queue.full():
        try:
            queue.get_nowait()
            queue.task_done()
        except asyncio.QueueEmpty:
            pass
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        pass


async def build_runtime(
    config: Settings | None = None,
    scenario: str | None = None,
    observer: PipelineObserver | None = None,
    seed: int | None = None,
) -> RuntimeComponents:
    """Construct and initialize all pipeline runtime components.

    Args:
        config: Optional Settings instance (defaults to get_settings()).
        scenario: Optional MockPerception scenario name.
        observer: Optional PipelineObserver for scenario instrumentation.
        seed: Optional RNG seed for scenario determinism.

    Returns:
        RuntimeComponents instance with ready components.
    """
    settings = config if config is not None else get_settings()

    memory = MemoryStore(settings.internal_dynamics.memory_db_path)
    llm_client: LLMClient | None = None
    try:
        await memory.init()

        drives = Drives(settings)
        oscillators = OscillatorBank(settings, seed=seed)
        chaos = LorenzAttractor(
            sigma=settings.internal_dynamics.lorenz_sigma,
            rho=settings.internal_dynamics.lorenz_rho,
            beta=settings.internal_dynamics.lorenz_beta,
            dt=settings.internal_dynamics.lorenz_dt,
        )
        meta_state_generator = MetaStateGenerator(
            config=settings,
            drives=drives,
            oscillators=oscillators,
            chaos=chaos,
            memory=memory,
        )

        llm_client = LLMClient(settings)
        strategist = Strategist(
            config=settings,
            llm_client=llm_client,
            memory=memory,
        )

        controller = Controller(dry_run=settings.executor.dry_run)
        perception_rng = np.random.default_rng(seed) if seed is not None else None
        perception = MockPerception(scenario=scenario, rng=perception_rng)
        watchdog = WatchdogProcess()
        idle_engine = IdleBehaviorEngine(rng=np.random.default_rng(seed))

        rc = RuntimeComponents(
            config=settings,
            memory=memory,
            drives=drives,
            oscillators=oscillators,
            chaos=chaos,
            meta_state_generator=meta_state_generator,
            llm_client=llm_client,
            strategist=strategist,
            controller=controller,
            perception=perception,
            watchdog=watchdog,
            idle_engine=idle_engine,
            observer=observer,
        )

        _wire_watchdog_shutdown(rc)
        return rc
    except BaseException:
        if llm_client is not None:
            try:
                await llm_client.close()
            except Exception:  # noqa: BLE001 - Cleanup must preserve the construction failure.
                logger.warning("LLM cleanup failed after runtime construction failure")
        try:
            await memory.close()
        except Exception:  # noqa: BLE001
            logger.warning("Memory cleanup failed after runtime construction failure")
        raise



async def perception_loop(
    components: RuntimeComponents,
    perception_queue: asyncio.Queue[GameState],
    shutdown_event: asyncio.Event,
) -> None:
    """Fetch GameState from MockPerception and publish to perception_queue."""
    interval_seconds = components.config.internal_dynamics.update_interval_ms / 1000.0
    logger.info(f"Perception loop started (cadence: {interval_seconds:.3f}s).")

    while not shutdown_event.is_set():
        game_state = await components.perception.get_state()
        if components.observer is not None:
            try:
                components.observer.on_game_state(game_state)
            except Exception as exc:
                logger.error(f"Observer error in on_game_state: {exc}")
                raise
        await perception_queue.put(game_state)
        await asyncio.sleep(interval_seconds)


async def dynamics_loop(
    components: RuntimeComponents,
    perception_queue: asyncio.Queue[GameState],
    executor_queue: asyncio.Queue[PipelineSnapshot],
    strategist_queue: asyncio.Queue[PipelineSnapshot],
    shutdown_event: asyncio.Event,
    runtime_state: dict[str, Any],
) -> None:
    """Process GameState through Internal Dynamics, publish PipelineSnapshot, update progress token."""
    logger.info("Dynamics loop started.")
    last_game_state: GameState | None = None
    default_dt = components.config.internal_dynamics.update_interval_ms / 1000.0

    while not shutdown_event.is_set():
        try:
            game_state = await asyncio.wait_for(perception_queue.get(), timeout=0.5)
        except TimeoutError:
            continue

        if runtime_state["session_start"] is None:
            runtime_state["session_start"] = float(game_state.timestamp)

        if last_game_state is None:
            dt = default_dt
        else:
            dt = float(game_state.timestamp) - float(last_game_state.timestamp)

        if not math.isfinite(float(game_state.timestamp)) or not math.isfinite(dt) or dt < 0.0:
            raise ValueError("Simulation timestamp must be finite and must not move backward")
        if dt == 0.0:
            perception_queue.task_done()
            continue

        last_game_state = game_state

        meta_state = await components.meta_state_generator.step(dt, game_state)
        snapshot = PipelineSnapshot(game_state=game_state, meta_state=meta_state)

        if components.observer is not None:
            try:
                components.observer.on_meta_state(meta_state)
            except Exception as exc:
                logger.error(f"Observer error in on_meta_state: {exc}")
                raise

        # Publish ordered snapshot to Executor queue
        await executor_queue.put(snapshot)

        # Publish latest snapshot to Strategist queue
        _put_latest(strategist_queue, snapshot)

        # Task 6.1 contract: Progress token advances ONLY on successfully completed Dynamics step
        runtime_state["progress_token"] += 1
        runtime_state["last_sim_timestamp"] = float(game_state.timestamp)

        if components.observer is not None:
            try:
                components.observer.on_progress_step()
            except Exception as exc:
                logger.error(f"Observer error in on_progress_step: {exc}")
                raise

        # Forward explicit canonical death events to Watchdog IPC queue
        for event in game_state.events:
            if event.type == DEATH:
                if components.observer is not None:
                    try:
                        components.observer.on_death_event(float(event.timestamp))
                    except Exception as exc:
                        logger.error(f"Observer error in on_death_event: {exc}")
                        raise

                try:
                    components.watchdog.message_queue.put_nowait(
                        DeathEventMessage(simulation_timestamp=float(event.timestamp))
                    )
                    logger.info(
                        f"Forwarded canonical death event to Watchdog at ts={event.timestamp:.3f}"
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Failed to queue death event to Watchdog: {exc}")

        perception_queue.task_done()


async def strategist_loop(
    components: RuntimeComponents,
    strategist_queue: asyncio.Queue[PipelineSnapshot],
    strategy_queue: asyncio.Queue[Strategy],
    shutdown_event: asyncio.Event,
    runtime_state: dict[str, Any],
) -> None:
    """Evaluate Strategy generation rules, call Strategist, enforce cooldowns and latest-value outputs."""
    logger.info("Strategist loop started.")
    last_planned_meta_state: MetaState | None = None
    next_refresh_allowed_at: float = 0.0

    while not shutdown_event.is_set():
        try:
            snapshot = await asyncio.wait_for(strategist_queue.get(), timeout=0.5)
        except TimeoutError:
            continue

        sim_ts = float(snapshot.game_state.timestamp)
        current_strategy = components.strategist.current_strategy

        # Check refresh conditions:
        # 1. No current strategy (bootstrap)
        # 2. Current strategy expired
        # 3. MetaState adaptive trigger fires
        has_strategy = current_strategy is not None
        expired = components.strategist.is_expired(sim_ts)
        adaptive_trigger = (
            last_planned_meta_state is not None
            and components.meta_state_generator.should_trigger_llm(
                snapshot.meta_state, last_planned_meta_state
            )
        )

        should_generate = (not has_strategy) or expired or adaptive_trigger

        if should_generate:
            if sim_ts < next_refresh_allowed_at:
                logger.debug(
                    f"Strategist refresh suppressed by cooldown until sim_ts={next_refresh_allowed_at:.3f} (current={sim_ts:.3f})"
                )
            else:
                session_start = runtime_state.get("session_start")
                if session_start is None:
                    session_start = sim_ts
                dynamic_context = DynamicContext(
                    now=sim_ts,
                    session_start=session_start,
                    previous_strategy=current_strategy,
                    available_regions=(),
                )

                logger.info(
                    f"Triggering Strategy generation: sim_ts={sim_ts:.3f} bootstrap={not has_strategy} "
                    f"expired={expired} adaptive={adaptive_trigger}"
                )

                prev_strategy_obj = current_strategy
                if components.observer is not None:
                    try:
                        components.observer.on_strategy_attempt(sim_ts)
                    except Exception as exc:
                        logger.error(f"Observer error in on_strategy_attempt: {exc}")
                        raise

                new_strategy = await components.strategist.generate_strategy(
                    snapshot.meta_state, dynamic_context
                )

                # Task 4.4 exact-object check: distinguish replacement from fallback
                if new_strategy is not prev_strategy_obj:
                    logger.info(
                        f"Genuinely new Strategy received: goal={new_strategy.goal!r} "
                        f"valid_until={new_strategy.valid_until:.3f}"
                    )
                    if components.observer is not None:
                        try:
                            components.observer.on_strategy_accepted(new_strategy, sim_ts)
                        except Exception as exc:
                            logger.error(f"Observer error in on_strategy_accepted: {exc}")
                            raise

                    next_refresh_allowed_at = 0.0
                    last_planned_meta_state = snapshot.meta_state
                    _put_latest(strategy_queue, new_strategy)
                else:
                    logger.warning(
                        "Strategy generation returned fallback/same strategy object. Setting 10s cooldown."
                    )
                    if components.observer is not None:
                        try:
                            components.observer.on_strategy_fallback(sim_ts)
                        except Exception as exc:
                            logger.error(f"Observer error in on_strategy_fallback: {exc}")
                            raise

                    failed_at = max(sim_ts, runtime_state.get("last_sim_timestamp", sim_ts))
                    next_refresh_allowed_at = failed_at + STRATEGIST_RETRY_COOLDOWN_SECONDS

        # Task 5.4 idle intent integration (symbolic evaluation)
        idle_intent = components.idle_engine.maybe_generate(
            curiosity=float(snapshot.meta_state.vector[2])
        )
        if idle_intent is not None:
            logger.info(
                f"Symbolic idle behavior intent observed: {idle_intent.behavior.name} "
                f"(emote={idle_intent.emote})"
            )
            if components.observer is not None:
                try:
                    components.observer.on_idle_intent(sim_ts, idle_intent.behavior.name)
                except Exception as exc:
                    logger.error(f"Observer error in on_idle_intent: {exc}")
                    raise

        strategist_queue.task_done()


async def executor_loop(
    components: RuntimeComponents,
    executor_queue: asyncio.Queue[PipelineSnapshot],
    strategy_queue: asyncio.Queue[Strategy],
    shutdown_event: asyncio.Event,
    runtime_state: dict[str, Any],
) -> None:
    """Consume initial Strategy, instantiate ExecutorFSM, update strategy dynamically, and tick FSM."""
    logger.info("Executor loop waiting for initial Strategy...")

    # Wait for initial Strategy before creating ExecutorFSM
    initial_strategy: Strategy | None = None
    while not shutdown_event.is_set():
        try:
            initial_strategy = await asyncio.wait_for(strategy_queue.get(), timeout=0.5)
            strategy_queue.task_done()
            break
        except TimeoutError:
            continue

    if initial_strategy is None or shutdown_event.is_set():
        logger.info("Executor loop exiting without FSM initialization (shutdown requested).")
        return

    def _fsm_on_transition(ts: float, from_st: Any, to_st: Any, reason: str) -> None:
        if components.observer is not None:
            components.observer.on_fsm_transition(
                ts, from_st.name if hasattr(from_st, "name") else str(from_st),
                to_st.name if hasattr(to_st, "name") else str(to_st),
                reason,
            )

    fsm = ExecutorFSM(
        config=components.config.executor,
        controller=components.controller,
        strategy=initial_strategy,
        on_transition=_fsm_on_transition,
    )
    runtime_state["fsm_ref"] = fsm
    logger.info(f"ExecutorFSM initialized with initial strategy goal={initial_strategy.goal!r}.")

    while not shutdown_event.is_set():
        # Check if a new strategy is pending in strategy_queue
        if not strategy_queue.empty():
            try:
                latest_strategy = strategy_queue.get_nowait()
                fsm.set_strategy(latest_strategy)
                strategy_queue.task_done()
                logger.info(f"Updated ExecutorFSM strategy to goal={latest_strategy.goal!r}.")
            except asyncio.QueueEmpty:
                pass

        try:
            snapshot = await asyncio.wait_for(executor_queue.get(), timeout=0.5)
        except TimeoutError:
            continue

        await fsm.tick(snapshot.game_state)
        if components.observer is not None:
            try:
                fatigue_val = float(snapshot.meta_state.vector[1])
                components.observer.on_fsm_tick(float(snapshot.game_state.timestamp), fatigue_val)
            except Exception as exc:
                logger.error(f"Observer error in on_fsm_tick: {exc}")
                raise
        executor_queue.task_done()


async def watchdog_heartbeat_loop(
    components: RuntimeComponents,
    shutdown_event: asyncio.Event,
    runtime_state: dict[str, Any],
) -> None:
    """Publish periodic runtime heartbeats to Watchdog message queue."""
    logger.info("Watchdog heartbeat loop started.")

    while not shutdown_event.is_set():
        sim_ts = runtime_state.get("last_sim_timestamp", 0.0)
        progress_token = runtime_state.get("progress_token", 0)
        fsm_ref: ExecutorFSM | None = runtime_state.get("fsm_ref")
        fsm_state_name = fsm_ref.state.name if fsm_ref is not None else "IDLE"

        hb = HeartbeatMessage(
            monotonic_sent_at=time.monotonic(),
            simulation_timestamp=sim_ts,
            fsm_state=fsm_state_name,
            progress_token=progress_token,
        )

        try:
            components.watchdog.message_queue.put_nowait(hb)
            logger.debug(
                f"Heartbeat sent: mono={hb.monotonic_sent_at:.3f} sim_ts={sim_ts:.3f} "
                f"fsm={fsm_state_name} progress={progress_token}"
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to queue heartbeat to Watchdog: {exc}")

        await asyncio.sleep(WATCHDOG_HEARTBEAT_INTERVAL_SECONDS)


class _WatchdogShutdownWire:
    """Internal helper binding WatchdogProcess callback to GracefulShutdown state."""

    def __init__(self, components: RuntimeComponents) -> None:
        self.components = components
        self.async_loop: asyncio.AbstractEventLoop | None = None
        self.shutdown_event: asyncio.Event | None = None
        self.graceful_shutdown: GracefulShutdown | None = None
        self.last_report: ShutdownReport | None = None

    def on_shutdown_request(self, reason: ShutdownReason) -> None:
        logger.error(f"Watchdog requested graceful shutdown with reason={reason.value!r}")
        if self.graceful_shutdown is None:
            self.graceful_shutdown = GracefulShutdown(
                safety=getattr(self.components, "safety", None),
                actuator=getattr(self.components, "actuator", None),
                session=getattr(self.components, "session", None),
                health=getattr(self.components, "health", None),
            )

        self.last_report = self.graceful_shutdown.run(reason)

        if self.shutdown_event is not None:
            if self.async_loop is not None and self.async_loop.is_running():
                self.async_loop.call_soon_threadsafe(self.shutdown_event.set)
            else:
                self.shutdown_event.set()


def _wire_watchdog_shutdown(components: RuntimeComponents) -> None:
    """Wire WatchdogProcess shutdown request callback to GracefulShutdown ONCE."""
    wire = _WatchdogShutdownWire(components)
    if hasattr(components.watchdog, "on_shutdown_request"):
        components.watchdog.on_shutdown_request(wire.on_shutdown_request)
    components._watchdog_shutdown_wire = wire


async def watchdog_shutdown_bridge(
    components: RuntimeComponents,
    shutdown_event: asyncio.Event,
) -> None:
    """Bridge Watchdog process shutdown_event IPC signal into local asyncio shutdown_event."""
    logger.info("Watchdog shutdown bridge started.")

    wire: _WatchdogShutdownWire | None = getattr(components, "_watchdog_shutdown_wire", None)
    if wire is not None:
        wire.async_loop = asyncio.get_running_loop()
        wire.shutdown_event = shutdown_event

    while not shutdown_event.is_set():
        if components.watchdog.shutdown_event.is_set():
            logger.error("Watchdog process requested emergency application shutdown!")
            shutdown_event.set()
            raise RuntimeError("Watchdog requested shutdown")
        if not components.watchdog.is_alive:
            raise RuntimeError("Watchdog process exited unexpectedly")
        await asyncio.sleep(0.2)


async def run_pipeline(
    components: RuntimeComponents,
    *,
    run_duration_seconds: float | None = None,
) -> None:
    """Execute the asynchronous application pipeline tasks under supervision.

    Args:
        components: Initialized RuntimeComponents container.
        run_duration_seconds: Optional positive finite duration limit in wall-clock seconds.

    Raises:
        ValueError: If run_duration_seconds is invalid.
    """
    if run_duration_seconds is not None and (
        isinstance(run_duration_seconds, bool)
        or not isinstance(run_duration_seconds, (int, float))
        or not math.isfinite(run_duration_seconds)
        or run_duration_seconds <= 0.0
    ):
        raise ValueError(
            f"run_duration_seconds must be a positive finite float, got {run_duration_seconds!r}"
        )

    shutdown_event = asyncio.Event()

    perception_queue: asyncio.Queue[GameState] = asyncio.Queue(maxsize=20)
    executor_queue: asyncio.Queue[PipelineSnapshot] = asyncio.Queue(maxsize=20)
    strategist_queue: asyncio.Queue[PipelineSnapshot] = asyncio.Queue(maxsize=1)
    strategy_queue: asyncio.Queue[Strategy] = asyncio.Queue(maxsize=1)

    runtime_state: dict[str, Any] = {
        "progress_token": 0,
        "last_sim_timestamp": 0.0,
        "session_start": None,
        "fsm_ref": None,
    }

    tasks: list[asyncio.Task[None]] = []
    duration_timer_task: asyncio.Task[None] | None = None
    shutdown_waiter: asyncio.Task[bool] | None = None
    first_exception: BaseException | None = None
    try:
        components.watchdog.start()
        logger.info("Started independent Watchdog process.")

        tasks = [
            asyncio.create_task(
                perception_loop(components, perception_queue, shutdown_event),
                name="perception_loop",
            ),
            asyncio.create_task(
                dynamics_loop(
                    components,
                    perception_queue,
                    executor_queue,
                    strategist_queue,
                    shutdown_event,
                    runtime_state,
                ),
                name="dynamics_loop",
            ),
            asyncio.create_task(
                strategist_loop(
                    components,
                    strategist_queue,
                    strategy_queue,
                    shutdown_event,
                    runtime_state,
                ),
                name="strategist_loop",
            ),
            asyncio.create_task(
                executor_loop(
                    components,
                    executor_queue,
                    strategy_queue,
                    shutdown_event,
                    runtime_state,
                ),
                name="executor_loop",
            ),
            asyncio.create_task(
                watchdog_heartbeat_loop(components, shutdown_event, runtime_state),
                name="watchdog_heartbeat_loop",
            ),
            asyncio.create_task(
                watchdog_shutdown_bridge(components, shutdown_event),
                name="watchdog_shutdown_bridge",
            ),
        ]

        if run_duration_seconds is not None:

            async def _duration_timer() -> None:
                await asyncio.sleep(float(run_duration_seconds))
                logger.info(
                    f"Bounded run duration limit ({run_duration_seconds}s) reached. Requesting graceful shutdown."
                )
                shutdown_event.set()

            duration_timer_task = asyncio.create_task(_duration_timer(), name="duration_timer")

        shutdown_waiter = asyncio.create_task(shutdown_event.wait(), name="shutdown_waiter")
        done, _pending = await asyncio.wait(
            [*tasks, shutdown_waiter], return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            if not task.cancelled() and task.exception() is not None:
                exc = task.exception()
                assert exc is not None
                if first_exception is None:
                    first_exception = exc
                logger.error(
                    f"Pipeline task '{task.get_name()}' failed with exception: {type(exc).__name__}"
                )
                shutdown_event.set()

        if first_exception is None and not shutdown_event.is_set():
            raise RuntimeError("Pipeline loop exited unexpectedly")
    finally:
        logger.info("Beginning graceful application pipeline cleanup...")
        shutdown_event.set()

        if duration_timer_task is not None and not duration_timer_task.done():
            duration_timer_task.cancel()

        for task in tasks:
            if not task.done():
                task.cancel()

        if shutdown_waiter is not None:
            shutdown_waiter.cancel()
        await asyncio.gather(
            *tasks,
            *([duration_timer_task] if duration_timer_task is not None else []),
            *([shutdown_waiter] if shutdown_waiter is not None else []),
            return_exceptions=True,
        )
        cleanup_errors: list[Exception] = []

        try:
            await components.controller.stop_all()
        except Exception as exc:  # noqa: BLE001
            cleanup_errors.append(exc)
            logger.warning(f"Error calling controller.stop_all() during cleanup: {exc}")

        try:
            await components.llm_client.close()
        except Exception as exc:  # noqa: BLE001
            cleanup_errors.append(exc)
            logger.warning(f"Error closing LLMClient during cleanup: {exc}")

        try:
            await components.memory.close()
        except Exception as exc:  # noqa: BLE001
            cleanup_errors.append(exc)
            logger.warning(f"Error closing MemoryStore during cleanup: {exc}")

        try:
            await asyncio.to_thread(components.watchdog.close)
            logger.info("Watchdog process joined cleanly.")
        except Exception as exc:  # noqa: BLE001
            cleanup_errors.append(exc)
            logger.warning(f"Error stopping Watchdog process during cleanup: {exc}")

        logger.info("Graceful application pipeline cleanup completed.")

        if first_exception is not None:
            raise first_exception
        if cleanup_errors and sys.exc_info()[0] is None:
            raise cleanup_errors[0]


async def main() -> None:
    """Application CLI entry point."""
    logger.info("Initializing WoW-Bot Async Application Pipeline (Task 7.1)...")
    components = await build_runtime()

    loop = asyncio.get_running_loop()
    main_shutdown_event = asyncio.Event()

    def _on_signal(sig_name: str) -> None:
        logger.info(f"Received signal {sig_name}. Initiating graceful shutdown...")
        main_shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal, sig.name)
        except NotImplementedError:  # Windows signal compatibility
            pass

    exit_code = 0
    try:
        pipeline_task = asyncio.create_task(run_pipeline(components))

        async def _signal_watcher() -> None:
            await main_shutdown_event.wait()
            if not pipeline_task.done():
                pipeline_task.cancel()

        watcher_task = asyncio.create_task(_signal_watcher())

        try:
            await pipeline_task
        finally:
            watcher_task.cancel()
            await asyncio.gather(watcher_task, return_exceptions=True)
    except asyncio.CancelledError:
        logger.info("Main pipeline cancelled.")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Pipeline exited with error: {type(exc).__name__}")
        exit_code = 1

    wire = getattr(components, "_watchdog_shutdown_wire", None)
    if wire is not None and wire.last_report is not None:
        exit_code = wire.last_report.exit_code

    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
