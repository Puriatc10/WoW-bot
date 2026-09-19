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
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any

from wow_bot.executor.controller import Controller
from wow_bot.executor.fsm import ExecutorFSM
from wow_bot.executor.idle_behaviors import IdleBehaviorEngine
from wow_bot.internal_dynamics.chaos import LorenzAttractor
from wow_bot.internal_dynamics.drives import Drives
from wow_bot.internal_dynamics.memory import MemoryStore
from wow_bot.internal_dynamics.meta_state import MetaStateGenerator
from wow_bot.internal_dynamics.oscillators import OscillatorBank
from wow_bot.mocks.mock_perception import MockPerception
from wow_bot.shared.config import Settings, get_settings
from wow_bot.shared.events import DEATH
from wow_bot.shared.interfaces import GameState, MetaState, Strategy
from wow_bot.shared.logger import get_logger
from wow_bot.strategist.llm_client import LLMClient
from wow_bot.strategist.orchestrator import Strategist
from wow_bot.strategist.prompts import DynamicContext
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


def _put_latest[T](queue: asyncio.Queue[T], item: T) -> None:
    """Put item into a latest-value queue (capacity 1), discarding stale item if full."""
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        pass


async def build_runtime(
    config: Settings | None = None,
    scenario: str | None = None,
) -> RuntimeComponents:
    """Construct and initialize all pipeline runtime components.

    Args:
        config: Optional Settings instance (defaults to get_settings()).
        scenario: Optional MockPerception scenario name.

    Returns:
        RuntimeComponents instance with ready components.
    """
    settings = config if config is not None else get_settings()

    memory = MemoryStore(settings.internal_dynamics.memory_db_path)
    await memory.init()

    drives = Drives(settings)
    oscillators = OscillatorBank(settings)
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
    perception = MockPerception(scenario=scenario)
    watchdog = WatchdogProcess()
    idle_engine = IdleBehaviorEngine()

    return RuntimeComponents(
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
    )


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

        if dt <= 0.0:
            logger.warning(
                f"Non-positive simulation dt detected ({dt:.4f}s). Skipping backward/static tick."
            )
            perception_queue.task_done()
            continue

        last_game_state = game_state

        meta_state = await components.meta_state_generator.step(dt, game_state)
        snapshot = PipelineSnapshot(game_state=game_state, meta_state=meta_state)

        # Publish ordered snapshot to Executor queue
        await executor_queue.put(snapshot)

        # Publish latest snapshot to Strategist queue
        _put_latest(strategist_queue, snapshot)

        # Task 6.1 contract: Progress token advances ONLY on successfully completed Dynamics step
        runtime_state["progress_token"] += 1
        runtime_state["last_sim_timestamp"] = float(game_state.timestamp)

        # Forward explicit canonical death events to Watchdog IPC queue
        for event in game_state.events:
            if event.type == DEATH:
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
                session_start = runtime_state.get("session_start") or sim_ts
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
                try:
                    new_strategy = await components.strategist.generate_strategy(
                        snapshot.meta_state, dynamic_context
                    )

                    # Task 4.4 exact-object check: distinguish replacement from fallback
                    if new_strategy is not prev_strategy_obj:
                        logger.info(
                            f"Genuinely new Strategy received: goal={new_strategy.goal!r} "
                            f"valid_until={new_strategy.valid_until:.3f}"
                        )
                        next_refresh_allowed_at = 0.0
                        last_planned_meta_state = snapshot.meta_state
                        _put_latest(strategy_queue, new_strategy)
                    else:
                        logger.warning(
                            "Strategy generation returned fallback/same strategy object. Setting 10s cooldown."
                        )
                        next_refresh_allowed_at = sim_ts + STRATEGIST_RETRY_COOLDOWN_SECONDS

                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        f"Strategy generation failed without fallback: {type(exc).__name__}: {exc}. "
                        f"Setting {STRATEGIST_RETRY_COOLDOWN_SECONDS}s cooldown."
                    )
                    next_refresh_allowed_at = sim_ts + STRATEGIST_RETRY_COOLDOWN_SECONDS

        # Task 5.4 idle intent integration (symbolic evaluation)
        idle_intent = components.idle_engine.maybe_generate(
            curiosity=float(snapshot.meta_state.vector[2])
        )
        if idle_intent is not None:
            logger.info(
                f"Symbolic idle behavior intent observed: {idle_intent.behavior.name} "
                f"(emote={idle_intent.emote})"
            )

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

    fsm = ExecutorFSM(
        config=components.config.executor,
        controller=components.controller,
        strategy=initial_strategy,
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


async def watchdog_shutdown_bridge(
    components: RuntimeComponents,
    shutdown_event: asyncio.Event,
) -> None:
    """Bridge Watchdog process shutdown_event IPC signal into local asyncio shutdown_event."""
    logger.info("Watchdog shutdown bridge started.")

    while not shutdown_event.is_set():
        if components.watchdog.shutdown_event.is_set():
            logger.error("Watchdog process requested emergency application shutdown!")
            shutdown_event.set()
            break
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

    duration_timer_task: asyncio.Task[None] | None = None
    if run_duration_seconds is not None:

        async def _duration_timer() -> None:
            await asyncio.sleep(float(run_duration_seconds))
            logger.info(
                f"Bounded run duration limit ({run_duration_seconds}s) reached. Requesting graceful shutdown."
            )
            shutdown_event.set()

        duration_timer_task = asyncio.create_task(_duration_timer(), name="duration_timer")

    first_exception: BaseException | None = None

    try:
        done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in done:
            if not task.cancelled() and task.exception() is not None:
                exc = task.exception()
                assert exc is not None
                if first_exception is None:
                    first_exception = exc
                logger.error(
                    f"Pipeline task '{task.get_name()}' failed with exception: {type(exc).__name__}: {exc}"
                )
                shutdown_event.set()

    finally:
        logger.info("Beginning graceful application pipeline cleanup...")
        shutdown_event.set()

        if duration_timer_task is not None and not duration_timer_task.done():
            duration_timer_task.cancel()

        for task in tasks:
            if not task.done():
                task.cancel()

        await asyncio.gather(*tasks, return_exceptions=True)

        try:
            await components.controller.stop_all()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Error calling controller.stop_all() during cleanup: {exc}")

        try:
            await components.llm_client.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Error closing LLMClient during cleanup: {exc}")

        try:
            await components.memory.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Error closing MemoryStore during cleanup: {exc}")

        try:
            components.watchdog.stop()
            components.watchdog.join(timeout=5.0)
            logger.info("Watchdog process joined cleanly.")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Error stopping Watchdog process during cleanup: {exc}")

        logger.info("Graceful application pipeline cleanup completed.")

        if first_exception is not None:
            raise first_exception


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

    try:
        pipeline_task = asyncio.create_task(run_pipeline(components))

        async def _signal_watcher() -> None:
            await main_shutdown_event.wait()
            if not pipeline_task.done():
                pipeline_task.cancel()

        watcher_task = asyncio.create_task(_signal_watcher())

        await pipeline_task
        watcher_task.cancel()
    except asyncio.CancelledError:
        logger.info("Main pipeline cancelled.")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Pipeline exited with error: {type(exc).__name__}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
