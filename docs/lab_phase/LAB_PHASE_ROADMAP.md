# LAB_PHASE_ROADMAP.md

Operational roadmap for evolving the research prototype from MOCK_MODE
into a LAB_MODE system with real actuation, real reflex, and real
navigation, while keeping MockPerception as the sole perception source.

This document is the source of truth for phase and task breakdown.
Each task is written to be directly handed to a coding agent (Jules)
as an implementation prompt. Do not merge tasks. Do not reorder
phases without updating this file.

## Global Rules (apply to every task)

1. MockPerception MUST remain the only producer of `GameState` in this
   roadmap. Its schema is frozen. Any task that touches perception MUST
   NOT change the schema.
2. Default mode is `MOCK_MODE`. `LAB_MODE` is opt-in and never runs in CI.
3. No task may introduce OS input, screen capture, or network calls
   outside the modules designated for them.
4. No task may read or write game process memory, inject DLLs, hook
   syscalls, or modify the game binary.
5. Every task MUST ship with tests that run in MOCK_MODE.
6. Every task MUST emit structured, append-only logs to
   `runs/lab/<session_id>/` when in LAB_MODE.
7. Per-component RNG only. No global `np.random.seed`.
8. Local Ollama only. No cloud LLM calls.
9. SafetyLayer is alive from Phase 0 and MUST NOT be bypassed.
10. All new config keys go into `config/lab.example.toml` with comments.

## Phase Overview

| Phase | Name                        | Primary Output                        |
|-------|-----------------------------|---------------------------------------|
| 0     | Foundation                  | Config, Safety, Session, Logging      |
| 1     | RealActuator v0             | Real movement, no combat              |
| 2     | Reactive Reflex Layer       | 10-20 Hz safety loop                  |
| 3     | Executor FSM v2             | Feedback-driven FSM                   |
| 4     | World Model                 | Spatial + entity memory               |
| 5     | Navigation                  | A* on graph, replan, stuck recovery   |
| 6     | Combat Engine               | Rotation, interrupt, flee             |
| 7     | Humanizer (real actuation)  | Lognormal timing on real input        |
| 8     | Behavioral Watchdog         | Progress-based supervision            |
| 9     | Strategist v2               | LLM grounded in real state            |
| 10    | Reporting & Analysis v2     | Lab-grade reports and metrics         |
| 11    | End-to-End Farm Loop        | Farm → loot → vendor → repair         |
| 12    | Soak & Final Analysis       | 24-72h stability, RESULTS.md          |

---

# Phase 0 — Foundation

**Goal:** Config, Safety, Session, and Logging infrastructure. No
actuation, no reflex. Everything that follows depends on this.

**Phase acceptance:**
- All tasks T0.1 through T0.6 merged.
- Kill switch latency <100 ms over 100 trials.
- Non-allowlisted address aborts within 1 s.
- External network reachable → startup fails.
- SIGTERM results in clean shutdown with flushed logs.

### T0.1 — Config System

**Depends on:** none
**Deliverables:**
- `config/lab.example.toml`
- `src/wow_bot/config.py`
- `tests/test_config.py`

**Contract:**
- `load_config(path: Path) -> Config` returns a frozen dataclass.
- `Config` fields (minimum): `lab_mode: bool`, `server_allowlist: list[str]`,
  `isolation_sentinel: str`, `kill_switch_key: str`, `session_root: Path`,
  `dry_run: bool`, `max_session_seconds: int`, `log_level: str`.
- Missing keys raise `ConfigError`. Unknown keys raise `ConfigError`.
- `lab.example.toml` includes a comment above every key.

**Acceptance:**
- [ ] Test: missing required key raises `ConfigError`.
- [ ] Test: unknown key raises `ConfigError`.
- [ ] Test: valid config loads without error.
- [ ] Test: `server_allowlist` containing a known Blizzard domain raises
  `ConfigError` at load time.

**Out of scope:** env-var overrides, hot reload, secret storage.

### T0.2 — Session Manager

**Depends on:** T0.1
**Deliverables:**
- `src/wow_bot/session.py`
- `tests/test_session.py`

**Contract:**
- `Session.start(config: Config) -> Session` creates
  `runs/lab/<session_id>/` where `session_id` is
  `YYYYMMDD-HHMMSS-<short_uuid>`.
- Writes `session.json` (config snapshot, start time, mode).
- `Session.close(reason: str) -> None` appends stop time and reason.
- `Session.write_event(event: dict) -> None` appends one JSON line to
  `events.jsonl`. Events are never truncated.
- `Session.write_crash(exc: BaseException) -> None` writes `crash.json`.
- All file handles are opened with `O_APPEND`.

**Acceptance:**
- [ ] Test: two sessions get distinct ids.
- [ ] Test: events append in order across multiple calls.
- [ ] Test: after `close()`, further `write_event` raises.
- [ ] Test: `write_crash` produces a valid JSON file with traceback.
- [ ] Test: no code path deletes or truncates any file.

**Out of scope:** rotation, compression, remote shipping.

### T0.3 — Safety Layer (skeleton)

**Depends on:** T0.1, T0.2
**Deliverables:**
- `src/wow_bot/safety.py`
- `tests/test_safety.py`

**Contract:**
- `SafetyLayer` exposes:
  - `arm() -> None` (called once at startup)
  - `check_allowlist(addr: str) -> None` (raises `AllowlistViolation`)
  - `check_isolation() -> None` (raises `IsolationViolation`)
  - `abort(reason: str) -> None` (idempotent; sets a flag and emits an event)
  - `is_aborted() -> bool`
  - `register_kill_switch(callback: Callable[[], None]) -> None`
- Kill switch hooks `SIGTERM` and `SIGINT`.
- `check_isolation` attempts a TCP connect to `isolation_sentinel` with
  1 s timeout; success → raise.

**Acceptance:**
- [ ] Test: `check_allowlist` passes for allowlisted, raises otherwise.
- [ ] Test: `check_isolation` raises when sentinel is reachable
  (use a local listener in the test).
- [ ] Test: `abort` is idempotent.
- [ ] Test: SIGTERM triggers `abort` and registered callback.

**Out of scope:** physical key binding (T0.4), OS input release (Phase 1).

### T0.4 — Physical Kill Switch

**Depends on:** T0.3
**Deliverables:**
- `src/wow_bot/kill_switch.py`
- `tests/test_kill_switch.py`

**Contract:**
- `KillSwitch(key: str, on_trigger: Callable[[], None])` starts a
  background thread that watches for the key.
- Backend selection is abstracted: `NullBackend` for CI, `PynputBackend`
  for lab (imported lazily; missing dependency fails closed with a clear
  error in LAB_MODE).
- On trigger: calls `on_trigger` once and stops.

**Acceptance:**
- [ ] Test: `NullBackend` never triggers.
- [ ] Test: simulated trigger calls callback exactly once.
- [ ] Test: missing pynput in LAB_MODE raises at construction.

**Out of scope:** modifier combos, sequences.

### T0.5 — Structured Logging

**Depends on:** T0.2
**Deliverables:**
- `src/wow_bot/logging_setup.py`
- `tests/test_logging_setup.py`

**Contract:**
- `setup_logging(session: Session, level: str) -> None`
- Console handler: human-readable, filtered to `WARNING`+ in LAB_MODE,
  `INFO`+ in MOCK_MODE.
- File handler: JSON lines to `runs/lab/<session_id>/app.log`.
- Every record includes: `ts`, `level`, `module`, `event`, `payload`.

**Acceptance:**
- [ ] Test: log file contains valid JSON per line.
- [ ] Test: log level filtering works.
- [ ] Test: no duplicate handlers after double call.

**Out of scope:** log shipping, rotation.

### T0.6 — MOCK/LAB Mode Gate

**Depends on:** T0.1, T0.3, T0.5
**Deliverables:**
- `src/wow_bot/mode.py`
- `tests/test_mode.py`

**Contract:**
- `enter_mode(config: Config, session: Session) -> ModeContext`
- If `config.lab_mode` is True:
  - `check_isolation()` MUST pass before returning.
  - All entries of `server_allowlist` MUST be validated.
  - Session MUST be in append-only mode.
- If `config.lab_mode` is False:
  - `dry_run=False` MUST raise.
  - No isolation check is performed.

**Acceptance:**
- [ ] Test: MOCK_MODE with `dry_run=False` raises.
- [ ] Test: LAB_MODE with unreachable sentinel passes.
- [ ] Test: LAB_MODE with reachable sentinel raises.
- [ ] Test: LAB_MODE populates ModeContext with allowlist and session.

**Out of scope:** runtime mode switching.

---

# Phase 1 — RealActuator v0 (Movement Only)

**Goal:** Real keyboard/mouse output for `move_to(point)` and
`turn(angle)`. No combat. No loot. Actuation gated by SafetyLayer.

**Phase acceptance:**
- T1.1 through T1.5 merged.
- In lab: reach a fixed waypoint 9/10 times.
- No key remains pressed after any session end.
- `focus_lost` detected in <50 ms.
- Intent-to-first-input latency <20 ms.

### T1.1 — Input Driver (abstract + null + real)

**Depends on:** Phase 0
**Deliverables:**
- `src/wow_bot/actuation/driver.py`
- `src/wow_bot/actuation/drivers/null.py`
- `src/wow_bot/actuation/drivers/pynput_backend.py`
- `src/wow_bot/actuation/drivers/interception_backend.py`
- `tests/test_driver.py`

**Contract:**
- `InputDriver` protocol: `key_down(k)`, `key_up(k)`, `mouse_move(x,y)`,
  `mouse_down(b)`, `mouse_up(b)`, `release_all()`, `now() -> float`.
- `NullDriver` records calls in a list for CI.
- Real drivers import lazily; missing dependency raises at construction.
- `release_all()` is idempotent and MUST be called on session close.

**Acceptance:**
- [ ] Test: NullDriver records calls.
- [ ] Test: real driver import fails cleanly when dep missing.
- [ ] Test: `release_all` after `key_down` results in no held keys.

**Out of scope:** game-specific key mapping.

### T1.2 — Focus Manager

**Depends on:** T1.1
**Deliverables:**
- `src/wow_bot/actuation/focus.py`
- `tests/test_focus.py`

**Contract:**
- `FocusManager(window_title: str)` resolves a window handle at start.
- `is_focused() -> bool`
- `assert_focused() -> None` raises `FocusLost` if not focused.
- Polling rate configurable (default 20 Hz).
- On focus loss, emits a `focus_lost` event to a registered callback.

**Acceptance:**
- [ ] Test: null backend always reports focused.
- [ ] Test: callback fires on simulated loss.
- [ ] Test: `assert_focused` raises after simulated loss.

**Out of scope:** multi-window, alt-tab recovery.

### T1.3 — Action Mapper (movement subset)

**Depends on:** T1.1, T1.2
**Deliverables:**
- `src/wow_bot/actuation/mapper.py`
- `tests/test_mapper.py`

**Contract:**
- Input: symbolic intents `move_to(x, y)` and `turn(angle_rad)`.
- Output: sequence of driver calls.
- Each intent returns `ActionResult(status, latency_ms, notes)` where
  `status ∈ {success, failed, timeout}`.
- Mapping table is data-driven (loaded from config or a Python dict).

**Acceptance:**
- [ ] Test: `move_to` produces a deterministic sequence on NullDriver.
- [ ] Test: `turn` respects sign and magnitude.
- [ ] Test: `ActionResult` carries latency.

**Out of scope:** path, combat, loot.

### T1.4 — RealActuator Facade

**Depends on:** T1.3
**Deliverables:**
- `src/wow_bot/actuation/actuator.py`
- `tests/test_actuator.py`

**Contract:**
- `RealActuator(driver, focus, mapper, safety, session)`.
- `execute(intent: Intent) -> ActionResult`:
  - Calls `safety.is_aborted()` first; if aborted, returns `failed`.
  - Calls `focus.assert_focused()`.
  - Delegates to mapper.
  - Logs the intent and result as a session event.
- `abort() -> None` calls `driver.release_all()` and must complete
  in <100 ms.

**Acceptance:**
- [ ] Test: aborted safety returns failed without driver calls.
- [ ] Test: focus loss prevents driver calls.
- [ ] Test: abort completes in <100 ms on NullDriver.

**Out of scope:** humanization (Phase 7), reflex coupling (Phase 2).

### T1.5 — Movement Scenario Harness

**Depends on:** T1.4
**Deliverables:**
- `scripts/lab/movement_demo.py`
- `tests/test_movement_harness.py`

**Contract:**
- Reads a scenario file listing waypoints.
- Uses MockPerception to provide position feedback.
- Exits cleanly on `max_session_seconds` or kill switch.
- Writes a session with `events.jsonl` and a summary in `session.json`.

**Acceptance:**
- [ ] Test: harness runs in MOCK_MODE with NullDriver and completes.
- [ ] Manual lab acceptance: reach fixed waypoint 9/10 times.
- [ ] Manual lab acceptance: no key held after exit.

**Out of scope:** autonomous navigation (Phase 5).

---

# Phase 2 — Reactive Reflex Layer

**Goal:** A 10-20 Hz loop that can abort, pause, and recover,
independent of LLM and FSM.

**Phase acceptance:**
- T2.1 through T2.4 merged.
- p99 tick jitter <5 ms over 1000 ticks.
- Kill switch reaction in <1 tick.
- Stuck detection active in <2 s.
- LLM call count from this layer is asserted to be zero.

### T2.1 — Reflex Loop Skeleton

**Depends on:** Phase 0, T1.4
**Deliverables:**
- `src/wow_bot/reflex/loop.py`
- `tests/test_reflex_loop.py`

**Contract:**
- `ReflexLoop(rate_hz, sinks, session)`.
- `start()`, `stop()`, `tick()` (tick callable from tests).
- Each tick: read signals, run rules, dispatch control signals.
- Deterministic given (state, tick index, per-component seed).

**Acceptance:**
- [ ] Test: 1000 ticks, p99 jitter <5 ms (NullClock).
- [ ] Test: stop() returns in <1 tick.
- [ ] Test: no import of strategist or LLM modules (static check).

**Out of scope:** signal sources.

### T2.2 — Signal Sources

**Depends on:** T2.1
**Deliverables:**
- `src/wow_bot/reflex/signals.py`
- `tests/test_signals.py`

**Contract:**
- Signals emitted: `focus_lost`, `position_stuck`, `session_timeout`,
  `kill_switch`, `safety_abort`.
- Each signal has `name`, `payload`, `ts`.
- Sources register with the loop; loop polls per tick.

**Acceptance:**
- [ ] Test: each signal fires in isolation.
- [ ] Test: multiple signals in one tick handled deterministically.

**Out of scope:** perception signals (Phase 5 reuses this).

### T2.3 — Control Signals & FSM Coupling

**Depends on:** T2.1, T2.2
**Deliverables:**
- `src/wow_bot/reflex/controls.py`
- `tests/test_controls.py`

**Contract:**
- Control signals: `abort_actuation`, `pause_fsm`, `resume_fsm`,
  `enter_recovery`.
- FSM exposes `pause()`, `resume()`, `enter_recovery()` (declared in
  Phase 3; in Phase 2 use a NullFSM stub).

**Acceptance:**
- [ ] Test: `abort_actuation` propagates to RealActuator.abort.
- [ ] Test: `pause_fsm` then `resume_fsm` restores state.
- [ ] Test: NullFSM records control signals.

**Out of scope:** real FSM (Phase 3).

### T2.4 — Stuck Detector (position-based)

**Depends on:** T2.2
**Deliverables:**
- `src/wow_bot/reflex/stuck.py`
- `tests/test_stuck.py`

**Contract:**
- Samples position at 1 Hz from GameState.
- If displacement < threshold over N seconds → `position_stuck`.
- Thresholds configurable.

**Acceptance:**
- [ ] Test: stationary position triggers stuck in <2 s.
- [ ] Test: moving position never triggers.
- [ ] Test: recovery resets the detector.

**Out of scope:** vision-based detection.

---

# Phase 3 — Executor FSM v2

**Goal:** Feedback-driven FSM with new states and integration with
Reflex, World Model hooks, and ActionResult feedback.

**Phase acceptance:**
- T3.1 through T3.5 merged.
- 1000 transitions on MockPerception without crash.
- Target-lost scenario recovers to IDLE.
- Every transition emits a structured event.

### T3.1 — State Schema

**Depends on:** Phase 2
**Deliverables:**
- `src/wow_bot/executor/states.py`
- `tests/test_states.py`

**Contract:**
- States: `IDLE, SCANNING, MOVING_TO_TARGET, TARGETING, PATHING,
  COMBAT, LOOTING, FLEEING, STUCK_RECOVERY, WAITING_GCD,
  RECOVERING, PAUSED`.
- `StateSpec` includes entry/exit hooks and allowed transitions.

**Acceptance:**
- [ ] Test: transition table is exhaustive.
- [ ] Test: disallowed transition raises.

**Out of scope:** behaviors.

### T3.2 — Feedback Intake

**Depends on:** T3.1, T1.4
**Deliverables:**
- `src/wow_bot/executor/feedback.py`
- `tests/test_feedback.py`

**Contract:**
- `Feedback(status: ActionResultStatus, reason: str, ts: float)`.
- FSM consumes feedback after each action.
- Feedback is logged as a session event.

**Acceptance:**
- [ ] Test: feedback with `failed` moves to RECOVERING.
- [ ] Test: feedback with `timeout` moves to STUCK_RECOVERY.
- [ ] Test: feedback with `success` continues.

**Out of scope:** multi-action feedback aggregation.

### T3.3 — FSM v2 Core

**Depends on:** T3.1, T3.2
**Deliverables:**
- `src/wow_bot/executor/fsm_v2.py`
- `tests/test_fsm_v2.py`

**Contract:**
- `FSM.tick(state: GameState, meta: MetaState) -> Intent | None`.
- `FSM.pause()`, `FSM.resume()`, `FSM.enter_recovery()`.
- Deterministic given input sequence and per-component seed.
- Emits `fsm_transition` event on every transition.

**Acceptance:**
- [ ] Test: 1000 transitions without crash.
- [ ] Test: pause/resume preserves state.
- [ ] Test: enter_recovery lands in STUCK_RECOVERY.

**Out of scope:** combat, navigation.

### T3.4 — Stuck Recovery Behaviors

**Depends on:** T3.3
**Deliverables:**
- `src/wow_bot/executor/recovery.py`
- `tests/test_recovery.py`

**Contract:**
- Behaviors: short backstep, camera sweep, jump, alternative waypoint.
- Chosen by per-component RNG.
- Counts attempts; after K fails emits `hard_stuck` and pauses FSM.

**Acceptance:**
- [ ] Test: each behavior executes in isolation.
- [ ] Test: K fails trigger hard_stuck.
- [ ] Test: hard_stuck emits an event.

**Out of scope:** perception-aware recovery.

### T3.5 — FSM ↔ Reflex Integration

**Depends on:** T3.3, T2.3
**Deliverables:**
- `src/wow_bot/executor/reflex_bridge.py`
- `tests/test_reflex_bridge.py`

**Contract:**
- Reflex `pause_fsm` → FSM.pause() in <1 tick.
- Reflex `abort_actuation` → FSM to PAUSED and actuator.abort().
- After resume, FSM resumes from the exact previous state.

**Acceptance:**
- [ ] Test: pause latency <1 tick.
- [ ] Test: resume restores state.
- [ ] Test: abort path ends in PAUSED.

**Out of scope:** humanizer (Phase 7).

---

# Phase 4 — World Model

**Goal:** Spatial and entity memory, separate from Internal Dynamics.

**Phase acceptance:**
- T4.1 through T4.5 merged.
- 10k entities stored, queries <10 ms.
- Load of full map in startup <500 ms.
- Schema isolation from Internal Dynamics asserted by test.

### T4.1 — Schema Design

**Depends on:** Phase 3
**Deliverables:**
- `migrations/world_model/001_init.sql`
- `src/wow_bot/world/schema.py`
- `tests/test_world_schema.py`

**Contract:**
- Tables: `map_nodes`, `map_edges`, `entities_seen`, `routes_taken`,
  `combat_history`.
- All tables prefixed with `wm_` to avoid collision with internal
  dynamics tables.
- Foreign keys enforced.

**Acceptance:**
- [ ] Test: migration applies cleanly.
- [ ] Test: schema isolation from internal dynamics.
- [ ] Test: foreign keys reject orphans.

**Out of scope:** migrations for other phases.

### T4.2 — Store API

**Depends on:** T4.1
**Deliverables:**
- `src/wow_bot/world/store.py`
- `tests/test_world_store.py`

**Contract:**
- `WorldModel(path: Path)` opens the DB.
- Methods: `add_node`, `add_edge`, `mark_seen`, `record_route`,
  `record_combat`, `query_nearest(kind, from_xy)`, `get_node(id)`.
- All methods are async (`aiosqlite`).

**Acceptance:**
- [ ] Test: 10k nodes, `query_nearest` <10 ms.
- [ ] Test: edges are bidirectional when requested.
- [ ] Test: concurrent writes serialize correctly.

**Out of scope:** caching layer.

### T4.3 — GameState ↔ World Sync

**Depends on:** T4.2
**Deliverables:**
- `src/wow_bot/world/sync.py`
- `tests/test_world_sync.py`

**Contract:**
- `sync_from_state(state: GameState, wm: WorldModel) -> None`.
- Called at configurable cadence (default 1 Hz).
- Records player position, nearby entities, and current target.

**Acceptance:**
- [ ] Test: state with N entities writes N rows on first call.
- [ ] Test: second call updates rather than duplicates.
- [ ] Test: sync is idempotent for identical state.

**Out of scope:** perception (MockPerception stays).

### T4.4 — Strategist Summary

**Depends on:** T4.2
**Deliverables:**
- `src/wow_bot/world/summary.py`
- `tests/test_world_summary.py`

**Contract:**
- `summarize(wm, around_xy, radius) -> WorldSummary` with nearest
  vendors, trainers, nodes, and last N combat events.
- Output is a plain dataclass, JSON-serializable.

**Acceptance:**
- [ ] Test: summary includes exactly the within-radius nodes.
- [ ] Test: JSON serialization round-trips.

**Out of scope:** strategist integration (Phase 9).

### T4.5 — World Model Loader

**Depends on:** T4.2
**Deliverables:**
- `src/wow_bot/world/loader.py`
- `tests/test_world_loader.py`

**Contract:**
- `load_map(path: Path) -> WorldModel` with startup budget <500 ms.
- In LAB_MODE, must complete before actuation is enabled.

**Acceptance:**
- [ ] Test: 100k nodes load in <500 ms (in-memory warm).
- [ ] Test: corrupt DB raises clean error.

**Out of scope:** remote sync.

---

# Phase 5 — Navigation

**Goal:** Real movement between waypoints, with replanning and
recovery.

**Phase acceptance:**
- T5.1 through T5.5 merged.
- 20 consecutive trips without hard_stuck.
- Unexpected obstacle → replan in <2 s.
- CPU <15% on one core during navigation.

### T5.1 — Graph Builder

**Depends on:** T4.2
**Deliverables:**
- `src/wow_bot/nav/graph.py`
- `tests/test_nav_graph.py`

**Contract:**
- Builds a graph from `map_nodes` and `map_edges`.
- Edge cost = Euclidean distance × directional penalty from config.
- Graph is immutable after build; rebuild on World Model change.

**Acceptance:**
- [ ] Test: disconnected nodes are not connected in the graph.
- [ ] Test: directional penalty applied correctly.
- [ ] Test: rebuild is O(E + V).

**Out of scope:** dynamic edge updates.

### T5.2 — A* Pathfinding

**Depends on:** T5.1
**Deliverables:**
- `src/wow_bot/nav/astar.py`
- `tests/test_astar.py`

**Contract:**
- `find_path(graph, start, goal) -> list[Node] | None`.
- Deterministic tie-breaking.
- Returns `None` if unreachable.

**Acceptance:**
- [ ] Test: known small graphs return expected paths.
- [ ] Test: unreachable returns None.
- [ ] Test: determinism across 100 runs.

**Out of scope:** hierarchical A*.

### T5.3 — Navigator

**Depends on:** T5.2, T1.4
**Deliverables:**
- `src/wow_bot/nav/navigator.py`
- `tests/test_navigator.py`

**Contract:**
- `Navigator(graph, actuator, fsm, world)`.
- `go_to(target_xy) -> NavResult`.
- Delegates each segment to FSM `MOVING_TO_TARGET` / `PATHING`.
- On deviation, requests replan from current position.

**Acceptance:**
- [ ] Test: single-segment path executes.
- [ ] Test: multi-segment path executes.
- [ ] Test: replan triggered on deviation.

**Out of scope:** combat-aware navigation.

### T5.4 — Replanner

**Depends on:** T5.3
**Deliverables:**
- `src/wow_bot/nav/replan.py`
- `tests/test_replan.py`

**Contract:**
- Triggered when position deviates > threshold from path.
- Replans from current position to same goal.
- If replan fails N times, returns `hard_nav_failure`.

**Acceptance:**
- [ ] Test: replan in <2 s (simulated).
- [ ] Test: N failures → hard_nav_failure.
- [ ] Test: replan uses updated graph.

**Out of scope:** online graph learning.

### T5.5 — Navigation Telemetry

**Depends on:** T5.3
**Deliverables:**
- `src/wow_bot/nav/telemetry.py`
- `tests/test_nav_telemetry.py`

**Contract:**
- Emits `nav_started`, `nav_segment`, `nav_replan`, `nav_completed`.
- CPU sampling during navigation; exposes `cpu_percent` per second.

**Acceptance:**
- [ ] Test: events emitted for each phase.
- [ ] Test: CPU sampling returns finite numbers.
- [ ] Manual lab: CPU <15% during a 5-minute navigation.

**Out of scope:** dashboards.

---

# Phase 6 — Combat Engine

**Goal:** Combat rotation on training dummy, then a simple mob.

**Phase acceptance:**
- T6.1 through T6.5 merged.
- Kill training dummy 10/10.
- Interrupt within window <500 ms.
- Flee at low HP 9/10.
- No out-of-range / no-resource casts.

### T6.1 — Rotation Table

**Depends on:** Phase 5
**Deliverables:**
- `src/wow_bot/combat/rotation.py`
- `tests/test_rotation.py`

**Contract:**
- `RotationTable` loaded from config.
- Rules: `(condition) -> spell` with priority.
- Conditions: `in_range`, `resource>=x`, `cd_ready`, `target_hp<%`,
  `self_hp<%`.

**Acceptance:**
- [ ] Test: priority order respected.
- [ ] Test: unmet conditions skip rule.
- [ ] Test: empty rotation returns None.

**Out of scope:** PvP, group, raid.

### T6.2 — Target Selector

**Depends on:** T6.1
**Deliverables:**
- `src/wow_bot/combat/targeting.py`
- `tests/test_targeting.py`

**Contract:**
- Selects a target from `GameState` based on threat, distance, HP.
- Deterministic tie-breaking.
- Emits `target_changed` event.

**Acceptance:**
- [ ] Test: nearest valid target chosen.
- [ ] Test: tie-breaking deterministic.
- [ ] Test: invalid targets excluded.

**Out of scope:** crowd control.

### T6.3 — Combat Loop

**Depends on:** T6.1, T6.2, T3.3
**Deliverables:**
- `src/wow_bot/combat/loop.py`
- `tests/test_combat_loop.py`

**Contract:**
- Runs on reflex tick (fast) and FSM `COMBAT` (slow).
- Reflex: interrupt, defensive, flee triggers.
- FSM: rotation decisions.
- LLM is asserted to be unused here.

**Acceptance:**
- [ ] Test: no LLM import (static check).
- [ ] Test: rotation cast when conditions met.
- [ ] Test: interrupt fires within one tick.

**Out of scope:** advanced mechanics.

### T6.4 — Interrupt & Defensive Logic

**Depends on:** T6.3
**Deliverables:**
- `src/wow_bot/combat/reactive.py`
- `tests/test_combat_reactive.py`

**Contract:**
- Watches `GameState` for cast events on target.
- Fires interrupt spell if ready and in range.
- Fires defensive if self HP < threshold.
- Uses reflex tick budget.

**Acceptance:**
- [ ] Test: interrupt fires when cast detected.
- [ ] Test: interrupt skipped on cooldown.
- [ ] Test: defensive fires under low HP.

**Out of scope:** dispel, purge.

### T6.5 — Flee Logic

**Depends on:** T6.3, T5.3
**Deliverables:**
- `src/wow_bot/combat/flee.py`
- `tests/test_flee.py`

**Contract:**
- Triggered when `self_hp < flee_threshold` or `adds > N`.
- Selects a flee waypoint from World Model.
- Delegates to Navigator.
- Emits `flee_started`, `flee_completed`.

**Acceptance:**
- [ ] Test: flee triggers under threshold.
- [ ] Test: waypoint selection deterministic.
- [ ] Test: flee completes or reports failure.

**Out of scope:** Feign Death, Vanish specifics.

---

# Phase 7 — Humanizer (Real Actuation)

**Goal:** Apply Task 8.2 statistical model to real input timing.

**Phase acceptance:**
- T7.1 through T7.4 merged.
- PIT test on recorded intervals: p > 0.05.
- KS vs Uniform(0,1): p > 0.05.
- CV of intervals > 0.3.
- All humanization parameters logged.

### T7.1 — Interval Model

**Depends on:** T1.4
**Deliverables:**
- `src/wow_bot/humanize/intervals.py`
- `tests/test_humanize_intervals.py`

**Contract:**
- `sample_interval(rng, mu, sigma, clip) -> float`.
- Uses clipped lognormal.
- Parameters loaded from config with defaults from Task 8.2.

**Acceptance:**
- [ ] Test: samples within clip bounds.
- [ ] Test: PIT p > 0.05 over 10k samples.
- [ ] Test: KS vs Uniform p > 0.05.

**Out of scope:** per-class tuning.

### T7.2 — Cursor Path Model

**Depends on:** T1.4
**Deliverables:**
- `src/wow_bot/humanize/cursor.py`
- `tests/test_humanize_cursor.py`

**Contract:**
- `cursor_path(start, end, rng) -> list[(x, y, t)]`.
- Bezier with jittered control points.
- Duration lognormal, same model as T7.1.

**Acceptance:**
- [ ] Test: path starts and ends at given points.
- [ ] Test: monotone time.
- [ ] Test: length within [d, 2d] for straight-line distance d.

**Out of scope:** multi-monitor.

### T7.3 — Micro-Pauses & Miss-Clicks

**Depends on:** T7.1
**Deliverables:**
- `src/wow_bot/humanize/imperfections.py`
- `tests/test_humanize_imperfections.py`

**Contract:**
- `maybe_pause(rng, p) -> bool`
- `maybe_miss_click(rng, p, max_offset) -> (dx, dy) | None`
- Probabilities configurable; every event logged.

**Acceptance:**
- [ ] Test: pause rate matches configured p within tolerance.
- [ ] Test: miss-click offset bounded.
- [ ] Test: events logged.

**Out of scope:** skill-based degradation.

### T7.4 — Humanizer Integration in Actuator

**Depends on:** T7.1, T7.2, T7.3, T1.4
**Deliverables:**
- `src/wow_bot/actuation/humanized.py`
- `tests/test_humanized_actuator.py`

**Contract:**
- `HumanizedActuator(RealActuator, humanizer_config)`.
- Wraps driver calls with humanizer.
- All samples logged with parameters.

**Acceptance:**
- [ ] Test: interval sampling affects driver call timing.
- [ ] Test: cursor path replaces direct jumps.
- [ ] Test: disable flag bypasses humanizer.

**Out of scope:** A/B statistical comparison (Phase 10).

---

# Phase 8 — Behavioral Watchdog

**Goal:** Progress-based supervision, replacing pure liveness.

**Phase acceptance:**
- T8.1 through T8.4 merged.
- Loop-with-no-progress → CRITICAL in <60 s.
- Zero false positives in 1 h healthy run.
- Clean shutdown preserving logs.

### T8.1 — Progress Metrics

**Depends on:** Phase 4, Phase 5
**Deliverables:**
- `src/wow_bot/watchdog/metrics.py`
- `tests/test_wd_metrics.py`

**Contract:**
- Metrics: position delta, inventory delta, level delta, successful
  actions per minute.
- Sampled at 1 Hz, windowed over 60 s.

**Acceptance:**
- [ ] Test: each metric computed correctly on synthetic input.
- [ ] Test: windowing does not leak.

**Out of scope:** ML anomaly detection.

### T8.2 — Health States

**Depends on:** T8.1
**Deliverables:**
- `src/wow_bot/watchdog/health.py`
- `tests/test_wd_health.py`

**Contract:**
- States: `HEALTHY`, `DEGRADED`, `CRITICAL`.
- Thresholds configurable.
- Transitions emit events.

**Acceptance:**
- [ ] Test: threshold crossing transitions correctly.
- [ ] Test: no oscillation under noisy input (hysteresis).

**Out of scope:** adaptive thresholds.

### T8.3 — Loop Detector

**Depends on:** T8.2
**Deliverables:**
- `src/wow_bot/watchdog/loops.py`
- `tests/test_wd_loops.py`

**Contract:**
- Detects repeated action patterns without state change.
- Sliding window 600 s by default.
- Emits `loop_detected`.

**Acceptance:**
- [ ] Test: synthetic loop triggers in <60 s.
- [ ] Test: normal variation does not trigger.

**Out of scope:** root-cause analysis.

### T8.4 — Graceful Shutdown

**Depends on:** T8.2, T0.3
**Deliverables:**
- `src/wow_bot/watchdog/shutdown.py`
- `tests/test_wd_shutdown.py`

**Contract:**
- CRITICAL → `safety.abort("watchdog_critical")`.
- Waits for actuation release, flushes logs, writes `shutdown.json`.
- Exit code 0 on graceful, non-zero on unclean.

**Acceptance:**
- [ ] Test: shutdown path flushes logs.
- [ ] Test: exit code 0 on graceful.
- [ ] Test: no key held after shutdown.

**Out of scope:** remote notify.

---

# Phase 9 — Strategist v2

**Goal:** LLM grounded in real state, with dynamic cooldown.

**Phase acceptance:**
- T9.1 through T9.4 merged.
- 100 calls, 100% valid JSON.
- No out-of-vocabulary decisions.
- p95 LLM latency <5 s.

### T9.1 — Grounded Prompt Builder

**Depends on:** Phase 4
**Deliverables:**
- `src/wow_bot/strategist/prompts_v2.py`
- `tests/test_prompts_v2.py`

**Contract:**
- Input: `MetaState`, `WorldSummary`, current `GameState` snapshot.
- Output: prompt string with strict JSON schema block.
- Deterministic given inputs.

**Acceptance:**
- [ ] Test: prompt contains all sections.
- [ ] Test: schema block matches parser expectations.
- [ ] Test: determinism across 100 runs.

**Out of scope:** few-shot examples.

### T9.2 — Dynamic Cooldown

**Depends on:** Phase 6
**Deliverables:**
- `src/wow_bot/strategist/cooldown.py`
- `tests/test_cooldown.py`

**Contract:**
- `should_call(state) -> bool`.
- Combat active → false. Idle → true after cooldown seconds.
- Cooldown configurable; default 10-30 s.

**Acceptance:**
- [ ] Test: no calls during combat.
- [ ] Test: calls spaced by cooldown.
- [ ] Test: manual override works.

**Out of scope:** adaptive cooldown by RL.

### T9.3 — Vocabulary Guard

**Depends on:** T9.1
**Deliverables:**
- `src/wow_bot/strategist/vocab.py`
- `tests/test_vocab.py`

**Contract:**
- Allowed goals: existing set + `sell_vendor`, `repair`, `travel_to`.
- Any parsed goal outside set → rejected with `VocabViolation`.
- Rejection logged.

**Acceptance:**
- [ ] Test: each allowed goal parses.
- [ ] Test: unknown goal raises.
- [ ] Test: rejection event emitted.

**Out of scope:** learning new goals.

### T9.4 — Strategist Orchestrator v2

**Depends on:** T9.1, T9.2, T9.3
**Deliverables:**
- `src/wow_bot/strategist/orchestrator_v2.py`
- `tests/test_orchestrator_v2.py`

**Contract:**
- Consumes `MetaState`, `WorldSummary`, `GameState`.
- Calls local Ollama only.
- Returns `Strategy` with goal, rationale, and target.
- Every call logged with prompt hash and latency.

**Acceptance:**
- [ ] Test: mock Ollama returns valid strategy.
- [ ] Test: invalid JSON → retry once, then fail closed.
- [ ] Manual lab: p95 latency <5 s.

**Out of scope:** cloud fallback.

---

# Phase 10 — Reporting & Analysis v2

**Goal:** Lab-grade reports and metrics feeding Task 8.x analyses.

**Phase acceptance:**
- T10.1 through T10.4 merged.
- Reports schema-validated.
- Spectral analysis on lab data runs.
- Soak 24 h without crash.

### T10.0 — Reporting & Analysis Reconciliation

### T10.1 — Report Schema v2

**Depends on:** Phases 6-9
**Deliverables:**
- `schemas/report_v2.json`
- `src/wow_bot/reporting/schema_v2.py`
- `tests/test_report_schema_v2.py`

**Contract:**
- Sections: perception, action, reflex, navigation, combat, humanizer,
  strategist.
- Every section optional; presence depends on mode.

**Acceptance:**
- [ ] Test: valid report passes.
- [ ] Test: missing required section fails.
- [ ] Test: extra fields rejected.

**Out of scope:** report diffing.

### T10.2 — Lab Reporting Pipeline

**Depends on:** T10.1
**Deliverables:**
- `src/wow_bot/reporting/lab_pipeline.py`
- `tests/test_lab_pipeline.py`

**Contract:**
- Reads `events.jsonl`, produces `report.json` per session.
- Versioned output path.

**Acceptance:**
- [ ] Test: 10k events → valid report.
- [ ] Test: idempotent re-run.

**Out of scope:** cross-session aggregation.

### T10.3 — Spectral & Timing Analysis on Lab Data

**Depends on:** T10.2
**Deliverables:**
- `src/wow_bot/analysis/lab_spectral.py`
- `src/wow_bot/analysis/lab_timing.py`
- `tests/test_lab_analysis.py`

**Contract:**
- Reuses Task 8.1 and 8.2 code paths on lab reports.
- Produces plots under `runs/lab/<session>/analysis/`.

**Acceptance:**
- [ ] Test: spectral slope in [-1.5, -0.5] target band.
- [ ] Test: PIT and KS on lab intervals.

**Out of scope:** cross-session stats.

### T10.4 — Soak Harness v2

**Depends on:** T10.2
**Deliverables:**
- `scripts/lab/soak_v2.py`
- `tests/test_soak_v2.py`

**Contract:**
- Runs N hours, samples CPU, RSS, log size, progress metrics.
- On crash, writes `crash.json` and exits non-zero.

**Acceptance:**
- [ ] Test: dry-run soak for 60 s in MOCK_MODE.
- [ ] Manual lab: 24 h with zero crash.

**Out of scope:** multi-node soak.

---

# Phase 11 — End-to-End Farm Loop

**Goal:** Farm → loot → vendor → repair → return.

**Phase acceptance:**
- T11.1 through T11.4 merged.
- 1 h farm loop without hard_stuck.
- ≥50 successful cycles.
- Stable resource usage.

### T11.0 — Farm Reconciliation

### T11.1 — Farm Profile

**Depends on:** Phase 9
**Deliverables:**
- `config/farm_profiles/example.toml`
- `src/wow_bot/farm/profile.py`
- `tests/test_farm_profile.py`

**Contract:**
- Profile: nodes, mobs, vendor, repair point, route preferences.
- Loaded and validated at session start.

**Acceptance:**
- [ ] Test: invalid profile rejected.
- [ ] Test: profile drives World Model queries.

**Out of scope:** multiple profiles per session.

### T11.2 — Loot & Inventory

**Depends on:** Phase 6
**Deliverables:**
- `src/wow_bot/farm/loot.py`
- `tests/test_loot.py`

**Contract:**
- `loot_corpse(state) -> ActionResult`.
- Inventory full → emit `inventory_full` to FSM.

**Acceptance:**
- [ ] Test: loot action generated.
- [ ] Test: full inventory triggers vendor decision.

**Out of scope:** item filtering.

### T11.3 — Vendor & Repair

**Depends on:** Phase 5
**Deliverables:**
- `src/wow_bot/farm/vendor.py`
- `tests/test_vendor.py`

**Contract:**
- `sell_junk()`, `repair_all()`.
- Uses Navigator to reach vendor/repair.

**Acceptance:**
- [ ] Test: sell reduces inventory.
- [ ] Test: repair completes.
- [ ] Test: failure routes to recovery.

**Out of scope:** auction house.

### T11.4 — Farm Loop Orchestrator

**Depends on:** T11.1, T11.2, T11.3
**Deliverables:**
- `src/wow_bot/farm/loop.py`
- `tests/test_farm_loop.py`

**Contract:**
- Cycle: select node → navigate → farm → loot → check inventory →
  sell/repair → return.
- Integrates Reflex, FSM v2, World Model, Humanizer, Watchdog.

**Acceptance:**
- [ ] Test: single cycle completes in MOCK_MODE.
- [ ] Manual lab: 1 h run without hard_stuck.
- [ ] ≥50 cycles in the run.

**Out of scope:** multi-zone routing.

---

# Phase 12 — Soak & Final Analysis

**Goal:** 24-72 h stability and research-ready results.

**Phase acceptance:**
- T12.1 through T12.3 merged.
- Zero crashes over 24 h.
- No upward memory trend.
- RESULTS.md with all metrics.

### T12.1 — Long Soak

**Depends on:** Phase 11, T10.4
**Deliverables:**
- `runs/lab/<session>/soak_report.json`
- `docs/SOAK_PROTOCOL.md`

**Contract:**
- 24-72 h run with telemetry at fixed cadence.
- Protocol documented: start, stop, abort conditions.

**Acceptance:**
- [ ] Manual lab: 24 h, zero crashes.
- [ ] RSS trend slope not significantly positive.

**Out of scope:** multi-machine.

### T12.2 — Aggregate Analysis

**Depends on:** T12.1
**Deliverables:**
- `src/wow_bot/analysis/aggregate.py`
- `tests/test_aggregate.py`

**Contract:**
- Across-session metrics: crash rate, cycle success rate, humanizer
  distribution fit, reflex latency distribution.

**Acceptance:**
- [ ] Test: aggregate on synthetic sessions.
- [ ] Test: outputs schema-valid.

**Out of scope:** ML clustering.

### T12.3 — Results Write-up

**Depends on:** T12.2
**Deliverables:**
- `docs/RESULTS.md`

**Contract:**
- Sections: setup, metrics, spectral, timing, soak, threats to validity,
  explicit non-claims (no retail, no anti-cheat bypass, no third-party).

**Acceptance:**
- [ ] All numbers traced to a session id.
- [ ] Non-claims section present.
- [ ] Reproducibility commands included.

**Out of scope:** publication formatting.

---

# Appendix A — Dependency Graph

    T0.1 → T0.2 → T0.3 → T0.4
    ↘ T0.5 → T0.6
    T1.1 → T1.2 → T1.3 → T1.4 → T1.5
    T2.1 → T2.2 → T2.3
    ↘ T2.4
    T3.1 → T3.2 → T3.3 → T3.4
    ↘ T3.5
    T4.1 → T4.2 → T4.3
    ↘ T4.4
    ↘ T4.5
    T5.1 → T5.2 → T5.3 → T5.4
    ↘ T5.5
    T6.1 → T6.2 → T6.3 → T6.4
    ↘ T6.5
    T7.1 → T7.4
    T7.2 → T7.4
    T7.3 → T7.4
    T8.1 → T8.2 → T8.3
    ↘ T8.4
    T9.0 → T9.1
    T9.1 → T9.4
    T9.2 → T9.4
    T9.3 → T9.4
    T10.1 → T10.2 → T10.3
    ↘ T10.4
    T11.1 → T11.4
    T11.2 → T11.4
    T11.3 → T11.4
    T12.1 → T12.2 → T12.3



# Appendix B — Jules Prompt Template

Each task is handed to Jules with this template:

    You are implementing task <TASK_ID> from LAB_PHASE_ROADMAP.md.

    Context:
    - Repo: WoW-bot (research prototype, MOCK_MODE default).
    - Read AGENTS.md and LAB_CONSTRAINTS.md before starting.
    - MockPerception is the only GameState producer. Do not change
      its schema.
    - SafetyLayer is always active. Do not bypass it.

    Deliverables:
    <copy from task>

    Contract:
    <copy from task>

    Acceptance:
    <copy from task>

    Out of scope:
    <copy from task>

    Constraints:
    - Python 3.12, asyncio, aiosqlite, pytest, mypy strict, Ruff.
    - Per-component RNG. No global seeds.
    - No OS input or memory access outside designated modules.
    - Tests must run in MOCK_MODE and pass in CI.

    Output:
    - All files listed under Deliverables.
    - Tests under tests/ with the specified names.
    - A brief PR description mapping changes to Acceptance items.