# Farm Reconciliation (T11.0)

This document is the normative reconciliation reference for Phase 11 ("End-to-End Farm Loop").
It audits all public symbols across `src/wow_bot/lab/` and `src/wow_bot/main.py`, classifies their
relationship to Phase 11 tasks (T11.1–T11.4), resolves structural conflicts, defines naming conventions,
establishes frozen interface declarations, and provides the authoritative Module Ownership Map for LAB_MODE execution.

---

## Inventory

Below is the complete inventory of public top-level symbols defined in:
- `src/wow_bot/lab/__init__.py`
- `src/wow_bot/lab/movement_harness.py`
- `src/wow_bot/lab/scenario.py`
- `src/wow_bot/main.py`

| Name | Kind | Module | Responsibility | Exported via `lab/__init__.py` | Exercised in `tests/` | Used by `main.py` | Used by lab-phase modules |
|---|---|---|---|---|---|---|---|
| `HarnessResult` | class | `src/wow_bot/lab/movement_harness.py` | Immutable result snapshot of a scenario movement harness run. | Yes | Yes (`tests/test_movement_harness.py`) | No | No |
| `MovementHarness` | class | `src/wow_bot/lab/movement_harness.py` | Outer scenario harness driving actuator waypoint by waypoint. | Yes | Yes (`tests/test_movement_harness.py`) | No | No |
| `ALLOWED_TOP_LEVEL_KEYS` | constant | `src/wow_bot/lab/scenario.py` | Frozen set of allowed top-level JSON keys for scenario specifications. | No | Yes (`tests/test_movement_harness.py`) | No | No |
| `ScenarioError` | exception | `src/wow_bot/lab/scenario.py` | Exception raised when scenario loading, parsing, or validation fails. | Yes | Yes (`tests/test_movement_harness.py`) | No | No |
| `Scenario` | class | `src/wow_bot/lab/scenario.py` | Immutable scenario specification for movement runs. | Yes | Yes (`tests/test_movement_harness.py`) | No | No |
| `validate_waypoints` | function | `src/wow_bot/lab/scenario.py` | Pure function validating waypoint tuple format, finiteness, and non-emptiness. | Yes | Yes (`tests/test_movement_harness.py`) | No | No |
| `load_scenario` | function | `src/wow_bot/lab/scenario.py` | Reads and validates a movement scenario specification from a JSON file. | Yes | Yes (`tests/test_movement_harness.py`) | No | No |
| `STRATEGIST_RETRY_COOLDOWN_SECONDS` | constant | `src/wow_bot/main.py` | Cooldown in simulation seconds after a failed/fallback Strategist refresh. | No | Yes (`tests/unit/test_main.py`) | Yes | No |
| `WATCHDOG_HEARTBEAT_INTERVAL_SECONDS` | constant | `src/wow_bot/main.py` | Watchdog heartbeat interval in real wall-clock seconds. | No | Yes (`tests/unit/test_main.py`) | Yes | No |
| `PipelineSnapshot` | class | `src/wow_bot/main.py` | Immutable integration snapshot coupling GameState and MetaState. | No | Yes (`tests/unit/test_main.py`) | Yes | No |
| `RuntimeComponents` | class | `src/wow_bot/main.py` | Container holding instantiated pipeline runtime components for MOCK_MODE. | No | Yes (`tests/unit/test_main.py`, `tests/unit/test_review_gate.py`) | Yes | No |
| `build_runtime` | async function | `src/wow_bot/main.py` | Constructs and initializes pre-lab MOCK_MODE pipeline runtime components. | No | Yes (`tests/unit/test_main.py`, `tests/unit/test_review_gate.py`) | Yes | No |
| `perception_loop` | async function | `src/wow_bot/main.py` | Fetches GameState from MockPerception and publishes to perception_queue. | No | Yes (`tests/unit/test_main.py`) | Yes | No |
| `dynamics_loop` | async function | `src/wow_bot/main.py` | Processes GameState through Internal Dynamics and publishes PipelineSnapshot. | No | Yes (`tests/unit/test_main.py`) | Yes | No |
| `strategist_loop` | async function | `src/wow_bot/main.py` | Evaluates strategy rules and calls pre-lab Strategist orchestrator. | No | Yes (`tests/unit/test_main.py`) | Yes | No |
| `executor_loop` | async function | `src/wow_bot/main.py` | Ticks pre-lab ExecutorFSM on incoming PipelineSnapshots. | No | Yes (`tests/unit/test_main.py`) | Yes | No |
| `watchdog_heartbeat_loop` | async function | `src/wow_bot/main.py` | Publishes periodic runtime heartbeats to Watchdog message queue. | No | Yes (`tests/unit/test_main.py`) | Yes | No |
| `watchdog_shutdown_bridge` | async function | `src/wow_bot/main.py` | Bridges Watchdog IPC shutdown signals to asyncio shutdown event. | No | Yes (`tests/unit/test_main.py`) | Yes | No |
| `run_pipeline` | async function | `src/wow_bot/main.py` | Executes pre-lab MOCK_MODE application pipeline loops under supervision. | No | Yes (`tests/unit/test_main.py`) | Yes | No |
| `main` | async function | `src/wow_bot/main.py` | Pre-lab application CLI entry point. | No | Yes (`tests/unit/test_main.py`) | Yes | No |

---

## Classification

Each public symbol is classified into exactly one of `{KEEP_AS_IS, KEEP_AND_EXTEND, RENAME, DEPRECATE, REPLACE, NEW}`:

1. `HarnessResult`: `KEEP_AS_IS` — Scenario movement result snapshot used by T1.5 scenario harness tests.
2. `MovementHarness`: `KEEP_AS_IS` — Scenario movement harness used by T1.5 scenario tests.
3. `ALLOWED_TOP_LEVEL_KEYS`: `KEEP_AS_IS` — Constant frozen set for scenario JSON parsing validation.
4. `ScenarioError`: `KEEP_AS_IS` — Exception class raised during scenario parsing and validation.
5. `Scenario`: `KEEP_AS_IS` — Scenario spec dataclass for movement tests.
6. `validate_waypoints`: `KEEP_AS_IS` — Pure validation function for movement waypoints.
7. `load_scenario`: `KEEP_AS_IS` — Scenario specification loader function.
8. `STRATEGIST_RETRY_COOLDOWN_SECONDS`: `KEEP_AS_IS` — Pre-lab MOCK_MODE constant retained frozen for `main.py`.
9. `WATCHDOG_HEARTBEAT_INTERVAL_SECONDS`: `KEEP_AS_IS` — Pre-lab MOCK_MODE constant retained frozen for `main.py`.
10. `PipelineSnapshot`: `KEEP_AS_IS` — Pre-lab integration snapshot coupling GameState and MetaState for `main.py`.
11. `RuntimeComponents`: `KEEP_AS_IS` — Pre-lab container of MOCK_MODE pipeline components for `main.py`.
12. `build_runtime`: `KEEP_AS_IS` — MOCK_MODE runtime initializer retained frozen for `main.py`.
13. `perception_loop`: `KEEP_AS_IS` — MOCK_MODE perception task loop retained frozen for `main.py`.
14. `dynamics_loop`: `KEEP_AS_IS` — MOCK_MODE internal dynamics task loop retained frozen for `main.py`.
15. `strategist_loop`: `KEEP_AS_IS` — MOCK_MODE strategist task loop retained frozen for `main.py`.
16. `executor_loop`: `KEEP_AS_IS` — MOCK_MODE executor task loop retained frozen for `main.py`.
17. `watchdog_heartbeat_loop`: `KEEP_AS_IS` — MOCK_MODE watchdog heartbeat task loop retained frozen for `main.py`.
18. `watchdog_shutdown_bridge`: `KEEP_AS_IS` — MOCK_MODE watchdog shutdown bridge task loop retained frozen for `main.py`.
19. `run_pipeline`: `KEEP_AS_IS` — MOCK_MODE pipeline supervisor retained frozen for `main.py`.
20. `main`: `KEEP_AS_IS` — MOCK_MODE CLI entry point retained frozen in `main.py`.

---

## Conflict Resolution

The following subsections address key architectural conflicts C1 through C7.

### C1. Entry Point Conflict (`main.py` vs LAB_MODE runner)
- **Affected symbols:** `main.py` (`build_runtime`, `run_pipeline`, `main`, `RuntimeComponents`).
- **Conflict:** `main.py` is a pre-lab entry point written for MOCK_MODE. It imports pre-lab components (`ExecutorFSM`, dry-run `Controller`, `Strategist`, pre-lab `PipelineObserver`, `MetaStateGenerator`). It does NOT import any lab-phase components built in Phases 0–10 (such as `fsm_v2.FSM`, `ReflexLoop`, `WorldModel`, `Navigator`, `CombatLoop`, `HumanizedActuator`, `OrchestratorV2`, `Session`, `SafetyLayer`). Therefore, `main.py` cannot serve as the LAB_MODE runner.
- **Resolution Strategy:** **OPTION A (RECOMMENDED)**. `main.py` remains frozen exclusively for MOCK_MODE pipeline execution. A new entry point `src/wow_bot/lab/runner_v2.py` will be declared (and implemented in T11.4) for LAB_MODE orchestration.
- **Migration Path:** MOCK_MODE runs continue using `main.py`. LAB_MODE runs will execute via `lab/runner_v2.py`.
- **Pre-lab tests impact:** None. All pre-lab tests in `tests/unit/test_main.py` continue to pass without modification.

### C2. FSM Implementation Coexistence
- **Affected symbols:** `wow_bot.executor.fsm.ExecutorFSM` vs `wow_bot.executor.fsm_v2.FSM`.
- **Conflict:** `main.py` uses pre-lab `ExecutorFSM`, whereas Phase 11 farm orchestration requires `fsm_v2.FSM` (from Task 3.3) with feedback intake and reflex control bridge.
- **Resolution Strategy:** `ExecutorFSM` remains untouched in `executor/fsm.py` for pre-lab `main.py`. The Phase 11 farm loop (`farm/loop.py`) and lab runner (`lab/runner_v2.py`) use `fsm_v2.FSM`.
- **Pre-lab tests impact:** None.

### C3. Strategist Implementation Coexistence
- **Affected symbols:** `wow_bot.strategist.orchestrator.Strategist` vs `wow_bot.strategist.orchestrator_v2.OrchestratorV2`.
- **Conflict:** `main.py` uses pre-lab `Strategist`, whereas Phase 11 requires grounded Strategist v2 (`OrchestratorV2` from Task 9.4) with dynamic cooldown gate and vocabulary guard.
- **Resolution Strategy:** `Strategist` remains untouched in `strategist/orchestrator.py` for `main.py`. Phase 11 farm loop uses `OrchestratorV2`.
- **Pre-lab tests impact:** None.

### C4. Actuation Entry Point Coexistence
- **Affected symbols:** `wow_bot.executor.controller.Controller` vs `wow_bot.actuation.actuator.RealActuator` / `HumanizedActuator`.
- **Conflict:** `main.py` uses pre-lab dry-run `Controller`, whereas LAB_MODE requires `HumanizedActuator` wrapping `RealActuator` backed by `InputDriver`.
- **Resolution Strategy:** `Controller` remains untouched in `executor/controller.py` for `main.py`. Phase 11 uses `HumanizedActuator` / `RealActuator`.
- **Pre-lab tests impact:** None.

### C5. Telemetry Path Coexistence
- **Affected symbols:** `wow_bot.reporting.scenario.PipelineObserver` vs `Session.write_event` / `events.jsonl`.
- **Conflict:** `main.py` uses pre-lab in-memory `PipelineObserver` (schema v1), whereas LAB_MODE requires append-only `Session` writing JSON lines to `events.jsonl`.
- **Resolution Strategy:** `PipelineObserver` remains intact for pre-lab scenario reporting. LAB_MODE farm orchestration emits events directly via `Session.write_event`.
- **Pre-lab tests impact:** None.

### C6. MovementHarness Scope Boundary
- **Affected symbols:** `wow_bot.lab.movement_harness.MovementHarness`.
- **Conflict:** `MovementHarness` (Task 1.5) is a scenario movement driver that executes waypoint navigation without combat, loot, vendor, or inventory logic.
- **Resolution Strategy:** `MovementHarness` is `KEEP_AS_IS`. Phase 11 farm orchestration will NOT build on or modify `MovementHarness`; instead, T11.4 will implement a dedicated farm loop in `src/wow_bot/farm/loop.py`.
- **Rationale:** `MovementHarness` lacks entity targeting, spell rotation, corpse looting, vendor selling/repairing, and FSM v2 integration. Extending it would break its single-purpose scenario testing contract.
- **Pre-lab tests impact:** None.

### C7. New `farm/` Package Structure
- **Affected symbols:** None (no pre-lab `farm/` package exists).
- **Conflict:** Phase 11 requires farm profile, loot, vendor, and loop orchestrator modules.
- **Resolution Strategy:** Tasks T11.1–T11.4 will create a new top-level package `src/wow_bot/farm/` containing:
  - `farm/profile.py` (T11.1)
  - `farm/loot.py` (T11.2)
  - `farm/vendor.py` (T11.3)
  - `farm/loop.py` (T11.4)
- **Naming Discipline:** Since no pre-lab `farm/` package exists, no versioning suffix (e.g. `_v2`) is needed for files inside `farm/`.

---

## Naming Discipline

1. **`farm/` Package:**
   - Modules in `src/wow_bot/farm/` use clean domain names without version suffixes:
     - `profile.py`
     - `loot.py`
     - `vendor.py`
     - `loop.py`
   - Reason: No pre-lab legacy counterparts exist for these domain responsibilities.

2. **LAB_MODE Entry Point (`lab/runner_v2.py`):**
   - The new LAB_MODE runner is named `src/wow_bot/lab/runner_v2.py`.
   - Reason: Follows the same versioning convention as Phase 10 (`lab_pipeline_v2.py`, `lab_spectral_v2.py`, `lab_timing_v2.py`, `lab_soak_v2.py`) to explicitly distinguish lab-phase infrastructure from pre-lab MOCK_MODE entry points (`main.py`).

---

## Interface Freeze Declarations

The following contracts specify the strict interface boundaries for authors implementing Phase 11 tasks T11.1 through T11.4.

### 1. `src/wow_bot/lab/__init__.py`
- **MUST NOT CHANGE:** `__all__` exports (`HarnessResult`, `MovementHarness`, `Scenario`, `ScenarioError`, `load_scenario`, `validate_waypoints`).
- **MAY ADD:** Re-export for `runner_v2` symbols when T11.4 is completed (not before).
- **MAY REFACTOR FREELY:** Module docstring and internal comments.

### 2. `src/wow_bot/lab/movement_harness.py`
- **MUST NOT CHANGE:** `HarnessResult`, `MovementHarness`, `MovementHarness.__init__`, `MovementHarness.run`.
- **MAY ADD:** None. Pre-lab scenario harness is frozen.
- **MAY REFACTOR FREELY:** Private helper methods `_make_result`.

### 3. `src/wow_bot/lab/scenario.py`
- **MUST NOT CHANGE:** `ALLOWED_TOP_LEVEL_KEYS`, `ScenarioError`, `Scenario`, `validate_waypoints`, `load_scenario`.
- **MAY ADD:** None. Pre-lab scenario module is frozen.
- **MAY REFACTOR FREELY:** Internal parsing loops inside `load_scenario`.

### 4. `src/wow_bot/main.py`
- **MUST NOT CHANGE:** Completely frozen. T11.x tasks MUST NOT modify `main.py`.

### 5. `src/wow_bot/lab/runner_v2.py` (Planned Public API for T11.4)
- **PLANNED EXPORTS:**
  - `build_lab_runtime(config: Config, session: Session) -> LabRuntime`
  - `run_lab_loop(runtime: LabRuntime) -> LabRunResult`
  - `LabRuntime` (dataclass)
  - `LabRunResult` (dataclass)
- **NOTE:** Implementation will occur in T11.4. Declarations in this document establish the contract.

---

## Module Ownership Map

This table is the **SINGLE source of truth** for capability ownership in LAB_MODE execution and T11.4 orchestration wiring:

| Capability / Layer | Owning Module / Class in LAB_MODE | Mode / Phase | Notes |
|---|---|---|---|
| **Perception** | `wow_bot.mocks.mock_perception.MockPerception` | MOCK / LAB | Unchanged; MockPerception remains sole GameState producer. |
| **Internal Dynamics** | `wow_bot.internal_dynamics.*` | MOCK only | Not used in LAB_MODE. |
| **FSM** | `wow_bot.executor.fsm_v2.FSM` | LAB (T3.3) | Feedback-driven FSM with state schema v2. |
| **Reflex** | `wow_bot.reflex.loop.ReflexLoop` | LAB (T2.1) | Fast loop running signal sources and control sinks. |
| **Navigation** | `wow_bot.nav.navigator.Navigator` | LAB (T5.3) | A* pathfinding graph navigation with replanning. |
| **World Model** | `wow_bot.world.store.WorldModel` | LAB (T4.2) | SQLite spatial and entity store (`wm_` prefix). |
| **Combat** | `wow_bot.combat.loop.CombatLoop`, `wow_bot.combat.reactive.ReactiveCombat` | LAB (T6.3, T6.4) | Fast/slow loop combat rotation and reactive interrupts. |
| **Flee** | `wow_bot.combat.flee.FleeController` | LAB (T6.5) | Low-HP and multi-add flee controller. |
| **Humanizer** | `wow_bot.actuation.humanized.HumanizedActuator` | LAB (T7.4) | Wraps `RealActuator` with lognormal timing and micro-pauses. |
| **Strategist** | `wow_bot.strategist.orchestrator_v2.OrchestratorV2` | LAB (T9.4) | High-level strategy selection via local Ollama. |
| **Cooldown Gate** | `wow_bot.strategist.cooldown_v2.CooldownGate` | LAB (T9.2) | Dynamic LLM query rate limiter and state gate. |
| **Vocabulary Guard** | `wow_bot.strategist.vocab_v2.VocabularyGuard` | LAB (T9.3) | Goal vocabulary and schema validator. |
| **Watchdog** | `wow_bot.watchdog.watchdog.WatchdogProcess` | LAB (T8.x) | Behavioral progress and loop supervisor process. |
| **Shutdown** | `wow_bot.watchdog.shutdown.GracefulShutdown` | LAB (T8.4) | Idempotent shutdown coordinator. |
| **Session** | `wow_bot.session.Session` | LAB (T0.2) | Append-only JSON line session event logger. |
| **Safety** | `wow_bot.safety.SafetyLayer` | LAB (T0.3) | Arming, allowlist, network isolation, and abort safety. |
| **Farm Profile** | `wow_bot.farm.profile.FarmProfile` | LAB (T11.1) | Farm route, target mob, vendor, repair configuration. |
| **Loot & Inventory** | `wow_bot.farm.loot.LootManager` | LAB (T11.2) | Corpse looting and inventory capacity detection. |
| **Vendor & Repair** | `wow_bot.farm.vendor.VendorManager` | LAB (T11.3) | Junk selling and item repair via Navigator. |
| **Farm Orchestrator** | `wow_bot.farm.loop.FarmLoop` | LAB (T11.4) | End-to-end cycle: select → navigate → farm → loot → sell → return. |
| **Entry Point (LAB)** | `wow_bot.lab.runner_v2` | LAB (T11.4) | Asynchronous LAB_MODE application runtime builder and runner. |
