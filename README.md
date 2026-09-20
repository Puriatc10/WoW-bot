# WoW-Bot Research Prototype

A **mock-first, dry-run academic research prototype** for studying layered autonomous-agent architecture, internal behavioral dynamics, reproducible simulation, local LLM planning, deterministic execution-state machines, supervision, and long-running statistical analysis.

---

## Scope

This is a research prototype for studying LLM-driven agent architectures
under human-like timing and reflex constraints.

- Default mode (`MOCK_MODE`) is fully synthetic and runs in CI.
- `LAB_MODE` runs against a private, self-owned server on an isolated
  network. It never connects to retail or third-party servers.
- See `LAB_CONSTRAINTS.md` for the hard boundaries.
- Anti-cheat bypass, memory inspection, and binary modification are
  explicitly out of scope.

---

## Current Project State

The project specification is frozen through **Task 8.3**.

- Phases 0–6 define the core architecture and component contracts.
- Phase 7 integrates the components and produces structured scenario reports.
- Phase 8 analyzes those reports and provides a long-running soak-test harness.
- Runtime/scientific acceptance for **Tasks 7.1–8.3 is intentionally local-only** and is documented in [`LOCAL_VALIDATION_ROADMAP.md`](./LOCAL_VALIDATION_ROADMAP.md).
- **Task 9.1 is PENDING** until the external perception dependency is available.
- A coding/reviewer agent must not infer that local validation passed merely because the implementation exists.

Read project documentation in this order:

1. [`README.md`](./README.md)
2. [`AGENTS.md`](./AGENTS.md)
3. [`ROADMAP.md`](./ROADMAP.md)
4. [`REVIEWER_GUIDE.md`](./REVIEWER_GUIDE.md)
5. [`LOCAL_VALIDATION_ROADMAP.md`](./LOCAL_VALIDATION_ROADMAP.md)

---

## Architecture

```text
                     ┌────────────────────────┐
                     │    MockPerception      │
                     │   → synthetic state    │
                     └───────────┬────────────┘
                                 │ GameState
                                 ▼
                     ┌────────────────────────┐
                     │   Internal Dynamics    │
                     │ Drives / Oscillators   │
                     │ Lorenz / Memory        │
                     └───────────┬────────────┘
                                 │ MetaState
                                 ▼
                     ┌────────────────────────┐
                     │      Strategist        │
                     │ local Ollama + parser  │
                     └───────────┬────────────┘
                                 │ Strategy
                                 ▼
                     ┌────────────────────────┐
                     │      Executor FSM      │
                     │ deterministic states   │
                     └───────────┬────────────┘
                                 │
                     symbolic / dry-run intent
                                 │
                                 ▼
                     ┌────────────────────────┐
                     │  Simulation Controller │
                     │ no physical OS input   │
                     └────────────────────────┘

                Runtime health metadata
                         │
                         ▼
              ┌─────────────────────┐
              │ Independent Watchdog│
              │ multiprocessing     │
              └─────────┬───────────┘
                        │ shutdown request
                        ▼
                 graceful cleanup

Scenario reports ──→ Spectrum analysis / Timing analysis / Soak analysis
```

### Layer responsibilities

| Layer | Owns | Produces | Must not own |
|---|---|---|---|
| Mock Perception | Synthetic scenario generation | `GameState` | strategy, executor behavior |
| Internal Dynamics | Drives, oscillators, Lorenz chaos, memory, trigger | `MetaState` | LLM calls, controller I/O |
| Strategist | Prompting, local LLM call, strict parse, fallback | `Strategy` | frame-by-frame action logic |
| Executor FSM | Deterministic state transitions | local execution state | LLM calls, wall-clock decision logic |
| Simulation helpers | timing/error/path/idle models | synthetic values/intents | physical input |
| Watchdog | liveness, progress, recovery/death-loop supervision | health + shutdown request | FSM business logic |
| Reporting | passive instrumentation | versioned JSON | changing runtime behavior |
| Analysis | offline scientific analysis | JSON + plots | tuning the model to pass |

---

## Stable Shared Contracts

The reviewer must inspect the actual definitions in `src/wow_bot/shared/interfaces.py`. The intended contract is:

### `GameState`

```text
timestamp
hp_pct
mana_pct
position
facing
in_combat
target
enemies
events
```

`timestamp` is the authoritative **simulation/domain time**.

### `MetaState`

```text
vector: shape (5,)
recent_events
timestamp
```

Canonical vector order:

```text
[hunger, fatigue, curiosity, aggression, social]
```

### `Strategy`

```text
goal
region
risk_tolerance
priority
constraints
valid_until
raw_llm_output = ""
```

Frozen goal vocabulary:

```text
farm_herbs
grind_humans
explore
flee
```

### `Event`

```text
type
timestamp
data
```

Public fields must not be removed or renamed silently. Additive changes must be backward-compatible and optional unless explicitly coordinated.

---

## Technical Stack

| Area | Choice |
|---|---|
| Language | Python 3.12+ |
| Concurrency | `asyncio` |
| Package manager | `uv` |
| LLM runtime | local Ollama |
| Default model | Qwen 2.5 7B |
| LLM client | OpenAI-compatible async client |
| HTTP transport | `httpx.AsyncClient(trust_env=False)` |
| Config/validation | Pydantic |
| Memory | SQLite via `aiosqlite` |
| Numerical/statistical work | NumPy + SciPy |
| Plotting | Matplotlib when Phase 8 requires it |
| Logging | Loguru |
| Testing | pytest + pytest-asyncio |
| Type checking | mypy strict |
| Linting | Ruff |
| Process supervision | `multiprocessing` |

A dependency must not be added only to work around an isolated agent environment.

---

## Repository Structure

```text
wow-bot/
├── README.md
├── ROADMAP.md
├── AGENTS.md
├── REVIEWER_GUIDE.md
├── LOCAL_VALIDATION_ROADMAP.md
├── pyproject.toml
├── config/
│   └── config.json
├── src/
│   └── wow_bot/
│       ├── shared/
│       ├── mocks/
│       ├── internal_dynamics/
│       ├── strategist/
│       ├── executor/
│       ├── watchdog/
│       ├── analysis/          # optional reusable helpers
│       ├── reporting/         # optional reusable helpers
│       └── main.py
├── scripts/
│   ├── run_scenario.py
│   ├── analyze_spectrum.py
│   ├── analyze_timing.py
│   └── run_soak_test.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
└── reports/                   # runtime artifacts; normally ignored
```

---

## Core Runtime Rules

### Dry-run only

The current controller is a **simulation recorder**.

- `dry_run=True` is the supported mode.
- `dry_run=False` must be rejected.
- The codebase must not depend on physical-input libraries such as `pynput`, `pyautogui`, OS keyboard hooks, DirectInput, `xdotool`, or `uinput`.
- FSM, idle behaviors, timing helpers, and path helpers must not bypass this boundary.

### Mock-first

Until Task 9.1 is explicitly unblocked:

- runtime perception comes from `MockPerception`;
- no real screenshot adapter is required;
- no direct game-process integration exists;
- Task 9.1 remains untouched.

### Local LLM only

The intended runtime uses Ollama on localhost. No cloud LLM is part of the runtime path.

### Evidence preservation

Shutdown paths preserve logs and diagnostics. No anti-forensic deletion or log truncation is allowed.

---

## Clock Domains

### Simulation/domain time

Use `GameState.timestamp` for:

- Dynamics `dt`;
- adaptive MetaState threshold;
- Strategy validity/expiry;
- Strategist retry cooldown;
- deterministic replay and scenario-domain timestamps.

### Real monotonic time

Use `time.monotonic()` only for operational concerns:

- Watchdog heartbeat age;
- progress stall duration;
- recovery supervision;
- soak-test elapsed runtime;
- resource sampling cadence.

### Async scheduling time

`asyncio.sleep()` may be used by runtime scheduling/harnesses, not as a replacement for domain time.

Clock-domain mixing is a high-risk defect.

---

## Development / Validation Model

### Tasks 0–6

Normal implementation tasks should have executable unit/integration evidence inside the regular test suite where practical.

### Tasks 7.1–8.3

These tasks are intentionally split:

```text
IMPLEMENTATION:
coding agent / Jules

RUNTIME OR SCIENTIFIC ACCEPTANCE:
developer local environment
```

An isolated coding agent must not:

- fake a 5-minute/10-minute/24-hour run;
- claim Ollama integration passed if Ollama was unavailable;
- alter thresholds to make synthetic tests green;
- mark implementation blocked only because long runtime validation is unavailable.

The local procedure is defined in [`LOCAL_VALIDATION_ROADMAP.md`](./LOCAL_VALIDATION_ROADMAP.md).

---

## Quality Gates

Typical static/offline gates:

```bash
uv run --extra dev ruff check .
uv run --extra dev mypy src
uv run --extra dev pytest
```

Use the repository-defined equivalent if configuration differs.

For Phase 7–8, these commands establish code quality but **do not replace** local runtime/scientific gates.

---

## Scenario Reports

Task 7.2 defines a versioned structured JSON research artifact.

Expected high-level shape:

```json
{
  "schema_version": 1,
  "run": {},
  "summary": {},
  "strategist": {},
  "fsm": {},
  "meta_state": {},
  "timing": {},
  "watchdog": {}
}
```

Important analysis contracts:

```text
meta_state.dimensions
meta_state.samples[*].simulation_timestamp
meta_state.samples[*].vector

timing.unit
timing.samples_ms
timing.samples[*]
```

Operational logs are not the authoritative scientific dataset.

---

## Phase 8 Analysis

### Spectrum

- consume Task 7.2 JSON;
- validate schema;
- derive sampling diagnostics from simulation timestamps;
- resample onto a uniform grid using median `dt`;
- analyze each drive independently;
- use FFT plus Welch PSD;
- fit slope in log-log space;
- target per-drive range: `[-1.5, -0.5]`;
- valid analysis outside the target is a scientific result, not a software exception.

### Timing

Timing model:

```text
mu = log(base_ms)
sigma = 0.4 + 0.2*chaos_component + 0.3*fatigue
delay = clip(LogNormal(mu, sigma), 50, 2000)
```

Primary validation:

- randomized probability-integral transform (PIT);
- boundary-aware clipping treatment;
- KS test against `Uniform(0,1)`;
- fixed PIT seed `8202`;
- roadmap target `p > 0.05`;
- CV computed from actual clipped delays;
- roadmap target `CV > 0.3`.

### Soak

Task 8.3 provides the harness. It must not invent a memory-leak threshold.

It records CPU, RSS memory, log growth, progress, termination reason, crash outcome, and memory trend.

The **24-hour run is local-only**. Zero crashes is objective; memory stability requires evidence review.

---

## Working With Agents

### Coding agent

Read `AGENTS.md`, `ROADMAP.md`, then current task dependencies. Implement exactly one task.

### Reviewer agent

Read all five documentation files, then compare actual code against frozen contracts.

Reviewer status must distinguish:

```text
implemented
static/offline validated
local runtime validated
scientifically accepted
```

---

## Project-Level Invariants

Flag violations of:

- simulation-only Controller;
- no physical OS input;
- no direct game-process/memory access;
- no cloud LLM fallback;
- logs preserved;
- shared contracts stable;
- canonical drive order unchanged;
- no backward simulation time acceptance;
- no fabricated initial Strategy;
- FSM reused across Strategy updates;
- Watchdog does not duplicate FSM stuck logic;
- no global F10/hotkey Watchdog core;
- report-based scientific analysis;
- no post-result threshold/seed tuning;
- Task 9.1 remains pending.

---

## Task 9.1 Status

```text
PENDING
```

Do not implement real Perception until the external dependency is delivered and a new explicit task is issued.
