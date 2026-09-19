# AGENTS.md — Project Context and Rules for Coding / Review Agents

> Read this file completely before modifying or reviewing the repository.

---

## 1. Project Identity

**Project:** WoW-Bot Research Prototype  
**Mode:** mock-first, dry-run, controlled simulation  
**Runtime:** Python 3.12+, asyncio  
**LLM:** local Ollama / Qwen 2.5 7B  
**Current specification:** frozen through Task 8.3  
**Task 9.1:** PENDING

The project studies software architecture, dynamical systems, local high-level planning, deterministic state machines, reproducible simulation, supervision, and statistical analysis.

Development and validation must remain on synthetic/mock/prerecorded/isolated inputs. The current executor must not generate physical OS input.

---

## 2. Source-of-Truth Order

For intended behavior:

1. `AGENTS.md` — global architecture and invariants
2. `ROADMAP.md` — task-specific contracts and acceptance
3. `LOCAL_VALIDATION_ROADMAP.md` — local-only runtime/scientific acceptance for Tasks 7.1–8.3
4. `REVIEWER_GUIDE.md` — audit procedure and required evidence
5. `README.md` — overview

The actual code is the subject being evaluated against these contracts.

If implementation and documentation conflict, do not silently reinterpret either side. Report the conflict with file/symbol evidence.

---

## 3. Agent Modes

### Coding mode

- work on exactly one requested task;
- inspect its dependencies before editing;
- make the smallest task-scoped change;
- preserve stable contracts;
- add behavioral evidence where appropriate;
- do not start the next task automatically.

### Reviewer mode

- do not modify code unless explicitly asked;
- audit the repository against `ROADMAP.md`;
- distinguish implementation defects from local-validation-pending items;
- never claim a local acceptance gate passed without artifacts/evidence.

---

## 4. Architecture

```text
MockPerception → GameState
                     │
                     ▼
              Internal Dynamics
                     │ MetaState
                     ▼
                 Strategist
                     │ Strategy
                     ▼
                ExecutorFSM
                     │
            simulation/dry-run only

Runtime health ──→ independent Watchdog ──→ shutdown request

Pipeline observer ──→ Scenario Report ──→ offline analysis
```

### Dependency direction

- Perception produces `GameState`.
- Internal Dynamics consumes `GameState`, produces `MetaState`.
- Strategist consumes `MetaState` + explicit context, produces `Strategy`.
- Executor consumes `GameState` + `Strategy`.
- Watchdog consumes health metadata, not full domain objects.
- Reporting is passive.
- Analysis consumes report files and does not mutate runtime behavior.

Avoid hidden cross-layer dependencies and shared mutable global state.

---

## 5. Stable Interface Contracts

### Canonical drive order

```text
[hunger, fatigue, curiosity, aggression, social]
```

Never reorder this vector silently.

### Strategy

Frozen fields:

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

### Contract evolution

Existing shared fields:

- must not be removed;
- must not be renamed;
- must not change meaning silently.

Additive fields must be optional with safe defaults unless an explicit coordinated contract change is approved.

---

## 6. Non-Negotiable Runtime Boundaries

### 6.1 Simulation Controller only

The current Controller is a dry-run command recorder.

Required:

```text
dry_run=True
```

Unsupported:

```text
dry_run=False
```

Do not add/use physical-input libraries or APIs such as:

```text
pynput
pyautogui
keyboard
mouse
DirectInput
Win32 input APIs
Quartz input APIs
xdotool
uinput
```

### 6.2 No direct game-process interaction

Do not add:

- DLL injection;
- process memory reading/writing;
- game hooks;
- addon manipulation;
- game-file manipulation;
- official-service deployment/validation.

### 6.3 Local network boundary

Runtime LLM traffic is local to Ollama. Do not introduce cloud LLM calls as a hidden fallback.

### 6.4 Preserve evidence

Do not delete/truncate logs or reports on failure. Emergency paths preserve diagnostic evidence.

---

## 7. Randomness and Determinism Rules

Randomness is allowed only where a task contract explicitly defines a stochastic simulation.

Required practices:

- use injectable `np.random.Generator`;
- use fixed seeds in tests/reproducible experiments;
- avoid global `np.random.seed`;
- do not search seeds to make statistical acceptance pass.

Deterministic components such as FSM and Watchdog health classification must remain deterministic for identical input/clock sequences.

---

## 8. Clock-Domain Rules

### Simulation time

Use `GameState.timestamp` for:

- Dynamics `dt`;
- adaptive LLM threshold;
- Strategy expiry;
- Strategist retry cooldown;
- deterministic scenario-domain calculations.

### Monotonic operational time

Use `time.monotonic()` for:

- heartbeat liveness;
- progress stall;
- Watchdog recovery windows;
- soak duration/resource sampling.

Never silently substitute one clock for the other.

---

## 9. Internal Dynamics Invariants

### Drives

- shape `(5,)`;
- `float64`;
- values remain `[0,1]`;
- baseline around `0.5`;
- event effects come from canonical registry;
- decay trends back toward baseline;
- callers must not receive a mutable reference that corrupts internal state.

### Oscillators

Frozen periods:

```text
5h, 90m, 20m, 5m, 1m
```

Frozen amplitudes:

```text
0.15, 0.08, 0.05, 0.03, 0.01
```

Seeded phase initialization; no claim that oscillators alone mathematically guarantee 1/f.

### Lorenz

- RK4;
- canonical Lorenz equations;
- deterministic;
- default `dt` around `0.001` unless the approved config says equivalent;
- finite bounded expected trajectory;
- deterministic projection/normalization.

### Memory

- async SQLite via `aiosqlite`;
- vectors stored safely, not with pickle;
- JSON event data;
- deterministic nearest-neighbor/tie behavior;
- explicit close lifecycle.

### MetaState step ordering

Expected conceptual order:

```text
chaos
→ oscillators
→ drives.step
→ apply events
→ decay
→ persist important events
→ construct MetaState
```

Adaptive trigger:

```text
threshold(t) = base + 0.1*sin(2π*t/7200)
```

using simulated elapsed time.

---

## 10. Strategist Invariants

### LLM Client

- `openai.AsyncOpenAI`;
- `httpx.AsyncClient(trust_env=False)`;
- local endpoint only;
- SDK retries disabled if custom retry policy owns retries;
- expected transient failures retried narrowly;
- cancellation preserved;
- explicit lifecycle close;
- no full prompt/raw response/secret logging.

### Prompt layer

Prompt construction is pure:

- no network;
- no database;
- no wall-clock read;
- no randomness.

`DynamicContext` supplies explicit runtime context.

### Parser

The current parser contract is strict.

Expected top-level fields:

```text
reasoning
goal
region
risk_tolerance
priority
constraints
```

`reasoning` is diagnostic and is not a Strategy field.

Parser rules:

- strict JSON contract;
- narrow fenced-JSON recovery;
- reject missing/extra fields;
- reject NaN/Infinity;
- reject bool where a real numeric/int is required;
- validate supported constraints;
- no silent coercion/repair;
- no LLM retry from parser;
- `valid_until` supplied explicitly by caller;
- failures raise the parser's narrow error type.

### Orchestrator

- one explicit generation request → at most one logical LLM query from orchestrator;
- no hidden orchestrator retry;
- expected refresh failure + existing previous Strategy → return the **same Strategy object** unchanged;
- no previous Strategy → propagate expected failure;
- cancellation/programming errors propagate;
- explicit `generate_strategy` means replan; it does not auto-skip solely because strategy is unexpired.

---

## 11. Executor Invariants

### Controller

Simulation-only command history.

`press_key`, `move_mouse`, `click`, and `stop_all` must not perform OS input.

### Timing model

```text
sigma = 0.4 + 0.2*chaos_component + 0.3*fatigue
mu = log(base_ms)
delay = clip(LogNormal(mu, sigma), 50, 2000)
```

- returns simulated milliseconds;
- no sleep;
- no Controller call;
- no global RNG.

Error probability:

```text
0.02 + 0.05*fatigue + 0.03*(1-curiosity)
```

Canonical indices:

```text
fatigue = vector[1]
curiosity = vector[2]
```

Actual formula range is `[0.02, 0.10]`; do not stretch it artificially to `0.15`.

### FSM

States exactly:

```text
IDLE
SCANNING
MOVING_TO_TARGET
COMBAT
LOOTING
FLEEING
STUCK_RECOVERY
```

Initial state:

```text
IDLE
```

Flee threshold:

```text
0.60 - 0.40*risk_tolerance
```

Enter flee when:

```text
in_combat and hp_pct <= threshold
```

No automatic flee solely for low HP outside combat.

Stuck detection:

- only movement-like intent;
- `GameState.timestamp`;
- exact threshold `>= 5.0s`;
- changed action resets timer;
- leaving movement clears timer;
- recovery clears tracker;
- entry to FLEEING/STUCK_RECOVERY may call simulated `stop_all()`;
- `press_key`, `move_mouse`, `click` are not called by FSM.

Strategy updates use `set_strategy`; they do not recreate/reset the FSM.

### Idle behavior

Exactly:

```text
CAMERA_WANDER
SUDDEN_PAUSE
INVENTORY_CHECK
SOCIAL_EMOTE
ANGLED_MOVEMENT
```

These are symbolic intents only.

Overall eligible-tick trigger probability:

```text
0.08
```

Curiosity changes selection weights, not trigger probability.

Allowed symbolic emotes:

```text
wave
laugh
```

No slash commands, physical input, camera geometry, or path coordinates.

### Path

Pure synthetic 2D geometry.

- `num_points` = total returned points;
- minimum `3`;
- endpoints exact;
- start != end;
- injectable RNG;
- smooth single-arc perpendicular offset;
- minimum curve amplitude `20px`;
- maximum `80px`;
- relative sample `10–20%` baseline length;
- canonical acceptance: std dev from direct line > `5px`;
- no Controller/FSM/timing coupling.

---

## 12. Watchdog Invariants

Watchdog is an independent supervisor, not an executor.

Architecture:

```text
main process → IPC messages → Watchdog process
Watchdog → shutdown Event → main process
```

Health states:

```text
HEALTHY
DEGRADED
CRITICAL
```

Frozen baseline thresholds unless the current approved code/config has an explicitly documented equivalent:

```text
startup grace                  30s
heartbeat degraded             >=10s
heartbeat critical             >=30s
progress degraded              >=30s
recovery window                60s
recovery degraded count        >=3
recovery critical count        >=5
continuous recovery critical   >=20s
death-loop window              600 simulation seconds
death-loop degraded count      >=3
death-loop critical count      >=5
graceful shutdown timeout      10s
poll interval                  0.5s
```

Critical rules:

- Watchdog does not reimplement FSM's 5-second stuck logic.
- Count entries into `STUCK_RECOVERY`, not repeated heartbeats while already there.
- DEGRADED does not automatically request shutdown.
- CRITICAL latches shutdown request.
- preserve logs;
- no F10/global hotkey listener in Watchdog core;
- no Controller calls from Watchdog.

---

## 13. Task 7.1 Pipeline Invariants

Expected logical loops:

```text
perception
dynamics
strategist
executor
watchdog heartbeat
watchdog shutdown bridge
```

Queue policy:

- ordered `GameState`/executor domain flow uses backpressure;
- strategist snapshot uses latest-value semantics;
- Strategy delivery may use latest-value semantics;
- avoid unbounded queues.

Progress token:

```text
one successfully completed Dynamics step
= +1
```

Strategist refresh reasons:

```text
no Strategy
OR expired Strategy
OR MetaState trigger
```

Failed refresh cooldown:

```text
10 simulation seconds
```

Initial Strategy failure is not replaced with a fabricated Strategy.

FSM is constructed lazily after first valid Strategy and reused thereafter.

Task 7.1 local runtime acceptance is **not** established by static/unit checks.

---

## 14. Task 7.2 Scenario Report Invariants

Report schema version:

```text
1
```

High-level shape:

```text
run
summary
strategist
fsm
meta_state
timing
watchdog
```

Required for Task 8.1:

```text
meta_state.dimensions
meta_state.samples[].simulation_timestamp
meta_state.samples[].vector
```

Required for Task 8.2:

```text
timing.unit
timing.samples_ms
```

Enhanced timing metadata should preserve `samples_ms` and add per-sample parameters rather than breaking the schema.

JSON must be strict (`NaN`/`Infinity` forbidden).

Instrumentation is passive and must not affect runtime decisions.

---

## 15. Task 8.1 Scientific Contract

- consume structured report, not logs;
- each drive analyzed independently;
- validate timestamps and dimensions;
- handle irregular sampling by deterministic uniform resampling;
- FFT diagnostics;
- Welch PSD;
- log-log slope fit;
- exclude DC/non-positive/non-finite PSD;
- target slope range `[-1.5, -0.5]`;
- result outside target is not a software exception;
- do not retune fit range or Dynamics after seeing the result.

---

## 16. Task 8.2 Scientific Contract

Primary exact model test:

```text
conditional clipped-lognormal
→ randomized PIT
→ KS against Uniform(0,1)
```

Fixed PIT seed:

```text
8202
```

Roadmap targets:

```text
KS p-value > 0.05
CV > 0.3
```

CV:

```text
sample_std(ddof=1) / sample_mean
```

Clipped samples remain included.

Legacy reports without per-sample model parameters may receive descriptive analysis but must not claim exact conditional-model acceptance.

---

## 17. Task 8.3 Soak Contract

Harness only; actual 24-hour execution is local.

Expected:

- configurable scenario/duration/sample interval/seed/output;
- default 24h duration may be 86400s;
- operational clock is monotonic;
- sample CPU, RSS memory, log size, progress, watchdog state where available;
- no auto-restart after crash;
- partial/checkpoint evidence preserved;
- strict JSON;
- objective crash criterion;
- no arbitrary memory-leak threshold.

Memory stability status remains:

```text
manual_review_required
```

until an explicit validated threshold is approved.

---

## 18. Phase 7–8 Validation Policy

For Tasks 7.1–8.3:

- an isolated coding agent may complete implementation with static/offline tests;
- local runtime/scientific gates are deferred;
- task status should say `LOCAL VALIDATION PENDING` where appropriate;
- reviewer must not misclassify this as missing implementation;
- reviewer must not claim final acceptance without the evidence described in `LOCAL_VALIDATION_ROADMAP.md`.

---

## 19. Task 9.1

```text
PENDING
```

Do not implement until the external perception dependency is delivered and explicitly unblocked.

---

## 20. Coding Quality

- Python 3.12+
- strict typing
- Ruff clean
- async cancellation preserved
- no broad `except Exception: continue`
- explicit lifecycle for long-lived resources
- no hidden blocking I/O inside event loop
- no secrets in logs
- no unrelated refactors
- task-scoped commits
- generated reports/logs/databases not committed

---

## 21. Definition of Done

### Tasks 0–6

A task normally requires:

- implementation;
- task-relevant executable evidence;
- regression safety;
- lint/type checks;
- no invariant regression.

### Tasks 7.1–8.3

A task can reach:

```text
IMPLEMENTED — LOCAL VALIDATION PENDING
```

when:

- implementation is complete;
- static/offline tests are credible;
- local-only checklist is documented.

Final acceptance requires `LOCAL_VALIDATION_ROADMAP.md`.

---

## 22. Reviewer Escalation

If a reviewer finds a mismatch:

1. cite task ID;
2. cite file/symbol;
3. explain expected contract;
4. show actual behavior;
5. classify whether it is:
   - implementation defect,
   - documentation mismatch,
   - local validation pending,
   - scientific result outside target,
   - external dependency pending.

Do not automatically "fix" scientific results by changing thresholds or models.
