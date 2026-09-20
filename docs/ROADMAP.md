# WoW-Bot Implementation Roadmap — Reviewer-Oriented Frozen Specification

> **Approach:** mock-first, dry-run, simulation/research only  
> **Runtime:** Python 3.12+, asyncio  
> **LLM:** local Ollama + Qwen 2.5 7B  
> **Specification baseline:** frozen through Task 8.3  
> **Task 9.1:** PENDING  
> **Validation policy:** Tasks 7.1–8.3 require separate local acceptance.

---

## How to Read This Roadmap

This document defines **what the repository is expected to implement**.

For every task, a reviewer should check:

1. dependency contract;
2. required output/module;
3. frozen behavior;
4. negative boundaries;
5. acceptance evidence;
6. whether validation is agent-executable or local-only.

Historical pseudocode that conflicts with the frozen rules below is superseded by this document.

---

# Phase 0 — Bootstrap

## Task 0.1 — Project Structure

**Output:** repository/package/test/script/config skeleton.

**Review criteria**

- package layout exists;
- Python package directories have `__init__.py`;
- docs/config/scripts/tests are separated;
- no generated runtime artifacts tracked.

---

## Task 0.2 — Python Project / Tooling

**Output:** `pyproject.toml`

**Core requirements**

- Python 3.12+;
- `uv`;
- OpenAI-compatible client + httpx;
- Pydantic;
- NumPy/SciPy;
- aiosqlite;
- Loguru;
- pytest / pytest-asyncio;
- Ruff;
- mypy strict.

Phase 8 may add Matplotlib when required. Task 8.3 may add psutil only if no existing resource probe can satisfy process CPU/RSS measurement.

**Acceptance**

- dependencies resolve;
- imports work;
- project tooling commands are defined.

---

## Task 0.3 — Configuration

**Output:** `config/config.json`, `src/wow_bot/shared/config.py`

**Required**

- typed Pydantic settings;
- local Ollama configuration;
- Dynamics parameters;
- memory DB path;
- executor dry-run defaults;
- logging configuration;
- clear validation failures.

**Invariant**

```text
executor dry-run remains enabled/supported;
physical execution is not enabled by config.
```

---

## Task 0.4 — Shared Logger

**Output:** `src/wow_bot/shared/logger.py`

**Required**

- Loguru wrapper;
- console + configured file;
- rotation;
- millisecond timestamps;
- tagged loggers;
- `get_logger(tag)`.

**Invariant**

No secrets/full prompts/raw model responses should be logged unnecessarily.

---

## Task 0.5 — `.gitignore`

Ignore at minimum:

```text
__pycache__/
*.py[cod]
.venv/
venv/
.env
logs/
data/
reports/
.pytest_cache/
.mypy_cache/
.ruff_cache/
*.db
```

Generated scientific outputs normally remain untracked.

---

# Phase 1 — Shared Contracts

## Task 1.1 — GameState and Subtypes

**Output:** `src/wow_bot/shared/interfaces.py`

Required conceptual contracts:

```text
TargetInfo
EnemyInfo
Event
GameState
```

`GameState` fields:

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

**Acceptance**

- valid object construction/serialization;
- strict typing;
- normalized percentage fields;
- no hidden OS/runtime dependency.

---

## Task 1.2 — MetaState and Strategy

`MetaState`:

```text
vector shape (5,)
recent_events
timestamp
```

Canonical vector order:

```text
[hunger, fatigue, curiosity, aggression, social]
```

`Strategy`:

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

**Acceptance**

- risk tolerance `[0,1]`;
- vector contract preserved;
- existing shared fields not removed/renamed.

---

## Task 1.3 — Event Registry

Canonical event types include:

```text
death
rare_loot
pvp_hit
pvp_kill
stuck
level_up
quest_complete
npc_interact
```

Event effects must define all five drive keys.

Reviewer should compare actual values with the committed canonical registry and ensure Dynamics uses that registry rather than duplicating effects.

---

# Phase 2 — Mock Perception

## Task 2.1 — Basic MockPerception

**Output:** `src/wow_bot/mocks/mock_perception.py`

**Required**

- async `get_state()`;
- valid bounded `GameState`;
- synthetic position/events/combat;
- no external game dependency.

---

## Task 2.2 — Scenario-Driven MockPerception

Canonical scenarios:

```text
peaceful_farm
combat_light
death_loop
rare_loot_drought
stuck_repeatedly
```

**Required**

- scenario selected explicitly;
- deterministic under seed;
- scenario events/state transitions support downstream tests.

---

# Phase 3 — Internal Dynamics

## Task 3.1 — Drives

**Output:** `src/wow_bot/internal_dynamics/drives.py`

Frozen invariants:

- vector shape `(5,)`;
- `float64`;
- baseline `0.5`;
- bounded `[0,1]`;
- fatigue drifts faster than social;
- chaos scalar may perturb drift subtly;
- event effects use shared registry;
- decay toward baseline;
- caller cannot mutate internal vector through returned reference.

---

## Task 3.2 — Oscillator Bank

**Output:** oscillator module

Frozen periods:

```text
5 hours
90 minutes
20 minutes
5 minutes
1 minute
```

Amplitudes:

```text
0.15
0.08
0.05
0.03
0.01
```

**Required**

- seeded local RNG phases;
- deterministic when seeded;
- bounded combined output;
- no false claim that this component alone guarantees a 1/f process.

---

## Task 3.3 — Lorenz Attractor

**Output:** Lorenz/chaos module

**Required**

- canonical Lorenz equations;
- RK4, not Euler;
- float64;
- default/frozen `dt` around `0.001`;
- deterministic;
- finite/bounded expected trajectory;
- fixed deterministic projection/normalization;
- no unstable online normalization that changes semantics unpredictably.

---

## Task 3.4 — Memory Store

**Output:** `src/wow_bot/internal_dynamics/memory.py`

**Required**

- async SQLite / aiosqlite;
- explicit init/close;
- safe float64 vector storage;
- JSON event data;
- deterministic Euclidean recall/ties;
- decay/removal of old records;
- no pickle.

---

## Task 3.5 — MetaState Generator

**Output:** `meta_state.py`

Frozen step order:

```text
chaos
→ oscillators
→ drives.step
→ apply GameState events
→ drives.decay
→ persist important events
→ build MetaState
```

`MetaState.vector` remains the drives vector.

No LLM call occurs here.

---

## Task 3.6 — Adaptive Trigger

Threshold:

```text
threshold(t) = base + 0.1*sin(2π*t/7200)
```

- simulated elapsed time;
- strict trigger comparison `norm(delta) > threshold`;
- no wall-clock dependency.

---

## Task 3.7 — Spectrum Integration Test

**Required**

- actual MetaState output;
- sufficiently long deterministic series;
- Welch PSD;
- per-drive log-log slope;
- scientific target `[-1.5, -0.5]`.

**Important**

Do not change Dynamics only to force a statistical test to pass without first establishing that the analysis is valid.

---

# Phase 4 — Strategist

## Task 4.1 — Local LLM Client

**Output:** `src/wow_bot/strategist/llm_client.py`

**Required**

- `openai.AsyncOpenAI`;
- `httpx.AsyncClient(trust_env=False)`;
- local endpoints only;
- timeout from config;
- SDK retries disabled if custom retries own policy;
- bounded custom transient retries;
- cancellation propagation;
- lifecycle close;
- no prompt/response/key logging.

---

## Task 4.2 — Prompt Construction

**Output**

```text
src/wow_bot/strategist/prompts.py
src/wow_bot/strategist/system_prompt.txt
```

Expected API:

```text
load_system_prompt()
DynamicContext
build_user_prompt(meta_state, dynamic_context)
```

`DynamicContext` carries explicit:

```text
now
session_start
available_regions
previous_strategy
timezone
is_weekend
sleep_window
```

**Purity**

- no network;
- no DB;
- no LLM;
- no random;
- no wall-clock read.

Prompt output contract contains:

```text
reasoning
goal
region
risk_tolerance
priority
constraints
```

---

## Task 4.3 — Strict Strategy Parser

**Output:** parser module

Frozen API semantics:

```text
parse raw response + explicit valid_until → Strategy
```

**Required**

- strict JSON fields;
- reasoning diagnostic only;
- narrow full-response fenced-JSON recovery;
- reject extra/missing fields;
- reject NaN/Infinity;
- reject bool masquerading as numbers/ints;
- validate supported constraints;
- no silent coercion/repair;
- no LLM retry;
- failures raise narrow parser error;
- `valid_until` is explicit; no wall-clock read.

Historical "return default Strategy on any parser failure" behavior is superseded.

---

## Task 4.4 — Strategist Orchestrator

Expected surface:

```text
generate_strategy(meta_state, dynamic_context)
current_strategy
is_expired(now)
```

**Frozen behavior**

- validate explicit context;
- build prompts;
- compute explicit validity;
- one logical query;
- parse strictly;
- on success replace current Strategy;
- expected transport/parser failure + previous Strategy → return exact same previous object unchanged;
- no previous Strategy → propagate;
- cancellation/programming errors propagate;
- no orchestrator retry;
- explicit generate means replan even if previous Strategy was not expired.

---

# Phase 5 — Executor / Simulation

## Task 5.1 — Dry-Run Controller

**Output:** `src/wow_bot/executor/controller.py`

Expected API:

```text
press_key
move_mouse
click
stop_all
commands
```

**Frozen boundary**

- `dry_run=True` only;
- `dry_run=False` rejected;
- no physical input libraries;
- immutable/deterministic command records;
- no wall-clock/random IDs;
- `stop_all` safely resets simulated held state;
- no FSM/Strategy/Perception coupling.

Historical real-input/Notepad acceptance is superseded.

---

## Task 5.2 — Stochastic Timing/Error Simulation

**Output:** `humanize.py`

Timing:

```text
sigma = 0.4 + 0.2*chaos_component + 0.3*fatigue
mu = log(base_ms)
delay = clip(LogNormal(mu, sigma), 50, 2000)
```

Error probability:

```text
0.02 + 0.05*fatigue + 0.03*(1-curiosity)
```

Jitter:

```text
uniform integer dx/dy in [-radius, +radius]
```

**Required**

- no sleep;
- no Controller;
- injectable RNG;
- no global seed;
- strict validation;
- vector shape exactly `(5,)`;
- actual error range `[0.02,0.10]`;
- statistical tests scientifically account for clipping.

---

## Task 5.3 — Deterministic Executor FSM

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

Initial:

```text
IDLE
```

Flee threshold:

```text
0.60 - 0.40*risk_tolerance
```

Entry condition:

```text
in_combat and hp_pct <= threshold
```

Stuck:

```text
same movement-like action signature for >=5 simulated seconds
```

using `GameState.timestamp`.

**Boundary rules**

- low HP outside combat does not automatically flee;
- state transitions deterministic;
- backward timestamps rejected;
- Strategy update does not reset FSM;
- no physical action mapping;
- only simulated `stop_all()` allowed on safety/recovery entry.

---

## Task 5.4 — Symbolic Idle Behaviors

Exactly:

```text
CAMERA_WANDER
SUDDEN_PAUSE
INVENTORY_CHECK
SOCIAL_EMOTE
ANGLED_MOVEMENT
```

Independent callables + optional `IdleBehaviorEngine`.

Frozen overall trigger probability:

```text
0.08 per eligible synthetic tick
```

Curiosity modifies behavior-selection weights, not trigger frequency.

`SOCIAL_EMOTE` metadata only:

```text
wave
laugh
```

No slash commands / key bindings / mouse coordinates.

---

## Task 5.5 — Synthetic Curved Path

Expected API:

```text
generate_path(start, end, num_points=5, *, rng=None)
```

**Frozen**

- `num_points` includes endpoints;
- minimum 3;
- start != end;
- exact endpoints;
- finite numeric 2D coordinates;
- perpendicular single-arc/sine geometry;
- side selected by RNG;
- relative amplitude `10–20%` baseline;
- minimum amplitude `20px`;
- maximum `80px`;
- monotonic baseline progress;
- canonical deviation std > `5px`;
- no Controller/FSM/timing integration.

---

# Phase 6 — Independent Watchdog Supervisor

## Task 6.1 — Watchdog

**Output:** `src/wow_bot/watchdog/watchdog.py`

Architecture:

```text
main → Queue/messages → independent multiprocessing Watchdog
Watchdog → shutdown Event → main
```

Health:

```text
HEALTHY
DEGRADED
CRITICAL
```

Frozen baseline constants:

```text
poll                         0.5s
startup grace                30s
heartbeat degraded           >=10s
heartbeat critical           >=30s
progress degraded            >=30s
recovery window              60s
recovery degraded            >=3 entries
recovery critical            >=5 entries
recovery max duration        >=20s
death-loop window            600 simulation seconds
death-loop degraded          >=3 deaths
death-loop critical          >=5 deaths
graceful shutdown timeout    10s
```

**Required**

- injected/testable monotonic clock;
- progress token monotonic;
- count entries into STUCK_RECOVERY;
- death loop uses simulation time;
- health severity = max active severity;
- DEGRADED does not shut down;
- CRITICAL latches shutdown;
- optional abstract ResourceProbe;
- graceful request before bounded escalation;
- preserve logs;
- no F10/global hotkey;
- no Controller;
- no duplicated 5-second FSM stuck algorithm.

Historical log-deletion / direct `sys.exit()` behavior is superseded.

---

# Phase 7 — Integration

> **Policy:** implementation may be completed in an isolated coding environment. Runtime acceptance is local-only.

## Task 7.1 — Async Pipeline

**Output:** `src/wow_bot/main.py`

Required logical loops:

```text
perception
dynamics
strategist
executor
watchdog heartbeat
watchdog shutdown bridge
```

Expected data flow:

```text
MockPerception
→ GameState
→ MetaState
→ Strategist
→ Strategy
→ ExecutorFSM
```

Queue policy:

- bounded;
- ordered/backpressure for domain flow;
- latest-value semantics for Strategist planning snapshots;
- avoid stale Strategy backlogs.

Clock rules:

- GameState time for Dynamics/Strategy/cooldown;
- Watchdog monotonic clock for liveness;
- async real time only for scheduling/bounded run.

Strategist refresh:

```text
no current Strategy
OR expired
OR adaptive MetaState trigger
```

Failed refresh cooldown:

```text
10 simulation seconds
```

FSM bootstrap:

- no fabricated Strategy;
- construct after first valid Strategy;
- reuse FSM;
- later Strategy uses `set_strategy`.

Watchdog:

- one process;
- heartbeats about once per second unless approved equivalent;
- progress token increments per completed Dynamics step;
- explicit death messages;
- shutdown Event bridged to graceful async cleanup.

Cleanup:

```text
stop loops
controller.stop_all
close LLM resources
close memory
stop/join Watchdog
preserve logs
```

### Local-only acceptance

Defined in `LOCAL_VALIDATION_ROADMAP.md`:

- structural `python -m wow_bot.main`;
- short smoke;
- 300-second MockPerception stability;
- full lifecycle/cleanup/Watchdog observation.

---

## Task 7.2 — Scenario Runner

**Output:** `scripts/run_scenario.py`

CLI supports at minimum:

```text
--scenario
--duration
--seed
--output
```

Roadmap command:

```bash
python scripts/run_scenario.py --scenario combat_light --duration 600 --seed 42
```

Report schema version:

```text
1
```

High-level JSON:

```text
run
summary
strategist
fsm
meta_state
timing
watchdog
```

Required raw data:

```text
MetaState timestamps + vectors
timing samples
structured FSM transitions
Strategist generation counts
death-event metrics
```

Strict JSON; no NaN/Infinity.

Instrumentation is passive.

Enhanced Task 8.2 metadata may add:

```text
timing.samples[].delay_ms
timing.samples[].base_ms
timing.samples[].fatigue
timing.samples[].chaos_component
timing.samples[].simulation_timestamp
```

without removing `timing.samples_ms`.

### Local-only acceptance

- 60-second smoke;
- 600-second `combat_light`;
- scenario matrix;
- report schema inspection.

---

# Phase 8 — Offline Analysis / Stability

> **Policy:** analyzer/harness implementation can be reviewed offline. Scientific/long-running acceptance is local-only.

## Task 8.1 — Spectrum Analysis

**Output:** `scripts/analyze_spectrum.py` plus optional reusable analysis module.

Input:

```text
Task 7.2 schema_version 1
meta_state.dimensions
meta_state.samples
```

Method:

1. validate canonical dimensions/data;
2. minimum data threshold;
3. compute timestamp deltas;
4. diagnostics: median/mean/std `dt`, sampling rate, `dt` CV;
5. resample each signal to uniform grid using median `dt`;
6. constant detrend;
7. FFT diagnostics;
8. Welch PSD;
9. positive finite bins only;
10. log10 frequency/PSD;
11. fit linear slope;
12. optionally report R².

Target per drive:

```text
-1.5 <= slope <= -0.5
```

A valid out-of-range result is scientific evidence, not an exception.

Outputs:

```text
spectrum_analysis.json
spectrum_psd.png
```

### Local-only acceptance

Run against real 7.2 reports and inspect all five slopes.

---

## Task 8.2 — Timing Distribution Analysis

**Output:** `scripts/analyze_timing.py` plus optional reusable analysis module.

Input:

```text
Task 7.2 timing section
```

Model:

```text
mu = log(base_ms)
sigma = 0.4 + 0.2*chaos + 0.3*fatigue
clip to [50,2000]
```

Primary method:

```text
per-sample conditional CDF
→ randomized boundary-aware PIT
→ one-sample KS against Uniform(0,1)
```

Fixed PIT seed:

```text
8202
```

PIT rules:

- interior: `u = F(delay)`;
- lower clipped: randomize in `[0, F(lower)]`;
- upper clipped: randomize in `[F(upper), 1]`.

Roadmap targets:

```text
p-value > 0.05
CV > 0.3
```

CV:

```text
np.std(samples, ddof=1) / mean(samples)
```

Legacy report without conditional metadata:

- descriptive analysis allowed;
- exact conditional KS marked unavailable;
- do not fake overall acceptance.

Outputs:

```text
timing_analysis.json
timing_distribution.png
```

### Local-only acceptance

Run against a freshly generated enhanced Task 7.2 report.

---

## Task 8.3 — 24-Hour Soak Harness

**Output:** `scripts/run_soak_test.py` plus optional reusable soak module.

This task implements the harness; it does not execute 24 hours in Jules.

CLI:

```text
--scenario
--duration
--sample-interval
--seed
--output
--log-path (if needed)
```

Recommended local command:

```bash
python scripts/run_soak_test.py \
  --scenario peaceful_farm \
  --duration 86400 \
  --sample-interval 30 \
  --seed 42 \
  --output reports/soak_24h.json
```

Metrics as available:

```text
elapsed_seconds
cpu_percent
memory_rss_mb
log_size_bytes
progress_token
watchdog_alive
shutdown_requested
```

Summary:

```text
initial/final/min/max/mean memory
memory delta
memory growth percent
memory slope MB/hour
average/max CPU
log growth
progress delta
termination reason
zero_crash_target_met
```

**Important**

No arbitrary automatic memory-leak threshold.

```text
memory_stability_status = manual_review_required
```

Requirements:

- real monotonic duration;
- periodic metrics;
- main process/process-tree scope documented;
- strict JSON;
- partial/checkpoint evidence survives interruption/failure when feasible;
- no auto-restart after crash;
- no runtime model tuning.

### Local-only acceptance

- 5-minute harness smoke;
- 1-hour intermediate soak;
- 24-hour run;
- zero crashes;
- manual memory-stability assessment.

---

# Phase 9 — External Perception

## Task 9.1 — Perception Adapter

Status:

```text
PENDING — external dependency not delivered/unblocked
```

Do not implement proactively.

When unblocked, a new explicit task must define:

- input contract of external perception;
- conversion to `GameState`;
- validation;
- prerecorded/lab fixture acceptance;
- no expansion into direct game-process access.

---

# Master Reviewer Status Rule

A reviewer must not equate these statuses:

```text
implemented
unit/static validated
local runtime validated
scientifically accepted
```

For Tasks 7.1–8.3 they are separate gates.

Use `LOCAL_VALIDATION_ROADMAP.md` to determine final acceptance.
