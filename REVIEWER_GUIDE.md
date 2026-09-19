# REVIEWER_GUIDE.md — Codebase Audit Contract

> Use this file when giving the repository to a reviewer agent.

The reviewer should act as an auditor. Do not modify code unless explicitly requested.

---

# 1. Reviewer Mission

Determine whether the current repository faithfully implements the frozen project specification.

The review must answer:

1. What is implemented correctly?
2. What violates a frozen contract?
3. What lacks executable/static evidence?
4. What is intentionally waiting for local validation?
5. What is scientifically outside target versus actually implemented incorrectly?
6. Has Task 9.1 remained pending?

---

# 2. Required Reading Order

Before reviewing code:

1. `README.md`
2. `AGENTS.md`
3. `ROADMAP.md`
4. `LOCAL_VALIDATION_ROADMAP.md`
5. current `pyproject.toml`
6. actual source/tests

Do not rely on old chat prompts, historical roadmap text, or stale comments when the updated documentation supersedes them.

---

# 3. Review Modes

## Static / implementation review

Can be completed from codebase:

- architecture boundaries;
- APIs;
- validation;
- deterministic behavior;
- queue policies;
- clock usage;
- cleanup design;
- report schema;
- analysis formulas;
- unit/offline tests;
- lint/type state.

## Local-evidence review

Requires external evidence:

- Task 7.1 300-second stability;
- Task 7.2 600-second real scenario report;
- Task 8.1 slopes on that report;
- Task 8.2 KS/CV on a fresh report;
- Task 8.3 24-hour stability.

Absence of local evidence is:

```text
LOCAL VALIDATION PENDING
```

not automatically an implementation defect.

---

# 4. Severity Model

Use:

```text
BLOCKER
HIGH
MEDIUM
LOW
INFO
```

Suggested interpretation:

- **BLOCKER** — violates a core runtime/safety/architecture boundary or prevents expected task operation.
- **HIGH** — wrong frozen contract, data corruption, lifecycle/concurrency failure, or invalid scientific method.
- **MEDIUM** — meaningful robustness/testability/documentation gap.
- **LOW** — maintainability issue without contract violation.
- **INFO** — pending local validation or optional observation.

---

# 5. Finding Format

Each finding should contain:

```text
ID:
Severity:
Task:
File / symbol:
Expected:
Actual:
Evidence:
Impact:
Recommended correction:
Validation required after correction:
```

Do not give vague findings like "could be cleaner."

---

# 6. Global Blocker Checks

Immediately flag if code:

- enables physical keyboard/mouse execution;
- accepts `dry_run=False` as normal supported mode;
- imports/uses OS input automation libraries;
- touches external game process memory/hooks;
- introduces non-local/cloud LLM fallback;
- deletes/truncates logs on shutdown;
- introduces Watchdog global F10/keyboard hook;
- implements Task 9.1 despite it remaining pending;
- hides a failed initial Strategy by fabricating a default;
- silently changes canonical drive order/shared fields.

---

# 7. Shared Contract Review

Verify:

```text
GameState
MetaState
Strategy
Event
TargetInfo
EnemyInfo
```

Check:

- fields preserved;
- vector shape/order;
- normalized domains;
- optional additive changes backward-compatible;
- no duplicated conflicting contract models.

Canonical drive order:

```text
hunger
fatigue
curiosity
aggression
social
```

---

# 8. Phase 3 Review

## Drives

Check:

- `(5,)`, float64;
- `[0,1]`;
- copy/immutability boundary;
- event registry used;
- decay;
- no accidental index-order mismatch.

## Oscillators

Check frozen periods/amplitudes and seeded Generator.

## Lorenz

Check RK4 and deterministic projection.

## Memory

Check async lifecycle, JSON-safe data, no pickle.

## MetaState

Check exact step ordering and simulation-time adaptive threshold.

## Spectrum test

Ensure it tests actual MetaState output and does not contain parameter-tuning hacks.

---

# 9. Phase 4 Review

## LLM client

Review:

- local-only base URL policy;
- `trust_env=False`;
- timeout/retry ownership;
- cancellation;
- close lifecycle;
- logging hygiene.

## Prompt

Review purity and explicit `DynamicContext`.

## Parser

This is high risk. Verify:

- strict exact fields;
- no default Strategy;
- no silent repair;
- no non-finite numerics;
- constraints strict;
- explicit `valid_until`;
- narrow fenced response support only.

## Strategist

Verify fallback identity semantics:

```text
expected failure + previous Strategy
→ exact same previous object
```

No previous Strategy → propagate.

---

# 10. Phase 5 Review

## Controller

Must remain simulation-only.

## Timing

Verify exact formula and no sleep.

## FSM

Verify:

```text
flee = 0.60 - 0.40*risk
```

Boundary:

```text
hp <= threshold
```

Verify low HP outside combat does not automatically flee.

Stuck:

```text
same movement signature >=5 simulation seconds
```

No wall-clock.

FSM should not call actual input methods.

## Idle behavior

Five symbolic intents only; trigger probability `0.08`; no physical semantics.

## Path

Pure 2D geometry; exact endpoints; deterministic under seed; no controller coupling.

---

# 11. Phase 6 Review

Check independent process design and testable monitor logic.

Verify:

- startup grace;
- heartbeat degraded/critical boundaries;
- progress semantics;
- repeated recovery entry counting;
- recovery duration;
- death-loop simulation-time window;
- health severity composition;
- DEGRADED does not shut down;
- CRITICAL latches;
- graceful timeout before escalation;
- evidence preserved.

High-risk bug:

```text
counting every STUCK_RECOVERY heartbeat as a new recovery entry
```

must be rejected.

---

# 12. Task 7.1 Review

Review implementation, not local runtime success.

Check:

### Loop boundaries

```text
perception
dynamics
strategist
executor
heartbeat
shutdown bridge
```

### Queues

- bounded;
- backpressure for ordered domain flow;
- latest-value only where stale data is intentionally disposable;
- no unbounded Strategist backlog.

### Time

- Dynamics/Strategy use simulation time;
- Watchdog liveness uses monotonic time.

### Strategist

- bootstrap/expiry/adaptive triggers;
- 10-simulation-second failed-refresh cooldown;
- no initial fabricated Strategy.

### Executor

- FSM lazy bootstrap after Strategy;
- `set_strategy` for updates.

### Cleanup

- cancellation propagation;
- controller stop;
- memory close;
- LLM close;
- Watchdog stop/join;
- primary exception preserved.

### Review status

Without local 300s evidence, report:

```text
IMPLEMENTATION REVIEWED — LOCAL 7.1 ACCEPTANCE PENDING
```

---

# 13. Task 7.2 Review

Verify report is a scientific contract, not log scraping.

Check schema version 1 and strict JSON.

Required:

```text
MetaState raw time series
timing samples
FSM transitions
Strategist counts
death metrics
```

Instrumentation must not alter runtime decisions.

If enhanced timing metadata was added, ensure `samples_ms` remains backward-compatible.

---

# 14. Task 8.1 Review

Verify analyzer:

- consumes report JSON;
- validates canonical dimensions;
- validates timestamps;
- handles irregular sampling by explicit deterministic resampling;
- uses each drive separately;
- computes FFT;
- uses Welch PSD for slope;
- fits positive finite bins only;
- does not select favorable frequency range after results;
- outputs strict JSON + plot.

Target:

```text
[-1.5,-0.5]
```

Do not mark code defective merely because real data is outside target if method is correct.

---

# 15. Task 8.2 Review

This is another high-risk scientific area.

Primary method must account for:

- sample-specific sigma;
- clipping;
- boundary probability mass.

Expected:

```text
conditional clipped lognormal
→ randomized PIT
→ KS Uniform
```

Fixed seed:

```text
8202
```

Check exact strict thresholds:

```text
p > 0.05
CV > 0.3
```

CV uses:

```text
sample std, ddof=1
```

Clipped samples must not be dropped.

Legacy report without conditional metadata must not claim exact KS acceptance.

---

# 16. Task 8.3 Review

Verify harness:

- reuses Task 7.1;
- does not duplicate pipeline;
- uses monotonic operational time;
- samples resource metrics;
- documents process scope;
- preserves partial evidence;
- no auto-restart after crash;
- no arbitrary leak threshold;
- strict JSON;
- refuses accidental overwrite or has explicit safe policy.

Memory conclusion must remain manual unless an explicit project threshold is later approved.

---

# 17. Tests and Quality

Run what the environment supports:

```bash
uv run --extra dev ruff check .
uv run --extra dev mypy src
uv run --extra dev pytest
```

Then targeted tests for changed/high-risk modules.

Do not weaken test assertions just to make them pass.

Tests should prove behavior/invariants, not private implementation structure.

---

# 18. Scientific Honesty Checks

Flag any implementation that:

- changes seed after seeing result;
- scans multiple seeds and chooses best;
- changes fit band after seeing slope;
- removes clipped timing samples;
- modifies Dynamics from an analysis script;
- automatically retunes timing to hit CV/KS;
- labels unavailable data as PASS;
- labels synthetic fixture output as real-data acceptance.

---

# 19. Local Acceptance Evidence

Use `LOCAL_VALIDATION_ROADMAP.md`.

Expected final evidence:

```text
7.1: 300-second clean runtime
7.2: 600-second combat_light report
8.1: five real per-drive slopes + plot
8.2: real conditional KS + CV + plot
8.3: 24-hour report + zero crashes + manual memory review
```

If missing, report it as pending.

---

# 20. Expected Reviewer Final Report

Use sections:

```text
Executive summary

Verified architecture/invariants

Findings
  BLOCKER
  HIGH
  MEDIUM
  LOW

Task-by-task compliance matrix

Static/offline validation performed

Local-validation evidence present/missing

Scientific acceptance evidence present/missing

Task 9.1 status

Final unresolved risks
```

Do not produce an overall "PASS" if mandatory local acceptance evidence is missing. Prefer:

```text
IMPLEMENTATION REVIEW COMPLETE
LOCAL ACCEPTANCE PENDING
```

when appropriate.
