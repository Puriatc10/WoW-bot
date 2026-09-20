# LOCAL_VALIDATION_ROADMAP.md

> Local-only acceptance plan for Tasks 7.1, 7.2, 8.1, 8.2, and 8.3.

This document exists because those tasks contain runtime, scientific, multiprocessing, local-LLM, or long-duration criteria that are not reliably executable inside an isolated coding-agent environment.

Do not ask Jules/reviewer agents to fabricate these results.

---

# 1. Validation Principles

1. Run against the exact commit being reviewed.
2. Preserve logs/reports from every failing run.
3. Do not change seeds/thresholds after seeing a failing result.
4. Stop at the first material infrastructure/code failure and fix that defect before drawing scientific conclusions.
5. A scientific result outside a target is not automatically a code defect.
6. Do not enable physical input.
7. Task 9.1 remains pending and is not part of this validation.
8. Record environment and command evidence.

Recommended evidence directory outside version control:

```text
reports/local-validation/
```

---

# 2. Record the Baseline

Before runtime testing:

```bash
git rev-parse HEAD
git status --short
python --version
uv --version
```

Record:

```text
commit
OS
Python version
uv version
Ollama version
model tag
config used
```

Static baseline:

```bash
uv sync --all-extras
uv run --extra dev ruff check .
uv run --extra dev mypy src
uv run --extra dev pytest
```

If the repository uses slightly different canonical commands, use the repository-defined variants and record them.

Do not proceed with long-running tests while ordinary regressions are failing.

---

# 3. Local LLM Prerequisite

Expected local model:

```bash
ollama pull qwen2.5:7b
ollama serve
```

Confirm configured local endpoint and model.

No cloud fallback should be required.

If the project exposes a health check, run it before Task 7 integration validation.

---

# 4. Task 7.1 — Async Pipeline

## Goal

Prove that the full mock pipeline runs end-to-end with real local process/event-loop/lifecycle behavior.

Expected chain:

```text
MockPerception
→ GameState
→ Internal Dynamics
→ MetaState
→ Strategist
→ Strategy
→ ExecutorFSM
```

with:

```text
Watchdog heartbeat/progress side-channel
```

and graceful cleanup.

---

## 4.1 Structural boot

Run the repository entry point:

```bash
uv run python -m wow_bot.main
```

or the exact documented equivalent.

Verify:

- settings load;
- memory initializes;
- Watchdog starts once;
- first GameState appears;
- MetaState processing begins;
- initial Strategy is generated;
- FSM is initialized only after valid Strategy;
- no physical input backend is loaded;
- no unhandled startup exception.

Stop manually after a short observation and verify cleanup.

---

## 4.2 Short bounded smoke

If `run_duration_seconds` or an equivalent bounded-run entry exists, run approximately 30–60 seconds.

Verify:

- progress token increases;
- heartbeat continues;
- FSM receives snapshots;
- no stale task/process remains after return;
- memory DB handle closes;
- LLM client closes;
- Watchdog child exits;
- Controller records only dry-run/simulation commands.

---

## 4.3 Strategist lifecycle

Observe at least:

- initial Strategy generation;
- no fabricated initial Strategy;
- current Strategy reaches FSM;
- later Strategy update uses `set_strategy`, not FSM reconstruction.

Where feasible with a controlled trigger, verify:

- expiry refresh;
- adaptive MetaState refresh;
- no unnecessary refresh for small unexpired changes.

---

## 4.4 Failed refresh cooldown

Induce a controlled local LLM refresh failure **after** a valid Strategy exists.

Expected:

- exact previous Strategy remains active;
- retry is not attempted every Dynamics tick;
- cooldown is 10 simulation seconds;
- `valid_until` is not silently extended.

Restore LLM after the observation.

---

## 4.5 Watchdog shutdown bridge

Use a safe test path to set the Watchdog shutdown Event or trigger a test critical condition.

Verify:

- local async shutdown begins;
- sibling tasks stop;
- `controller.stop_all()` is invoked;
- resources close;
- Watchdog is joined;
- logs remain present.

---

## 4.6 Five-minute acceptance

Run:

```text
MockPerception for 300 real seconds
```

using the implemented bounded-run command/API.

Acceptance:

```text
no crash
progress continues
Watchdog remains operational
clean termination
no leaked child process/tasks
```

Record:

```text
Task 7.1 local acceptance: PASS / FAIL
```

---

# 5. Task 7.2 — Scenario Runner

## 5.1 60-second smoke

```bash
uv run python scripts/run_scenario.py   --scenario combat_light   --duration 60   --seed 42
```

Use exact CLI syntax implemented in the repo.

Verify output file exists and parses as JSON.

---

## 5.2 Schema checks

The report should have:

```text
schema_version == 1
run
summary
strategist
fsm
meta_state
timing
watchdog
```

Verify:

```text
run.scenario == combat_light
run.seed == 42
completed_normally == true
```

MetaState:

```text
dimensions == [hunger,fatigue,curiosity,aggression,social]
samples not empty
timestamps ordered
every vector length == 5
all values finite
```

Timing:

```text
unit == ms
samples_ms not empty
all values finite
```

JSON must contain no NaN/Infinity.

---

## 5.3 600-second roadmap run

```bash
uv run python scripts/run_scenario.py   --scenario combat_light   --duration 600   --seed 42
```

Acceptance:

- exits normally;
- report complete;
- LLM/Strategist counts are plausible;
- FSM transitions are present;
- timing samples are present;
- MetaState series is present.

Preserve the report for Tasks 8.1 and 8.2.

---

## 5.4 Scenario matrix

Run shorter smoke reports for:

```text
peaceful_farm
death_loop
rare_loot_drought
stuck_repeatedly
```

Suggested duration:

```text
60–180 seconds initially
```

Verify scenario-specific evidence:

- `death_loop`: death events appear;
- `stuck_repeatedly`: recovery/stuck evidence appears when scenario timing permits;
- `peaceful_farm`: no unexpected combat/death pattern;
- `rare_loot_drought`: no unexpected rare-loot events.

Do not force every scenario to have the same FSM statistics.

---

# 6. Task 8.1 — Spectrum Analysis

Use a real report generated by Task 7.2.

Example:

```bash
uv run python scripts/analyze_spectrum.py   --input reports/<combat-light-600s-report>.json   --output-dir reports/local-validation/spectrum-combat
```

Verify outputs:

```text
spectrum_analysis.json
spectrum_psd.png
```

---

## 6.1 Sampling diagnostics

Inspect:

```text
original sample count
resampled sample count
median dt
sampling rate
dt CV
```

The analyzer must explicitly resample irregular data rather than silently assuming uniform sampling.

---

## 6.2 Per-drive results

Verify results for all five dimensions:

```text
hunger
fatigue
curiosity
aggression
social
```

For each inspect:

```text
slope
fit point count
fit policy
within_target_range
```

Roadmap target:

```text
-1.5 <= slope <= -0.5
```

Record the actual five values.

Do not change frequency bands/seed/Dynamics to rescue an unfavorable result without first investigating validity.

---

## 6.3 Interpretation

Two separate statuses:

```text
analysis execution: PASS / FAIL
scientific target: PASS / OUTSIDE TARGET
```

An out-of-range slope is not automatically an analyzer bug.

---

# 7. Task 8.2 — Timing Distribution Analysis

Generate a **fresh Task 7.2 report after enhanced per-sample timing metadata is implemented**.

If the report only has `samples_ms`, exact conditional-model KS validation is unavailable and the report should be regenerated.

Example:

```bash
uv run python scripts/run_scenario.py   --scenario combat_light   --duration 600   --seed 42
```

Then:

```bash
uv run python scripts/analyze_timing.py   --input reports/<fresh-combat-light-600s-report>.json   --output-dir reports/local-validation/timing-combat
```

Verify:

```text
timing_analysis.json
timing_distribution.png
```

---

## 7.1 Conditional KS

Expected primary method:

```text
conditional clipped-lognormal
→ randomized PIT
→ KS vs Uniform(0,1)
```

Fixed PIT seed:

```text
8202
```

Roadmap target:

```text
p-value > 0.05
```

At exactly `0.05`, target is not met.

Record:

```text
KS statistic
p-value
sample count
PIT seed
clipping counts
```

---

## 7.2 CV

Expected:

```text
CV = sample_std(ddof=1) / sample_mean
```

Roadmap target:

```text
CV > 0.3
```

At exactly `0.3`, target is not met.

Record:

```text
mean
sample std
CV
min/max
median
p05/p95
clipped fraction
```

---

## 7.3 Interpretation

Separate:

```text
analysis execution
KS target
CV target
overall roadmap target
```

Do not tune the timing model automatically after a statistical failure.

Investigate:

- instrumentation consistency;
- per-sample parameters;
- temporal dependence;
- clipping fraction;
- true model mismatch.

---

# 8. Task 8.3 — Soak Harness

Run in increasing duration.

## 8.1 Five-minute harness smoke

```bash
uv run python scripts/run_soak_test.py   --scenario peaceful_farm   --duration 300   --sample-interval 5   --seed 42   --output reports/local-validation/soak_smoke_5m.json
```

Verify:

- harness itself works;
- samples are periodically written;
- metrics are finite/nullable as documented;
- pipeline exits normally;
- report survives completion;
- no orphan Watchdog process.

---

## 8.2 One-hour intermediate soak

```bash
uv run python scripts/run_soak_test.py   --scenario peaceful_farm   --duration 3600   --sample-interval 30   --seed 42   --output reports/local-validation/soak_1h.json
```

Inspect:

```text
completed_normally
zero_crash_target_met
progress_delta
memory samples
CPU samples
log growth
memory slope MB/hour
```

This is a preflight, not final acceptance.

If obvious monotonic memory growth or lifecycle failures appear, stop and investigate before 24h.

---

## 8.3 Twenty-four-hour roadmap run

```bash
uv run python scripts/run_soak_test.py   --scenario peaceful_farm   --duration 86400   --sample-interval 30   --seed 42   --output reports/local-validation/soak_24h.json
```

Supply the implemented `--log-path` argument if automatic log discovery is unavailable.

---

## 8.4 Objective crash acceptance

Required:

```text
completed_normally == true
completed_duration_seconds >= 86400
zero_crash_target_met == true
progress_delta > 0
```

Any unexpected pipeline/Watchdog failure before completion fails the zero-crash criterion.

No automatic restart is allowed to hide crashes.

---

## 8.5 Memory stability review

The project intentionally has no invented leak threshold.

Review:

```text
initial_memory_mb
final_memory_mb
minimum_memory_mb
maximum_memory_mb
average_memory_mb
memory_change_mb
memory_growth_percent
memory_growth_slope_mb_per_hour
first-vs-last-quarter means if available
full time series
```

Look for:

- warm-up then plateau;
- bounded oscillation;
- persistent monotonic growth;
- stepwise unbounded growth;
- late-run drift.

Record human conclusion:

```text
memory stability: PASS / NEEDS INVESTIGATION
reason:
evidence:
```

A single positive slope does not automatically prove a leak; inspect the time series.

---

# 9. Failure Triage Policy

When a local gate fails:

1. preserve report/log artifacts;
2. record commit/environment/command;
3. classify failure:
   - infrastructure/config;
   - implementation bug;
   - lifecycle/concurrency bug;
   - instrumentation/report bug;
   - scientific target outside range;
   - resource/stability issue;
4. do not start the next long-running gate until resolved;
5. create a small targeted Jules fix task using the exact evidence.

Do not give Jules a broad "fix everything" prompt.

---

# 10. Evidence Bundle for Reviewer

For final review, retain:

```text
git commit hash
static quality-gate output
Task 7.1 300s result
Task 7.2 600s report
Task 8.1 analysis JSON + plot
Task 8.2 analysis JSON + plot
Task 8.3 5m report
Task 8.3 1h report
Task 8.3 24h report
relevant logs
manual memory-stability conclusion
```

The reviewer should not claim project-level Phase 8 acceptance without these artifacts.

---

# 11. Final Status Template

```text
Commit:
Environment:

7.1 Async Pipeline:
  Implementation:
  300s local acceptance:
  Notes:

7.2 Scenario Runner:
  Implementation:
  600s combat_light:
  Report path:
  Notes:

8.1 Spectrum:
  Execution:
  hunger slope:
  fatigue slope:
  curiosity slope:
  aggression slope:
  social slope:
  Target:

8.2 Timing:
  Execution:
  KS p-value:
  CV:
  Clipped fraction:
  Target:

8.3 Soak:
  5m:
  1h:
  24h:
  Zero crashes:
  Memory stability:
  Evidence:

9.1:
  PENDING
```
