# AGENTS.md — WoW-Bot Project Context for AI Agents

> This file provides context for AI coding agents (Jules, DeepSeek Harness, Cursor, etc.) working on this project. Read it fully before starting any task.

---

## 🎯 Project Overview

**Name:** WoW-Bot
**Type:** Research prototype for game bot architecture
**Purpose:** Academic study of anti-cheat evasion techniques — how modern detection systems (Warden, behavioral analysis, ML-based) can and cannot detect automated gameplay.
**Status:** Research-only prototype. Use synthetic/mock data, prerecorded screenshots, or isolated lab environments you control. Do not deploy against official game services.

## 📚 Source of Truth and Agent Startup Protocol

Before changing code, every agent must:

1. Read this file completely.
2. Read `ROADMAP.md` and identify the **single current task**.
3. Read the current implementation and the outputs of every dependency listed for that task.
4. Confirm that the previous task/phase acceptance criteria pass.
5. Implement only the current task unless explicitly instructed otherwise.

Authority:

- `ROADMAP.md` → task order, dependencies, deliverables, acceptance criteria.
- `AGENTS.md` → architecture, interfaces, invariants, coding/testing rules, safety boundaries.

If a requirement is ambiguous or the two documents conflict, **do not guess**. Report the conflict and request clarification.

### Core Thesis

> Modern anti-cheat systems still have exploitable gaps in short and medium time windows (hours to days) if a bot uses **internal dynamics** (chaos + oscillators + memory) instead of explicit randomization. The real gap closes in long-term horizons (weeks to months) via server-side analysis.

---

## 🏗️ Architecture Summary

Five-layer architecture, each layer has strict interfaces:

```
┌─────────────────────────────────────────┐
│  Perception Layer (teammate's work)     │  → GameState
├─────────────────────────────────────────┤
│  Internal Dynamics Layer                │  → MetaState + trigger flag
│  (Drives + Oscillators + Chaos + Memory)│
├─────────────────────────────────────────┤
│  Strategist Layer (LLM)                 │  → Strategy (JSON)
│  (Local Qwen 2.5 7B via Ollama)         │
├─────────────────────────────────────────┤
│  Executor Layer (FSM + Humanize)        │  → Actions
├─────────────────────────────────────────┤
│  Watchdog Layer (separate process)      │  → Kill switch
└─────────────────────────────────────────┘
```

**Key design principle:** LLM is a **strategist**, not an actor. It's called only when `MetaState` changes significantly (typically every 20-40 minutes). All moment-to-moment decisions happen in the fast local Executor.

### Layer Responsibilities and Allowed Dependencies

| Layer | Owns | Consumes | Produces | Must not do |
|---|---|---|---|---|
| Perception | Input interpretation | screenshot/mock frame | `GameState` | strategy, long-term memory, direct execution policy |
| Internal Dynamics | drives, oscillators, chaos, memory, trigger decision | `GameState` | `MetaState`, trigger signal | keyboard/mouse I/O, prompt policy |
| Strategist | high-level strategy | `MetaState`, recent memory, previous strategy | `Strategy` | frame-by-frame control |
| Executor | local FSM/action planning | `GameState`, `Strategy`, dynamics-derived context | controller actions | changing contracts or making LLM calls per action |
| Watchdog | health and emergency stop | health/heartbeat/metrics | stop signal, diagnostics | business/strategy decisions |

Data flows forward through contracts. Avoid hidden cross-layer imports and shared mutable globals.

---

## 🔒 Non-Negotiable Constraints

These constraints define the project's integrity. Never violate them:

1. **NO contact with game process.**
   - No DLL injection.
   - No `ReadProcessMemory` / `WriteProcessMemory`.
   - No addons or in-game scripts.
   - No manipulation of game files.

2. **NO network calls to non-local services.**
   - LLM must be local (Ollama at `localhost:11434`).
   - No cloud APIs (OpenAI, Anthropic, etc.).
   - Reason: eliminate the network-traffic signature.

3. **NO explicit randomization.**
   - No fixed timers ("every N hours").
   - No constant probabilities ("50% chance").
   - Variation must come from **internal dynamics** (chaos, oscillators, memory).

   **Implementation clarification:** the roadmap still uses deterministic seeded randomness in mocks/tests and sampling where a task explicitly specifies a statistical distribution. The prohibition above means production behavior must not be driven by hard-coded periodic schedules or constant-probability decision rules; strategic variation should remain state/event driven.

4. **NO hardcoded UI coordinates.**
   - All coordinates come from `config/config.json`.
   - Reason: UI scales differ across resolutions.

5. **NO skipped tests.**
   - Every acceptance criterion must pass before moving to the next task.
   - Tests use fixed seeds; production uses entropy.

6. **DRY-RUN is the default executor mode.**
   - Agents must not silently flip `executor.dry_run` to `false`.
   - Any non-dry-run validation must be explicitly requested and performed only in a harmless local test application or isolated lab environment.

7. **Preserve auditability.**
   - Emergency shutdown must flush/close resources and preserve logs needed to reproduce failures.
   - Do not add anti-forensic cleanup or log deletion behavior.

---

## 🧬 Why Internal Dynamics Matter (For Agents)

**Problem:** Most bot detection systems flag patterns that are "designed random." A timer set to "every 5-10 hours" produces a statistically detectable signature (uniform distribution, low CV, memorylessness).

**Solution:** Generate variation through **coupled dynamical systems** that produce pink noise (1/f spectrum):

- **Lorenz attractor** — deterministic chaos, produces fractal time series.
- **Coupled oscillators** (incommensurable frequencies) — produces quasi-periodic, never-repeating patterns.
- **Drives** with drift + decay — physiological-like state variables.
- **Memory store** — makes future behavior correlated with past events (humans have inertia; bots don't).

The signature of this approach in the frequency domain: **power spectral density slope ≈ -1** (pink noise), matching human behavior.

---

## 🎭 Mock-First Development

**Perception Layer is mocked until Phase 9.** All development until then uses `MockPerception`, which emits synthetic `GameState` data. This means:

- Agents can build and test Dynamics, Strategist, and Executor independently.
- No game installation is needed during development.
- Scenarios (`combat_light`, `death_loop`, etc.) drive test coverage.

When a teammate delivers the real Perception Layer, only `src/wow_bot/perception/adapter.py` needs to change.

---

## 🧩 Critical Design Decisions & Rationale

### Why `httpx.AsyncClient(trust_env=False)` for LLM?
Windows systems often have system proxies that intercept `localhost` traffic. `trust_env=False` disables env-based proxy resolution, preventing `502 Bad Gateway` errors from the OpenAI-compatible Ollama endpoint.

### Why Pydantic for Config?
Silent config errors are the #1 source of "the bot is behaving weird" bugs. Pydantic fails loudly at startup with a clear message.

### Why `aiosqlite` for Memory?
Memory writes/reads happen inside the async event loop. Synchronous `sqlite3` would block. `aiosqlite` runs on a thread pool and returns awaitables.

### Why `loguru` over stdlib `logging`?
Simpler API, better default formatting, built-in rotation. Less boilerplate for tagged loggers.

### Why log-normal for timing?
Human reaction times are log-normal (right-skewed). Uniform or Gaussian delays are statistical fingerprints of automation.

### Why Lorenz RK4 instead of Euler?
Euler diverges or distorts the attractor at large `dt`. RK4 preserves the shape and energy of the attractor, ensuring consistent statistical properties.

---

## 📐 Interface Contracts (DO NOT CHANGE WITHOUT COORDINATION)

These dataclasses are the borders between layers. Modifying them breaks teammates' work.

### `GameState`
Emitted by Perception Layer, consumed by Internal Dynamics and Executor.
Fields: `timestamp`, `hp_pct`, `mana_pct`, `position`, `facing`, `in_combat`, `target`, `enemies`, `events`.

### `MetaState`
Emitted by Internal Dynamics, consumed by Strategist.
Fields: `vector` (np.ndarray shape (5,)), `recent_events`, `timestamp`.

### `Strategy`
Emitted by Strategist, consumed by Executor.
Fields: `goal`, `region`, `risk_tolerance`, `priority`, `constraints`, `valid_until`.

### `Event`
Produced by Perception, consumed by Dynamics and Memory.
Fields: `type`, `timestamp`, `data` (dict).

If you need a new field, add it as optional with a default. **Never remove or rename existing fields.**

---

## 🧱 Implementation Rules

- Prefer small, task-scoped changes. No opportunistic rewrites.
- Preserve public contracts and file/module boundaries unless the current task explicitly changes them.
- Never remove or rename a contract field without coordination; additive fields must be optional with safe defaults.
- Use configuration for tunable values and UI coordinates; avoid magic numbers in behavior code.
- Keep blocking I/O out of the asyncio event loop.
- Avoid shared mutable global state. Pass dependencies explicitly.
- All long-lived resources must have explicit lifecycle/cleanup (`close`, context manager, or shutdown hook).
- Log state transitions and failures with enough context to reproduce the issue, but never log secrets.
- Add a dependency only when the standard library/current dependencies are insufficient; document why.
- Do not weaken tests, validation, type checking, or linting just to make CI pass.
- Do not change acceptance thresholds to fit the implementation without explicit approval.

## 🧪 Testing Philosophy

Every task in `ROADMAP.md` has acceptance criteria. These are non-negotiable. Tests must be:

1. **Deterministic** — use fixed seeds in `tests/fixtures/`.
2. **Fast** — unit tests < 1s each, integration tests < 10s each.
3. **Isolated** — no shared global state between tests.
4. **Async-aware** — use `pytest-asyncio` with `asyncio_mode = "auto"`.
5. **Behavioral evidence** — tests should prove acceptance criteria and architecture/domain invariants, not implementation details.
6. **Canonical** — every normal task test must be reachable from the standard `pytest` test path; no hidden manual-only test unless explicitly marked as a long-running Phase 8 check.
7. **Negative paths** — cover malformed LLM output, timeout/fallback behavior, invalid config/data, shutdown, and blocked I/O paths where relevant.

### Special tests
- **Spectrum test** (`test_spectrum.py`): validates 1/f signature via FFT + slope fit.
- **Timing test** (`test_timing.py`): validates log-normal distribution via KS test.

---

## 🚦 Working Order for Agents

**Always follow `ROADMAP.md` order.** Never skip ahead. Work on one task at a time unless the user explicitly changes the workflow. If a task is blocked:

1. Check if dependency is truly complete (read its output files).
2. If blocked, comment on the task, do NOT start unrelated work.
3. Ask for clarification rather than assuming.

### Git conventions
- One commit per task: `feat(phase-X.Y): short description`.
- Never commit `.env`, `logs/`, `data/`, `*.db`.
- Run task-relevant `pytest`, then `ruff check` and `mypy --strict` before committing.
- Keep commits task-scoped; do not mix unrelated cleanup.
- Never rewrite history or force-push unless explicitly requested.

---

## 🎯 What Success Looks Like

At the end of Phase 8, the system should:

- ✅ Run continuously for 24h without crash or memory leak.
- ✅ Produce MetaState time series with FFT slope in `[-1.5, -0.5]` (pink noise).
- ✅ Produce `human_delay` samples that pass KS test vs log-normal (p > 0.05).
- ✅ Call the LLM only when MetaState changes significantly (not on a timer).
- ✅ Never touch the game process, never make non-local network calls.
- ✅ Have >80% test coverage on `internal_dynamics/`, `strategist/`, `executor/`.

---

## ❓ Common Questions Agents Ask

**Q: Can I use a different LLM model than Qwen 2.5 7B?**
A: Yes, if the Ollama endpoint is OpenAI-compatible. Update `config.json`. But keep it **local**.

**Q: Can I use multithreading instead of asyncio?**
A: No. The whole project is `asyncio`-based. Mixed threading will cause subtle race conditions.

**Q: What if a dependency is missing?**
A: Add it to `pyproject.toml`, run `uv sync --all-extras`, then proceed. Do not install via `pip install` directly.

**Q: Can I add a new event type?**
A: Yes — add it to `EVENT_EFFECTS` in `src/wow_bot/shared/events.py`. Update the road map task 1.3 tests accordingly.

**Q: How do I know when Phase X is "done"?**
A: All tasks in Phase X have passing acceptance criteria, `ruff check` and `mypy` clean, and all commits pushed.

**Q: The spectrum test fails with slope = -0.3. What's wrong?**
A: Check oscillator frequencies and amplitudes. More low-frequency components → steeper slope. Check Lorenz dt not too large (should be `0.001`).

**Q: LLM output is not valid JSON. What do I do?**
A: `parse_strategy()` should fall back to a default Strategy. Check the raw output in logs — if it's consistently non-JSON, the system prompt needs strengthening (add "Respond ONLY with JSON").

---

## 🗂️ File-by-File Reference

| Path | Purpose |
|---|---|
| `src/wow_bot/shared/interfaces.py` | Dataclasses: `GameState`, `MetaState`, `Strategy`, `Event`, `TargetInfo`, `EnemyInfo` |
| `src/wow_bot/shared/events.py` | Event type constants + effect table |
| `src/wow_bot/shared/config.py` | Pydantic Settings |
| `src/wow_bot/shared/logger.py` | Loguru wrapper with tags |
| `src/wow_bot/internal_dynamics/drives.py` | Five drives with drift/decay |
| `src/wow_bot/internal_dynamics/oscillators.py` | Coupled oscillator bank |
| `src/wow_bot/internal_dynamics/chaos.py` | Lorenz attractor (RK4) |
| `src/wow_bot/internal_dynamics/memory.py` | SQLite-backed memory store |
| `src/wow_bot/internal_dynamics/meta_state.py` | Combines drives + oscillators + chaos + memory |
| `src/wow_bot/strategist/llm_client.py` | Ollama client |
| `src/wow_bot/strategist/prompts.py` | System + user prompts |
| `src/wow_bot/strategist/parser.py` | JSON → Strategy parser with fallback |
| `src/wow_bot/strategist/orchestrator.py` | Strategist coordinator |
| `src/wow_bot/executor/controller.py` | Keyboard/mouse I/O (dry-run capable) |
| `src/wow_bot/executor/humanize.py` | Log-normal timing, error, jitter |
| `src/wow_bot/executor/fsm.py` | Parameterized FSM |
| `src/wow_bot/executor/idle_behaviors.py` | Camera turns, random emotes |
| `src/wow_bot/executor/path.py` | Curved path generation |
| `src/wow_bot/watchdog/watchdog.py` | Independent watchdog process |
| `src/wow_bot/mocks/mock_perception.py` | Synthetic GameState source |
| `src/wow_bot/main.py` | Async entry point |
| `scripts/run_scenario.py` | Scenario runner |
| `scripts/analyze_spectrum.py` | FFT analysis |
| `scripts/analyze_timing.py` | Timing distribution analysis |

---

## ✅ Definition of Done for Every Task

A task is complete only when all of the following are true:

- Required output files exist and are integrated with the current codebase.
- Every acceptance criterion for the task has executable evidence.
- Relevant unit/integration tests pass.
- `ruff check` passes for changed code.
- `mypy --strict` passes for changed/covered modules (and project-wide when practical).
- No contract/invariant was weakened.
- No unrelated files or behavior changed.
- The agent reports exactly what changed, validation commands/results, and any known limitation.

## 🔗 Related Documents

- `ROADMAP.md` — task-by-task implementation plan.
- `README.md` — public-facing overview.
- `config/config.json` — all tunable parameters.

---

## ⚠️ Ethical & Legal Notice

This project is **strictly for academic research and controlled experimentation**. Use mock/synthetic data, prerecorded screenshots, or isolated environments you control. Do not deploy or validate the automation against official game services. Keep `dry_run` enabled unless an explicitly approved harmless local test requires input generation.

---

**END OF AGENTS.md**
