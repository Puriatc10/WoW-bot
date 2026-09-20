# AGENTS.md

Operating rules for any agent (human or automated) contributing to this
repository. This file is normative. If anything here conflicts with a
task prompt, the conflict MUST be resolved by updating this file first,
not by silently diverging.

---

## 1. What This Project Is

A research prototype for studying LLM-driven agent architectures under
human-like timing and reflex constraints. The project supports two
execution modes:

- **MOCK_MODE** (default): fully synthetic. Runs in CI. No OS input, no
  screen capture, no network to any game server.
- **LAB_MODE** (opt-in): runs against a private, self-owned server on an
  isolated network. Real actuation, real reflex, real navigation.
  Perception remains mocked in this phase of the roadmap.

**Non-goals, permanently out of scope:**
- Any connection to retail WoW, Blizzard services, or any third-party
  server the operator does not own.
- Any form of anti-cheat bypass, evasion, or detection avoidance.
- Any read/write of game process memory, DLL injection, syscall hooking,
  or binary modification.

See `LAB_CONSTRAINTS.md` for the full boundary specification.

---

## 2. Source of Truth

The following documents are authoritative. Read them before starting any
task.

| Document | Purpose |
|---|---|
| `LAB_PHASE_ROADMAP.md` | **The** phase and task breakdown. Single source for what to build next. |
| `LAB_CONSTRAINTS.md` | Hard boundaries of LAB_MODE. Normative. |
| `ARCHITECTURE.md` | Layer table and data flow (both modes). |
| `PERCEPTION.md` | RealPerception design (future; not in current roadmap). |
| `ACTUATION.md` | RealActuator design. |
| `SAFETY.md` | Safety, kill switch, recovery, log integrity. |
| `LOCAL_VALIDATION_ROADMAP.md` | Original mock-phase validation tasks (8.x). |
| `README.md` | Project overview and scope statement. |

If any of these files disagree, the order of precedence is:

1. `LAB_CONSTRAINTS.md`
2. `LAB_PHASE_ROADMAP.md`
3. `AGENTS.md` (this file)
4. `ARCHITECTURE.md`
5. Everything else.

Any change to a higher-precedence file MUST be reflected downward.

---

## 3. Execution Modes

### 3.1 MOCK_MODE (default)

- Perception: `MockPerception` produces synthetic `GameState`.
- Actuation: `SimulationController` records symbolic intents only.
- `dry_run=False` MUST be rejected.
- No OS input, no screen capture, no network to game server.
- Runs in CI. All tests run in this mode by default.

### 3.2 LAB_MODE (opt-in)

Activated only when ALL of the following hold:

- `LAB_MODE=1`
- A valid `LAB_SERVER_ALLOWLIST` entry exists in config.
- Network isolation check passes (see `SAFETY.md`).
- SafetyLayer is armed.
- A `Session` has been opened with append-only logging.

In LAB_MODE:
- Perception: still `MockPerception` (per current roadmap).
- Actuation: `RealActuator` (real OS input).
- Reflex layer runs at 10-20 Hz.
- Watchdog runs in behavioral mode.
- Kill switch armed on startup.
- Session logs immutable.
- MUST NOT run in CI. MUST NOT run unattended.

---

## 4. Layering Rules (both modes)

### 4.1 GameState Contract

`GameState` is the single contract between perception and the rest of
the pipeline. Producers may differ (`MockPerception` vs future
`RealPerception`); consumers MUST NOT branch on producer identity.

**Any task that changes the `GameState` schema MUST be halted and
reviewed.** Schema changes ripple through every downstream layer and
invalidate existing reports.

### 4.2 Fast vs Slow Loops

The system has two decision loops that MUST remain separate:

- **Fast loop (Reflex):** 10-20 Hz, no LLM, deterministic given
  `(state, tick_index, per-component_seed)`. Handles safety, abort,
  stuck detection, focus loss, interrupt windows.
- **Slow loop (Strategist):** seconds-scale, LLM-backed, may be
  nondeterministic across runs. Handles goal selection and high-level
  planning.

The Combat Engine runs on the fast loop. The FSM runs on the slow loop
but consumes fast-loop control signals.

**No layer other than the Strategist may call the LLM.** This is
enforced by static import checks in tests.

### 4.3 Memory Separation

Two distinct stores exist and MUST NOT share tables:

- **Internal Dynamics:** drives, oscillators, chaos, agent internal
  state. Owned by `internal_dynamics`.
- **World Model:** map nodes, edges, entities, routes, combat history.
  Owned by `world`. All tables prefixed `wm_`.

Cross-writes are forbidden. Cross-reads go through explicit APIs.

### 4.4 Safety Layer is Always Alive

`SafetyLayer` is instantiated in Phase 0 and MUST be active in every
mode. No task may bypass it. The following calls are mandatory:

- `safety.check_allowlist(addr)` before any outbound game connection.
- `safety.check_isolation()` at LAB_MODE startup.
- `safety.is_aborted()` at the top of every actuation.
- `safety.abort(reason)` on any critical failure.

---

## 5. Task Workflow

### 5.1 Source of Tasks

Tasks are defined exclusively in `LAB_PHASE_ROADMAP.md`. Do not invent
tasks. Do not merge tasks. Do not reorder tasks without updating the
roadmap first.

Each task has:
- Identifier (e.g., `T0.1`)
- Dependencies
- Deliverables (files)
- Contract (interfaces, behavior)
- Acceptance criteria (testable)
- Out-of-scope items

### 5.2 PR Rules

Every PR MUST:

- Address exactly one task from the roadmap.
- Reference the task id in the PR title (e.g., `T0.1: Config System`).
- Include all deliverables listed in the task.
- Pass all acceptance criteria, with tests demonstrating each.
- Run tests in MOCK_MODE.
- Not touch files outside the task's declared scope. If a shared file
  needs a change, that change belongs to a separate task.
- Include a short PR description mapping changes to acceptance items.

PRs that expand scope, combine tasks, or skip acceptance tests MUST be
rejected.

### 5.3 Prompt Template for Coding Agents

When handing a task to an automated agent (Jules or equivalent), use the
template in Appendix B of `LAB_PHASE_ROADMAP.md`. Do not paraphrase the
task; copy the deliverables, contract, acceptance, and out-of-scope
sections verbatim.

### 5.4 When to Stop and Ask

An agent MUST stop and request review if any of the following occur:

- The task requires changing `GameState` schema.
- The task requires touching `SafetyLayer`, `Session`, or `Config` in
  ways not declared in the task.
- The task requires network access, OS input, or filesystem access
  outside the designated modules.
- A dependency task is incomplete or failing.
- A test cannot be written in MOCK_MODE.
- The task's acceptance criteria cannot be met without violating
  `LAB_CONSTRAINTS.md`.

Stopping is always preferable to silently diverging.

---

## 6. Coding Standards

### 6.1 Language and Tooling

- Python 3.12+
- `asyncio` for concurrency
- `aiosqlite` for DB access
- `pytest` + `pytest-asyncio` for tests
- `mypy` strict for types
- `Ruff` for lint and format
- `uv` for package management

### 6.2 Determinism

- Per-component RNG only. Each component receives its own
  `numpy.random.Generator` seeded from config.
- **Global `np.random.seed` is forbidden.** Any task that introduces it
  MUST be rejected.
- Deterministic components (FSM, Reflex, Watchdog, Navigation) MUST
  produce identical outputs given identical inputs and seeds.

### 6.3 LLM Usage

- Local Ollama only. No cloud fallback, ever.
- Default model: Qwen 2.5 7B (configurable).
- Only `strategist` may call the LLM.
- All prompts and responses are logged with a stable prompt hash.

### 6.4 Logging

- Structured, append-only, JSON lines.
- No log truncation or deletion on error. Ever.
- Every session writes `session.json`, `events.jsonl`, and (on crash)
  `crash.json`.
- Every event has `ts`, `event`, `payload`.

### 6.5 Config

- All config keys live in `config/lab.example.toml` with a comment.
- Missing or unknown keys raise `ConfigError` at load time.
- Any task that adds a config key MUST update the example file.

---

## 7. Forbidden Actions (any mode)

The following are forbidden and MUST cause a task to be rejected:

- Reading or writing game process memory.
- Injecting DLLs, hooking syscalls, or patching the game binary.
- Connecting to any server outside `LAB_SERVER_ALLOWLIST`.
- Including any Blizzard-owned domain or IP in the allowlist.
- Calling a cloud LLM.
- Using global `np.random.seed`.
- Using `pynput`, `pyautogui`, `keyboard`, or any OS input library
  outside `src/wow_bot/actuation/drivers/`.
- Using screen capture (`mss`, `dxcam`, etc.) outside
  `src/wow_bot/perception/` (currently empty; reserved for future).
- Truncating, rotating, or deleting logs on error.
- Adding tests that require a live game client.
- Adding CI steps that run in LAB_MODE.
- Hardcoding secrets, tokens, or credentials anywhere.

---

## 8. Phase Discipline

Phases in `LAB_PHASE_ROADMAP.md` MUST be completed in order.

- A phase is "complete" only when all its tasks pass acceptance.
- A new phase MUST NOT start until the previous phase's acceptance
  criteria are met and merged.
- Within a phase, tasks may proceed in parallel only if their declared
  dependencies allow it.
- Phase 0 (Foundation) is non-negotiable and MUST be fully complete
  before any LAB_MODE actuation is written.

---

## 9. Safety Escalation

If at any point during a task it becomes clear that the implementation
would require violating `LAB_CONSTRAINTS.md`, the agent MUST:

1. Stop immediately.
2. Do not implement the violating behavior, not even behind a flag.
3. Open an issue describing the conflict.
4. Wait for explicit resolution before resuming.

"Do not implement, even behind a flag" is literal. Flags get flipped.
Half-measures get shipped. Stop and ask.

---

## 10. References Quick Index

- Phase and task list: `LAB_PHASE_ROADMAP.md`
- Hard boundaries: `LAB_CONSTRAINTS.md`
- Layer table: `ARCHITECTURE.md` §Layers
- Data flow: `ARCHITECTURE.md` §Data Flow
- Safety spec: `SAFETY.md`
- Actuation spec: `ACTUATION.md`
- Perception spec (future): `PERCEPTION.md`
- Legacy validation tasks: `LOCAL_VALIDATION_ROADMAP.md`
- Project scope statement: `README.md`

---

## 11. Versioning

This file is versioned with the repo. Changes MUST be made by explicit
edit, not by agent inference. When a change is made, the section that
changed MUST be listed in the commit message body.