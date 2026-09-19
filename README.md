# WoW-Bot Research Prototype

A **mock-first academic research prototype** for studying autonomous game-agent architecture, behavioral dynamics, and the limitations of modern anti-cheat detection approaches.

> **Research scope:** This repository is intended for synthetic inputs, prerecorded screenshots, and isolated lab environments under your control. It is not intended for deployment against official game services.

---

## Overview

The project explores a layered autonomous-agent architecture in which high-level strategy, internal state, perception, and execution are separated behind explicit contracts.

The main architectural idea is that the language model acts only as a **high-level strategist**. Fast, moment-to-moment decisions are handled locally by deterministic components such as the internal-dynamics engine and executor state machine.

Development follows a **Mock-First** approach: the real perception layer is intentionally postponed until the rest of the system can be developed, tested, and analyzed against synthetic `GameState` inputs.

---

## Architecture

```text
┌──────────────────────────────────────────┐
│              Perception Layer            │
│      screenshot / mock → GameState       │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│         Internal Dynamics Layer          │
│ Drives + Oscillators + Chaos + Memory    │
│          → MetaState + trigger           │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│             Strategist Layer             │
│          Local LLM via Ollama            │
│              → Strategy                  │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│              Executor Layer              │
│       FSM + timing + path planning       │
│                → Actions                 │
└────────────────────┬─────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────┐
│              Watchdog Layer              │
│     health monitoring + emergency stop   │
└──────────────────────────────────────────┘
```

### Layer responsibilities

| Layer | Responsibility | Main output |
|---|---|---|
| Perception | Convert synthetic or visual input into normalized game state | `GameState` |
| Internal Dynamics | Maintain long-lived internal state, memory, oscillators, and trigger logic | `MetaState` |
| Strategist | Produce high-level strategy from state and recent history | `Strategy` |
| Executor | Translate strategy and current state into local FSM actions | Controller actions |
| Watchdog | Observe health, stuck conditions, and shutdown signals | Diagnostics / stop signal |

The layer boundaries are intentional. Cross-layer shortcuts and shared mutable global state should be avoided.

---

## Core Contracts

The main interfaces between layers are defined in `src/wow_bot/shared/interfaces.py`.

### `GameState`

Produced by Perception and consumed by Internal Dynamics and Executor.

Typical fields include:

- timestamp
- HP / mana percentage
- approximate position and facing
- combat state
- target information
- nearby enemies
- emitted events

### `MetaState`

Produced by Internal Dynamics and consumed by Strategist.

It contains the five-dimensional internal-state vector:

```text
[hunger, fatigue, curiosity, aggression, social]
```

plus recent events and a timestamp.

### `Strategy`

Produced by Strategist and consumed by Executor.

It contains high-level intent such as:

- goal
- region
- risk tolerance
- priorities
- constraints
- validity window

### `Event`

Used to communicate meaningful state changes between Perception, Dynamics, and Memory.

The public contracts are treated as stable. Existing fields should not be removed or renamed without coordination.

---

## Technical Stack

| Area | Choice |
|---|---|
| Language | Python 3.12+ |
| Concurrency | `asyncio` |
| Package manager | `uv` |
| Local LLM runtime | Ollama |
| Default model | Qwen 2.5 7B |
| LLM client | OpenAI-compatible async client over local Ollama |
| Validation / config | Pydantic |
| Memory store | SQLite via `aiosqlite` |
| Numerical work | NumPy + SciPy |
| Logging | Loguru |
| Testing | Pytest + pytest-asyncio |
| Type checking | mypy, strict mode |
| Linting | Ruff |

---

## Repository Structure

```text
wow-bot/
├── README.md
├── ROADMAP.md
├── AGENTS.md
├── pyproject.toml
├── .gitignore
├── .env.example
├── config/
│   └── config.json
├── src/
│   └── wow_bot/
│       ├── shared/
│       ├── internal_dynamics/
│       ├── strategist/
│       ├── executor/
│       ├── perception/
│       ├── watchdog/
│       ├── mocks/
│       └── main.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── scripts/
└── reports/
```

Important files:

| File | Purpose |
|---|---|
| `ROADMAP.md` | Authoritative task order, dependencies, outputs, and acceptance criteria |
| `AGENTS.md` | Architecture, invariants, coding rules, testing rules, and agent workflow |
| `config/config.json` | Runtime configuration and tunable parameters |
| `src/wow_bot/shared/interfaces.py` | Stable data contracts between layers |

---

## Development Model

The project is implemented phase-by-phase.

```text
Phase 0  Bootstrap
Phase 1  Interfaces & Data Contracts
Phase 2  Mock Perception
Phase 3  Internal Dynamics
Phase 4  Strategist
Phase 5  Executor
Phase 6  Watchdog
Phase 7  Integration
Phase 8  Analysis & Long-Running Tests
Phase 9  Perception Adapter / Lab Integration
```

The complete task-level plan is in [`ROADMAP.md`](./ROADMAP.md).

**Rule:** do not start a phase while the previous phase has failing acceptance criteria.

---

## Mock-First Approach

Until Phase 9, the system uses `MockPerception` instead of a real perception implementation.

This allows the core architecture to be developed independently of a game installation and provides reproducible scenarios for automated testing.

Planned mock scenarios include:

- `peaceful_farm`
- `combat_light`
- `death_loop`
- `rare_loot_drought`
- `stuck_repeatedly`

Tests use fixed seeds where reproducibility is required. Production-style simulation uses runtime entropy.

---

## Installation

### Prerequisites

- Python 3.12+
- `uv`
- Ollama, for Strategist-related phases

### Install dependencies

```bash
uv sync --all-extras
```

### Local LLM

Install the default model:

```bash
ollama pull qwen2.5:7b
```

Start Ollama if it is not already running:

```bash
ollama serve
```

The default configuration expects an OpenAI-compatible endpoint at:

```text
http://127.0.0.1:11434/v1/
```

All language-model requests are expected to remain local.

---

## Configuration

Runtime settings live in:

```text
config/config.json
```

The configuration is loaded and validated through Pydantic.

The default executor configuration must remain in dry-run mode during ordinary development:

```json
{
  "executor": {
    "dry_run": true
  }
}
```

Do not silently switch dry-run off in automated-agent changes.

---

## Running the Project

Once the integration phase is implemented, the main entry point is:

```bash
uv run python -m wow_bot.main
```

During earlier phases, individual modules and tests are the primary execution path.

A scenario runner is planned at:

```bash
uv run python scripts/run_scenario.py --scenario combat_light --duration 600
```

Scenario execution should produce structured reports under `reports/` when that phase is implemented.

---

## Testing and Quality Gates

Run the test suite:

```bash
uv run pytest
```

Run linting:

```bash
uv run ruff check .
```

Run strict type checking:

```bash
uv run mypy src tests
```

Tests are expected to be:

- deterministic when reproducibility matters
- isolated
- async-aware
- focused on behavior and acceptance criteria
- reachable from the canonical Pytest path

Important integration checks include:

- MetaState spectral analysis
- strategy fallback behavior
- malformed LLM output handling
- async resource cleanup
- executor dry-run behavior
- long-running stability tests

---

## Analysis Goals

The research prototype includes explicit measurable outputs rather than relying only on visual inspection.

Examples include:

- power spectral density analysis of internal-state time series
- log-normal timing-distribution analysis
- strategy evolution across state changes
- LLM invocation frequency
- FSM transition statistics
- memory and CPU stability during long-running simulation

The relevant scripts are developed in Phase 8.

---

## Working With Coding Agents

AI coding agents must read the project documents before modifying code.

Recommended startup instruction:

```text
Read AGENTS.md completely, then ROADMAP.md.
Inspect the repository and all dependencies of the first incomplete task.
Work on exactly one task at a time.
Do not skip acceptance criteria or change public contracts without coordination.
```

For every task, the expected workflow is:

1. Read the relevant task and dependencies.
2. Inspect the existing implementation.
3. Make the smallest task-scoped change.
4. Add tests that prove the acceptance criteria.
5. Run narrow tests.
6. Run project-level checks where applicable.
7. Report changed files, commands run, evidence, and remaining risks.

See [`AGENTS.md`](./AGENTS.md) for the complete agent protocol.

---

## Project Invariants

The following rules are intentional architectural constraints:

- no direct access to the game process
- no DLL injection or process-memory access
- no addons or game-file manipulation
- no cloud LLM dependency in the runtime path
- no hardcoded UI coordinates
- no skipped acceptance tests
- no unrelated refactors inside task-scoped changes
- no hidden blocking I/O inside the asyncio event loop
- no removal or renaming of stable contract fields without coordination
- dry-run execution by default
- preserve diagnostic logs and reproducibility data

---

## Research and Usage Boundary

This repository is an academic software-architecture study.

Development and evaluation should use:

- synthetic `GameState` data
- `MockPerception`
- prerecorded screenshots
- isolated environments under the researcher's control

The project documentation does not treat successful operation against official services as a development or acceptance goal.

---

## Documentation

Read these files in this order when joining the project:

1. [`README.md`](./README.md) — project overview and setup
2. [`AGENTS.md`](./AGENTS.md) — architecture and engineering rules
3. [`ROADMAP.md`](./ROADMAP.md) — implementation sequence and acceptance criteria

---

## License

No license has been selected yet. Add an explicit `LICENSE` file before distributing or accepting external contributions.
