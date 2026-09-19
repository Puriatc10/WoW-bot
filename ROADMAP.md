# WoW-Bot Implementation Roadmap

> **Project:** WoW-Bot — Research prototype for game bot architecture study
> **Approach:** Mock-First; dry-run by default; real-input integration is restricted to isolated lab validation
> **Target Runtime:** Python 3.12+, asyncio
> **LLM Runtime:** Ollama + Qwen 2.5 7B (local)
> **Last Updated:** 2026-09-19

> **Scope boundary:** This roadmap is for an academic research prototype. Development and evaluation before Phase 9 use synthetic/mock inputs. Any Phase 9 validation must use prerecorded screenshots, synthetic fixtures, or an isolated lab environment under your control; do not deploy against official game services.

## Source of Truth

- `ROADMAP.md` is authoritative for **task order, dependencies, outputs, and acceptance criteria**.
- `AGENTS.md` is authoritative for **architecture, contracts, coding rules, safety boundaries, and agent workflow**.
- If the two files conflict, stop and ask for clarification instead of guessing.
- Do not silently reinterpret an acceptance criterion. If it is impossible or internally inconsistent, report the issue before changing scope.

## Per-Task Execution Protocol

For every task, the implementing agent must follow this sequence:

1. Read `AGENTS.md` completely, then read this roadmap and the current task.
2. Inspect the existing implementation and all direct dependencies before editing.
3. Make the smallest change that satisfies the task; do not refactor unrelated code.
4. Add or update tests that directly prove the task acceptance criteria and relevant interface/architecture invariants.
5. Run the narrow tests for the task first, then the canonical project checks (`pytest`, `ruff check`, `mypy --strict`) as applicable.
6. Verify no unrelated generated/runtime files are staged.
7. Commit exactly one task-scoped commit unless the repository workflow says otherwise.
8. Report: files changed, acceptance criteria evidence, commands run, and any remaining risk/blocker.

**Do not start the next task while the current task has failing acceptance criteria.**

---

## 📌 Roadmap Principles

1. **Each task ≤ 60 minutes of agent work.** If bigger, split it.
2. **Each task has explicit output file(s).** No ambiguity about what to create.
3. **Each task has acceptance criteria.** Must be testable and pass/fail.
4. **Dependencies are explicit.** No ordering by guessing.
5. **Mock first, real later.** Perception is mocked until Phase 9.

---

## 🛠️ Locked Technical Decisions

| Item | Choice |
|---|---|
| Language | Python 3.12+ |
| Concurrency | `asyncio` |
| LLM Runtime | Ollama + Qwen 2.5 7B |
| Chaos System | Lorenz attractor (RK4) |
| Memory Store | SQLite (via `aiosqlite`) |
| Logging | `loguru` |
| Config | JSON + Pydantic validation |
| Package Manager | `uv` |
| Testing | `pytest` + `pytest-asyncio` |
| Type Checking | `mypy` (strict mode) |
| Linting | `ruff` |

---

## 📦 Phase 0: Bootstrap

**Goal:** Project skeleton, dependencies, tooling.

### Task 0.1 — Project Directory Structure
**Dependencies:** none
**Output:** Folder tree under project root.

```
wow-bot/
├── pyproject.toml
├── README.md
├── ROADMAP.md
├── AGENTS.md
├── .gitignore
├── .env.example
├── config/
│   └── config.json
├── src/
│   └── wow_bot/
│       ├── __init__.py
│       ├── shared/
│       ├── internal_dynamics/
│       ├── strategist/
│       ├── executor/
│       ├── perception/
│       ├── watchdog/
│       └── mocks/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
└── scripts/
```

**Acceptance Criteria:**
- All directories created.
- Every package directory has `__init__.py` (except `tests/`, `scripts/`, `config/`).
- `README.md` exists with one-line project description and install command.

---

### Task 0.2 — `pyproject.toml` Setup
**Dependencies:** 0.1
**Output:** `pyproject.toml`

**Required dependencies:**
```toml
[project]
name = "wow-bot"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "openai>=1.50.0",
    "httpx>=0.27.0",
    "loguru>=0.7.2",
    "pydantic>=2.9.0",
    "pydantic-settings>=2.5.0",
    "numpy>=2.0.0",
    "aiosqlite>=0.20.0",
    "scipy>=1.14.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=6.0.0",
    "ruff>=0.6.0",
    "mypy>=1.11.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.mypy]
strict = true
python_version = "3.12"
```

**Acceptance Criteria:**
- `uv sync --all-extras` succeeds without errors.
- `python -c "import openai, httpx, loguru, pydantic, numpy, aiosqlite, scipy"` runs clean.

---

### Task 0.3 — Config File + Settings
**Dependencies:** 0.2
**Output:** `config/config.json`, `src/wow_bot/shared/config.py`

**`config.json` structure:**
```json
{
  "llm": {
    "base_url": "http://127.0.0.1:11434/v1/",
    "api_key": "ollama",
    "model": "qwen2.5:7b",
    "timeout_seconds": 60,
    "max_retries": 3
  },
  "internal_dynamics": {
    "update_interval_ms": 100,
    "lorenz_sigma": 10,
    "lorenz_rho": 28,
    "lorenz_beta": 2.667,
    "lorenz_dt": 0.001,
    "trigger_threshold_base": 0.3,
    "memory_db_path": "data/memory.db"
  },
  "executor": {
    "human_delay_base_ms": 200,
    "human_delay_sigma_base": 0.4,
    "dry_run": true
  },
  "logging": {
    "level": "INFO",
    "log_file": "logs/bot.log",
    "rotation": "1 day",
    "retention": "7 days"
  }
}
```

**Requirements:**
- `Settings` class using Pydantic `BaseSettings`.
- Loads from `config/config.json` by default.
- `get_settings()` singleton accessor.
- All fields typed with sensible defaults.

**Acceptance Criteria:**
- `python -c "from wow_bot.shared.config import get_settings; print(get_settings().llm.model)"` prints `qwen2.5:7b`.
- Invalid config raises a clear Pydantic error.

---

### Task 0.4 — Shared Logger
**Dependencies:** 0.3
**Output:** `src/wow_bot/shared/logger.py`

**Requirements:**
- Wrapper around `loguru`.
- Console output with colors.
- File output with daily rotation.
- Millisecond timestamps.
- Context tag per module (e.g., `[DYNAMICS]`, `[LLM]`, `[EXEC]`).
- `get_logger(tag: str)` factory function.

**Acceptance Criteria:**
```python
from wow_bot.shared.logger import get_logger
log = get_logger("DYNAMICS")
log.info("test")
```
- One line printed to console.
- One line appended to `logs/bot.log`.

---

### Task 0.5 — `.gitignore`
**Dependencies:** 0.1
**Output:** `.gitignore`

**Must ignore:**
```
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

**Acceptance Criteria:**
- `git status` doesn't show `logs/`, `data/`, `__pycache__/`.

---

## 🔗 Phase 1: Interfaces & Data Contracts

**Goal:** Define `GameState`, `MetaState`, `Strategy`, `Event` as frozen contracts between layers.

### Task 1.1 — `GameState` and Sub-types
**Dependencies:** 0.3
**Output:** `src/wow_bot/shared/interfaces.py`

**Required classes:**
```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TargetInfo:
    name: str
    hp_pct: float                    # 0..1
    reaction: str                    # "hostile" | "neutral" | "friendly"
    distance_estimate: float         # yards, approximate

@dataclass
class EnemyInfo:
    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float                # 0..1
    distance_estimate: float         # yards, approximate

@dataclass
class Event:
    type: str
    timestamp: float
    data: dict = field(default_factory=dict)

@dataclass
class GameState:
    timestamp: float
    hp_pct: float
    mana_pct: float
    position: tuple[float, float]
    facing: float
    in_combat: bool
    target: Optional[TargetInfo]
    enemies: list[EnemyInfo]
    events: list[Event]
```

**Acceptance Criteria:**
- `tests/unit/test_interfaces.py` builds a `GameState` and serializes it to JSON.
- All type hints pass `mypy --strict`.

---

### Task 1.2 — `MetaState` and `Strategy`
**Dependencies:** 1.1
**Output:** extend `src/wow_bot/shared/interfaces.py`

```python
import numpy as np

@dataclass
class MetaState:
    vector: np.ndarray           # shape (5,): [hunger, fatigue, curiosity, aggression, social]
    recent_events: list[Event]
    timestamp: float

@dataclass
class Strategy:
    goal: str                    # "farm_herbs" | "grind_humans" | "explore" | "flee"
    region: str
    risk_tolerance: float        # 0..1
    priority: list[str]
    constraints: dict
    valid_until: float
    raw_llm_output: str = ""
```

**Acceptance Criteria:**
- `Strategy` round-trips through JSON (to/from dict).
- Pydantic validator enforces `0 <= risk_tolerance <= 1`.

---

### Task 1.3 — Event Types Registry
**Dependencies:** 1.1
**Output:** `src/wow_bot/shared/events.py`

**Required constants:**
```python
DEATH = "death"
RARE_LOOT = "rare_loot"
PVP_HIT = "pvp_hit"
PVP_KILL = "pvp_kill"
STUCK = "stuck"
LEVEL_UP = "level_up"
QUEST_COMPLETE = "quest_complete"
NPC_INTERACT = "npc_interact"

EVENT_EFFECTS = {
    DEATH:        {"hunger": 0.0,   "fatigue": +0.10, "curiosity": 0.0,   "aggression": -0.15, "social": -0.05},
    RARE_LOOT:    {"hunger": +0.20, "fatigue": 0.0,   "curiosity": +0.15, "aggression": 0.0,   "social": +0.05},
    PVP_HIT:      {"hunger": 0.0,   "fatigue": +0.05, "curiosity": -0.05, "aggression": +0.10, "social": -0.10},
    PVP_KILL:     {"hunger": 0.0,   "fatigue": 0.0,   "curiosity": 0.0,   "aggression": +0.05, "social": +0.05},
    STUCK:        {"hunger": 0.0,   "fatigue": +0.15, "curiosity": -0.10, "aggression": -0.05, "social": 0.0},
    LEVEL_UP:     {"hunger": +0.10, "fatigue": -0.10, "curiosity": +0.10, "aggression": +0.05, "social": +0.05},
    QUEST_COMPLETE: {"hunger": +0.10, "fatigue": -0.05, "curiosity": +0.10, "aggression": 0.0, "social": +0.05},
    NPC_INTERACT: {"hunger": 0.0,   "fatigue": -0.05, "curiosity": 0.0,   "aggression": 0.0,   "social": +0.15},
}
```

**Acceptance Criteria:**
- Every event type in `EVENT_EFFECTS`.
- `mypy --strict` passes.

---

## 🎭 Phase 2: Mock Perception

**Goal:** Build a simulator that emits realistic `GameState`, so we can develop all other layers independently.

### Task 2.1 — Simple Mock Perception
**Dependencies:** 1.1, 1.3
**Output:** `src/wow_bot/mocks/mock_perception.py`

**Requirements:**
- `MockPerception` class with `async def get_state() -> GameState`.
- HP/Mana fluctuate randomly between 0.3 and 1.0.
- Random events emitted occasionally.
- Position follows a slow random walk.
- `in_combat` toggles True for 10s every ~30s.

**Acceptance Criteria:**
```python
async def test():
    p = MockPerception()
    for _ in range(10):
        state = await p.get_state()
        assert 0 <= state.hp_pct <= 1
        assert isinstance(state.timestamp, float)
        await asyncio.sleep(0.1)
```
Runs without error.

---

### Task 2.2 — Scenario-driven Mock Perception
**Dependencies:** 2.1
**Output:** extend `mock_perception.py`

**Supported scenarios:**
- `"peaceful_farm"` — no combat, HP stable.
- `"combat_light"` — short combats, single enemy.
- `"death_loop"` — dies every ~2 minutes.
- `"rare_loot_drought"` — no rare loot events.
- `"stuck_repeatedly"` — stuck events every few minutes.

**Requirements:**
- Scenario selected via config or constructor arg.
- Reproducible via seed.

**Acceptance Criteria:**
- Test runs each scenario for at least 10 frames.
- All relevant event types appear in respective scenarios.

---

## 🧠 Phase 3: Internal Dynamics

**Goal:** Implement the heart of the system — natural variation with pink noise.

### Task 3.1 — Drives Core
**Dependencies:** 1.2, 1.3
**Output:** `src/wow_bot/internal_dynamics/drives.py`

**Required API:**
```python
class Drives:
    def __init__(self, config): ...
    @property
    def vector(self) -> np.ndarray: ...              # [hunger, fatigue, curiosity, aggression, social]
    def step(self, dt: float, chaos_component: float) -> None: ...
    def apply_event(self, event: Event) -> None: ...
    def decay(self, dt: float) -> None: ...
```

**Requirements:**
- Each drive clamped to `[0, 1]`.
- Drift rate differs per drive (`fatigue` faster than `social`).
- Event effects applied immediately, decay slowly to baseline (0.5).
- Baseline itself can drift subtly over hours (optional, low priority).

**Acceptance Criteria:**
- Start all drives at 0.5, run 1000 steps with no events, all stay in `[0, 1]`.
- Emit `death` event → `aggression` drops immediately, recovers slowly over ~10 minutes of simulated time.

---

### Task 3.2 — Coupled Oscillators
**Dependencies:** 3.1
**Output:** `src/wow_bot/internal_dynamics/oscillators.py`

**Required API:**
```python
class OscillatorBank:
    def __init__(self, config): ...
    def step(self, dt: float) -> float: ...          # sum of all oscillators
    @property
    def frequencies(self) -> list[float]: ...
    @property
    def phases(self) -> list[float]: ...
```

**Requirements:**
- 5 oscillators with incommensurable frequencies (Hz): `1/18000, 1/5400, 1/1200, 1/300, 1/60`.
- Amplitudes: `[0.15, 0.08, 0.05, 0.03, 0.01]`.
- Initial phases from config or seed.
- Output sum bounded to `[-0.3, +0.3]`.

**Acceptance Criteria:**
- After 10,000 steps, FFT of output resembles 1/f spectrum.
- Max absolute value never exceeds 0.35.

---

### Task 3.3 — Lorenz Attractor
**Dependencies:** 0.3
**Output:** `src/wow_bot/internal_dynamics/chaos.py`

**Required API:**
```python
class LorenzAttractor:
    def __init__(self, sigma=10, rho=28, beta=2.667, dt=0.001): ...
    def step(self) -> np.ndarray: ...                # returns new (x, y, z)
    def normalized(self) -> float: ...               # in [-1, +1]
```

**Requirements:**
- Use **RK4** integration (not Euler).
- Initial condition: `(1.0, 1.0, 1.0)` or from seed.
- Normalize: `(x - mean_x) / std_x` approximate.

**Acceptance Criteria:**
- After 1000 steps, no NaN or Inf.
- Trajectory remains bounded in phase space.
- Two initial conditions differing by 0.001 diverge after 1000 steps (chaos signature).

---

### Task 3.4 — Memory Store (SQLite)
**Dependencies:** 1.1
**Output:** `src/wow_bot/internal_dynamics/memory.py`

**Required API:**
```python
class MemoryStore:
    def __init__(self, db_path: str): ...
    async def init(self) -> None: ...
    async def add(self, event: Event, state_vector: np.ndarray) -> None: ...
    async def recall_similar(self, state: np.ndarray, k: int = 5) -> list[Event]: ...
    async def decay_old(self, max_age_hours: float = 72) -> None: ...
    async def close(self) -> None: ...
```

**SQLite schema:**
```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    timestamp REAL NOT NULL,
    state_vector BLOB NOT NULL,
    data TEXT
);
CREATE INDEX idx_event_type ON memories(event_type);
CREATE INDEX idx_timestamp ON memories(timestamp);
```

**Acceptance Criteria:**
- Store event with vector `[0.5]*5`, recall with `[0.51]*5` → returns it.
- `decay_old` removes events older than threshold.
- All operations async-safe.

---

### Task 3.5 — MetaState Generator
**Dependencies:** 3.1, 3.2, 3.3, 3.4
**Output:** `src/wow_bot/internal_dynamics/meta_state.py`

**Required API:**
```python
class MetaStateGenerator:
    def __init__(self, config, drives, oscillators, chaos, memory): ...
    async def step(self, dt: float, game_state: GameState) -> MetaState: ...
    def should_trigger_llm(self, current: MetaState, last: MetaState) -> bool: ...
```

**Algorithm:**
1. `chaos.step()` → scalar
2. `oscillators.step(dt)` → scalar
3. `drives.step(dt, chaos_val)`
4. For each event in `game_state.events`: `drives.apply_event(event)`
5. `drives.decay(dt)`
6. `memory.add()` for important events
7. Build `MetaState` from `drives.vector`
8. Trigger check: `norm(current - last) > adaptive_threshold`

**Acceptance Criteria:**
- 100 steps at `dt=0.1`: MetaState vector always in `[0,1]^5`.
- Emit `death` → visible vector shift.
- `should_trigger_llm`: large change → True, small change → False.

---

### Task 3.6 — Adaptive Trigger Threshold
**Dependencies:** 3.5
**Output:** extend `meta_state.py`

**Requirements:**
- Threshold not constant; follows a slow oscillator.
- `threshold = 0.3 + 0.1 * sin(2π * t / 7200)` (2-hour period).

**Acceptance Criteria:**
- Over 2 simulated hours (72,000 steps at `dt=0.1`), threshold passes through `[0.2, 0.4]` at least once.

---

### Task 3.7 — Spectrum Analysis Test
**Dependencies:** 3.5
**Output:** `tests/integration/test_spectrum.py`

**Requirements:**
- Run 10,000 steps of `MetaStateGenerator`.
- FFT of each drive's time series.
- Compute PSD via `scipy.signal.welch`.
- Fit slope on log-log plot.

**Acceptance Criteria:**
- Slope in `[-1.5, -0.5]` (close to 1/f).
- Test fails if outside range.

---

## 🎯 Phase 4: Strategist Layer

**Goal:** Connect to local LLM, design effective prompts.

### Task 4.1 — LLM Client
**Dependencies:** 0.3
**Output:** `src/wow_bot/strategist/llm_client.py`

**Required API:**
```python
class LLMClient:
    def __init__(self, config): ...
    async def query(self, system_prompt: str, user_prompt: str) -> str: ...
    async def health_check(self) -> bool: ...
```

**Requirements:**
- Use `openai.AsyncOpenAI` with `http_client=httpx.AsyncClient(trust_env=False)`.
- Timeout from config.
- Retry up to 3 times with exponential backoff.
- Log every call: prompt length, response time, token count.

**Acceptance Criteria:**
- `health_check()` returns True with running Ollama.
- `query("You are helpful.", "Say hi")` returns a string.

---

### Task 4.2 — Prompt Templates
**Dependencies:** 4.1
**Output:** `src/wow_bot/strategist/prompts.py`

**Required:**
- `SYSTEM_PROMPT`: persistent player persona (constant).
- `build_user_prompt(meta_state, recent_events, previous_strategy) -> str`.

**System Prompt template:**
```
You are roleplaying as a casual World of Warcraft player with these persistent traits:
- You sometimes get distracted and lose focus
- When tired, you make riskier decisions
- You enjoy fighting groups of enemies
- When low on gold, you become greedy
- You get bored of repetitive patterns quickly
- You speak only in JSON when asked for a strategy

Your job is NOT to make moment-to-moment decisions.
Your job is to define a HIGH-LEVEL strategy for the next 20-40 minutes.
Another system will execute your strategy using fast local decisions.
```

**User Prompt template:**
```
[INTERNAL STATE]
hunger: {hunger:.2f}
fatigue: {fatigue:.2f}
curiosity: {curiosity:.2f}
aggression: {aggression:.2f}
social: {social:.2f}

[RECENT EVENTS]
{events_list}

[PREVIOUS STRATEGY]
Goal: {prev_goal}
Region: {prev_region}
Risk tolerance: {prev_risk}

[YOUR TASK]
Produce a NEW strategy that EVOLVES from the previous one.
Do not start from scratch.
Respond ONLY with valid JSON in this exact schema:
{
  "goal": "string",
  "region": "string",
  "risk_tolerance": 0.0-1.0,
  "priority": ["string", ...],
  "constraints": {"max_deaths_per_hour": int},
  "reasoning": "one short sentence"
}
```

**Acceptance Criteria:**
- `build_user_prompt` returns a string containing all required sections.
- Sample MetaState produces prompt with all fields filled.

---

### Task 4.3 — Response Parser + Validator
**Dependencies:** 4.2, 1.2
**Output:** `src/wow_bot/strategist/parser.py`

**Required API:**
```python
def parse_strategy(raw: str) -> Strategy:
    """Build a Strategy from raw LLM output."""
```

**Requirements:**
- Clean markdown fences (` ```json ... ``` `).
- Find first `{` and last `}` to extract JSON.
- Validate with Pydantic.
- On failure: return a default fallback Strategy.
- Set `valid_until = time.time() + 30*60`.

**Acceptance Criteria:**
- Handles: pure JSON, JSON in markdown fence, JSON with leading/trailing text, incomplete JSON (→ fallback).
- No exceptions in any case.

---

### Task 4.4 — Strategist Orchestrator
**Dependencies:** 4.1, 4.2, 4.3
**Output:** `src/wow_bot/strategist/orchestrator.py`

**Required API:**
```python
class Strategist:
    def __init__(self, config, llm_client, memory): ...
    async def generate_strategy(self, meta_state: MetaState) -> Strategy: ...
    @property
    def current_strategy(self) -> Strategy | None: ...
    def is_expired(self) -> bool: ...
```

**Acceptance Criteria:**
- With sample MetaState, returns valid Strategy.
- On LLM timeout, falls back to previous strategy.

---

## ⚡ Phase 5: Executor Layer

**Goal:** Convert Strategy to real keyboard/mouse actions with human-like timing.

### Task 5.1 — Controller Base
**Dependencies:** 0.2
**Output:** `src/wow_bot/executor/controller.py`

**Required API:**
```python
class Controller:
    async def press_key(self, key: str, duration_ms: int = 50) -> None: ...
    async def move_mouse(self, x: int, y: int) -> None: ...
    async def click(self, button: str = "left") -> None: ...
    async def stop_all(self) -> None: ...
```

**Requirements:**
- Use `pynput` or `pyautogui`.
- **Dry-run mode** via config flag: log actions without executing.

**Acceptance Criteria:**
- `dry_run=True`: no actual input, but logs produced.
- `dry_run=False` on Notepad: text is typed.

---

### Task 5.2 — Human-like Timing
**Dependencies:** 5.1, 3.1
**Output:** `src/wow_bot/executor/humanize.py`

**Required API:**
```python
def human_delay(base_ms: int = 200, fatigue: float = 0.5, 
                chaos_component: float = 0.0) -> float:
    """Log-normal delay correlated with fatigue."""

def human_error_probability(drives_vector: np.ndarray) -> float:
    """Error probability correlated with fatigue and boredom."""

def jitter_coordinates(x: int, y: int, radius: int = 5) -> tuple[int, int]:
    """Random jitter for clicks."""
```

**Implementation:**
```python
sigma = 0.4 + 0.2 * chaos_component + 0.3 * fatigue
mu = np.log(base_ms)
delay = np.clip(np.random.lognormal(mu, sigma), 50, 2000)
```

**Acceptance Criteria:**
- 1000 `human_delay` calls: distribution is log-normal (KS test p > 0.05).
- Mean within `[150, 300]` for `base_ms=200`.
- `human_error_probability` in `[0.02, 0.15]`.

---

### Task 5.3 — FSM Core
**Dependencies:** 5.1, 5.2, 1.2
**Output:** `src/wow_bot/executor/fsm.py`

**Required API:**
```python
class State(Enum):
    IDLE = auto()
    SCANNING = auto()
    MOVING_TO_TARGET = auto()
    COMBAT = auto()
    LOOTING = auto()
    FLEEING = auto()
    STUCK_RECOVERY = auto()

class ExecutorFSM:
    def __init__(self, config, controller, strategy): ...
    async def tick(self, game_state: GameState) -> None: ...
    def set_strategy(self, strategy: Strategy) -> None: ...
```

**Requirements:**
- Transition rules parameterized by strategy (e.g., `risk_tolerance` affects FLEE threshold).
- Every transition logged.
- `STUCK_RECOVERY` auto-triggered if same action repeats for 5s.

**Acceptance Criteria:**
- With MockPerception, FSM transitions correctly in each scenario.
- `risk_tolerance=0.9` → later FLEE; `risk_tolerance=0.1` → earlier FLEE.

---

### Task 5.4 — Idle Behaviors
**Dependencies:** 5.3
**Output:** `src/wow_bot/executor/idle_behaviors.py`

**Required behaviors:**
- Camera rotation without purpose (when `curiosity` low).
- Sudden stop mid-path.
- Open Inventory without reason.
- Random emotes (`/wave`, `/laugh`).
- Angled path movement.

**Acceptance Criteria:**
- Each behavior is an independent callable from FSM.
- In 100 ticks with `dry_run=True`, at least 3 idle behaviors fire.

---

### Task 5.5 — Path Variation
**Dependencies:** 5.3
**Output:** `src/wow_bot/executor/path.py`

**Required API:**
```python
def generate_path(start: tuple, end: tuple, num_points: int = 5) -> list[tuple]:
    """Curved path with random control points."""
```

**Acceptance Criteria:**
- Path is not straight (std dev from line > 5 px).
- Path always passes through start and end.

---

## 🐕 Phase 6: Watchdog

### Task 6.1 — Watchdog Process
**Dependencies:** 5.3
**Output:** `src/wow_bot/watchdog/watchdog.py`

**Requirements:**
- Independent `multiprocessing.Process`.
- Monitors: stuck detection, death loop detection, high CPU usage.
- Kill switch: `F10` → `sys.exit()` immediately.
- Emergency shutdown must stop input generation, flush/close resources, and preserve audit logs for reproducibility. Do not delete logs as part of shutdown.

**Acceptance Criteria:**
- Pressing F10 kills the process in <1s.
- Watchdog detects if bot does nothing for 30s.

---

## 🔌 Phase 7: Integration

### Task 7.1 — Async Pipeline
**Dependencies:** all prior phases
**Output:** `src/wow_bot/main.py`

**Required:**
```python
async def main():
    config = get_settings()
    # 1. Build all components
    # 2. Start asyncio tasks:
    #    - perception_loop (or mock)
    #    - dynamics_loop
    #    - strategist_loop
    #    - executor_loop
    # 3. Wait for Ctrl+C
```

**Acceptance Criteria:**
- `python -m wow_bot.main` runs.
- With MockPerception, runs for 5 minutes without crash.
- Logs show full cycle working.

---

### Task 7.2 — Scenario Runner
**Dependencies:** 7.1
**Output:** `scripts/run_scenario.py`

**Requirements:**
- Pick scenario from MockPerception.
- Run for 10 minutes.
- Collect stats: LLM call count, state transitions, time delay distribution.
- Output: JSON report to `reports/`.

**Acceptance Criteria:**
- `python scripts/run_scenario.py --scenario combat_light --duration 600`
- JSON output with complete stats.

---

## 📊 Phase 8: Analysis & Final Testing

### Task 8.1 — FFT Analysis on Scenario/Recorded Data
**Dependencies:** 7.2
**Output:** `scripts/analyze_spectrum.py`

**Requirements:**
- From scenario report, extract MetaState time series.
- FFT + PSD + slope calculation.
- Output: plot + slope value.

**Acceptance Criteria:**
- Slope in `[-1.5, -0.5]`.

---

### Task 8.2 — Timing Distribution Analysis
**Dependencies:** 7.2
**Output:** `scripts/analyze_timing.py`

**Requirements:**
- Extract all `human_delay` values from logs.
- KS test vs log-normal.
- CV calculation.

**Acceptance Criteria:**
- p-value > 0.05 (fits log-normal).
- CV > 0.3.

---

### Task 8.3 — 24-Hour Test
**Dependencies:** all
**Output:** final report

**Requirements:**
- Run bot for 24h with MockPerception continuously.
- Measure: memory usage, CPU usage, log size.
- Check for leaks or crashes.

**Acceptance Criteria:**
- Zero crashes in 24h.
- Memory usage stable (no leak).

---

## 🎁 Phase 9: Real Perception Integration

### Task 9.1 — Perception Adapter
**Dependencies:** teammate's Perception delivered
**Output:** `src/wow_bot/perception/adapter.py`

**Requirements:**
- Adapter converts teammate's output to `GameState`.
- Validation of incoming data.

**Acceptance Criteria:**
- From a prerecorded/lab screenshot fixture, produces a valid `GameState`.

---

## 📋 Master Checklist for Agent

```
Phase 0 (Bootstrap):       Tasks 0.1 → 0.2 → 0.3 → 0.4 → 0.5
Phase 1 (Interfaces):      Tasks 1.1 → 1.2 → 1.3
Phase 2 (Mock):            Tasks 2.1 → 2.2
Phase 3 (Dynamics):        Tasks 3.1 → 3.2 → 3.3 → 3.4 → 3.5 → 3.6 → 3.7
Phase 4 (Strategist):      Tasks 4.1 → 4.2 → 4.3 → 4.4
Phase 5 (Executor):        Tasks 5.1 → 5.2 → 5.3 → 5.4 → 5.5
Phase 6 (Watchdog):        Tasks 6.1
Phase 7 (Integration):     Tasks 7.1 → 7.2
Phase 8 (Analysis):        Tasks 8.1 → 8.2 → 8.3
Phase 9 (Real Perception): Task 9.1 (blocked until teammate delivers)
```

**Golden Rule:** Do not start a phase until the previous phase is complete and its tests pass.

---

## 🚫 What NOT to Do

- ❌ Do NOT touch the game process (no DLL injection, no ReadProcessMemory, no addons).
- ❌ Do NOT deploy or validate the automation against official game services; use mock/synthetic/prerecorded or isolated lab inputs.
- ❌ Do NOT make network calls to non-local services (LLM must be local).
- ❌ Do NOT use fixed random seeds in production (only in tests).
- ❌ Do NOT hardcode UI coordinates (use config).
- ❌ Do NOT skip acceptance criteria tests.
- ❌ Do NOT commit `.env`, `logs/`, `data/`, or `*.db`.

---

## 🎯 Success Criteria (Project Level)

By end of Phase 8, the system must demonstrate:

1. **Spectral signature:** 1/f noise in MetaState time series (slope in `[-1.5, -0.5]`).
2. **Human-like timing:** Log-normal delay distribution (KS p > 0.05).
3. **Adaptive strategy:** LLM produces evolving strategies (not replacing from scratch).
4. **No process contamination:** Zero contact with game process.
5. **Stable operation:** 24h continuous run without crash or leak.

---

**END OF ROADMAP**
