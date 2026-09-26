# HAMBERGER_PORT_PLAN.md

Plan for bringing the real perception logic from the `hamberger` branch into
the `master` architecture.

This document is an investigation record plus an execution plan. It answers
two questions:

1. **How** do the `hamberger` readers land on `master`?
2. **Which** pre-real-perception roadmap tasks are actually required first —
   are all of them needed?

Task identifiers below are added to
`docs/lab_phase/PRE_REAL_PERCEPTION_FIX_ROADMAP.md`, which remains the source
of truth for task definitions. This document carries the analysis.

---

## 1. Investigation record

### 1.1 Branch topology — this is not a merge candidate

| Property | Value |
|---|---|
| Local branch | `hamberger` → `9b968a4` |
| Remote | `origin/hamberger` → `9b968a4` (identical) |
| Commit message | `Update project` (author Amirm227) |
| Parent count | **0** — this is a root commit |
| `rev-list --count` | **1** — the branch is a single commit |
| Merge base with `master` | **none** (`git merge-base` exits 1: "no merge base") |
| `master` | `2f48240 T-FIX-21` (221 commits ahead, unrelated) |

`hamberger` is an **orphan history**: an independent root commit with no
ancestry shared with `master`. `git merge hamberger` is therefore not
possible in the ordinary sense — it would be an `--allow-unrelated-histories`
merge with **zero overlap** between the trees.

### 1.2 What the orphan tree contains

```
.gitignore                     perception/__init__.py   core/__init__.py
capture/__init__.py            perception/bars.py       core/action.py
capture/screen_capture.py      perception/combat.py     core/extractor.py
main.py                        perception/enemies.py    core/state.py
models/*.png  (3 templates)    perception/events.py
weights/yolo26n.pt (5.5 MB)    perception/minimap.py
yolov8n.pt (6.5 MB)            perception/target.py
yolov8s.pt (22.6 MB)
datasets/  974 files   (YOLO training frames)
runs/       24 files   (training artifacts incl. best.pt 22.5 MB, last.pt 22.5 MB)
```

Per-directory counts: `datasets` 974, `runs` 24, `perception` 7, `core` 4,
`models` 3, `capture` 2.

**Source code is only ~30 KB across 12 files.** Everything else is datasets,
training output, and model weights (~57 MB across 4 `.pt` files).

### 1.3 Verdict on the merge

> **Do not merge the branch.** Port 12 source files and 3 PNG templates.

Concretely:

- The histories are unrelated, so a merge buys no shared ancestry and no
  conflict resolution — only a wholesale tree union.
- A merge would import **974 dataset images and ~57 MB of binary weights**
  into a repository that has never tracked them.
- A merge would also import `.gitignore`, `AGENTS.md`-adjacent metadata and
  `main.py` at the repo root, colliding with the existing `src/wow_bot/`
  layout and AGENTS.md §1's module placement rules.
- `core/state.py` defines its **own `GameState` dataclass** — a direct name
  collision with `wow_bot.shared.interfaces.GameState`. Merging would put two
  incompatible types called `GameState` in the tree.

**Mechanism:** copy the files explicitly out of the orphan commit with
`git show hamberger:<path>` / `git checkout hamberger -- <paths>` for the
source and template paths only, then reshape them (Stage 3 below).

---

## 2. Inventory of the perception logic

The `hamberger` prototype is a **pull-mode state extractor**: a background
capture thread grabs frames, and `StateExtractor.extract()` reads every
component off one frame and assembles a state. That shape maps cleanly onto
`PerceptionBackend.snapshot() -> GameState`.

| Source file | Class | Extraction method | Raw output |
|---|---|---|---|
| `capture/screen_capture.py` | `ScreenCapture` | `mss` grab on a daemon thread, frame pool, `idle_fps=10` / `combat_fps=30` switching | BGR `(h, w, 4)` `uint8` frame |
| `perception/bars.py` | `BarReader` | HSV saturation/brightness ratio over two ROIs, column-fill heuristic | HP and mana ratio in `[0,1]` |
| `perception/combat.py` | `CombatDetector` | red-hue pixel ratio on the four screen edges + cooldown latch | `in_combat: bool`, ratio |
| `perception/target.py` | `TargetReader` | template match for the target-frame nameplate → OCR (Tesseract, 3 threshold strategies) → `KNOWN_ENEMIES` whitelist; HP from green→golden pixel scan | `name`, `hp_pct` **int 0..100**, match confidence; keeps a position lock |
| `perception/enemies.py` | `EnemyDetector` | **YOLO** (`ultralytics`) with `conf=0.5` | list of dicts: `name`, `bbox=(x1,y1,x2,y2)`, `conf`, `center`, `width`, `height`, `area` |
| `perception/minimap.py` | `MinimapTracker` | template match of the player arrow + white-pixel centroid → tip angle, with circular mean smoothing over 5 frames | `position` (arrow centre in ROI pixels), `angle` in **degrees 0..360**, confidence |
| `perception/events.py` | `EventDetector` | OCR of combat-log and chat regions under 4 colour filters + keyword classification with dedup set | list of `{type, text, source}` dicts |
| `core/extractor.py` | `StateExtractor` | composes all of the above over one frame; flips capture FPS between `combat`/`idle` | its own `GameState` |
| `core/state.py` | `GameState` (dataclass) | **9 fields**, `hp_pct: int`, `position: Optional`, `target: Optional[str]` | — |
| `main.py` | — | monitor loop printing the state | ROIs, template and YOLO paths as module constants |
| `core/action.py` | `ActionController` | `pydirectinput` key/mouse actuation | **not perception** — see §5 |

### 2.1 What is genuinely valuable here

- **Bars** (`BarReader`) — direct, clean match to `hp_pct` / `mana_pct`.
- **Combat** (`CombatDetector`) — direct match to `in_combat`.
- **Target nameplate** (`TargetReader`) — template + OCR + a position lock
  that avoids re-OCR every frame. This is the only source of target identity.
- **Enemies** (`EnemyDetector`) — a *trained* detector. `docs/PERCEPTION.md`
  currently documents only template match / segmentation / OCR, so this is a
  method the roadmap never anticipated, and for mob detection it is strictly
  better than the documented alternatives.
- **Minimap facing** (`MinimapTracker`) — genuine heading estimate with
  proper circular-angle smoothing (sin/cos mean, not naive averaging). This
  is well-written code worth keeping.
- **Events** (`EventDetector`) — colour-filtered OCR with keyword
  classification and dedup; maps onto `GameState.events`.
- **Capture** (`ScreenCapture`) — `mss`, pooled buffers, mode-driven FPS.

### 2.2 What must not be ported

| Path | Why |
|---|---|
| `core/action.py` | `pydirectinput` is an OS-input library. AGENTS.md §7 forbids OS-input libraries outside `src/wow_bot/actuation/drivers/`. It is also perception-irrelevant. |
| `core/state.py` | Defines a conflicting `GameState`; the master schema is authoritative. |
| `datasets/` (974 files) | Training data, not source. |
| `runs/` (24 files, ~45 MB) | Training artifacts; `best.pt` is referenced by `EnemyDetector`'s default path. |
| `yolov8n.pt`, `yolov8s.pt`, `weights/yolo26n.pt` | ~57 MB of binaries. |
| `main.py` (as-is) | Module-level constants and an infinite loop; becomes config + a component instead. |
| `.gitignore` | Would replace the repo's own ignore rules. |

---

## 3. Gap analysis: `hamberger` output vs the canonical `GameState`

Master's `GameState` carries **20 fields** (nine pre-existing plus the eleven
added by T-FIX-03.6). Below is what `hamberger` can and cannot supply.

| `GameState` field | Type / range | `hamberger` source | Status |
|---|---|---|---|
| `timestamp` | `float` | `time.time()` | ⚠️ needs the monotonic clock `docs/PERCEPTION.md` §1 requires |
| `hp_pct` | `float [0,1]` | `BarReader.read_hp` | ✅ direct (already a ratio) |
| `mana_pct` | `float [0,1]` | `BarReader.read_mana` | ✅ direct |
| `position` | `(float, float)` world | minimap arrow centre | ❌ **screen pixels, not world** — see F-1 |
| `facing` | radians | `MinimapTracker` angle | ⚠️ **degrees 0..360** — conversion required |
| `in_combat` | `bool` | `CombatDetector` | ✅ direct |
| `target` | `TargetInfo` | `TargetReader` | ⚠️ `name` ✅, `hp_pct` needs /100, **`reaction` ✗**, **`distance_estimate` ✗** |
| `enemies` | `list[EnemyInfo]` | `EnemyDetector` | ⚠️ see §3.2 |
| `events` | `list[Event]` | `EventDetector` | ⚠️ dicts → `Event(type, timestamp, data)` |
| `player_z` | `float` | — | ❌ **no source** |
| `target_x` | `float` | — | ❌ **no source** |
| `target_y` | `float` | — | ❌ **no source** |
| `entities` | `tuple[EnemyInfo, ...]` | `EnemyDetector` | ⚠️ same as `enemies` |
| `perception_confidence` | `dict[str, float]` | YOLO `conf`, template match values | ⚠️ data exists but is not currently routed (T-FIX-22) |
| `inventory_count` | `int` | — | ❌ no channel (Bag frame, T-FIX-31) |
| `inventory_max` | `int` | — | ❌ no channel (Bag frame, T-FIX-31) |
| `level_or_xp` | `float` | — | ❌ no channel (XP bar, T-FIX-31) |
| `durability_fraction` | `float [0,1]` | — | ❌ no channel (Character frame, T-FIX-31) |
| `target_is_lootable` | `bool` | — | ❌ no channel (loot sparkle, T-FIX-31) |
| `incoming_casts` | `tuple[IncomingCast, ...]` | — | ❌ no channel (Enemy cast bar, T-FIX-31) |

### 3.1 Unit and convention mismatches (silent-bug class)

These are the traps — each would either crash or, worse, quietly corrupt
state:

1. **HP unit.** `hamberger` `TargetReader` returns an **`int` 0..100**;
   `TargetInfo.hp_pct` and `GameState.hp_pct` are **fractions in `[0,1]`**.
   `BarReader` already returns a ratio, so only the target reader needs
   normalizing. Mixing them would make `TargetInfo.__post_init__` raise for
   values ≤ 1 only, so a 0 would pass and a 55 would fail — an inconsistent,
   hard-to-read failure.
2. **Facing unit.** Master uses `math.cos(self._facing)` /
   `math.sin(self._facing)` (`mock_perception.py:322-323`) → **radians**.
   `MinimapTracker` returns **degrees 0..360**. Feeding degrees in as radians
   would make `target_x`/`target_y` projections nonsense at a plausible rate
   rather than failing loudly.
3. **Bounding-box convention.** `EnemyDetector` emits `bbox = (x1, y1, x2,
   y2)` (corner pair). `EnemyInfo.bbox` is documented as `x, y, w, h`
   (origin + size). Direct assignment silently produces wrong geometry.
4. **Position space.** See F-1.
5. **`target` type.** A bare `str` in `hamberger` vs `TargetInfo`; `reaction`
   must come from a colour heuristic (hostile/red, neutral/yellow,
   friendly/green) that `hamberger` does **not** implement, and
   `distance_estimate` has no source at all.
6. **`events` type.** `{"type","text","source"}` dicts vs
   `Event(type, timestamp, data)`; the timestamp must be attached and the
   free text moved into `data`.

### 3.2 `EnemyInfo` coverage (13 fields)

| Field | Source in `hamberger` |
|---|---|
| `bbox` | ⚠️ needs `(x, y, w, h)` conversion from `(x1, y1, x2, y2)` |
| `confidence` | ✅ YOLO `conf` |
| `distance_estimate` | ❌ **no source** (could be estimated from bbox area/height, but that is an unvalidated inference) |
| `entity_id` | ❌ **no stable id** — needs ADR-002 unresolved question 1 (stability policy) |
| `kind` | ⚠️ class name exists; must be mapped onto `VALID_NODE_KINDS` or world sync skips the entity |
| `x`, `y` | ⚠️ only **screen** centre available; no world coordinates |
| `z` | ❌ no source |
| `hp_fraction` | ❌ per-enemy HP is not read (only the *selected target's* HP is) |
| `threat` | ❌ no source (ADR-002 unresolved question 3) |
| `is_attackable` | ⚠️ inferable from class/whitelist membership |
| `is_alive` | ⚠️ inferable from a per-enemy HP channel, which does not exist; today only "detected ⇒ assumed alive" |
| `is_in_combat_with_self` | ⚠️ inferable from `in_combat` + detection, but ADR-002 unresolved question 5 says this is unverified |

---

## 4. The three hard findings

### F-1 — `hamberger` does not observe world position. This blocks 7 of 8 views.

`MinimapTracker` returns the **centre of the matched arrow template inside the
minimap ROI**, i.e. screen pixels in roughly `[0..206] × [0..217]` for
`MINIMAP_ROI = (1705, 33, 206, 217)`.

Two independent problems:

1. **Wrong space.** `GameState.position` is consumed as world coordinates by
   `world/sync.py`, which deduplicates player nodes at a
   `player_node_min_distance_units = 5.0` radius (`world/sync.py:70`,
   `:132-150`). Minimap pixels are not world units; feeding them in would
   rebuild the graph around a 200-unit-wide screen rectangle.
2. **Almost certainly constant.** In WoW the player's own arrow is pinned to
   the **centre of the minimap** — the map scrolls under it, not vice versa.
   A template match of that arrow therefore returns ≈ the same point every
   frame. It encodes *facing* (the arrow rotates), not *position*. This must
   be validated on a real frame sequence before the field is trusted; if it is
   constant, it is not a position source at all.

**Consequence, and this is the headline of the whole plan:** every one of the
eight consumer views except `TargetView` requires `player_x`/`player_y` (or
`self_x`/`self_y`) as **non-Optional**. Without a genuine world-pose channel:

> A naive port of `hamberger` yields **0 of 8 projectable views**.

World position is therefore **the** critical path item, ahead of everything
else in this document.

Viable sources, none of which `hamberger` contains:

| Option | What it needs | Assessment |
|---|---|---|
| A. Addon coordinate OCR | A new documented channel reading an on-screen coordinate frame (TomTom/MapCoords style) | Most honest; needs T-FIX-30 + a `docs/PERCEPTION.md` row |
| B. Dead reckoning from actuation | Integrating W/A/S/D input with a speed model; needs T-FIX-08 + T-FIX-20 | Not perception; drifts; needs periodic correction |
| C. Minimap scroll offset + map anchor | Calibrating the scrolling map texture against a known node | Requires map imagery and a calibration step; not present |
| D. Feed minimap pixels as if they were world | nothing | **Reject.** Silently corrupts the world model. |

### F-2 — Target channel is incomplete: no distance, no reaction

`TargetInfo` requires `name`, `hp_pct`, `reaction`, `distance_estimate`.
`TargetReader` supplies only `name` and an int HP percentage.

- **`distance_estimate` missing** → `_derive_target_in_range` has no input →
  `ReactiveView` and `CombatView` stay blocked. A possible estimator is
  nameplate/target-frame size or a per-whitelist-member reference, but that is
  an *inference* needing validation, not an observation.
- **`reaction` missing** → `TargetInfo.__post_init__` will **raise**
  (`reaction must be one of ['friendly', 'hostile', 'neutral']`) if left
  unset or empty. A colour heuristic on the nameplate border is the obvious
  source and is a new channel (T-FIX-30).
- **`threat` has no source anywhere** (ADR-002 unresolved question 3), so
  `TargetView` stays blocked regardless. ADR-002 already notes the escape
  hatch is to change `TargetConfig.priority` so threat is not the primary sort
  key — that is a `combat/targeting.py` change needing its own exception.

### F-3 — The six channel-pending fields have no implementation

`docs/PERCEPTION.md` §2.1 now documents Bag frame, XP bar, Character frame,
Enemy cast bar, and Lootable-corpse indicator. `hamberger` implements
**none** of them. Until they exist:

- `StrategistView` and `CombatView` stay blocked (`inventory_count`,
  `level_or_xp`, plus derived `resource_max`)
- `LootView` stays blocked (`target_is_lootable`)
- `VendorView` stays blocked (`inventory_count`)
- `ReactiveView`'s `incoming_casts` defaults to an honest empty tuple (OK)

---

## 5. Net projection outcome if ported as-is

| View | Projectable after a naive port? | Blocking field |
|---|---|---|
| `WorldSyncView` | ❌ | `player_x`, `player_y`, `player_z` |
| `StrategistView` | ❌ | `player_x/y/z`, `inventory_count`, `level_or_xp`, `resource_max` |
| `CombatView` | ❌ | `self_x`, `self_y`, `resource_max`, `target_in_range` |
| `TargetView` | ❌ | `threat`, `distance_estimate` |
| `ReactiveView` | ❌ | `self_x`, `self_y`, `target_in_range` |
| `FleeView` | ❌ | `self_x`, `self_y`, `adds_count` |
| `LootView` | ❌ | `self_x`, `self_y`, `target_is_lootable`, `inventory_count` |
| `VendorView` | ❌ | `self_x`, `self_y`, `inventory_count` |

**0 of 8.** This is the single most important number in this document: the
`hamberger` code is good, well-factored perception code, but *it observes a
different, smaller world than the master architecture requires*. Bringing the
files in is the easy 10%; supplying the missing pose and channel fields is
the work.

With **T-FIX-30** (pose + target distance + reaction) landed, the table
becomes 6 of 8 (`WorldSyncView` needs `player_z`, and `StrategistView` /
`CombatView` still need `resource_max`). With **T-FIX-31** (the five UI
channels) it reaches 8 of 8 in principle.

---

## 6. Compliance audit

### 6.1 Governance: two documented rules must be resolved first

| Rule | Source | Status |
|---|---|---|
| "**MockPerception MUST remain the only producer of `GameState` in this roadmap.**" | `LAB_PHASE_ROADMAP.md` Global Rule 1 | **Conflicts** with making a real backend live |
| LAB_MODE perception is "`RealPerception`" | `docs/LAB_CONSTRAINTS.md` §2 table | Permits it |
| LAB_MODE perception is "still `MockPerception` (per current roadmap)" | `AGENTS.md` §3.2 | Permits it now, contradicts `LAB_CONSTRAINTS` |

AGENTS.md §2 precedence is **1. LAB_CONSTRAINTS, 2. LAB_PHASE_ROADMAP,
3. AGENTS.md**. So `LAB_CONSTRAINTS` already names `RealPerception` for
LAB_MODE and outranks the others — but `LAB_PHASE_ROADMAP` Global Rule 1
(precedence 2) still freezes `MockPerception` as the sole producer.

**Resolution required from the operator before Stage 4:** either amend
`LAB_PHASE_ROADMAP.md` Global Rule 1 (explicit, high-precedence doc change)
or keep the real backend *dormant* — present, unit-tested, but never the
configured producer — until Phase 13 formally starts. The plan below is
written so **Stages 0–3 need no such change**, because they add modules and
tests without wiring the backend as a producer.

AGENTS.md §5.5 applies: if the chosen path requires bypassing a frozen
document, stop and ask rather than diverging silently.

### 6.2 AGENTS.md §7 forbidden actions check

| Item | Verdict |
|---|---|
| `mss` screen capture | ✅ allowed — AGENTS §7 requires capture to live **inside** `src/wow_bot/perception/` |
| `core/action.py` / `pydirectinput` | ❌ **must not be ported** — OS-input library outside `actuation/drivers/` |
| Reading game process memory / DLL injection | ✅ absent from `hamberger` (it is pure CV) |
| Cloud LLM | ✅ absent |
| Hardcoded secrets | ✅ none found, but **hardcoded absolute paths** present (see 6.3) |

### 6.3 Configuration and hardcoding

`AGENTS.md` §6.5 requires every config key in `config/lab.example.toml` with
a comment. Currently hardcoded in `hamberger`:

| Value | Location | Becomes |
|---|---|---|
| `C:\Program Files\Tesseract-OCR\tesseract.exe` | `perception/target.py`, `perception/events.py` (+ `bars.py`) | `perception.tesseract_cmd` |
| `HP_ROI (514,771,123,18)`, `MANA_ROI (514,793,123,17)` | `main.py` | `perception.hp_roi`, `perception.mana_roi` |
| `MINIMAP_ROI (1705,33,206,217)` | `main.py` | `perception.minimap_roi` |
| `models/target_name_template.png`, `models/arrow_template.png` | `main.py` | `perception.*_template` |
| `runs/detect/train-2/weights/best.pt` | `enemies.py` default | `perception.yolo_weights` |
| `combat_region`, `chat_region` tuples | `events.py` | config |
| `KNOWN_ENEMIES` whitelist | `target.py` | `config/perception_enemies.toml` or config list |
| `edge_size=150`, `red_ratio_thresh=0.02`, `cooldown=1.0` | `combat.py` | config |
| ROI/threshold constants inside `_find_target` (0.35/0.80/0.50/0.85) | `target.py` | config |

### 6.4 Dependencies (none declared today)

`pyproject.toml` currently has **no** `opencv-python`, `pytesseract`,
`ultralytics`, `mss`, or `pydirectinput`. Adding them:

- `mss` — required, lightweight, documented in `docs/PERCEPTION.md` §1.
- `opencv-python` — required; note it must be the headless build
  (`opencv-python-headless`) since no GUI is used, and it is large.
- `pytesseract` — Python wrapper; the **Tesseract binary is a separate
  system install**. CI must not depend on it (MOCK_MODE tests must not need
  a live OCR install).
- `ultralytics` — the riskiest addition. It pulls `torch` and is known to
  perform online asset/model fetching and version checks at runtime. It must
  be pinned and configured to run fully offline, otherwise a
  `docs/LAB_CONSTRAINTS.md` "no network" reading could be violated
  *implicitly*. Treat this as a reviewable decision, not a routine
  dependency bump.

### 6.5 Assets: do not commit the weights

`yolov8n.pt` (6.5 MB), `yolov8s.pt` (22.6 MB), `weights/yolo26n.pt`
(5.5 MB), `runs/detect/train-2/weights/{best,last}.pt` (22.5 MB each) —
~57 MB. Plus 974 dataset images and 24 training artifacts.

Options: (a) Git LFS, (b) an ignored `models/` directory populated by the
operator, (c) a config key pointing at an absolute path outside the repo.
Recommended: **(b)+(c)** — gitignore `*.pt` and resolve the weight path from
config, failing closed with a clear error if it is absent.

The three small PNG templates (`models/*.png`, collectively a few KB) **should**
be committed: they are required for template matching and are not regenerable
without the original client resolution.

### 6.6 Performance: the current loop cannot run at reflex rate

`EventDetector.detect` runs **6 Tesseract calls per frame**
(`_read_combat_multi` ×2 + `_read_region_multi` ×4), and `_extract_name` in
`TargetReader` runs **3 more**. At `combat_fps=30` that is ~270 OCR calls per
second. The fast loop is 10–20 Hz, and `docs/PERCEPTION.md` §1 specifies 20 Hz
capture.

The port therefore needs an explicit budgeting strategy — OCR throttled to
N Hz per channel with caching between samples, YOLO inference sampled rather
than per-frame, and only cheap channels (bars, combat edges) run every frame.
This is a design requirement for T-FIX-27, not an optimisation for later: without
it the backend will simply not meet its own frame budget.

---

## 7. The plan

Five new task IDs, appended to the roadmap alongside the existing T-FIX
tasks. Stages 0–2 can start today with **no prerequisite roadmap work**.

### Stage 0 — Foundation (start immediately)

#### T-FIX-29 — Dependencies, assets, and perception configuration

**Depends on:** none
**Status:** PENDING (proposed)
**Deliverables:**
- `pyproject.toml` — `mss`, `opencv-python-headless`, `pytesseract`; the
  YOLO stack (`ultralytics`/`torch`) behind an optional extra, not a base
  dependency.
- `config/lab.example.toml` — every key listed in §6.3, each with a comment.
- `.gitignore` — `*.pt`, `models/weights/`.
- `models/*.png` — the three template PNGs, committed.
- Guarded imports so MOCK_MODE needs neither Tesseract nor a YOLO weights
  file to import or test the package.

**Contract:** missing optional dependencies degrade to an explicit
`ImportError`/`ConfigError` with a named remedy, never a silent pass; the
scanner configuration is injected, never module-level; `tesseract_cmd` is
set from config at construction time, not at import time.

**Acceptance:**
- [ ] `uv run pytest` in a clean MOCK_MODE environment passes with neither
      Tesseract nor YOLO installed.
- [ ] Every new key is present in `config/lab.example.toml` with a comment,
      and an unknown key still raises `ConfigError`.
- [ ] `git status` shows no `.pt` file staged.
- [ ] `ruff check` and `mypy src` stay clean.

**Out of scope:** implementing any reader; wiring a backend.

---

### Stage 1 — Port the readers

#### T-FIX-27 — Port capture and readers into `src/wow_bot/perception/`

**Depends on:** T-FIX-29
**Status:** PENDING (proposed)
**Deliverables** (source copied from `hamberger` with `git show`, then
reshaped):
- `src/wow_bot/perception/capture.py` ← `capture/screen_capture.py`
- `src/wow_bot/perception/bars.py` ← `perception/bars.py`
- `src/wow_bot/perception/combat.py` ← `perception/combat.py`
- `src/wow_bot/perception/target.py` ← `perception/target.py`
- `src/wow_bot/perception/enemies.py` ← `perception/enemies.py`
- `src/wow_bot/perception/minimap.py` ← `perception/minimap.py`
- `src/wow_bot/perception/events.py` ← `perception/events.py`
- Tests synthesising frames with NumPy (bars as coloured rectangles, combat
  as a red edge strip, templates stamped onto a canvas).

**Not ported:** `core/action.py`, `core/state.py`, `main.py`,
`.gitignore`, `datasets/`, `runs/`, `*.pt`.

**Contract:**
- Every reader is **frame-in**; no reader reaches for a global capture
  singleton. Composition happens in one place (T-FIX-28).
- All values from §6.3 come from injected config. No module-level
  `pytesseract.pytesseract.tesseract_cmd` assignment.
- **Frame budget:** cheap channels (bars, combat edges) run every frame;
  OCR and YOLO run on their own throttled schedule with cached results
  between samples (§6.6).
- Deterministic given identical frames — no `time.time()` inside readers;
  the combat cooldown latch takes the clock as a parameter so tests can
  inject it.
- YOLO import is lazy and optional.

**Acceptance:**
- [ ] Each reader has unit tests over synthesised frames; no test needs a
      live client, Tesseract, or YOLO weights.
- [ ] `BarReader` returns a ratio in `[0,1]` for a synthetic bar of known
      fill.
- [ ] `CombatDetector` returns `True` for a red edge strip and `False` for a
      neutral one.
- [ ] `MinimapTracker` converts degrees → radians **or** returns degrees
      with the conversion explicitly deferred to T-FIX-28 (one location only).
- [ ] OCR call count per second is asserted to stay under the configured
      budget.
- [ ] No hardcoded absolute path remains in `src/wow_bot/perception/`.

**Out of scope:** building a `GameState`; wiring the lab runner; any
actuation.

---

### Stage 2 — Assemble the canonical state

#### T-FIX-28 — Real state builder: readers → canonical `GameState`

**Depends on:** T-FIX-27
**Status:** PENDING (proposed)
**Deliverables:**
- `src/wow_bot/perception/builder.py` — composes one frame into the
  canonical `wow_bot.shared.interfaces.GameState`.
- Tests covering every unit/convention conversion in §3.1.

**Contract — every conversion is explicit and tested:**

| Conversion | Rule |
|---|---|
| target HP | `int 0..100` → `fraction [0,1]` by `/100.0` |
| facing | degrees → radians, normalised to `[-π, π)`; applied in **one** place |
| bbox | `(x1,y1,x2,y2)` → `(x, y, w, h)` |
| events | dict → `Event(type, timestamp, data)` with the frame's monotonic timestamp |
| `TargetInfo` | build only when `name`, `hp_pct`, `reaction`, `distance_estimate` are all present; otherwise `target = None` (an honest absence) — never a partial `TargetInfo`, because `__post_init__` would raise mid-pipeline |
| `entities` / `enemies` | one list, both populated from the same detections, per ADR-002 Decision 3 |
| `kind` | mapped to `VALID_NODE_KINDS`, or the entity is omitted rather than defaulted |
| `perception_confidence` | populated from YOLO `conf`, template-match scores, and OCR confidence |
| `timestamp` | the monotonic clock, per `docs/PERCEPTION.md` §1 |
| unobserved fields | left `None` / empty; never fabricated |

**Acceptance:**
- [ ] The emitted `GameState` passes `__post_init__` and a
      `to_dict`/`from_dict` round trip.
- [ ] One test per row of the conversion table.
- [ ] A frame set with no target yields `target=None` rather than raising.
- [ ] `perception_confidence` has an entry for every field the frame set
      actually observed.
- [ ] The builder imports nothing from `wow_bot.lab` or `wow_bot.main`.

**Out of scope:** supplying `player_z`, `position`, `distance_estimate`,
`reaction` — those are T-FIX-30 channels.

---

### Stage 3 — Close the observation gaps (the actual critical path)

#### T-FIX-30 — World pose, target distance, and reaction channels

**Depends on:** T-FIX-28, T-FIX-23
**Status:** PENDING (proposed)
**Why this is the critical path:** without `player_x`/`player_y`, seven of
the eight views stay blocked regardless of how good the readers are (F-1).

**Deliverables:**
- `docs/PERCEPTION.md` §2.1 — a new **World pose** channel row, a **Target
  distance** row, and a **Target reaction** row, each with source channel,
  extraction method, accuracy class, and confidence semantics.
- The corresponding reader(s) under `src/wow_bot/perception/`.
- `player_z` — if the chosen method cannot observe height, `None` is
  documented as the permanent LAB_MODE value and `WorldSyncView` /
  `StrategistView` stay blocked **by design** until a separate decision
  changes their contract. Do not invent a z.

**Decision needed before implementation** (pick one and record it in the row):

| Candidate | Source | Notes |
|---|---|---|
| A | On-screen coordinate frame from an addon | Honest, per-frame, needs OCR of a small stable region; recommended |
| B | Dead reckoning from actuation | Needs T-FIX-08/T-FIX-20; drifts; never self-corrects |
| C | Minimap scroll offset against a map anchor | Highest accuracy, highest effort; requires map imagery |

**Contract:** the chosen method states its units and error characteristics;
the adapter and `world/sync` see world units only; a low-confidence pose
leaves `position=None` rather than a plausible guess, because a wrong pose
writes a wrong node into the world model.

**Acceptance:**
- [ ] Each of the three new channels has a `docs/PERCEPTION.md` row with all
      four required attributes.
- [ ] A test proves `WorldSyncView` projects once pose and `player_z` are
      supplied.
- [ ] A test proves `ReactiveView` projects once `distance_estimate` exists
      (so `target_in_range` derives).
- [ ] Confidence below threshold leaves `position=None`, and no world node
      is created.

**Out of scope:** the six UI-panel channels (T-FIX-31); navigation.

---

#### T-FIX-31 — UI panel channels (implements the T-FIX-23 documentation)

**Depends on:** T-FIX-27, T-FIX-23
**Status:** PENDING (proposed)
**Deliverables:** readers for Bag frame, XP bar, Character frame, Enemy cast
bar, and Lootable-corpse indicator, exactly as `docs/PERCEPTION.md` §2.1
already specifies, plus their calibration anchors.

**Contract:** each reader obeys the confidence rule already written for it —
below threshold the field is `None`, never a partial or guessed value
(`inventory_count` in particular: a partial count silently suppresses
full-bag handling).

**Acceptance:**
- [ ] Each reader has a synthesised-frame unit test.
- [ ] Below-threshold behaviour matches the §2.1 rule for that channel.
- [ ] The §6 precision/recall gate is either measured against a corpus or
      explicitly recorded as still unmeasurable — not faked.

**Out of scope:** pose/distance (T-FIX-30); `resource_max` provenance
(ADR-002 unresolved question 2).

---

### Stage 4 — Assemble the backend (gated)

#### T-FIX-32 — `RealPerceptionBackend` behind a config gate

**Depends on:** T-FIX-28, T-FIX-04, T-FIX-20, T-FIX-22
**Status:** PENDING (proposed)
**Deliverables:**
- `src/wow_bot/perception/real_backend.py` — `RealPerceptionBackend(
  PerceptionBackend)` composing capture → readers → builder, exposing
  `async def snapshot() -> GameState`.
- An integration test running the whole path over a frozen frame corpus.

**Contract:**
- It is **never the configured default producer.** `MockPerception` stays the
  producer in `MOCK_MODE` and in `LAB_MODE`, so `LAB_PHASE_ROADMAP.md`
  Global Rule 1 is not violated (see §6.1).
- Frame staleness is explicit; a frame that cannot be captured yields an
  honest error rather than a silently reused stale state.
- The fast loop is not blocked — the T-FIX-20 port owns scheduling.

**Acceptance:**
- [ ] `RealPerceptionBackend` satisfies `isinstance(..., PerceptionBackend)`.
- [ ] An integration test feeds recorded/synthetic frames end to end and
      asserts the eight projection outcomes.
- [ ] `MockPerception` is still the producer in both modes; no default config
      selects the real backend.
- [ ] Full suite green; `ruff` and `mypy` clean.

**Out of scope:** enabling it — that is Phase 13, and requires amending
`LAB_PHASE_ROADMAP.md` Global Rule 1 first (§6.1).

---

## 8. Answer: are all pre-real-perception tasks needed first?

**No.** Roughly half the roadmap is unrelated to perception, and only two
pending tasks are true prerequisites for *landing* this port.

### 8.1 Classification

| Group | Tasks | Needed to port `hamberger`? | Needed to *use* it live? |
|---|---|---|---|
| **Already done** | T-FIX-01, 02, 03, 03.5, 03.6, 21, 23 | ✅ already satisfied | ✅ |
| **Perception contract** | T-FIX-04, 20, 22, 24 | **Yes — required** before the backend is wired | ✅ |
| **Port-specific (new)** | T-FIX-27, 28, 29, 30, 31, 32 | ✅ this *is* the port | ✅ |
| **Tier 2 — Execution** | T-FIX-05, 06, 07, 08 | ❌ **No** | ✅ **Yes** — before any live actuation |
| **Tier 3 — Science** | T-FIX-09, 10 | ❌ No | ❌ No (independent research gate) |
| **Tier 4 — Telemetry** | T-FIX-11, 12, 13, 14 | ❌ No | ⚠️ Before trusting a live run's numbers |
| **Tier 5 — World/Nav** | T-FIX-15, 16, 17 | ❌ No | ⚠️ Needed to *act* on the pose, not to read it |
| **Tier 6 — Aggregate** | T-FIX-18, 19 | ❌ No | ❌ No (Phase 12 reporting) |
| **Tier 7 — Deferred** | T-FIX-25, 26 | ❌ No | ⚠️ Before a coherent farm loop |

### 8.2 Critical path

```
T-FIX-29 (deps/config) ─┬─> T-FIX-27 (readers) ─> T-FIX-28 (builder) ─┬─> T-FIX-32 (backend, gated)
                        │                                              │
                        ├─> T-FIX-31 (UI panels) ──────────────────────┤
                        │                                              │
                        └─> T-FIX-30 (pose/distance/reaction) ─────────┘  <- THE BLOCKER
                                                                             needs T-FIX-04, 20, 22
```

**Prerequisites for simply landing and testing the port: none.** T-FIX-29 and
T-FIX-27 have no roadmap dependency and touch no frozen module (config keys
go in `config/lab.example.toml`, which tasks are explicitly allowed to edit).

**Prerequisites for the backend to drive the pipeline:** T-FIX-04,
T-FIX-20, T-FIX-22 (all existing) **plus T-FIX-30**.

**Prerequisites for live play:** all of Tier 2 (T-FIX-05…08), because they
govern FSM progression, reflex/watchdog lifecycle, safety arming, and real
input. Tier 2 has **no dependency on perception**, so it can proceed in
parallel.

### 8.3 What is genuinely not needed for this goal

- **T-FIX-09 / T-FIX-10** (spectral acceptance) — an internal-dynamics
  science question, unrelated to whether pixels become state.
- **T-FIX-18 / T-FIX-19** (aggregate reporting) — Phase 12 reporting only.
- **T-FIX-11 / T-FIX-12 / T-FIX-13** (telemetry truth) — matter before you
  *believe* a live run, not before you capture one.
- **T-FIX-14** (cross-platform resource metrics) — soak metrics only.

### 8.4 Summary answer

> You do **not** need the whole pre-real-perception roadmap before bringing
> `hamberger` in. **Stages 0–2 (T-FIX-29, 27, 28) start today with zero
> prerequisites.** The genuine blockers are **T-FIX-30 (world pose)** — which
> alone gates 7 of 8 views — plus **T-FIX-04/20/22** for wiring. Tiers 3, 4,
> and 6 are unrelated to perception, and Tier 2 is needed only for live
> actuation, where it can run in parallel.

---

## 9. Extraction procedure

The histories are unrelated, so copy files rather than merging. Prefer a
temporary worktree, which is byte-exact and avoids two traps: PowerShell's
`>` redirection defaults to UTF-16 in Windows PowerShell 5.1 (silently
corrupting text), and a naive `git show ... > file` can mangle non-ASCII
comments — the source contains Persian comments throughout.

```powershell
# 1. Materialise the orphan tree read-only
git worktree add --detach "$env:TEMP\hb" hamberger

# 2. Copy only the source we want (byte-exact)
New-Item -ItemType Directory -Force models | Out-Null
Copy-Item "$env:TEMP\hb\capture\screen_capture.py" src\wow_bot\perception\capture.py
Copy-Item "$env:TEMP\hb\perception\bars.py"         src\wow_bot\perception\bars.py
Copy-Item "$env:TEMP\hb\perception\combat.py"       src\wow_bot\perception\combat.py
Copy-Item "$env:TEMP\hb\perception\target.py"       src\wow_bot\perception\target.py
Copy-Item "$env:TEMP\hb\perception\enemies.py"      src\wow_bot\perception\enemies.py
Copy-Item "$env:TEMP\hb\perception\minimap.py"      src\wow_bot\perception\minimap.py
Copy-Item "$env:TEMP\hb\perception\events.py"       src\wow_bot\perception\events.py
Copy-Item "$env:TEMP\hb\models\*.png"               models\

# 3. Do NOT copy: core/, main.py, .gitignore, datasets/, runs/, *.pt

# 4. Clean up
git worktree remove --force "$env:TEMP\hb"
```

**Provenance.** Copying files out of an orphan commit discards their git
history and authorship. Record it explicitly in the first port commit body,
e.g. `Source: hamberger@9b968a4 (author Amirm227), ported and reshaped`.

---

## 10. Risks and open questions

| # | Risk / question | Handling |
|---|---|---|
| R-1 | **World pose has no source.** Gates 7 of 8 views. | T-FIX-30; decide between options A/B/C in §4 before writing code |
| R-2 | `player_z` may be unobservable by any screen method | Document `None` as the permanent value and accept the blocked views, or change the `WorldSyncView` contract under a separate ADR |
| R-3 | `threat` has no source (ADR-002 unresolved q3) | `TargetView` stays blocked; consider changing `TargetConfig.priority` so threat is not the primary sort key |
| R-4 | `entity_id` stability policy unknown (ADR-002 unresolved q1) | Needs an experiment on a recorded frame sequence |
| R-5 | `resource_max` needs a class channel that does not exist (unresolved q2) | Table stays empty; `StrategistView`/`CombatView` blocked |
| R-6 | `ultralytics` may perform network I/O | Pin it, disable auto-install/asset fetch, verify offline; otherwise reject the dependency and swap in a local-only detector |
| R-7 | No recorded frame corpus exists | Unit tests use synthesised frames; §6 accuracy gates stay explicitly unmeasurable until a corpus is committed |
| R-8 | Tesseract is a separate system binary | Guard the import; MOCK_MODE must not require it (T-FIX-29 acceptance) |
| R-9 | YOLO weights are not in the repo | Config path + fail-closed error; gitignore `*.pt` |
| R-10 | ROIs are hardcoded for **one** screen resolution | All ROIs become config; document the resolution they were calibrated at |
| R-11 | `KNOWN_ENEMIES` holds only 2 entries | Name matching generalises poorly; either expand the list or treat the name as advisory and rely on YOLO classes |
| R-12 | OCR/YOLO cannot sustain 20 Hz as written (§6.6) | Per-channel sampling budgets are a T-FIX-27 contract item |
| R-13 | Governance conflict between `LAB_CONSTRAINTS` and `LAB_PHASE_ROADMAP` Global Rule 1 | Operator decision before Stage 4; §6.1 |

---

## 11. Recommended immediate next steps

1. **Accept or amend the six new task IDs** (T-FIX-27 … T-FIX-32) into
   `docs/lab_phase/PRE_REAL_PERCEPTION_FIX_ROADMAP.md`.
2. **Start T-FIX-29** (deps/config/gitignore) — no prerequisites, no frozen
   module, unblocks everything else.
3. **Start T-FIX-27** (port readers) immediately after — this is where the
   `hamberger` code actually lands.
4. **Decide the world-pose source** (§4 options A/B/C). This is the one
   decision that determines whether the port can drive the pipeline at all,
   and it cannot be answered from the repository — it needs a look at a real
   client.
5. Keep **Tier 2 (T-FIX-05…08)** running in parallel: it is independent of
   perception and is the other half of "ready to go live".

---

## 12. Document history

Created as the investigation record for `hamberger@9b968a4` and the plan to
port it. Analysis performed against `master` at `2f48240 T-FIX-21`.
Documentation only: no source file, config key, test, or artifact was
modified by this document.








