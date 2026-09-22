# ADR-002: GameState Extension for Real Perception

## Status

**Proposed.** This ADR is design-only. It must not be implemented until the
STRATEGY A exception proposed in the "STRATEGY A exception scope" subsection
below is recorded verbatim in AGENTS.md section 5 by a separate, explicitly
scoped follow-up task.

Pass 1 left three confirmations open. Pass 2 (this revision) resolves item 1 in
full via the "Field provenance" subsection below: all seven fields that had no
vision component are now classified, and two of them turned out to be derivable
from fields already on `GameState`, so they left the schema. Items 2 and 3
remain open and are listed under "Unresolved questions".

## Context

### What T-FIX-03 established

T-FIX-03 delivered a perception contract layer that is deliberately
incomplete: `PerceptionBackend` (`src/wow_bot/perception/protocol.py:10-16`),
eight frozen view dataclasses (`src/wow_bot/perception/views.py`), and
`GameStateAdapter` (`src/wow_bot/perception/adapter.py`). The adapter's
contract is fail-loud: when a view declares a field non-Optional and the
snapshot cannot supply it, the adapter raises `AdapterIncompleteError`
rather than fabricating a plausible value.

Pass 3 verified this against a strictly canonical `GameState` (built only
from the nine fields in `src/wow_bot/shared/interfaces.py:147-155`) and
proved all eight projections raise. ADR-001 records the table in its
subsection "The adapter cannot produce a successful projection today".

### The pass-3 table (verified)

Fixture: canonical `GameState` carrying a `TargetInfo` target, the shape
`MockPerception` emits during combat.

| View name | Succeeds | Raises AdapterIncompleteError | First missing field |
|---|---|---|---|
| WorldSyncView | No | Yes | `player_z` |
| StrategistView | No | Yes | `resource_max` |
| CombatView | No | Yes | `resource_max` |
| TargetView (`to_targeting_views`) | No | Yes | `threat` |
| ReactiveView | No | Yes | `target_in_range` |
| FleeView | No | Yes | `adds_count` |
| LootView | No | Yes | `target_is_alive` |
| VendorView | No | Yes | `inventory_count` |

Each row is produced by an explicit raise in the adapter, for example
`adapter.py:87` (`player_z`), `:137` (`resource_max`), `:252` (`threat`),
`:302` (`target_in_range`), `:333` (`adds_count`), `:359`
(`target_is_alive`), and `:402` (`inventory_count`).

### Why T-FIX-04 is blocked

T-FIX-04 (reference `MockPerceptionAdapter`, per
`docs/lab_phase/PRE_REAL_PERCEPTION_FIX_ROADMAP.md` Tier 1) exists to test
the contract by projecting mock snapshots through all eight views. Pass 3
left it meaningless: `test_canonical_game_state_end_to_end_projection`
(`tests/test_perception_adapter.py:674`) is marked `xfail(strict=True)`, so
any implementation that makes a canonical projection succeed fails the
suite until the marker is removed. There is nothing for a mock adapter to
project successfully today.

### The eight consumers and their protocols

The eight structural views the lab pipeline requires are listed in
`docs/reviews/PRE_PHASE_13_REVIEW.md` lines 94-103, with definitions at
`world/sync.py:43`, `strategist/prompts_v2.py:30`, `combat/loop.py:20`,
`combat/targeting.py:14`, `combat/reactive.py:31`, `combat/flee.py:31`,
`farm/loot.py:38`, and `farm/vendor.py:45`.

Two structural facts dominate this ADR:

- The lab HP/resource views are **percentages in [0,100]**, while canonical
  `GameState` carries fractions in `[0,1]` (`shared/interfaces.py:148-149`).
  ADR-001's adapter already converts with banker's rounding.
- `fsm_state` is required non-Optional by `GameStateView`
  (`prompts_v2.py:46`), but it is the Python controller's own state, not an
  observation. Review Q6 states plainly that "vision cannot observe the
  Python controller's actual FSM state"
  (`docs/reviews/PRE_PHASE_13_REVIEW.md:140`). The adapter already treats it
  as an injected parameter, not a snapshot field (`adapter.py:113`).

## Decision

### Decision 1: Where do the missing fields go?

**We choose Option A: extend `GameState` in place with additive `Optional`
fields carrying `None` defaults.** The placement of individual fields then
follows a three-way classification (observed / derived / internal), which is
the useful content of Option C, implemented without a second type.

Four concrete reasons:

1. `docs/PERCEPTION.md:7-8` fixes the output contract for any real backend:
   "The output MUST conform to the existing GameState schema used by
   MockPerception. No downstream layer may branch on producer identity."
   A separate `PerceptionSnapshot` type (Option B) would contradict this and
   require rewriting that document.
2. `shared/interfaces.py:4-6` documents its own evolution policy: the
   dataclasses "must not be renamed or have fields removed; new fields are
   added as optional with safe defaults." Addition is the sanctioned change
   path; a rename or a parallel type would breach it.
3. The blast radius is small and already measured. In all of `src/`,
   `GameState(...)` is constructed in exactly one place
   (`mocks/mock_perception.py:313`). Positional construction appears in
   exactly one test (`tests/unit/test_review_gate.py:51`). The Phase 12
   artifacts serialize no `GameState` at all: `soak_report.json` carries
   CPU/RSS/log counters and per-sample deltas only,
   `runs/lab/aggregate-1h/aggregate_v1.json` carries perception-agnostic
   counters and non-claims, and `runs/lab/soak-1h/events.jsonl` carries
   event names with small payloads. So additive fields appended at the end
   leave every existing consumer and artifact untouched.
4. Option A is the choice that keeps the forcing function honest. The
   `xfail` at `tests/test_perception_adapter.py:660` flips to XPASS exactly
   when a mock producer populates the new fields and the adapter supplies
   the derived ones, so T-FIX-04 becomes self-verifying.

**Rejected: Option B** (a new `PerceptionSnapshot` type alongside
`GameState`). It would require changing `PerceptionBackend.snapshot()`'s
return type (`protocol.py:14`), `perception_loop`'s queue typing
(`main.py:208`), and the `RuntimeComponents.perception` annotation
(`main.py:100`) — three frozen signatures — while leaving `GameState`
permanently impoverished and forcing `MockPerception` to emit a type the
real path never uses. It also creates exactly the producer-identity
branching `docs/PERCEPTION.md:8` forbids.

**Rejected: Option C as a second channel.** The split rule is correct and is
adopted here, but implementing it as a separate object carried alongside
`GameState` doubles the queue plumbing without changing the content of any
single field. Classifying each field (observed / derived / internal) and
routing derived and internal quantities through the adapter achieves the
same separation with no new type.

### Decision 2: Field inventory

See the "Field inventory" section below. Every field is classified as
**observed** (a real backend could see it), **derived** (computable from
observations plus game rules or config), or **internal** (Python/controller
state). Fields classified derived or internal are deliberately **not** added
to `GameState`; the adapter derives or injects them.

Pass 2 resolved the seven fields that pass 1 had classified observed without
any backing vision component. Two of the seven are **derived from fields
already on `GameState`** and therefore left the schema entirely. The result:
the additive list shrinks from thirteen to **eleven**, of which **five are
genuinely observed** and **six are channel-pending observed fields** — fields
a real backend cannot produce until their vision channels are added to
`docs/PERCEPTION.md`. See "Field provenance" for the per-field rules.

### Decision 3: EnemyInfo vs EntityLike vs TargetEntityLike

**We choose to extend `EnemyInfo` in place into a superset type, keeping the
name `EnemyInfo`, so that one entity list serves both views.**

Today the situation is worse than a shape mismatch: `enemies` has no
production consumer at all. The only `src/` reference to `self.enemies`
outside the type itself is the serializer (`shared/interfaces.py:174`). The
world sync layer instead reads a `state.entities` attribute
(`world/sync.py:162-189`) that canonical `GameState` does not possess, so
the adapter's `to_world_sync_view` falls back to an empty tuple. The
combat consumers read `to_targeting_views()`, which projects from `target`.

Three reasons for the superset:

1. Both consumers observe the same physical entities. Two parallel lists
   (an `enemies` list for combat and an `entities` list for world sync)
   would drift: an entity present in one and absent from the other would
   silently change combat targeting without changing the world model, or
   vice versa. T-FIX-03 existed precisely to eliminate divergent structural
   views; duplicating the entity channel would reintroduce that failure.
2. `EnemyInfo` currently has no `entity_id` (`shared/interfaces.py:78-95`),
   so it can never satisfy `EntityLike` (`world/sync.py:23-39`) regardless
   of extension. `entity_id` must be added under some assignment policy;
   adding it once, on the single canonical entity type, is strictly cheaper
   than adding it twice.
3. Extending in place respects the module's own evolution policy
   (`shared/interfaces.py:4-6`): no rename, no removal, only addition.

**Exact field set for the extended `EnemyInfo`.** Existing fields stay:
`bbox`, `confidence`, `distance_estimate`. New fields, all `Optional` with
`None` defaults: `entity_id: str | None`, `kind: str | None`, `x: float |
None`, `y: float | None`, `z: float | None`, `hp_fraction: float | None`,
`threat: float | None`, `is_attackable: bool | None`, `is_alive: bool |
None`, `is_in_combat_with_self: bool | None`.

**bbox and confidence stay**, but as detection provenance, not world state.
`bbox` is screen-space; `x`/`y`/`z` are world-space. Neither view consumes
`bbox` or `confidence`, and neither should: keeping them is for the
`perception_confidence` map and offline debugging, and they are the fields
`docs/PERCEPTION.md:32` refers to when it says all vision outputs carry
confidence. `kind` must be one of `VALID_NODE_KINDS`
(`world/store.py:20-27`) or world sync will skip the entity
(`world/sync.py:163-166`).

**entity_id assignment** is assigned by perception, not read from the game,
and must be **stable across frames for the same mob**, because
`WorldSync.mark_seen` (`world/sync.py:181-189`) keys world-model entities on
it. A per-frame random id would flood the world model. The exact stability
scheme (nameplate-text hash plus spawn position, or a tracked id table) is
not determined by the repository and is listed under Unresolved questions.

**What breaks during transition.** `EnemyInfo` construction sites
(`mock_perception.py:458`, and test fixtures at `tests/unit/test_interfaces.py:39-40`)
plus `GameState.to_dict`/`from_dict` must serialize the new fields.
Tests that read `state.enemies` (`tests/unit/test_mock_perception.py:165-173`,
`tests/unit/test_mock_scenarios.py:143-303`) keep passing because they only
read `bbox` and `distance_estimate`, which are unchanged. Positional
`EnemyInfo` construction must be avoided in new code since all new fields
must come after the existing three.

### Decision 4: Unknown-value policy at the schema level

**We choose `Optional[X]` fields with `None` meaning "not observed this
frame", plus the parallel confidence map that `docs/PERCEPTION.md` already
specifies.**

`docs/PERCEPTION.md:38-40` is explicit: the state builder "Fills missing
fields with `None` / `unknown`; never fabricates" and "Attaches
`perception_confidence` map to the state for downstream use". Line 33 adds
that low-confidence fields are "marked `unknown` rather than guessed". So
the repository already decides this question; this ADR adopts it.

Concretely:

- Every observed field added to `GameState` is `X | None` with default
  `None`.
- `None` means "not observed". A genuine zero is a real value: `inventory_count
  = 0` means the bags were seen empty; `None` means the bags were not
  inspected. This distinction is exactly what review Q6 warns is missing
  today, where missing inventory silently defaults to zero
  (`docs/reviews/PRE_PHASE_13_REVIEW.md:138`).
- `perception_confidence: dict[str, float]` maps a field path to a
  confidence in `[0.0, 1.0]`. Absence from the map means not attempted;
  presence at low confidence means seen but uncertain, which per
  `docs/PERCEPTION.md:33` still surfaces as `None` in the field itself.
- The adapter composes with this unchanged: a `None` on a non-Optional view
  field still raises `AdapterIncompleteError` (ADR-001 decision 4).

**Rejected: a typed sentinel `NullValue`.** It doubles the type algebra of
every field and forces all eight views and the adapter to handle a third
state. `None` already composes with the fail-loud policy for free.

**Rejected: confidence map alone (no `Optional`).** Confidence cannot
express "not measured". ADR-001's fail-loud adapter needs a way to say the
field is absent; a confidence of `0.0` is ambiguous with "measured, fully
uncertain".

**Real backend reporting semantics.** "I could not see X" is `None`, with
no confidence entry. "I saw X and it is zero" is the value `0`/`0.0` with a
confidence entry. "I saw X but I am unsure" is `None` plus a low confidence
entry, per `docs/PERCEPTION.md:33`. Downstream, the reflex layer applies
confidence thresholds and the strategist receives thresholded aggregates
(`docs/PERCEPTION.md:46-47`), so no consumer ever reasons over a raw
low-confidence value.

### Decision 5: STRATEGY A exception scope

Because Decision 1 is Option A, a STRATEGY A exception is required. It is
proposed here, not applied.

**Files that would change.**

1. `src/wow_bot/shared/interfaces.py` — additive fields only.
2. `src/wow_bot/mocks/mock_perception.py` — populate the new fields. This is
   a pre-lab module consumed by frozen `main.py`, so it is covered by the
   same exception.

**No other file changes.** In particular, no consumer Protocol, no
`PerceptionBackend` signature, no `main.perception_loop` signature, and no
`lab/runner_v2.py` signature changes. `PerceptionBackend.snapshot()` still
returns `GameState` (`protocol.py:14`) — a decisive advantage of Option A
over Option B.

**Fields added to `GameState`, all appended after `events` (the last
existing field), all `Optional`, all defaulting to `None` or an empty
factory. Eleven fields in total — five observed, six channel-pending:**

Observed today (no new vision channel required):

- `player_z: float | None = None`
- `target_x: float | None = None`
- `target_y: float | None = None`
- `entities: list[EnemyInfo] = field(default_factory=list)` (the extended
  `EnemyInfo` from Decision 3; supersedes the role of `enemies`)
- `perception_confidence: dict[str, float] = field(default_factory=dict)`

Channel-pending observed fields (require the vision tasks in Decision 6):

- `inventory_count: int | None = None`
- `inventory_max: int | None = None`
- `level_or_xp: float | None = None`
- `durability_fraction: float | None = None` (fraction `[0.0, 1.0]`, per
  `farm/vendor.py:146-152`)
- `target_is_lootable: bool | None = None`
- `incoming_casts: list[IncomingCast] = field(default_factory=list)` (new
  small frozen dataclass in the same module)

**Removed from the list in pass 2** because they are derived, not observed:
`target_is_alive` (from `target.hp_pct`, already canonical) and `resource_max`
(from a class/level table). Both are computed by the adapter and never stored.

**Deliberately not added** (derived or internal, handled by the adapter):
`target_in_range`, `adds_count`, `gcd_ready`, `threat` as a top-level
field (it lives on the extended `EnemyInfo`), `fsm_state`, and now
`target_is_alive` and `resource_max`.

**No existing field's type, default, or position changes.** This is a hard
requirement, not a preference: `tests/unit/test_review_gate.py:51` constructs
`GameState` positionally with nine arguments, so any new field appended
after `events` preserves that call, and any insertion before it breaks it.

**`to_dict`/`from_dict`.** `to_dict` (`shared/interfaces.py:165-176`) emits
the new keys; `from_dict` (`:178-196`) reads them with `data.get(...)`
defaults so pre-extension payloads still parse. The round-trip tests at
`tests/unit/test_interfaces.py:63-95` then continue to pass, and old
artifacts (none of which serialize `GameState`) remain readable.

**MockPerception and main.perception_loop.** `perception_loop` needs no
change: it forwards `GameState` objects (`main.py:216-224`) and the new
fields ride along. `MockPerception` must populate the new fields from its
component RNG (AGENTS.md section 6.2 forbids global seeding) and should
emit `None` on some frames so the adapter's fail-loud path stays covered by
tests. Its behavior on the nine existing fields is unchanged.

**Phase 12 artifact parsers.** Unaffected. Verified by reading
`runs/lab/soak-1h/soak_report.json`, `runs/lab/aggregate-1h/aggregate_v1.json`,
and `runs/lab/soak-1h/events.jsonl`: none contains a serialized
`GameState`. No parser reads `GameState` from these artifacts; the only
`on_game_state` consumers are a counter increment
(`src/wow_bot/reporting/scenario.py:109`) and a no-op
(`src/wow_bot/analysis/soak.py:179`).

**Proposed AGENTS.md section 5.6 entry, verbatim:**

> ### 5.6 Bounded exception: T-FIX-02.5 GameState extension
>
> The task T-FIX-02.5 authorizes a single, bounded exception to STRATEGY A.
> Its sole purpose is to add the eleven observed fields enumerated in
> `docs/decisions/ADR-002-gamestate-extension.md` (five observed today, six
> channel-pending) to `GameState` and `EnemyInfo` in
> `src/wow_bot/shared/interfaces.py` as `Optional` fields with `None` or
> empty-factory defaults, appended after all existing fields, together with
> the matching `to_dict`/`from_dict` round trip and the population of those
> fields by `MockPerception` in `src/wow_bot/mocks/mock_perception.py`.
> `resource_max` and `target_is_alive` are NOT added: they are derived by
> the perception adapter and never stored on `GameState`.
>
> It does NOT authorize renaming or removing any existing field, changing
> any existing field's type or default, changing the positional argument
> order of any existing constructor, modifying `MetaState`, `Strategy`, or
> `TargetInfo`, changing any of the eight consumer Protocols, changing
> `PerceptionBackend.snapshot()`, `main.perception_loop`, or
> `lab/runner_v2.py`, or fabricating values for fields a producer cannot
> observe. It does NOT authorize changes to NON_CLAIMS wording, the T12.0
> scope split, the aggregate contract, or any Phase 12 artifact.
>
> After T-FIX-02.5 merges, this exception is closed. Any future edit to a
> module that STRATEGY A treats as FROZEN requires its own explicit
> exception recorded in this file.

### Decision 6: Migration path and ordering

Ordered so that no task depends on a later one. Inputs and outputs are
files; the acceptance criterion is the single test that proves the step.

1. **T-FIX-23 (new): PERCEPTION.md vision channel extensions.** Adds the
   four missing channels to the `docs/PERCEPTION.md:23-30` table.
   - Inputs: the channel rules specified in "Field provenance" below.
   - Outputs: `docs/PERCEPTION.md` only — four new component rows (bag
     frame, XP bar, enemy cast bar, lootable-corpse indicator) plus the
     anchor entries for them in the calibration list at
     `docs/PERCEPTION.md:51-56`.
   - Acceptance: each new row names source channel, extraction method,
     expected accuracy class, and confidence semantics; no existing row
     changes. Doc-only, so no test flips.
2. **T-FIX-02.5 (new): GameState schema extension.** Implements this ADR.
   - Inputs: `docs/decisions/ADR-002-gamestate-extension.md`, the AGENTS.md
     5.6 entry above, and the channels T-FIX-23 documents.
   - Outputs: extended `shared/interfaces.py`, extended
     `mocks/mock_perception.py`.
   - Acceptance: `tests/unit/test_interfaces.py` round-trip tests pass with
     the new fields; positional construction at
     `tests/unit/test_review_gate.py:51` still passes; `ruff check src
     tests scripts` and `mypy src` are clean.
3. **T-FIX-21 (new): adapter derivation and runtime context.** Teach the
   adapter to derive the non-stored quantities and accept the internal
   context.
   - Inputs: extended `GameState`; config values such as
     `engage_distance_units` (`combat/loop.py:56`, default `30.0`) and
     `adds_threshold` (`combat/flee.py:65`, default `3`); the class/level
     resource table for `resource_max`.
   - Outputs: `perception/adapter.py` deriving `target_in_range`,
     `adds_count`, `target_hp_percent`, `resource_max`, and
     `target_is_alive`; a small runtime context object supplying
     `gcd_ready`, `spell_cooldown_ready`, and `target_has_debuff` so the
     three `NotImplementedError` stubs
     (`perception/views.py:117-123`, `:152-154`) get a real backing.
   - Acceptance: against an extended mock snapshot, `to_combat_view`,
     `to_reactive_view`, `to_flee_view`, and `to_targeting_views` return
     without raising.
4. **T-FIX-24 (new): xfail test split.** Restructure the forcing function
   so it expresses per-view expectations instead of one all-or-nothing
   marker.
   - Inputs: the projection-by-projection analysis in "xfail impact
     analysis" below.
   - Outputs: `tests/test_perception_adapter.py` only. The single
     end-to-end `xfail` at `:660-674` becomes one test per view: each
     unblocked view asserts success, each still-blocked view asserts the
     specific `AdapterIncompleteError` it must keep raising.
   - Acceptance: every one of the eight views has a named test; the suite
     is green; no view's expected outcome is left implicit.
5. **T-FIX-04: Reference MockAdapter.** Now meaningful.
   - Inputs: the extended mock producer and the deriving adapter.
   - Outputs: the reference adapter path exercised end to end.
   - Acceptance: the unblocked projections pass with no `xfail` marker, and
     the blocked ones fail loudly and on purpose. This is the forcing
     function delivered by this ADR.
6. **T-FIX-22 (new): perception_confidence plumbing.**
   - Inputs: the `perception_confidence` map added in T-FIX-02.5.
   - Outputs: mock population of the map; adapter thresholding that lets a
     low-confidence observation surface as `None`.
   - Acceptance: a test where a low-confidence `inventory_count` yields
     `None` in `VendorView` and raises `AdapterIncompleteError`.
7. **T-FIX-05: FSM IDLE->SCANNING progression** (existing, Tier 2). Blocked
   until T-FIX-04, because the strategist view must be projectable first,
   and it intersects the Q6 finding that the prompt cooldown trusts
   `state.fsm_state` (`strategist/orchestrator_v2.py:266`) rather than the
   runtime FSM state.
6. **T-FIX-06: Reflex/Watchdog lifecycle** (existing, Tier 2).
7. **T-FIX-07: Safety lifecycle wiring** (existing, Tier 2).
8. **T-FIX-08: Gameplay primitives** (existing, Tier 2).
9. **Phase 13 real perception**: a `RealPerception` implementing
   `PerceptionBackend`, emitting the extended `GameState` per
   `docs/PERCEPTION.md`. Only after steps 1-10.

Ordering justification: T-FIX-23 precedes T-FIX-02.5 because a field's
channel must be specified before the schema names it, otherwise the field
is unimplementable by a real backend from day one. T-FIX-02.5 must precede
T-FIX-21 because the adapter cannot derive fields the schema lacks.
T-FIX-21 must precede T-FIX-24 and T-FIX-04 because the derived fields are
what make any projection succeed. T-FIX-24 precedes T-FIX-04 so the
reference adapter lands against a suite that already expresses per-view
expectations. T-FIX-04 must precede T-FIX-05 because the FSM progression
consumes a working strategist view. T-FIX-22 may run in parallel with
T-FIX-05 through T-FIX-08 since it only constrains perception semantics.
Phase 13 is last by definition in `docs/SOAK_PROTOCOL.md:58-72`.

## Field inventory

Classification key: **O** = observed (a real backend could see it),
**D** = derived (computable from observations plus rules/config),
**I** = internal Python/controller state. "Not stored" means the field is
computed or injected by the adapter and never placed on `GameState`.
**O-pending** marks a field that is observable in principle but has no row
in the `docs/PERCEPTION.md:23-30` vision table yet; its rule is specified in
"Field provenance" and its channel is added by task T-FIX-23.

### Self-state

| Field | Type | Unit and range | Required by | Class | Unknown looks like | MockPerception today |
|---|---|---|---|---|---|---|
| `player_z` | `float` | world units, any finite | WorldSyncView, StrategistView | **O** | `None` | needs change |
| `resource_max` | `float` | percent, `[0.0, 100.0]` | StrategistView, CombatView | **D** | `None`; see "Field provenance" | needs change |
| `inventory_count` | `int` | slots, `>= 0` | StrategistView, LootView, VendorView | **O-pending** | `None` (not "0") | needs change |
| `inventory_max` | `int` | slots, `> 0` | LootView (Optional there) | **O-pending** | `None` | needs change |
| `level_or_xp` | `float` | level points | StrategistView | **O-pending** | `None` | needs change |
| `durability_fraction` | `float` | fraction, `[0.0, 1.0]` | VendorView (Optional there) | **O-pending** | `None`; consumer already handles it (`farm/vendor.py:443-446`) | needs change |

`inventory_count` semantics matter: `farm/loot.py:325` compares
`inventory_count >= full_at` and `farm/vendor.py:319,400` branch on `== 0`
and `> 0`, so a fabricated `0` silently means "bags verified empty".
`level_or_xp` is read as `level` then `xp` by the runner
(`lab/runner_v2.py:958`), which review Q6 flags as a mock-only assumption.

### Target-state

| Field | Type | Unit and range | Required by | Class | Unknown looks like | MockPerception today |
|---|---|---|---|---|---|---|
| `target_is_alive` | `bool` | boolean | LootView | **D** | `None`; derived, so never stored | needs change |
| `target_is_lootable` | `bool` | boolean | LootView | **O-pending** | `None` | needs change |
| `threat` | `float` | threat points, `>= 0` | TargetView | **D** | `None`; only a sort metric (`combat/targeting.py:98`), never a gate | needs change |
| `is_attackable` | `bool` | boolean | TargetView | **O** | `None`; consumer filters on it (`combat/targeting.py:134`) | needs change |
| `is_alive` | `bool` | boolean | TargetView | **O** | `None`; consumer filters on it (`combat/targeting.py:132`) | needs change |
| `is_in_combat_with_self` | `bool` | boolean | TargetView | **D** | `None`; consumer filters on it (`combat/targeting.py:136`) | needs change |
| `target_x`, `target_y` | `float` | world units | CombatView, FleeView, LootView (all Optional there) | **O** | `None`; consumer degrades gracefully (`farm/loot.py:232-233`) | needs change |
| `target_in_range` | `bool` | boolean | CombatView, ReactiveView | **D** | not stored; adapter derives from distance vs `engage_distance_units` | needs change (or derivation) |

`target_is_alive` and `target_is_lootable` are hard gates: `farm/loot.py:206`
skips looting while the target is alive and `:216` skips when it is not
lootable, so fabricating either would cause the bot to attempt looting a
live or empty corpse. `threat` is used only as a priority sort key
(`combat/targeting.py:98`), so an unknown threat is far less dangerous than
an unknown `is_alive` — a useful asymmetry the adapter could exploit by
treating `None` threat as neutral rather than failing.

### Combat and cooldown

| Field | Type | Unit and range | Required by | Class | Unknown looks like | MockPerception today |
|---|---|---|---|---|---|---|
| `gcd_ready` | `bool` | boolean | CombatView | **I** | not stored; supplied by a runtime cooldown context | needs change |
| `target_has_debuff(debuff_id)` | `bool` (method) | boolean per spell id | CombatView | **I** | today `NotImplementedError` (`perception/views.py:117-119`) | needs change |
| `spell_cooldown_ready(spell_id)` | `bool` (method) | boolean per spell id | CombatView, ReactiveView | **I** | today `NotImplementedError` (`perception/views.py:121-123`, `:152-154`) | needs change |

These are query methods, not fields, and they are state owned by the
cooldown subsystem, not by perception. `combat/loop.py:209` gates the GCD
wait on `gcd_ready` and `combat/reactive.py:169` calls
`spell_cooldown_ready` before retreating, so a wrong answer changes combat
behavior, but the correct source is the runtime cooldown tracker, not a
vision frame.

### Environmental

| Field | Type | Unit and range | Required by | Class | Unknown looks like | MockPerception today |
|---|---|---|---|---|---|---|
| `adds_count` | `int` | count, `>= 0` | FleeView | **D** | not stored; adapter counts entities in combat with self | needs change |
| `incoming_casts` | `list[IncomingCast]` | one entry per cast | ReactiveView | **O-pending** | empty list, meaning "no casts observed" (honest, unlike `None`) | needs change |

`adds_count` is compared as `state.adds_count >= self._config.adds_threshold`
(`combat/flee.py:193`), so a fabricated `0` would permanently suppress
fleeing. `incoming_casts` is iterated for interrupt decisions
(`combat/reactive.py:184-197`); the current empty-tuple default is honest
because "no casts visible" is a legitimate observation, which is why the
empty collection is the right unknown shape here rather than `None`.

### Internal Python

| Field | Type | Unit and range | Required by | Class | Unknown looks like | MockPerception today |
|---|---|---|---|---|---|---|
| `fsm_state` | `FSMState` | enum, `executor/states.py:11-25` | StrategistView | **I** | not stored; already an adapter parameter (`perception/adapter.py:113`) | no change needed |

This is the field review Q6 singles out
(`docs/reviews/PRE_PHASE_13_REVIEW.md:140`): the prompt cooldown at
`strategist/orchestrator_v2.py:266` reads `state.fsm_state`, a synthetic
snapshot hardcodes `IDLE` (`scripts/lab/full_soak.py:108`), and vision
cannot observe the controller's own state. Keeping it an injected parameter
is the correct architecture; the follow-up is T-FIX-05's job, not a schema
change.

## Field provenance

Pass 2 resolved the seven fields that pass 1 had classified as observed
without any backing row in `docs/PERCEPTION.md:23-30`. Each of the seven is
settled below with one classification and one rule. None remain "assumed".

The governing constraint, stated in `docs/PERCEPTION.md:25-30`: the only
extraction methods available are color segmentation plus Hough, template
match, template match plus centroid, and OCR. There is no memory access
(`docs/LAB_CONSTRAINTS.md:29`), so every observed value must come from a
screenshot through one of those four methods.

### OBSERVED-WITH-VISION (six of the seven)

These are observable in principle but four of the five have no channel row
yet. Each entry specifies the rule task T-FIX-23 must add to
`docs/PERCEPTION.md`.

**`inventory_count`** — channel-pending. New "Bag frame" component.

- Source channel: the bag window's slot grid, captured when the window is
  open. The bag window is not always open, so this is an opportunistic
  observation, not a per-frame one.
- Extraction method: template match per slot against empty-slot and
  occupied-slot templates; the count is the number of occupied matches.
- Accuracy class: medium. Slot templates are small and the grid shifts with
  bag size, but the decision is binary per slot.
- Confidence semantics: mean per-slot match score. Below threshold the
  whole field is `None` rather than a partial count, because the consumer
  gates a full/not-full decision on it (`farm/loot.py:325`).
- Rejected alternative: accumulating "You receive item" lines from the
  chat/events channel. Chat lines scroll out of the OCR region while the
  region is occluded, so the accumulator silently drops lines and drifts.
  A per-frame absolute count does not drift.

**`inventory_max`** — channel-pending. Same "Bag frame" component.

- Source channel: the same bag window slot grid.
- Extraction method: count of slot positions in the grid, occupied or
  empty. This is a layout property, so it is stable for a session and can
  be captured once when the window opens and cached.
- Accuracy class: high. A grid count is a small integer with a clear
  template footprint.
- Confidence semantics: single high-confidence reading; otherwise `None`,
  and `farm/loot.py:290-294` falls back to the configured
  `inventory_full_absolute` threshold, which is the existing safe path.

**`level_or_xp`** — channel-pending. New "XP bar" component.

- Source channel: the experience bar, a thin fill bar at the bottom edge of
  the viewport.
- Extraction method: color segmentation plus Hough, the same method the
  health/mana row already documents at `docs/PERCEPTION.md:25`, yielding a
  `[0,1]` fill fraction. The integer level is OCR'd from the bar's level
  label with the same Tesseract method documented at `:27`.
- Emitted value: `level + xp_fraction`, a single monotone scalar, so the
  runner's `level`-then-`xp` read (`lab/runner_v2.py:958`) and the
  strategist's `level_or_xp` field agree on one number.
- Accuracy class: medium for the fraction, high for the small integer.
- Confidence semantics: bar-occlusion confidence; below threshold `None`,
  never a guessed level, because review Q6 warns a constant level hides XP
  progress (`docs/reviews/PRE_PHASE_13_REVIEW.md:138`).

**`durability_fraction`** — channel-pending. New "Character frame" component.

- Source channel: the character sheet's per-slot durability readouts. This
  field is the only one of the seven whose consumer already handles the
  unknown: `farm/vendor.py:443-446` treats `None` as `"durability_unknown"`
  and skips repairing, so an honest `None` is a safe degradation, not a
  block.
- Extraction method: OCR of the per-slot percentage text (the `:27`
  Tesseract method), then a mean over the slots the state builder read
  confidently.
- Accuracy class: low to medium. The values are small text on a busy
  frame, so OCR is the limiting factor; the mean over slots smooths one
  misread but not many.
- Confidence semantics: the mean is weighted by per-slot OCR confidence,
  and the field is emitted only when a quorum of slots read above
  threshold. Below quorum it is `None`, which routes to the existing safe
  path rather than fabricating an average.
- Emitted scale: `[0,1]`, the scale `farm/vendor.py:146-152` validates.
- Note on aggregation: the per-slot readings are the observation; the mean
  is a trivial reduction done in the state builder. The field stays on
  `GameState` because the consumer is a frozen Protocol that reads it as a
  scalar, and pushing the mean into the adapter would not change what is
  observed.

**`target_is_lootable`** — channel-pending. New "Lootable-corpse indicator".

- Source channel: the loot sparkle on a corpse in the game world. This is
  the weakest channel of the seven and is the main reason LootView stays
  blocked after the schema change.
- Extraction method: color segmentation for the sparkle hue in the region
  around a dead unit. The technique is documented at
  `docs/PERCEPTION.md:25`; the channel is new.
- Accuracy class: low. The sparkle is small, transient, and camera-angle
  dependent, so false negatives dominate.
- Confidence semantics: a high threshold to assert `True` and no confident
  negative at all. Absent a confident positive, the field is `None`, never
  `False`, because `farm/loot.py:216` skips looting on a false negative and
  the bot would walk away from loot it could have taken.
- Validation gate: task T-FIX-23 must record precision and recall on the
  frozen frame corpus that `docs/PERCEPTION.md:60` already requires. Until
  that measurement exists, the field is emitted as `None` and LootView
  keeps raising. This is a measured uncertainty, not an assumption.

**`incoming_casts`** — channel-pending. New "Enemy cast bar" component.

- Source channel: the cast bar that appears on the target frame and above
  enemy nameplates while a mob is casting.
- Extraction method: OCR for the spell name (the `:27` Tesseract method)
  plus template match for the interruptibility indicator and the bar's fill
  fraction. `spell_id` is the OCR'd name mapped through the spell-id table;
  `remaining_cast_time_s` is the fill fraction times the spell's known cast
  duration.
- Accuracy class: low to medium. Spell names are short and the cast bar is
  small, so OCR precision is the limiting factor.
- Confidence semantics: per-cast confidence. A cast below threshold is
  omitted from the list rather than included with a guessed spell id,
  because `combat/reactive.py:184` iterates the list for interrupt
  decisions and a wrong spell id produces a wrong interrupt.
- Unknown sub-field: `is_interruptible` is a template-match on a border
  color. If that match fails the whole cast entry is dropped, since
  `combat/reactive.py:186` gates the interrupt on it directly.
- Empty list is honest: with no confident casts, `()` means "no casts
  observed", which is a legitimate observation, not a fabricated absence.

### DERIVED (two of the seven)

Both are computed by the adapter and removed from the `GameState` additive
list, so the schema shrinks from thirteen fields to eleven.

**`target_is_alive`** — derived. Not stored on `GameState`.

- Input fields: `target.hp_pct`, already canonical
  (`shared/interfaces.py:148` region, `TargetInfo` definition). No new
  observation is needed.
- Inference rule: `alive = target.hp_pct > 0.0`. The target frame reads
  zero health for a dead unit, and the field exists only to express that
  threshold.
- Unknown input: when `target` is `None` or `target.hp_pct` is unknown, the
  adapter yields `None`, and because `LootView` declares the field
  non-Optional (`perception/views.py:182`), the projection raises
  `AdapterIncompleteError` rather than guessing alive.
- Where the derivation lives: `perception/adapter.py`, in `to_loot_view`,
  next to the existing `target.hp_pct` read at `adapter.py:357`. It is a
  one-line threshold, not a new module.
- Note on the mock: `MockPerception` samples target `hp_pct` in
  `[0.2, 1.0]` (`mock_perception.py:463`), so the mock never produces a
  dead target and this derivation always yields `True` in mock mode. That
  is a mock-coverage gap for T-FIX-04 to close with a scenario that emits
  `hp_pct == 0.0`, not a flaw in the rule.

**`resource_max`** — derived. Not stored on `GameState`.

- Input fields: the character's class and level, which come from the
  "Target frame" identity slot and the "XP bar" level label respectively.
  Neither exists as a `GameState` field today; the adapter reads them from
  the confidence map and the level channel.
- Inference rule: a class-and-level to max-resource lookup table. This is a
  game-data constant, not a per-frame quantity, so a vision channel would
  be the wrong place for it. `docs/PERCEPTION.md:25` specifies only
  normalized `[0,1]` bar output, which cannot yield a maximum.
- Unknown input: when class or level is unknown, the adapter yields `None`,
  and because both `StrategistView` (`perception/views.py:72`) and
  `CombatView` (`:109`) declare it non-Optional, both projections raise.
- Where the derivation lives: the table belongs in a new small module under
  `src/wow_bot/perception/` (for example `resource_table.py`), with the
  adapter calling it. It must not live in `strategist` or `combat`, because
  two consumers need the same constant and duplicating a game-data table
  across consumers is exactly the divergence ADR-001's single-adapter
  design exists to prevent.

### INTERNAL (none of the seven)

None of the seven are Python-side state. The three internal fields the
pipeline needs — `fsm_state`, `gcd_ready`, and the two cooldown query
methods — are already handled as adapter parameters or runtime context and
are listed in the "Combat and cooldown" and "Internal Python" tables above.
Pass 2 changes nothing about them.

### Unresolved after pass 2

None of the seven remain unresolved. The two open items below were already
open in pass 1 and are unrelated to the seven; both are restated in
"Unresolved questions" with their concrete next actions.

1. The `entity_id` stability policy for the extended `EnemyInfo` — a
   perception-implementation decision the repository does not contain.
2. Whether the class-and-level resource table can be transcribed from
   game data available on the private lab server, which determines whether
   `resource_max` is ever non-`None` in LAB_MODE.

## Consequences

**Positive:**

- All eight views become projectable from one canonical snapshot, so
  T-FIX-04 has real work to do and the `xfail` forcing function becomes
  live.
- `PerceptionBackend.snapshot()` keeps returning `GameState`
  (`protocol.py:14`), `main.perception_loop` is untouched, and
  `RuntimeComponents.perception` stays typed as `MockPerception`
  (`main.py:100`). Zero frozen signatures change.
- Unknown values are honest at the schema level: `None` for unobserved,
  real zeros for observed zeros, plus a confidence channel. This directly
  answers review Q6's warning that missing observations silently become
  plausible values today.
- A real backend has one contract to satisfy, already documented at
  `docs/PERCEPTION.md:7-8`, and the fail-loud adapter still tells it
  exactly which field it failed to supply.

**Negative:**

- A STRATEGY A exception is required, and even bounded it touches
  `shared/interfaces.py`, the most imported module in the repository. Any
  mistake in the round trip or the field ordering propagates widely.
- `GameState` grows from nine fields to twenty, and `EnemyInfo`
  from three to thirteen. Six of the new fields have no vision component
  in `docs/PERCEPTION.md` today, so a real backend cannot produce them
  until task T-FIX-23 adds the channels; until then the adapter keeps
  raising for them, and seven of the eight views stay blocked. See "xfail
  impact analysis".
- The adapter becomes stateful in a new way: deriving `target_in_range`
  requires config, and supplying `gcd_ready` requires a runtime context.
  The current adapter is a pure projection of one snapshot; that purity is
  reduced.
- `MockPerception` must grow synthetic emitters for the new fields. A mock
  that populates everything confidently would hide the unknown-value
  semantics, so it must deliberately emit `None` sometimes, which is more
  test surface.

**Neutral:**

- This ADR does not change any consumer, does not wire the adapter into any
  pipeline, and does not change the aggregate contract, NON_CLAIMS, or the
  T12.0 scope split.

## Backwards compatibility

**Does ADR-002 break any test that exists today?**

The ADR itself is design-only and changes no file, so nothing breaks on its
own. The implementation tasks above would affect tests as follows:

- `tests/unit/test_interfaces.py:63-95` — round-trip equality would break
  if `to_dict` emitted the new fields and `from_dict` did not read them.
  Mitigation is part of T-FIX-02.5, not an afterthought.
- `tests/unit/test_review_gate.py:51` — positional `GameState` construction.
  Safe only if every new field is appended after `events`. This is the
  single most fragile line in the repository for this change.
- `tests/test_perception_adapter.py:660-674` — the `xfail(strict=True)`
  test. It no longer describes reality after this pass, because only four
  of the eight projections can succeed. It is *designed* to be replaced:
  task T-FIX-24 splits it into one named test per view. Until that split
  lands, the marker must stay, because removing it would make the suite
  red for the four views that still raise.
- `tests/test_perception_adapter.py:62-105` — `_make_well_formed_state`
  uses `object.__setattr__` to inject fourteen extended attributes onto a
  canonical `GameState`. Once the fields exist for real, the helper becomes
  redundant but not incorrect; it should be simplified in T-FIX-04.
- `tests/unit/test_mock_perception.py:165-173` and
  `tests/unit/test_mock_scenarios.py:143-303` — read `state.enemies` and
  `EnemyInfo.bbox`/`distance_estimate`. These keep passing because the
  Decision 3 extension is additive and changes no existing field.

**Does it invalidate the Phase 12 `aggregate_v1.json` artifact?**

No. `runs/lab/aggregate-1h/aggregate_v1.json` contains
`internal_counters_not_research_findings`, `non_claims`, and
`perception_agnostic` counters only. It serializes no `GameState` fields,
and neither does `soak_report.json` nor `events.jsonl`. The artifact should
be **frozen as-is**, not regenerated: it is the Phase 12 MOCK_MODE stability
record whose non-claims at `docs/non_claims.json` are protected by
`docs/SOAK_PROTOCOL.md:77-85`, and regenerating it under an extended schema
would misrepresent what was measured.

**Does it change any public function signature in a frozen module?**

No. `MockPerception.get_state()` still returns `GameState`.
`main.perception_loop` keeps its signature. `PerceptionBackend.snapshot()`
keeps returning `GameState` (`protocol.py:14`). `GameStateAdapter` gains
derivation and an optional runtime context, but `GameStateAdapter` is a
T-FIX-03 module, not a frozen one. The only frozen-module constructor
signatures that change shape are `GameState` and `EnemyInfo` in
`shared/interfaces.py`, both strictly additive at the end, which is exactly
what the proposed AGENTS.md 5.6 entry authorizes.

**Does it change how MockPerception behaves today?**

Yes, in a bounded way. Its behavior on the nine existing fields is
unchanged: the random walk, scenario profiles, combat cadence, and event
emission all stay. It additionally populates the new fields from its
component RNG, and it must sometimes emit `None` to keep the fail-loud
adapter path under test. Its `_make_combat_entities`
(`mock_perception.py:443-467`) would assign stable `entity_id` values to
enemies, which it does not do today. Tests asserting on the nine existing
fields are unaffected; tests asserting exact equality of full snapshots
would need updating.

## xfail impact analysis

After the field-count change, the eight projections split into two groups
against a canonical `GameState` carrying the eleven extended fields. Five
observed fields are populated; the six channel-pending fields are `None`
until their vision channels ship; the derived fields are computed by the
adapter.

**Can succeed once the adapter derives the fields it now owns:**

| View | Blocker status |
|---|---|
| WorldSyncView | **Unblocked.** `player_z` and `entities` are both populated observed fields. |
| StrategistView | Blocked on `resource_max`, now derived from class and level. Raises until the resource table lands; the level input is itself channel-pending. |
| CombatView | Blocked on `resource_max` for the same reason, plus the cooldown methods. |
| TargetView (`to_targeting_views`) | Blocked on `threat` (derived, no derivation exists yet) and `is_in_combat_with_self` (derived, same). Neither has an input field, so both stay `None`. |
| ReactiveView | Blocked on the cooldown method only: `target_in_range` is derived and `incoming_casts` defaults to an honest empty tuple. |
| FleeView | Blocked on `adds_count`, which is derived from the entity list and `is_in_combat_with_self`; the latter has no derivation, so `adds_count` is `None`. |
| LootView | Blocked on `target_is_lootable`, which is channel-pending with no current channel. `target_is_alive` resolves by derivation from `target.hp_pct`, so it is not the blocker. |
| VendorView | Blocked on `inventory_count`, which is channel-pending. `durability_fraction` is `Optional` and already handled as `None` (`farm/vendor.py:443-446`). |

Net effect: **one of the eight views can succeed** (WorldSyncView) and
**seven cannot**. This is a sharper result than pass 1's "all eight raise",
and it is the honest one: the derived fields resolve nothing that was
channel-pending, because a derivation cannot manufacture its own inputs.
The single all-or-nothing `xfail` at
`tests/test_perception_adapter.py:660-674` no longer describes the truth,
because flipping it wholesale would require all eight to succeed, which
the table above shows cannot happen.

**Required test restructuring, specified but not applied here.** Task
T-FIX-24 replaces the one end-to-end marker with one named test per view:

- Each unblocked view gets a test asserting it projects without raising;
  after this pass that is WorldSyncView alone.
- Each blocked view gets a test asserting it raises
  `AdapterIncompleteError` and naming the exact field that must remain
  unsupplied, so the block is documented in code rather than in this ADR.
- The suite stays green throughout, because no view's expected outcome is
  left implicit: success is asserted where achievable and failure is
  asserted where intended.

The strict `xfail` marker is the wrong shape for a partial success, and
leaving it in place would make T-FIX-04 fail the suite for doing exactly
what this ADR requires. This pass does not modify the test; T-FIX-24 owns
the change, and it must land before the reference adapter in T-FIX-04.

## Migration path

Numbered identically to Decision 6, restated as the execution list:

1. **T-FIX-23** — PERCEPTION.md vision channel extensions (doc-only).
2. **T-FIX-02.5** — GameState schema extension (implements this ADR).
3. **T-FIX-21** — adapter derivation and runtime context.
4. **T-FIX-24** — split the `xfail` into per-view tests.
5. **T-FIX-04** — Reference MockAdapter.
6. **T-FIX-22** — `perception_confidence` map plumbing and thresholding.
7. **T-FIX-05** — FSM IDLE->SCANNING progression.
8. **T-FIX-06** — Reflex/Watchdog lifecycle.
9. **T-FIX-07** — Safety lifecycle wiring.
10. **T-FIX-08** — Gameplay primitives (cast/loot/vendor/jump).
11. **Phase 13** — `RealPerception` implementing `PerceptionBackend`.

No step depends on a later step. Steps 7-10 are the existing Tier 2 tasks
from `docs/lab_phase/PRE_REAL_PERCEPTION_FIX_ROADMAP.md`; steps 1, 3, 4, and
6 are new task IDs proposed by this ADR because the repository has no
existing task for the vision channels, the schema extension, the adapter
derivation, the test split, or the confidence channel.

## Alternatives considered

**Option B: a new `PerceptionSnapshot` type alongside `GameState`.**
Rejected. `docs/PERCEPTION.md:7-8` requires real backends to conform to the
existing `GameState` schema and forbids branching on producer identity, so
a second type would require rewriting that document and would produce the
producer-identity branching it forbids. It also forces three frozen
signature changes (`protocol.py:14`, `main.py:208`, `main.py:100`) and
leaves `MockPerception` emitting a type the real path never uses, so the
mock would stop validating the real pipeline.

**Option C as a separate derived/internal channel.** Rejected as a
mechanism, adopted as a classification. Routing derived quantities through
a second object carried next to `GameState` doubles the plumbing through
`perception_queue` and `PipelineSnapshot` without changing the content of
any field. The three-way classification itself is sound and is used in the
field inventory to decide what never lands on `GameState`.

**A typed sentinel `NullValue` instead of `None`.** Rejected. It triples
the state space of every field (value / `None` / sentinel) and forces all
eight views and the adapter to handle a case that `None` already expresses.
`docs/PERCEPTION.md:38-40` already prescribes `None` plus a confidence map.

**Confidence map without `Optional` fields.** Rejected. Confidence alone
cannot distinguish "not measured" from "measured at zero", which is the
exact distinction review Q6 reports is silently lost today
(`docs/reviews/PRE_PHASE_13_REVIEW.md:138`).

**Extending `EnemyInfo` versus a separate `entities` channel versus two new
types.** A separate channel keeps `EnemyInfo` frozen but duplicates the
entity list, letting the combat view and the world model disagree about
which mobs exist. Two new types (`WorldEntity` and `CombatEntity`) are
cleaner per view but require `GameState` to carry two lists describing the
same mobs, and require renaming `enemies`, which
`shared/interfaces.py:4-6` forbids. The superset extension is the only
option that satisfies both views with one list and no rename.

## Unresolved questions

These could not be determined from the repository and are deliberately not
guessed:

1. **The `entity_id` stability policy.** `WorldSync.mark_seen` keys the
   world model on `entity_id` (`world/sync.py:181-189`), so unstable ids
   flood the store. Whether stability should come from nameplate text,
   spawn position, or a tracked id table is a perception-implementation
   decision the repository does not contain. Minimum experiment: run the
   nameplate OCR component over a recorded frame sequence on the frozen
   corpus `docs/PERCEPTION.md:60` describes and measure how often the
   same mob OCRs to the same text. That decides between a text-derived id
   and a tracked table.
2. **The class-and-level resource table.** `resource_max` is now derived
   from class and level, so the table must exist before either
   `StrategistView` or `CombatView` can project. Whether the values can be
   transcribed from game data available on the private lab server is not
   determinable from this repository. Minimum action: read the class and
   level columns from one character on the lab server and cross-check the
   max-resource value against a full resource bar. Implementation of the
   `resource_max` derivation cannot proceed until that reading is made.
3. **`threat` provenance.** It is a sort metric only
   (`combat/targeting.py:98`) and no vision component observes it. It may
   need to be derived from the combat log or from a damage-event aggregate,
   or the `TargetConfig` default priority may need to change so threat is
   not the primary sort key. Not decidable here.
4. **Adapter config plumbing.** Deriving `target_in_range` needs
   `engage_distance_units` (`combat/loop.py:56`) and deriving `adds_count`
   needs `adds_threshold` (`combat/flee.py:65`). Whether the adapter should
   take these as constructor arguments or read them from `Settings` is left
   to T-FIX-21.
5. **`is_in_combat_with_self` semantics.** Whether a real backend can see
   that a specific mob is engaged with the player (as opposed to merely
   being hostile) is not established by any document in the repository. It
   is classified **D** on the assumption it can be inferred from the
   mob's target or from the combat indicator, but that inference is
   unverified. Until it is, `FleeView` and `TargetView` stay blocked
   through `adds_count` and the sort filters respectively.

## References

- `src/wow_bot/shared/interfaces.py:4-6` (evolution policy), `:143-155`
  (`GameState` fields), `:78-95` (`EnemyInfo`), `:165-196`
  (`to_dict`/`from_dict`)
- `src/wow_bot/perception/protocol.py:10-16`
- `src/wow_bot/perception/views.py:51-55` (WorldSyncView), `:67-77`
  (StrategistView), `:87-93` (TargetView), `:104-115` (CombatView),
  `:117-123` and `:152-154` (the three `NotImplementedError` stubs),
  `:144-150` (ReactiveView), `:165-171` (FleeView), `:182-191` (LootView),
  `:202-205` (VendorView)
- `src/wow_bot/perception/adapter.py:87`, `:113`, `:137`, `:252`, `:302`,
  `:333`, `:359`, `:402` (the eight raise sites behind the pass-3 table)
- `src/wow_bot/mocks/mock_perception.py:301-323` (`get_state`), `:313` (the
  only `GameState` construction in `src/`), `:443-467`
  (`_make_combat_entities`)
- `src/wow_bot/main.py:100` (`RuntimeComponents.perception`), `:206-224`
  (`perception_loop`)
- `src/wow_bot/world/sync.py:23-39` (`EntityLike`), `:43-62`
  (`GameStateLike`), `:162-199` (entity and target consumption)
- `src/wow_bot/world/store.py:20-27` (`VALID_NODE_KINDS`)
- `src/wow_bot/strategist/prompts_v2.py:30-46` (`GameStateView`), `:21-27`
  (`MetaStateLike`)
- `src/wow_bot/strategist/orchestrator_v2.py:264-266` (cooldown trusts
  `state.fsm_state`)
- `src/wow_bot/combat/loop.py:20-41` (`CombatStateView`), `:56`
  (`engage_distance_units`), `:209-214` (`gcd_ready`, `target_in_range`)
- `src/wow_bot/combat/targeting.py:14-23` (`TargetEntityLike`), `:98`
  (threat as metric), `:132-141` (hard filters)
- `src/wow_bot/combat/reactive.py:31-41` (`ReactiveStateView`), `:169`
  (cooldown call), `:184-197` (cast iteration)
- `src/wow_bot/combat/flee.py:31-40` (`FleeStateView`), `:65`
  (`adds_threshold`), `:193` (`adds_count` comparison)
- `src/wow_bot/farm/loot.py:38-50` (`LootStateView`), `:206-233` (alive and
  lootable gates), `:305-316` (inventory validation), `:325`
- `src/wow_bot/farm/vendor.py:45-51` (`VendorStateView`), `:116`
  (`repair_durability_threshold`), `:319`, `:400`, `:443-448`
- `src/wow_bot/executor/states.py:11-25` (`FSMState`)
- `src/wow_bot/lab/runner_v2.py:290-304` (`_extract_position` fallback),
  `:307-317` (`_extract_heading` fallback), `:958` and `:982` (level over
  xp), `:959-960` (fabricated action/reflex totals)
- `src/wow_bot/reporting/scenario.py:107-109` and
  `src/wow_bot/analysis/soak.py:178-179` (the only `on_game_state`
  consumers; a counter and a no-op)
- `tests/test_perception_adapter.py:62-105` (`_make_well_formed_state`),
  `:660-674` (the `xfail(strict=True)` forcing function)
- `tests/unit/test_interfaces.py:63-95` (round-trip tests)
- `tests/unit/test_review_gate.py:51` (positional `GameState` construction)
- `docs/decisions/ADR-001-perception-contract.md` (pass-3 table and the
  fail-loud policy)
- `docs/reviews/PRE_PHASE_13_REVIEW.md:94-103` (the eight protocols),
  `:133-144` (Q6), `:146-152` (Q7), `:169` (divergent-contract row)
- `docs/PERCEPTION.md:7-8` (output contract), `:23-30` (vision components),
  `:32-33` (confidence policy), `:38-40` (state builder), `:46-47`
  (thresholding)
- `docs/SOAK_PROTOCOL.md:7-41` (Phase 12 soak), `:58-72` (Phase 13 soak),
  `:77-85` (non-claims)
- `docs/lab_phase/PRE_REAL_PERCEPTION_FIX_ROADMAP.md` (T-FIX-01 through
  T-FIX-19, Tier 0 through Tier 6)
- `docs/LAB_CONSTRAINTS.md:19-36` (hard constraints, including the memory
  and log rules that bound `entity_id` assignment)
- `runs/lab/soak-1h/session.json`, `runs/lab/soak-1h/soak_report.json`,
  `runs/lab/soak-1h/events.jsonl`, `runs/lab/aggregate-1h/aggregate_v1.json`
  (Phase 12 artifacts; verified to contain no `GameState` serialization)
