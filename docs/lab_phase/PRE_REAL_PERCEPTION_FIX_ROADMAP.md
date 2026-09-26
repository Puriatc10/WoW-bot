# PRE_REAL_PERCEPTION_FIX_ROADMAP.md

Roadmap for everything that must be fixed **before** Phase 13
(`RealPerception`) can be merged into the lab pipeline.

This document is the source of truth for the pre-real-perception fix
tasks. It is subordinate to `LAB_CONSTRAINTS.md`,
`docs/lab_phase/LAB_PHASE_ROADMAP.md`, and `AGENTS.md`; see AGENTS.md §2
for the full precedence order. Each task below is written to be handed to
a coding agent as an implementation prompt: do not merge tasks, and do not
reorder tasks without updating this file first.

Phase 13 itself is defined in `docs/lab_phase/LAB_PHASE_ROADMAP.md` and is
**out of scope for this document**. Its entry conditions are listed in
"Phase 13 Entry Gate" at the end of this file.

Verification snapshot: HEAD `49b1510` on `master`. Every status below was
re-verified against that revision: `ruff check src tests scripts` → clean,
`mypy src` → clean, targeted `pytest` runs recorded in
"Current red tests" immediately below.

---

## Current red tests (must reach zero before Phase 13)

| Test | Cause | Owning task |
|---|---|---|
| `tests/integration/test_spectrum.py::test_project_spectral_acceptance` | Pre-existing baseline failure. All five drives fit a log-log slope of ≈ `-6.1` (R² ≈ 0.998) against the documented scientific target `[-1.5, -0.5]` (`docs/ROADMAP.md:405`). | T-FIX-10 |

**Resolved since the snapshot.** The
`tests/test_perception_adapter.py::test_combat_view_structural_and_type_compatibility`
failure introduced by T-FIX-03.6 was fixed by **T-FIX-21**, which makes an
empty entity channel fall back to the selected target instead of reporting a
fabricated empty world.

**Latest full-suite result:** `1 failed, 2258 passed, 7 skipped, 1 xfailed`.
The single failure is the spectral gate above. `tests/test_perception_adapter.py`
and `tests/test_perception_derivation.py` together report
`53 passed, 1 xfailed`.

---

## Status legend

| Marker | Meaning |
|---|---|
| `DONE` | Merged; acceptance re-verified at the snapshot above. |
| `DONE*` | Acceptance verified in the working tree; not yet committed. |
| `PARTIAL` | Merged with a known residual that another task owns. |
| `PENDING` | Not started. |
| `PROPOSED` | Not present in `LAB_PHASE_ROADMAP.md`, ADR-001, or ADR-002. Proposed by this document to close a documented gap; requires ratification before implementation starts. |

## Global rules (inherited; apply to every task)

1. Default mode is `MOCK_MODE`; `LAB_MODE` is opt-in and never runs in CI.
2. No task may add OS input, screen capture, or network calls outside the
   modules designated for them.
3. No task may read or write game process memory, inject DLLs, hook
   syscalls, or modify the game binary.
4. Memory separation holds: `wm_`-prefixed World Model tables and Internal
   Dynamics tables never share tables and never cross-write.
5. The fast loop (reflex, 10-20 Hz, no LLM) and the slow loop (strategist,
   LLM) stay separate. No layer other than the strategist calls the LLM.
6. Per-component RNG only. Global `np.random.seed` is forbidden.
7. Local Ollama only. No cloud LLM.
8. `SafetyLayer` is alive in every mode and MUST NOT be bypassed
   (AGENTS.md §4.4).
9. Every task ships tests that run in `MOCK_MODE`.
10. Every new config key goes into `config/lab.example.toml` with a comment.
11. Logs are append-only and are never truncated, rotated, or deleted on
    error.
12. **STRATEGY A:** pre-lab modules are frozen. A task that must edit a
    frozen module requires its own bounded exception recorded verbatim in
    `AGENTS.md` §5 before merge. `src/wow_bot/perception/` is a T-FIX-03
    module and is **not** frozen.
13. Phase 12 artifacts (`runs/lab/soak-1h/`, `runs/lab/aggregate-1h/`) are
    frozen records. They MUST NOT be regenerated or invalidated.
14. Any change to the `GameState` schema is a stop-and-ask event
    (AGENTS.md §4.1). The only sanctioned extension is the one already
    merged as T-FIX-03.6.

---

## Tier overview

Tier names are unchanged from the original revision of this document.

| Tier | Task | Title | Status |
|---|---|---|---|
| 0 — Static Debt (پاک‌سازی، سریع، ایزوله) | T-FIX-01 | Ruff/Mypy cleanup | DONE |
| 0 | T-FIX-02 | NON_CLAIMS reconciliation | DONE |
| 1 — Perception Contract (پایه Phase 13) | T-FIX-03 | Unified Perception Protocol | DONE |
| 1 | T-FIX-03.5 | ADR-002 GameState extension (design) | DONE |
| 1 | T-FIX-03.6 | GameState extension (implementation) | DONE |
| 1 | T-FIX-04 | Reference MockAdapter | PENDING |
| 1b — Perception Contract completion (ADR-002) | T-FIX-23 | PERCEPTION.md vision channel extensions | DONE* |
| 1b | T-FIX-21 | Adapter derivation and runtime context | DONE* |
| 1b | T-FIX-24 | Per-view projection expectations (xfail split) | PENDING |
| 1b | T-FIX-22 | perception_confidence plumbing and thresholding | PENDING |
| 1c — Perception → runtime bridge | T-FIX-20 | Async perception port and loop scheduling | PROPOSED |
| 2 — Execution Path (تیک → اکشن) | T-FIX-05 | FSM IDLE→SCANNING progression | PENDING |
| 2 | T-FIX-06 | Reflex/Watchdog lifecycle | PENDING |
| 2 | T-FIX-07 | Safety lifecycle wiring | PENDING |
| 2 | T-FIX-08 | Gameplay primitives (cast/loot/vendor/jump) | PENDING |
| 3 — Scientific Model | T-FIX-09 | Oscillator→drives coupling investigation | PENDING |
| 3 | T-FIX-10 | Spectral acceptance resolution | PENDING |
| 4 — Telemetry & Observability (صداقت داده) | T-FIX-11 | ProgressSample contract alignment | PENDING |
| 4 | T-FIX-12 | Reflex tick truth | PENDING |
| 4 | T-FIX-13 | Health counter truth | PENDING |
| 4 | T-FIX-14 | Cross-platform resource metric semantics | PENDING |
| 5 — World & Navigation | T-FIX-15 | Graph refresh after world sync | PENDING |
| 5 | T-FIX-16 | Signal routing completeness | PENDING |
| 5 | T-FIX-17 | travel_to / vendor target correctness | PENDING |
| 6 — Aggregate & Reporting | T-FIX-18 | Aggregate stability summaries | PENDING |
| 6 | T-FIX-19 | Percentile proxy semantics documentation | PENDING |
| 7 — Deferred / unowned gaps | T-FIX-25 | Strategist plan refresh and lifetime | PROPOSED |
| 7 | T-FIX-26 | Farm profile as an executed cycle plan | PROPOSED |
| 8 — Real perception port (from `hamberger`) | T-FIX-29 | Dependencies, assets, perception configuration | PROPOSED |
| 8 | T-FIX-27 | Port capture and readers | PROPOSED |
| 8 | T-FIX-28 | Real state builder → canonical `GameState` | PROPOSED |
| 8 | T-FIX-30 | World pose, target distance, reaction channels | PROPOSED |
| 8 | T-FIX-31 | UI panel channels (implements T-FIX-23) | PROPOSED |
| 8 | T-FIX-32 | `RealPerceptionBackend`, gated | PROPOSED |

### Dependency graph

```
T-FIX-23  (doc-only, independent)

T-FIX-03.6 ──┬──> T-FIX-21 ──┬──> T-FIX-24 ──> T-FIX-04
             │              └──> T-FIX-22
             └──> T-FIX-22   (also requires T-FIX-21)

T-FIX-04 ──┬──> T-FIX-05 ──┬──> T-FIX-08
           │               └──> T-FIX-25
           └──> T-FIX-20 ──> T-FIX-06 ──┬──> T-FIX-07
                                       └──> T-FIX-16

T-FIX-09 ──> T-FIX-10
T-FIX-12 ──> T-FIX-13
T-FIX-15 ──> T-FIX-17 ──> T-FIX-26

T-FIX-29 ──> T-FIX-27 ──> T-FIX-28 ─┬─> T-FIX-30 ──┐
                │                    │               ├─> T-FIX-32
                └──> T-FIX-31 ───────┘               │
                       (T-FIX-32 also needs T-FIX-04, T-FIX-20, T-FIX-22)

Independent: T-FIX-11, T-FIX-14, T-FIX-18, T-FIX-19
```

Ordering justification (from ADR-002 §Decision 6): a field's vision channel
must be documented (`T-FIX-23`) before the schema names it; the schema must
exist (`T-FIX-03.6`) before the adapter can derive from it (`T-FIX-21`);
the derivation must exist before per-view expectations can be truthful
(`T-FIX-24`); the reference adapter (`T-FIX-04`) lands against a suite
that already asserts per-view expectations; the FSM progression
(`T-FIX-05`) needs a projectable strategist view; and the runtime bridge
(`T-FIX-20`) plus lifecycle (`T-FIX-06`) need a working reference path.

---

# Tier 0 — Static Debt (پاک‌سازی، سریع، ایزوله)

## T-FIX-01 — Ruff/Mypy cleanup

**Status:** DONE (commit `5c4ae69`)
**Depends on:** none
**Deliverables:**
- Lint and typing fixes across `src/**`, `tests/**`, `scripts/**`.

**Contract:**
- `ruff check src tests scripts` reports zero diagnostics.
- `mypy src` reports zero errors.
- No behavior change beyond what the listed rules require. No blanket
  `# type: ignore`, no `noqa` that hides behavior, no test weakening, no
  NON_CLAIMS or schema edits.

**Acceptance:**
- [x] `ruff check src tests scripts` → `All checks passed!`
- [x] `mypy src` → `Success: no issues found in 115 source files`

**Out of scope:** behavior changes, schema, NON_CLAIMS, the aggregate
contract, the T12.0 scope split.

---

## T-FIX-02 — NON_CLAIMS reconciliation

**Status:** DONE (commit `ac881b5`)
**Depends on:** T-FIX-01
**Deliverables:**
- `docs/non_claims.json` (single source of truth)
- `docs/SOAK_PROTOCOL.md`
- `src/wow_bot/analysis/aggregate.py`

**Contract:**
- The four non-claim statements are aligned with `docs/non_claims.json`.
- `analysis/aggregate.py` continues to hardcode its `NON_CLAIMS` tuple and
  MUST NOT read files at import time.
- No change to the T12.0 scope split, schema definitions, or the aggregate
  runtime contract.

**Acceptance:**
- [x] `tests/test_non_claims_consistency.py` passes.

**Out of scope:** T12.0 scope split; aggregate runtime contract.

---

# Tier 1 — Perception Contract (پایه Phase 13)

## T-FIX-03 — Unified Perception Protocol

**Status:** DONE (commits `6d213bc`, `9ce5d92`)
**Depends on:** none
**Deliverables:**
- `src/wow_bot/perception/protocol.py` — `PerceptionBackend` ABC with a
  single abstract `async def snapshot(self) -> GameState`.
- `src/wow_bot/perception/views.py` — frozen dataclass views for all eight
  consumer Protocols, plus `WorldSyncEntityView` and `EnemyCastView`.
- `src/wow_bot/perception/adapter.py` — `GameStateAdapter`,
  `AdapterIncompleteError`, `fraction_to_percent`.
- `src/wow_bot/perception/__init__.py` — public exports.
- `tests/test_perception_adapter.py`
- `docs/decisions/ADR-001-perception-contract.md`

**Contract:**
- `GameStateAdapter` is a pure, frozen projection of exactly one
  `GameState` snapshot; it holds no state between calls.
- Eight projections: `to_world_sync_view`, `to_strategist_view`,
  `to_combat_view`, `to_targeting_views`, `to_reactive_view`,
  `to_flee_view`, `to_loot_view`, `to_vendor_view`.
- Fail-loud: when a target Protocol declares a field non-Optional and the
  snapshot cannot supply it, raise `AdapterIncompleteError`. Never
  fabricate a plausible scalar (no `(0,0)`, no zero heading, no synthetic
  entity ids).
- Collection-typed non-Optional fields are the one bounded deviation: an
  empty tuple means "no observations of this kind available".
- Fraction → percent conversion uses round-half-to-even (banker's) at one
  decimal place.
- No consumer Protocol is imported or inherited. The only import is a
  typing-only `FSMState` from `executor.states`, required by the
  `strategist.prompts_v2.GameStateView` Protocol.

**Acceptance:**
- [x] Per-view structural and type compatibility tests pass.
- [x] Determinism test passes (identical input → identical output).
- [x] No-mutation test passes (the input snapshot is unchanged).
- [x] `test_no_file_reads` passes (the adapter reads no files).
- [x] `test_static_ast_import_restrictions` passes.
- [x] `test_perception_backend_abc` passes.

**Out of scope:** derived fields, runtime context, runner wiring, any
`GameState` schema change.

**Note:** pass 3 proved all eight projections raise
`AdapterIncompleteError` against a strictly canonical `GameState`. That
result is what blocked `T-FIX-04` and triggered ADR-002 / `T-FIX-03.6`.

---

## T-FIX-03.5 — ADR-002: GameState extension for real perception (design)

**Status:** DONE (commit `b628ea9`)
**Depends on:** T-FIX-03
**Deliverables:**
- `docs/decisions/ADR-002-gamestate-extension.md`

**Contract (design-only; no code changes):**
- Decides Option A: extend `GameState` in place with additive `Optional`
  fields rather than introducing a parallel `PerceptionSnapshot` type.
- Field inventory: eleven new `GameState` fields (five observed today, six
  channel-pending), all `Optional`, appended after `events`.
- `EnemyInfo` is extended in place into a superset type so one entity list
  serves both `world.sync.EntityLike` and
  `combat.targeting.TargetEntityLike`.
- Unknown-value policy: `None` means "not observed"; a genuine zero is a
  real value; `perception_confidence` maps field path → confidence.
- `resource_max` and `target_is_alive` are derived, never stored.
- Proposes the STRATEGY A exception text and the migration order.

**Acceptance:**
- [x] ADR merged with status `Proposed`.
- [x] No `src/` file changed by this task.

**Out of scope:** implementation; schema edits.

---

## T-FIX-03.6 — GameState extension (implements ADR-002)

**Status:** PARTIAL (commit `49b1510`) — merged, with one residual failure
owned by T-FIX-24 / T-FIX-21.
**Depends on:** T-FIX-03.5
**Deliverables:**
- `src/wow_bot/shared/interfaces.py` — 11 additive `GameState` fields, 10
  additive `EnemyInfo` fields, the frozen `IncomingCast` dataclass, and the
  matching `to_dict` / `from_dict` round trip.
- `src/wow_bot/mocks/mock_perception.py` — population of the new fields
  from synthetic state; fields with no mock model stay `None`.
- `AGENTS.md` §5.6 — the bounded STRATEGY A exception.

**Contract:**
- Additive only. No existing field is renamed, removed, retyped,
  reordered, or given a new default. New fields are appended after
  `events`.
- `resource_max` and `target_is_alive` are NOT added (derived by the
  adapter).
- `to_dict` emits the new keys; `from_dict` reads them with `data.get(...)`
  defaults so pre-extension payloads still parse.
- `MockPerception` uses per-component RNG only.

**Acceptance:**
- [x] `tests/unit/test_interfaces.py` round-trip tests pass.
- [x] Positional construction at `tests/unit/test_review_gate.py:51` still
      passes.
- [x] `ruff` and `mypy` clean.
- [x] `tests/test_perception_adapter.py` fully green — **previously one
      failure** (`test_combat_view_structural_and_type_compatibility`);
      resolved by T-FIX-21.

**Residual (explicitly not this task's scope):** `entities` now exists with
an empty-tuple default, so `to_combat_view` stopped falling back to
`to_targeting_views()` and reported the honest empty channel. **T-FIX-21
fixed this** by falling back whenever the channel is empty. The assertion
change that T-FIX-21 required in `test_loot_view_...` is recorded under
T-FIX-21's notes; the systematic per-view restructure remains T-FIX-24's.

**Out of scope:** renaming/removing/retyping existing fields, positional
constructor order, `MetaState`/`Strategy`/`TargetInfo`, the eight consumer
Protocols, `PerceptionBackend.snapshot()`, `main.perception_loop`,
`lab/runner_v2.py`, fabricating unobservable values, NON_CLAIMS, the T12.0
scope split, the aggregate contract, or any Phase 12 artifact.

---

## T-FIX-04 — Reference MockAdapter

**Status:** PENDING — gated by T-FIX-21 and T-FIX-24.
**Depends on:** T-FIX-03.6, T-FIX-21, T-FIX-24
**Deliverables:**
- `src/wow_bot/perception/mock_backend.py` — a `MockPerceptionBackend`
  (`PerceptionBackend` subclass) that wraps `MockPerception.get_state()`
  and exposes it as `async def snapshot() -> GameState`.
- `tests/test_perception_adapter.py` — simplification of
  `_make_well_formed_state`: the `object.__setattr__` injection of fourteen
  attributes is redundant once the fields exist for real.
- A mock scenario emitting `target.hp_pct == 0.0`, so the
  `target_is_alive` derivation is exercised in both polarities.
- End-to-end tests: `snapshot → GameStateAdapter → each of the eight
  views`.

**Contract:**
- The reference path must project every view that T-FIX-21 unblocks, with
  no `xfail` marker anywhere.
- Views that remain blocked (channel-pending fields with no channel) must
  raise `AdapterIncompleteError` naming the exact unsupplied field.
- `MockPerceptionBackend` must be a genuine `PerceptionBackend` instance
  (the ABC is used for `isinstance` checks).
- The mock stays deterministic under a fixed seed and never mutates a
  snapshot after emission.

**Acceptance:**
- [ ] `isinstance(MockPerceptionBackend(...), PerceptionBackend)` is true.
- [ ] No `xfail` marker remains in `tests/test_perception_adapter.py`.
- [ ] Every one of the eight views has an asserted expected outcome.
- [ ] A complete end-to-end projection run succeeds for every unblocked
      view and raises for every blocked view.
- [ ] `pytest` is green.

**Out of scope:** real capture or vision; changing the view or adapter
contracts; wiring the backend into the lab runner (T-FIX-20).

---

# Tier 1b — Perception Contract completion (ADR-002 §Decision 6)

## T-FIX-23 — PERCEPTION.md vision channel extensions

**Status:** DONE* — implemented and verified in the working tree; not yet
committed.
**Depends on:** none
**Deliverables:**
- `docs/PERCEPTION.md`

**Contract:**
- Add the missing component rows to the vision table. ADR-002 §Decision 6
  says "four" (bag frame, XP bar, enemy cast bar, lootable-corpse
  indicator), but its own §Field provenance requires five channels, because
  `durability_fraction` needs a "Character frame" channel that Decision 6
  omits. **Five rows are added**, since the acceptance criterion below is
  that every channel-pending field has a named channel: **Bag frame**
  (`inventory_count`, `inventory_max`), **XP bar** (`level_or_xp`),
  **Character frame** (`durability_fraction`), **Enemy cast bar**
  (`incoming_casts`), and **Lootable-corpse indicator**
  (`target_is_lootable`).
- Add the matching anchor entries to the calibration list.
- Each new row states: source channel, extraction method, expected accuracy
  class, and confidence semantics — per the per-field rules ADR-002
  §"Field provenance" already specifies.
- No existing row changes. No code changes.

**Acceptance:**
- [x] Each of the six channel-pending `GameState` fields
      (`inventory_count`, `inventory_max`, `level_or_xp`,
      `durability_fraction`, `target_is_lootable`, `incoming_casts`) has a
      named channel with stated confidence semantics — §2.1 of
      `docs/PERCEPTION.md`.
- [x] The six pre-existing vision rows are byte-identical to the
      pre-task version. `git diff --unified=0` shows the table change as a
      pure insertion after the last existing row.
- [x] No `src/` file changed — `git status --porcelain -- src` is empty.
- [x] §5 calibration lists the five new anchors and marks them
      opportunistic (absent anchor → field `None`, session does not fail).
- [x] §6 records that the frozen frame corpus is absent, so no
      precision/recall target is measurable yet, and states that every new
      channel therefore reports `None` rather than a guess.

**Out of scope:** implementing any vision component; the frozen frame
corpus; changing the extraction methods already documented.

**Notes / deviations:**
- ADR-002 §Decision 6 says "four new component rows". **Five** were added,
  because `durability_fraction` needs the "Character frame" channel that
  Decision 6 omits while §Field provenance specifies it. The acceptance
  criterion (every channel-pending field owns a named channel) is what
  governs, so the ADR's count is the part that is incomplete, not the
  document.
- ADR-002 also assigns this task a precision/recall measurement on the
  frozen frame corpus `docs/PERCEPTION.md` §6 requires. That corpus does
  not exist in the repository, so the measurement cannot be performed. It
  is recorded as an open validation gate (§6) rather than being
  fabricated. Consequence: `target_is_lootable` stays `None` and `LootView`
  keeps raising until the corpus exists.

---

## T-FIX-21 — Adapter derivation and runtime context

**Status:** DONE* — implemented and verified in the working tree; not yet
committed.
**Depends on:** T-FIX-03.6
**Deliverables:**
- `src/wow_bot/perception/resource_table.py` — the class-and-level →
  `resource_max` lookup table (a game-data constant, deliberately not a
  vision channel).
- `src/wow_bot/perception/context.py` — a small runtime context object
  supplying `gcd_ready`, `spell_cooldown_ready(spell_id)`, and
  `target_has_debuff(debuff_id)`.
- `src/wow_bot/perception/adapter.py` — deriving `target_in_range`,
  `adds_count`, `target_is_alive`, `target_hp_percent`, and `resource_max`;
  accepting the runtime context.
- `src/wow_bot/perception/views.py` — replace the three
  `NotImplementedError` stubs (`CombatView.target_has_debuff`,
  `CombatView.spell_cooldown_ready`, `ReactiveView.spell_cooldown_ready`)
  with delegation to the context.
- Tests for each derivation and for the context delegation.

**Contract:**
- Each derived value's inputs are explicit and documented in the module
  docstring.
- When an input is unknown, the derived value is `None` and the projection
  raises `AdapterIncompleteError` as before. Derivation never manufactures
  a missing input.
- `target_in_range` is derived from distance versus `engage_distance_units`
  (default `30.0`); `adds_count` from the entity list versus
  `adds_threshold` (default `3`). Neither may be hardcoded: they are
  injected (constructor argument or `Settings`), and the choice must be
  documented.
- `target_is_alive` derives from `target.hp_pct`.
- No global RNG. No consumer Protocol imported or inherited.
- The three previously stub methods must not return a fabricated `False`:
  when the context cannot answer, they must fail loudly, exactly as today.

**Acceptance:**
- [x] Against a fully-populated snapshot plus injected derivation config and
      context, **four of eight** views project: `to_world_sync_view`,
      `to_targeting_views`, `to_reactive_view`, `to_flee_view`
      (`test_projection_outcome_with_fully_populated_snapshot`).
- [x] The remaining four stay fail-loud, each naming one unobserved field:
      `resource_max` (StrategistView, CombatView), `target_is_lootable`
      (LootView), `inventory_count` (VendorView).
- [x] Against a snapshot with a missing derivation input, the projection
      still raises `AdapterIncompleteError` naming that field — each
      derivation has a dedicated negative test.
- [x] Tests prove `gcd_ready`, `spell_cooldown_ready`, and
      `target_has_debuff` are answered by the injected context, not by a
      constant, and that no context yields `NotImplementedError` rather than
      a fabricated `False`.
- [x] Determinism test: identical snapshot + config + context yield identical
      views; the snapshot is not mutated.
- [x] ADR-002 unresolved questions 2 and 5 are recorded, not guessed:
      `resource_max` returns `None` (no class channel exists), and an
      unobserved per-entity `is_in_combat_with_self` makes `adds_count`
      `None`.
- [x] `ruff check src tests` and `mypy src` are clean.

**Out of scope:** the `GameState` schema; the eight consumer Protocols;
runner wiring; any new vision channel.

**Notes / decisions:**
- **`AdapterDerivationConfig.engage_distance_units` is required, not
  defaulted.** ADR-002's wording mentions "default `30.0`", but that default
  belongs to `combat.loop.CombatLoopConfig`. Defaulting it a second time in
  the adapter would let the two silently diverge, which is the duplication
  ADR-001's single-adapter design exists to prevent. Without a config,
  `target_in_range` is not derivable and the projection stays fail-loud.
- **`adds_threshold` is not consumed by the adapter.** ADR-002 §Decision 6
  lists it as a derivation input, but counting adds does not need the
  threshold: the comparison belongs to the flee evaluator. Taking an unused
  parameter would be dead surface, so it is deliberately omitted.
- **`adds_count` is now derivable, contrary to ADR-002's xfail table.**
  That table says `adds_count` stays `None` because `is_in_combat_with_self`
  "has no derivation" — but ADR-002's own Decision 3 added
  `EnemyInfo.is_in_combat_with_self`. The adapter therefore counts engaged
  entities, and returns `None` when the entity channel is empty or any
  entity's engagement is unobserved.
- **Entity channel conversion.** `to_combat_view` previously placed raw
  `EnemyInfo` objects into a `tuple[TargetView, ...]` field — a type lie a
  consumer would hit as `AttributeError`. The adapter now projects entities
  through `_entity_to_target_view`, deriving `hp_percent` and `is_alive` from
  the canonical `hp_fraction`, and raises on the first unobserved required
  field rather than dropping a candidate silently.
- **Empty entity channel now falls back to the selected target.** This is
  what makes `test_combat_view_structural_and_type_compatibility` pass again
  (it was red after T-FIX-03.6), so T-FIX-24 no longer needs to change that
  assertion.
- **One existing assertion was updated, in `tests/test_perception_adapter.py`.**
  `test_loot_view_missing_non_optional_fields_raises` expected a raise after
  deleting an injected `target_is_alive`; the value is now *derived* from
  `target.hp_pct`, so the test asserts the derivation and keeps a
  no-target branch that still raises. This is a consequence of implementing
  the derivation, not a weakening: coverage increased.

---

## T-FIX-24 — Per-view projection expectations (xfail split)

**Status:** PENDING
**Depends on:** T-FIX-21
**Deliverables:**
- `tests/test_perception_adapter.py`

**Contract:**
- The single `@pytest.mark.xfail(strict=True)` end-to-end test
  (`test_canonical_game_state_end_to_end_projection`) is replaced by one
  named test per view.
- Each unblocked view gets a test asserting it projects without raising.
- Each still-blocked view gets a test asserting the exact
  `AdapterIncompleteError` and naming the specific field that must remain
  unsupplied — so the block is documented in code rather than in a design
  document.
- The stale `len(view.entities) == 1` assertion in
  `test_combat_view_structural_and_type_compatibility` **was already
  corrected by T-FIX-21**, which made an empty entity channel fall back to
  the selected target. This task therefore only needs the per-view
  restructure and the `xfail` removal.
- No test is deleted to make the suite pass; no acceptance is weakened.

**Acceptance:**
- [ ] All eight views have a named test.
- [ ] No `xfail` marker remains.
- [ ] `pytest` is green, including the currently failing
      `test_combat_view_structural_and_type_compatibility`.
- [ ] No view's expected outcome is left implicit.

**Out of scope:** adapter behaviour (T-FIX-21); the reference backend
(T-FIX-04).

---

## T-FIX-22 — perception_confidence plumbing and thresholding

**Status:** PENDING
**Depends on:** T-FIX-03.6 (map exists), T-FIX-21 (deriving adapter)
**Deliverables:**
- `src/wow_bot/mocks/mock_perception.py` — populate
  `perception_confidence` from the component RNG, and deliberately emit a
  low-confidence field as `None` on some frames so the fail-loud path
  stays covered.
- `src/wow_bot/perception/adapter.py` — confidence thresholding: an
  observation whose confidence is below the configured threshold surfaces
  as `None`.
- `config/lab.example.toml` — the threshold key(s), each with a comment.
- Tests.

**Contract:**
- Absence from the map means "not attempted"; presence at low confidence
  means "seen but uncertain", which per `docs/PERCEPTION.md` still
  surfaces as `None` in the field itself.
- A low-confidence field is treated exactly like an unobserved field: on a
  non-Optional view field it raises `AdapterIncompleteError`.
- A genuine observed zero (e.g. `inventory_count == 0` with good
  confidence) is NOT converted to `None`. The
  measured-zero-versus-not-measured distinction is preserved end to end.
- Thresholds are config keys with defaults, validated at load time;
  unknown or missing keys raise `ConfigError` (AGENTS.md §6.5).
- No global RNG.

**Acceptance:**
- [ ] Test: a low-confidence `inventory_count` yields `None` in
      `VendorView`, and the projection raises `AdapterIncompleteError`.
- [ ] Test: `inventory_count == 0` with good confidence stays `0` and does
      not raise.
- [ ] Test: the new config key loads, and an unknown key still raises
      `ConfigError`.
- [ ] `ruff` and `mypy` clean.

**Out of scope:** real vision confidence values; the `docs/PERCEPTION.md`
channel rows (T-FIX-23); the `GameState` schema.

---

# Tier 1c — Perception → runtime bridge

## T-FIX-20 — Async perception port and loop scheduling

**Status:** PROPOSED — not present in `LAB_PHASE_ROADMAP.md`, ADR-001, or
ADR-002. Requires ratification before implementation.
**Depends on:** T-FIX-04
**Why this exists:** the review records that `lab/runner_v2.py` accepts a
**synchronous** `game_state_source: Callable[[], object]`, while every
perception backend is **asynchronous** (`PerceptionBackend.snapshot`).
Passing an async `get_state` to the runner produces a coroutine, not a
state. The same review requires "a scheduling design that preserves the
fast loop and allows cancellation", because `time.sleep` and navigation
calls currently block the async loop. Without this task, a real
`RealPerception` cannot be dropped into the lab pipeline at all — which is
precisely the merge the whole roadmap exists to enable.

**Deliverables:**
- `src/wow_bot/perception/port.py` — an explicit bridge that owns the
  latest snapshot: a background `snapshot()` pump producing a
  latest-value holder, plus a synchronous `sample()` accessor for the
  runner's existing `Callable[[], object]` slot.
- Bounded change to `src/wow_bot/lab/runner_v2.py`: accept a
  `PerceptionBackend` (or the port) while keeping the existing synchronous
  callable path working unchanged.
- `config/lab.example.toml` — snapshot staleness bound, with a comment.
- Tests.

**Contract:**
- The port never fabricates a snapshot. A missing or stale snapshot is an
  explicit, observable condition (a typed result or a raised error), not a
  silently reused value.
- The fast loop must not be blocked by synchronous navigation or
  `time.sleep`. The design must allow cancellation of an in-flight cycle.
- Downstream layers must not branch on producer identity: the port
  delivers `GameState`, and consumers still read it only through
  `GameStateAdapter`.
- No global clock or seed. Per-component RNG only.
- Existing synchronous-source tests and behaviour remain unchanged.

**Acceptance:**
- [ ] Test: a real-shaped async backend drives the runner through the port
      and a cycle observes the backend's snapshots in order.
- [ ] Test: staleness is reported explicitly; a stale snapshot is never
      silently substituted.
- [ ] Test: cancelling a cycle returns within a bounded time even while a
      synchronous navigation or sleep call is in flight.
- [ ] Test: the pre-existing synchronous `game_state_source` path still
      works and its tests are untouched.
- [ ] No consumer or frozen signature changes beyond the bounded runner
      change, which carries its own AGENTS.md §5 exception.

**Out of scope:** real screen capture or vision; implementing
`RealPerception` (Phase 13); changing the fast/slow loop split.

---

# Tier 2 — Execution Path (تیک → اکشن)

## T-FIX-05 — FSM IDLE→SCANNING progression

**Status:** PENDING
**Depends on:** T-FIX-04
**Evidence:** the archived soak reached no action intent. `_CompositeBehavior.decide`
handles only recovery, combat, and looting, returning `None` for `IDLE`
(`lab/runner_v2.py:262`); `FSM.tick` calls the behaviour but does not
progress `IDLE` into `SCANNING` (`executor/fsm_v2.py:166`). The strategist
prompt cooldown trusts a synthetic `state.fsm_state` rather than the
runtime FSM state (`strategist/orchestrator_v2.py:266`).

**Deliverables:**
- `src/wow_bot/executor/fsm_v2.py` — `IDLE → SCANNING` progression.
- `src/wow_bot/lab/runner_v2.py` — an IDLE/SCANNING behaviour that can emit
  a target-selection intent.
- Tests.

**Contract:**
- The progression is driven by the FSM transition table
  (`executor/states.py`), not by a side channel or a special case in the
  runner. Illegal transitions must still raise.
- `SCANNING` must be able to produce at least one real intent
  (e.g. `MOVING_TO_TARGET` / `TARGETING`), so the cycle is no longer inert.
- The strategist prompt must reflect runtime FSM state, not a hardcoded
  synthetic value.
- No global RNG; the FSM stays deterministic given identical inputs.

**Acceptance:**
- [ ] Test: a cycle progresses `IDLE → SCANNING` and reaches a subsequent
      state, emitting at least one intent.
- [ ] Test: an illegal transition still raises.
- [ ] Test: the value the strategist sees for `fsm_state` tracks the
      runtime FSM state.
- [ ] Test: the pre-existing "constructs but does not advance" runner test
      is updated to assert the new, real progression rather than being
      deleted.

**Out of scope:** combat rotation; navigation algorithm changes; real
actuation; plan refresh/lifetime (T-FIX-25).

---

## T-FIX-06 — Reflex/Watchdog lifecycle

**Status:** PENDING
**Depends on:** T-FIX-20
**Evidence:** `include_reflex_loop=True` only *constructs* the loop
(`lab/runner_v2.py:564`); `run_lab_loop_async` never starts or ticks it
(`:875`), and construct-without-startup is the currently tested behaviour.
The watchdog is likewise never started or stopped. `reflex_ticks_total=0`
in the archived soak is expected and does not validate these paths.

**Deliverables:**
- `src/wow_bot/lab/runner_v2.py` — lifecycle ownership for reflex and
  watchdog: explicit start, stop, and drain.
- Tests.

**Contract:**
- The reflex loop owns a real thread or scheduler with explicit start and
  stop. Construction alone must no longer be a terminal state.
- `ReflexLoop.tick_index` is the authority for tick counts.
- Shutdown is deterministic and idempotent; stopping mid-cycle does not
  leave a live thread or a hanging sleep.
- The watchdog runs in behavioural mode once started.
- The loop's rate stays within the 10-20 Hz band in LAB_MODE.
- Both components must be startable in `MOCK_MODE` tests using the
  injected fake clock (`NullClock`) — no reliance on real wall time.

**Acceptance:**
- [ ] Test: with reflex enabled, `run_lab_loop_async` produces more than
      zero ticks and stops cleanly.
- [ ] Test: after stop, no reflex or watchdog thread is alive.
- [ ] Test: a stop issued during a blocked sleep returns within a bounded
      time.
- [ ] Test: both components start in `MOCK_MODE` with a fake clock.

**Out of scope:** reflex rule content and routing (T-FIX-16); health
counter truth (T-FIX-13); reflex tick telemetry (T-FIX-12).

---

## T-FIX-07 — Safety lifecycle wiring

**Status:** PENDING
**Depends on:** T-FIX-06
**Evidence:** the generic lab builder does not enforce `dry_run`, call
`enter_mode`, check isolation, arm the `SafetyLayer`, or install a
physical kill switch before creating the actuator
(`lab/runner_v2.py:411`, `:434`). There is no unconditional input release
on success, stop, or runtime error. `make_driver` accepts but does not use
`lab_mode` (`actuation/driver.py:124`). `RealActuator.execute` has no
independent dry-run gate (`:116`).

**Deliverables:**
- `src/wow_bot/lab/runner_v2.py` — the startup/shutdown safety sequence.
- `src/wow_bot/actuation/driver.py` — honour or explicitly document
  `lab_mode` backend restriction.
- Tests.

**Contract — the mandatory calls from AGENTS.md §4.4 are all wired:**
- `safety.check_allowlist(addr)` before any outbound game connection.
- `safety.check_isolation()` at `LAB_MODE` startup.
- `safety.is_aborted()` at the top of every actuation.
- `safety.abort(reason)` on any critical failure.
- `SafetyLayer` is armed **before** actuator construction.
- Input release is unconditional on every exit path: success, stop,
  cancel, and exception.
- `dry_run=False` in `MOCK_MODE` is still rejected.
- The existing `mode.py` gate is invoked by the lab builder rather than
  duplicated.

**Acceptance:**
- [ ] Test: `LAB_MODE` startup fails when the isolation check fails, with
      no actuator constructed.
- [ ] Test: all held keys/buttons are released on success, on stop, and on
      an injected exception.
- [ ] Test: `dry_run=False` in `MOCK_MODE` raises.
- [ ] Test: a critical failure invokes `safety.abort` with a reason.
- [ ] Test: `SafetyLayer` is armed before the actuator exists (ordering
      asserted, not assumed).

**Out of scope:** changing `SafetyLayer` semantics; real network probes;
the kill-switch key implementation.

---

## T-FIX-08 — Gameplay primitives (cast/loot/vendor/jump)

**Status:** PENDING
**Depends on:** T-FIX-05
**Evidence:** the mapper's `Intent` union is only `MoveTo | Turn`
(`actuation/mapper.py:38`). Combat casts are expressed as a zero-distance
`MoveTo` (`combat/loop.py:136`); loot and sell/repair explicitly reuse the
same no-op convention (`farm/loot.py:6`, `farm/vendor.py:6`); the recovery
jump is a zero-angle `Turn` (`executor/recovery.py:139`). Movement mapping
also picks world-axis keys without using heading (`actuation/mapper.py:139`).

**Deliverables:**
- `src/wow_bot/actuation/mapper.py` — distinct intents for cast, loot,
  vendor interaction, and jump; heading-aware movement mapping.
- `src/wow_bot/combat/loop.py`, `src/wow_bot/farm/loot.py`,
  `src/wow_bot/farm/vendor.py`, `src/wow_bot/executor/recovery.py` — emit
  the real primitives instead of the no-op conventions.
- Tests.

**Contract:**
- Each primitive is a distinct intent carrying explicit parameters (spell
  id, target id, vendor entity, jump direction). No behaviour is
  expressed as a degenerate `MoveTo` or `Turn`.
- Movement key selection accounts for heading; no blind world-axis
  mapping.
- In `MOCK_MODE` the `SimulationController` records these intents
  symbolically. In `LAB_MODE` they reach real input strictly through the
  existing driver modules under `actuation/drivers/`.
- `dry_run=False` in `MOCK_MODE` remains rejected.

**Acceptance:**
- [ ] Test: each primitive produces its own distinct intent/command.
- [ ] Test: loot and vendor no longer emit a disguised `MoveTo`.
- [ ] Test: movement key selection is heading-correct (a rotated facing
      produces different keys).
- [ ] Test: `MOCK_MODE` records all four primitives symbolically.
- [ ] No CI test requires a live game client.

**Out of scope:** rotation content and priority policy; real client
testing; cooldown modelling (T-FIX-21).

---

# Tier 3 — Scientific Model

## T-FIX-09 — Oscillator→drives coupling investigation

**Status:** PENDING
**Depends on:** none
**Evidence:** `MetaStateGenerator.step` computes the oscillator output and
stores it in `_last_oscillator_value`, but never adds it to the drive
input or output (`internal_dynamics/meta_state.py:137`). The drive vector
receives only normalized chaos, drift, and events, so in the spectral test
(no events) changing oscillator seeds cannot repair a missing coupling.
Separately, two time scales are in play: each generator step advances
Lorenz by its fixed `dt = 0.001` (`chaos.py:34`) while the drives advance
by the test `dt = 0.1`, so the collection spans ~10 units of Lorenz time
over 1000 seconds of simulated collection time.

**Deliverables:**
- `docs/reviews/OSCILLATOR_COUPLING.md` (or an equivalent decision record)
  stating the finding, the candidate couplings, and a recommendation.
- If — and only if — the decision record authorizes it: a minimal coupling
  implementation plus determinism tests.

**Contract:**
- The investigation is evidence-based and line-referenced. It must state
  whether the advertised "oscillator drives" composition is intended to
  include oscillator output at all.
- **No tuning to satisfy the spectral target.** Changing seeds, the fit
  band, or the acceptance threshold to make `T-FIX-10` pass is forbidden
  (`docs/local_validation_docs/LOCAL_VALIDATION_ROADMAP.md` §6.2-6.3).
- Deterministic components use per-component seeds only.

**Acceptance:**
- [ ] The record names the exact missing coupling with file:line references.
- [ ] It lists at least two candidate couplings and their expected spectral
      consequences.
- [ ] It states a recommendation and whether the two time scales are
      intended.
- [ ] If any code changed, determinism tests pass and `ruff`/`mypy` are
      clean.

**Out of scope:** changing the spectral target (T-FIX-10); band/seed
tuning.

---

## T-FIX-10 — Spectral acceptance resolution

**Status:** PENDING
**Depends on:** T-FIX-09
**Evidence:** `tests/integration/test_spectrum.py::test_project_spectral_acceptance`
fails. All five drives fit log-log slopes of ≈ `-6.1` with R² ≈ 0.998,
against the documented scientific target `[-1.5, -0.5]`
(`docs/ROADMAP.md:405`). Test configuration is 10,000 samples, 1,000
warmup steps, `dt = 0.1`, seed 42, fit band `0.005`–`2.5 Hz`. The
implementation records a first-order smoothing filter on the drive
response, and the advertised oscillator composition is incomplete
(T-FIX-09). `docs/local_validation_docs/LOCAL_VALIDATION_ROADMAP.md`
§6.2-6.3 prohibits rescuing an unfavourable result by changing band, seed,
or dynamics.

**Deliverables — exactly one of:**
- (a) A reviewed **model fix**, traceable to the T-FIX-09 record, that makes
  the measured slope fall inside the target, with the test unchanged; or
- (b) A documented **target revision** via an explicit ADR (and the
  corresponding `docs/ROADMAP.md` update), stating why `[-1.5, -0.5]` is
  not the right acceptance criterion for the implemented dynamics.

**Contract:**
- The resolution must not be achieved by moving the fit band, reseeding,
  widening the assertion to whatever the model currently produces without
  justification, or adding `skip`/`xfail`/flaky markers.
- Acceptance evidence is the test itself, plus the ADR or model-change
  record.

**Acceptance:**
- [ ] `tests/integration/test_spectrum.py::test_project_spectral_acceptance`
      passes, **or** the target is formally revised with a rationale
      traceable to `docs/ROADMAP.md` and the test asserts the revised
      target.
- [ ] No `skip`, `xfail`, or retry marker exists on this test.
- [ ] The resolution is recorded in `docs/ROADMAP.md` and/or a new ADR.

**Out of scope:** other analysis modules; the aggregate/reporting tiers.

---

# Tier 4 — Telemetry & Observability (صداقت داده)

## T-FIX-11 — ProgressSample contract alignment

**Status:** PENDING
**Depends on:** none
**Evidence:** the progress callback is annotated `ProgressSample`, which
carries `position` and `inventory_count`, but the consumer reads
`position_delta`/`inventory_delta` (`watchdog/metrics.py:57`). Separately,
the soak summary subtracts first-from-last for the inventory delta while
summing per-sample position deltas
(`analysis/lab_soak_v2.py:499`), so the intended cumulative-versus-
incremental contract is ambiguous.

**Deliverables:**
- `src/wow_bot/watchdog/metrics.py` — read the fields the contract actually
  defines.
- `src/wow_bot/shared/interfaces.py` — additive-only change to
  `ProgressSample` **if** deltas are the chosen contract, or an explicit
  documented adapter if absolutes are chosen.
- `scripts/lab/full_soak.py` — a progress source that actually supplies the
  agreed data.
- Tests.

**Contract:**
- Exactly one semantics is chosen and stated: either the producer computes
  deltas, or the consumer computes them from absolutes. The producer and
  consumer must agree, and disagreement must be impossible rather than
  merely unlikely.
- Any `ProgressSample` change is additive: no rename, no removal, no
  retyping, no positional reordering.
- The summary's inventory and position deltas use the **same** aggregation
  rule (both cumulative-consistent or both incremental-consistent).
- No fabricated values when the source is absent; an absent source is
  explicit.

**Acceptance:**
- [ ] Test: the sampler reads real values from a producer that supplies
      them (no silent zero).
- [ ] Test: a cumulative producer and an incremental consumer cannot
      silently disagree — the mismatch is either impossible by type or
      raises.
- [ ] Test: summary position and inventory deltas use the same rule.
- [ ] `ruff` and `mypy` clean.

**Out of scope:** health counter truth (T-FIX-13); reflex tick truth
(T-FIX-12); aggregate/reporting semantics (T-FIX-18/19).

---

## T-FIX-12 — Reflex tick truth

**Status:** PENDING
**Depends on:** T-FIX-06
**Evidence:** quiet ticks are omitted from session logging
(`reflex/loop.py:178-190`), so counting `reflex_tick` events is not a
total-tick counter and not an unbiased jitter series. The runner
fabricates health tick counts instead of reading `tick_index`
(`lab/runner_v2.py:962`).

**Deliverables:**
- `src/wow_bot/reflex/loop.py` — total-tick telemetry that is independent of
  whether a tick is "interesting".
- `src/wow_bot/lab/runner_v2.py` — report the loop's own `tick_index`
  rather than a synthesized counter.
- Tests.

**Contract:**
- The total tick count is available and correct regardless of signal or
  control activity.
- Logging volume stays bounded (for example, a periodic summary plus a
  running counter) without truncating, rotating, or deleting logs on error.
- The tick-jitter series used for timing analysis is unbiased: it must not
  be derived from a filtered event subset.
- No fabricated counters: every reported number traces to
  `ReflexLoop.tick_index` or an equivalent authority.

**Acceptance:**
- [ ] Test: N ticks with zero signals report N.
- [ ] Test: the jitter series over a synthetic schedule is unbiased (no
      systematic omission).
- [ ] Test: the runner's reported tick count equals the loop's
      `tick_index`.
- [ ] Test: log volume stays bounded over a long quiet run.

**Out of scope:** health counters (T-FIX-13); reflex rules and routing
(T-FIX-16); lifecycle (T-FIX-06).

---

## T-FIX-13 — Health counter truth

**Status:** PENDING
**Depends on:** T-FIX-12
**Evidence:** counter increments happen even when no actions or reflex ticks
occur, because the runner's health counters are fabricated from the cycle
count (`lab/runner_v2.py:961`). The cycle index makes each action signature
different, which suppresses genuine repetition detection.

**Deliverables:**
- `src/wow_bot/lab/runner_v2.py` — health counters sourced from real
  events.
- `src/wow_bot/watchdog/health.py` — if the counter definitions live there.
- Tests.

**Contract:**
- Every health counter is traceable to a real, observed event. Counters
  must not increment on cycle iteration alone.
- The loop-detector's action signature must exclude any monotonically
  changing quantity (notably the cycle index), so a genuinely repeated
  action with no state change is detectable.
- Counters agree with the session event stream: an empty event stream means
  zero action counters.
- No deletion or truncation of health evidence on error.

**Acceptance:**
- [ ] Test: a run with no actions yields zero action and health counters.
- [ ] Test: a repeated identical action with unchanged state is detected as
      repetition.
- [ ] Test: each reported counter matches the underlying event stream
      count.
- [ ] Test: the cycle index is not part of the action signature.

**Out of scope:** reflex tick counting (T-FIX-12); humanizer timing
analysis.

---

## T-FIX-14 — Cross-platform resource metric semantics

**Status:** PENDING
**Depends on:** none
**Evidence:** the POSIX sampler reports `ru_maxrss` (peak resident usage)
while the Windows sampler reports `WorkingSetSize` (current working set) —
different physical quantities compared as one series
(`analysis/lab_soak_v2.py:79`, `analysis/windows_sampler.py:115`). The CLI
also samples `app.log` (`scripts/lab/full_soak.py:443`), but the harness
never installs its file logger, so the archived log-size samples are zero
while `events.jsonl` actually grows.

**Deliverables:**
- `src/wow_bot/analysis/lab_soak_v2.py`
- `src/wow_bot/analysis/windows_sampler.py`
- `scripts/lab/full_soak.py` — install the file logger before sampling log
  size.
- Documentation of the metric definition.

**Contract:**
- The reported metric is documented, and is either (a) genuinely equivalent
  across platforms, or (b) explicitly typed and labelled per platform and
  never pooled into a single comparable series.
- The peak-versus-current distinction is stated where the value is
  reported, so a slope over the series is not silently comparing two
  different quantities.
- The sampled log file is one the harness actually writes; a zero sample
  means a zero-size file, never "never written".
- The frozen Phase 12 artifacts are **not** regenerated.
- POSIX-only skips on Windows remain skips, not failures.

**Acceptance:**
- [ ] Test: a peak-vs-current mismatch is either corrected or explicitly
      typed/labelled with a test asserting the label.
- [ ] Test: the log-size sample is non-zero once the logger writes.
- [ ] Test: the metric definition is asserted against the documentation.
- [ ] The frozen `runs/lab/aggregate-1h/aggregate_v1.json` still parses
      unchanged.

**Out of scope:** opt-in real CPU telemetry; regenerating any Phase 12
artifact; the aggregate contract.

---

# Tier 5 — World & Navigation

## T-FIX-15 — Graph refresh after world sync

**Status:** PENDING
**Depends on:** none
**Evidence:** `WorldSync` stores entities in `wm_entities_seen`, but
`WorldSummary` queries map nodes and combat history
(`world/sync.py:181`, `world/summary.py:147`). No promotion of synced
entities into mob/vendor map nodes is shown, and there is no stale-entity
expiry. Graph construction happens before subsequent world syncs, so later
observations never reach the graph. Unbounded exploration growth is also
observed.

**Deliverables:**
- `src/wow_bot/world/sync.py` — promotion of observed entities into graph
  nodes.
- `src/wow_bot/world/store.py` and `src/wow_bot/world/schema.py` — the
  queries and (if needed) a migration; all tables stay `wm_`-prefixed.
- Consumers of `nav/graph.py` updated where needed.
- Tests.

**Contract:**
- A synced entity reaches the graph on a defined, documented path, using
  the `VALID_NODE_KINDS` label the entity carries; entities with an
  unknown kind are skipped (as world sync already does), not guessed.
- A staleness/expiry policy is explicit and configurable, with a stated
  default.
- Node growth is bounded; repeated syncs of the same stable `entity_id`
  update rather than duplicate.
- World Model tables remain `wm_`-prefixed and never cross-write to
  Internal Dynamics tables.
- No promotion of entities whose `entity_id` is unstable: stability is
  keyed on `entity_id` as `WorldSync.mark_seen` already requires.

**Acceptance:**
- [ ] Test: a synced entity becomes a graph node with a valid kind.
- [ ] Test: a synced entity with an unknown kind is skipped, not
      defaulted.
- [ ] Test: a stale entity expires per the configured policy.
- [ ] Test: repeated syncs of one stable id do not duplicate nodes.
- [ ] Test: node growth stays bounded over many syncs.

**Out of scope:** the A* algorithm; navigator replanning; `entity_id`
stability policy (a perception-implementation concern, ADR-002 unresolved
question 1).

---

## T-FIX-16 — Signal routing completeness

**Status:** PENDING
**Depends on:** T-FIX-06
**Evidence:** `default_rules` handles kill/safety/timeout/focus but ignores
`position_stuck`, `position_clear`, and the `combat_*` signals emitted by
registered sources (`reflex/rules.py:9`, `combat/reactive.py:265`,
`lab/runner_v2.py:569`). `RecoverySink` and `FSMSink` both target the
bridge for recovery, but a rule must first emit the signal. Source errors
are not fail-closed.

**Deliverables:**
- `src/wow_bot/reflex/rules.py` — a defined disposition for every signal
  type the registered sources produce.
- `src/wow_bot/reflex/sources.py`, `src/wow_bot/reflex/sinks.py` — as
  needed.
- Tests.

**Contract:**
- Every signal type a registered source can emit has a rule disposition:
  either handled, or **explicitly** declared ignored. Silent dropping is
  not allowed.
- The routing is declarative and testable: the set of produced signal
  types can be enumerated and compared against the set of routed types.
- A stuck signal reaches the recovery sink.
- Source errors are fail-closed (or logged and escalated) per
  `docs/SAFETY.md`, rather than swallowed.
- The reflex loop stays LLM-free and deterministic given the same inputs
  and seed.

**Acceptance:**
- [ ] Test: enumerating every signal type produced by the registered
      sources, each has a rule disposition.
- [ ] Test: a `position_stuck` signal reaches the recovery sink.
- [ ] Test: a source failure is fail-closed (or escalated) and observable.
- [ ] Test: an unknown signal type raises or is routed to an explicit
      catch-all that is asserted.

**Out of scope:** loop scheduling and lifecycle (T-FIX-06); rule policy
tuning; combat rotation.

---

## T-FIX-17 — travel_to / vendor target correctness

**Status:** PENDING
**Depends on:** T-FIX-15
**Evidence:** `_run_cycle` only calls the orchestrator while
`prev_goal == "farm"` (`lab/runner_v2.py:740`); strategy targets are
discarded and travel goes to the origin (`:809`); vendor selection uses the
nearest vendor rather than the requested entity (`:786`).

**Deliverables:**
- `src/wow_bot/lab/runner_v2.py` — resolve the strategy's named target to a
  concrete node/entity and use it.
- `src/wow_bot/nav/navigator.py` — as needed.
- Tests.

**Contract:**
- The target named by the active `Strategy` is resolved to a concrete
  destination and actually used for travel.
- Falling back to the origin (or any other destination) must be **explicit
  and logged with a reason**, never silent.
- Vendor selection honours the requested entity when one is named.
- No LLM call outside the strategist.

**Acceptance:**
- [ ] Test: a strategy carrying a target changes the travel destination.
- [ ] Test: an unresolvable target produces an explicit, logged fallback
      (not a silent origin trip).
- [ ] Test: a named vendor entity is selected over a merely nearer one.
- [ ] `ruff` and `mypy` clean.

**Out of scope:** plan refresh and lifetime (T-FIX-25); graph refresh
(T-FIX-15); profile execution (T-FIX-26).

---

# Tier 6 — Aggregate & Reporting

## T-FIX-18 — Aggregate stability summaries

**Status:** PENDING
**Depends on:** none
**Evidence:** `analysis/aggregate.py` loads soak reports but uses only their
count in the aggregate metrics (`:431`, `:482`, `:666`); it does not
aggregate crash/RSS/log trends or humanizer PIT/KS fits. The archived
aggregate has `report_count=0` and `has_report_v2=false`, so its zero event
counters do not mean the underlying event stream was empty.

**Deliverables:**
- `src/wow_bot/analysis/aggregate.py`
- `src/wow_bot/analysis/lab_soak_v2.py` — as needed.
- Tests with at least one real soak report fixture.

**Contract:**
- Aggregate the stability summaries the protocol promises: crash trends,
  RSS trends, log trends, and humanizer PIT/KS fits.
- **Do not change the T12.0 scope split or the `NON_CLAIMS` tuple.**
- **Do not regenerate or invalidate `runs/lab/aggregate-1h/aggregate_v1.json`.**
- Schema definitions (field names, types, defaults) in
  `reporting/schema_v2.py`, `analysis/lab_soak_v2.py`, and
  `analysis/aggregate.py` stay untouched unless a separate AGENTS.md §5
  exception is recorded first (per the T-FIX-01 precedent).
- Zero counts must be distinguishable from "no input reports".

**Acceptance:**
- [ ] Test: with ≥1 soak report, crash/RSS/log trends and humanizer fits
      are summarized with the correct input report count.
- [ ] Test: the frozen `aggregate_v1.json` still parses unchanged.
- [ ] Test: `report_count=0` is reported distinctly from a populated
      aggregate.
- [ ] `tests/test_non_claims_consistency.py` still passes.
- [ ] `tests/test_aggregate.py` passes.

**Out of scope:** rewriting T12.2 or T12.3; regenerating artifacts;
percentile semantics documentation (T-FIX-19).

---

## T-FIX-19 — Percentile proxy semantics documentation

**Status:** PENDING
**Depends on:** none
**Evidence:** the implementation computes weighted means of per-session
medians and maxima of per-session percentiles
(`analysis/aggregate.py:103`, `:615`). Duplicate session inputs can also add
metrics twice while session IDs are deduplicated (`:433`, `:678`).

**Deliverables:**
- A documentation page (for example `docs/REPORTING_RECONCILIATION.md` or a
  new `docs/aggregate_semantics.md`) describing what the aggregate
  percentile fields actually are, and how duplicate sessions are handled.
- Docstring updates in `src/wow_bot/analysis/aggregate.py`.
- A test asserting the documented dedup behaviour matches the code.

**Contract:**
- Consumers are told plainly that these are **proxies**, not pooled
  percentiles, and how to interpret them.
- The duplicate-session deduplication behaviour is stated exactly as
  implemented (metrics may be added twice while ids are deduplicated).
- `docs/SOAK_PROTOCOL.md` and the `NON_CLAIMS` wording are **protected** —
  do not change them; add a separate document instead.
- No schema or aggregate runtime-contract change.

**Acceptance:**
- [ ] The document distinguishes proxy values from pooled percentiles with
      concrete field names.
- [ ] Test: the documented duplicate-session behaviour matches the code
      (or the behaviour is fixed to match the documentation).
- [ ] `docs/non_claims.json`, `docs/SOAK_PROTOCOL.md`, and the `NON_CLAIMS`
      tuple are unchanged.

**Out of scope:** changing the aggregate runtime contract; T12 artifacts;
aggregate stability summaries (T-FIX-18).

---

# Tier 7 — Deferred / unowned gaps

These two findings are documented in the readiness review but were not
assigned to any task in the original roadmap or in ADR-002. They are
promoted to explicit tasks here so that no B-class finding is left
ownerless. Both are `PROPOSED` and require ratification.

## T-FIX-25 — Strategist plan refresh and lifetime

**Status:** PROPOSED
**Depends on:** T-FIX-05
**Evidence:** `_run_cycle` only calls the orchestrator while
`prev_goal == "farm"` (`lab/runner_v2.py:740`), so a successful
`explore`/`herb`/`grind` goal persists indefinitely with no expiry and no
replanning. `Strategy.valid_until` exists on the contract but is not
enforced in the lab loop.

**Deliverables:**
- `src/wow_bot/lab/runner_v2.py` — request a strategy when it is absent or
  expired.
- `src/wow_bot/strategist/orchestrator_v2.py` — as needed.
- Tests.

**Contract:**
- `Strategy.valid_until` is honoured: an expired or absent strategy
  triggers a new planning request; a valid one does not.
- The `farm` sentinel no longer gates strategic refresh.
- No LLM call outside the strategist; the prompt cooldown still applies.
- Planning requests are logged with the prompt hash (AGENTS.md §6.3).

**Acceptance:**
- [ ] Test: an expired strategy triggers a new planning request.
- [ ] Test: a valid strategy is not re-requested before `valid_until`.
- [ ] Test: the `farm` sentinel is not required for refresh.
- [ ] Test: the cooldown still prevents a request storm.

**Out of scope:** prompt or vocabulary content; cooldown policy values;
travel destination resolution (T-FIX-17).

---

## T-FIX-26 — Farm profile as an executed cycle plan

**Status:** PROPOSED
**Depends on:** T-FIX-17
**Evidence:** the builder stores the loaded profile
(`lab/runner_v2.py:643`), but `_run_cycle` does not select its configured
nodes or routes and does not implement its stop-after-cycles policy. The
iteration counter measures loop passes, not farm outcomes.

**Deliverables:**
- `src/wow_bot/lab/runner_v2.py` — drive the cycle from the loaded profile.
- `src/wow_bot/farm/profile.py` — as needed.
- Tests.

**Contract:**
- The executed plan is derived from the loaded profile's configured nodes
  and routes, not from an implicit origin/nearest heuristic.
- The stop-after-cycles policy is enforced against **farm outcomes**
  (completed cycles), not against loop passes.
- The reported iteration counter reflects farm outcomes.
- Profile validation stays strict: an invalid profile fails at load, not
  mid-run.

**Acceptance:**
- [ ] Test: a profile's configured nodes/routes determine the cycle.
- [ ] Test: stop-after-cycles halts the loop at the configured outcome
      count.
- [ ] Test: the iteration counter equals completed cycles, not loop passes.
- [ ] Test: an invalid profile fails at load time.

**Out of scope:** navigation algorithm changes (T-FIX-15/17); profile
schema changes beyond additive.

---

# Tier 8 — Real perception port (from `hamberger`)

Source branch `hamberger@9b968a4` is an **orphan root commit with no merge
base** to `master`, so these tasks port individual files rather than merging
the branch. Full analysis, gap tables, findings and the extraction procedure
are in `docs/lab_phase/HAMBERGER_PORT_PLAN.md`; the definitions below are
normative.

## T-FIX-29 — Dependencies, assets, and perception configuration

**Status:** PENDING (proposed)
**Depends on:** none
**Deliverables:**
- `pyproject.toml` — `mss`, `opencv-python-headless`, `pytesseract`; the YOLO
  stack (`ultralytics`/`torch`) behind an optional extra, not a base
  dependency.
- `config/lab.example.toml` — every key from plan doc §6.3, each commented.
- `.gitignore` — `*.pt`.
- `models/*.png` — the three template PNGs from `hamberger`, committed.
- Guarded imports so MOCK_MODE needs neither Tesseract nor YOLO weights.

**Contract:**
- Missing optional dependencies raise `ImportError`/`ConfigError` with a
  named remedy, never a silent pass.
- Scanner configuration is injected; `tesseract_cmd` is set at construction
  time, never at import time.
- `ultralytics` must be pinned and verified to run offline; if it cannot,
  the dependency is rejected rather than tolerated (plan doc R-6).

**Acceptance:**
- [ ] Full suite passes in MOCK_MODE with neither Tesseract nor YOLO installed.
- [ ] Every new key is in `config/lab.example.toml` with a comment; an
      unknown key still raises `ConfigError`.
- [ ] No `.pt` file is staged.
- [ ] `ruff` and `mypy` clean.

**Out of scope:** implementing any reader; wiring a backend.

---

## T-FIX-27 — Port capture and readers into `src/wow_bot/perception/`

**Status:** DONE* — implemented and verified in the working tree; not yet
committed.
**Depends on:** T-FIX-29 (also implemented in the working tree, not yet
committed).
**Deliverables** (copied from `hamberger` then reshaped — plan doc §9):
- `src/wow_bot/perception/capture.py`, `bars.py`, `combat.py`, `target.py`,
  `enemies.py`, `minimap.py`, `events.py`.
- Tests synthesising frames with NumPy.

**Not ported:** `core/action.py` (`pydirectinput` — AGENTS.md §7 forbids OS
input outside `actuation/drivers/`), `core/state.py` (conflicting
`GameState`), `main.py`, `.gitignore`, `datasets/`, `runs/`, `*.pt`.

**Contract:**
- Every reader is frame-in/frame-out; no reader owns a global capture
  singleton. Composition happens in T-FIX-28.
- All §6.3 constants come from injected config; no hardcoded absolute path
  and no module-level `pytesseract` assignment remains.
- **Frame budget:** cheap channels (bars, combat edges) run every frame; OCR
  and YOLO run on their own throttled schedule with cached results, so the
  20 Hz capture budget is met (plan doc §6.6).
- Deterministic given identical frames: no `time.time()` inside readers; the
  combat cooldown latch takes the clock as a parameter.
- YOLO import is lazy and optional.

**Implementation notes (recorded as built):**
- The frame budget is a per-channel `Throttle` (`capture.py`) whose `allow()`
  is a pure predicate and whose `record()` commits a sample, so a channel
  never consumes its budget on an early return. Each reader accepts
  `sampling_hz` and an injected `clock` (default `time.monotonic`).
- Six new config keys carry the budgets: `[bars]`, `[combat]`, `[target]`,
  `[enemies]`, `[events]`, `[minimap]` each gained `sampling_hz`
  (`inf` = every frame for the cheap channels). `[minimap].match_thresh` and
  `[target].name_refresh_frames` were added for the same reason: the two
  remaining hamberger magic numbers the port would otherwise hardcode.
- `[target]`'s four thresholds are enforced, not decorative:
  `max(match_thresh, ocr_thresh)` gates candidate search, `confirm_thresh`
  gates taking the position lock, and `ocr_thresh`/`ocr_confirm_thresh`
  gate OCR acceptance.
- `EnemyDetector` no longer ports hamberger's `click_position`: it is
  actuation, and AGENTS.md §7 keeps OS input out of this package.
- `MinimapReading.position_px` is documented as **screen pixels**, never a
  world pose (plan doc finding F-1); `degrees_to_facing_radians` is the
  single degrees→radians site.

**Acceptance:**
- [x] Per-reader unit tests over synthesised frames; none needs a live
      client, Tesseract, or YOLO weights.
- [x] `BarReader` returns a ratio in `[0,1]` for a synthetic bar of known fill.
- [x] `CombatDetector` returns `True` for a red edge strip and `False` for a
      neutral one.
- [x] Degrees→radians conversion exists in exactly one location.
- [x] OCR calls per second are asserted to stay under the configured budget.
- [x] No hardcoded absolute path under `src/wow_bot/perception/`.

**Out of scope:** building a `GameState`; wiring the runner; any actuation.

---

## T-FIX-28 — Real state builder: readers → canonical `GameState`

**Status:** PENDING (proposed)
**Depends on:** T-FIX-27
**Deliverables:**
- `src/wow_bot/perception/builder.py`
- Tests covering every unit/convention conversion below.

**Contract — all conversions explicit and tested:**

| Conversion | Rule |
|---|---|
| target HP | `int 0..100` → fraction by `/100.0` |
| facing | degrees → radians normalised to `[-π, π)`, in **one** place |
| bbox | `(x1,y1,x2,y2)` → `(x, y, w, h)` |
| events | dict → `Event(type, timestamp, data)` with the frame's monotonic timestamp |
| `TargetInfo` | built only when all four fields are present, else `target=None`; never partial, or `__post_init__` raises mid-pipeline |
| `entities`/`enemies` | one list, both populated from the same detections (ADR-002 Decision 3) |
| `kind` | mapped to `VALID_NODE_KINDS`, or the entity is omitted, never defaulted |
| `perception_confidence` | populated from YOLO `conf`, template-match scores, OCR confidence |
| `timestamp` | monotonic clock, per `docs/PERCEPTION.md` §1 |
| unobserved fields | left `None`/empty; never fabricated |

**Acceptance:**
- [ ] Emitted `GameState` passes `__post_init__` and a
      `to_dict`/`from_dict` round trip.
- [ ] One test per row above.
- [ ] A frame set with no target yields `target=None` rather than raising.
- [ ] `perception_confidence` has an entry for every field actually observed.
- [ ] No import of `wow_bot.lab` or `wow_bot.main`.

**Out of scope:** `player_z`, `position`, `distance_estimate`, `reaction` —
these are T-FIX-30 channels.

---

## T-FIX-30 — World pose, target distance, and reaction channels

**Status:** PENDING (proposed)
**Depends on:** T-FIX-28, T-FIX-23
**Critical path:** without `player_x`/`player_y`, seven of the eight views
stay blocked regardless of reader quality (plan doc finding F-1).

**Deliverables:**
- `docs/PERCEPTION.md` §2.1 rows for **World pose**, **Target distance**, and
  **Target reaction**, each with source channel, extraction method, accuracy
  class, and confidence semantics.
- The corresponding readers under `src/wow_bot/perception/`.
- `player_z`: if the method cannot observe height, record `None` as the
  permanent value and leave `WorldSyncView`/`StrategistView` blocked **by
  design**. Do not invent a z.

**Decision required before implementation** (record the choice in the row):
(A) OCR an on-screen addon coordinate frame — recommended; (B) dead reckoning
from actuation (needs T-FIX-08/T-FIX-20, drifts); (C) minimap scroll offset
against a map anchor (highest effort).

**Contract:** the method states its units and error characteristics; the
adapter and `world/sync` see world units only; a low-confidence pose leaves
`position=None`, because a wrong pose writes a wrong node into the world
model.

**Acceptance:**
- [ ] Three new `docs/PERCEPTION.md` rows, each with all four attributes.
- [ ] `WorldSyncView` projects once pose and `player_z` are supplied.
- [ ] `ReactiveView` projects once `distance_estimate` exists (so
      `target_in_range` derives).
- [ ] Low confidence leaves `position=None` and creates no world node.

**Out of scope:** the UI panel channels (T-FIX-31); navigation.

---

## T-FIX-31 — UI panel channels (implements the T-FIX-23 documentation)

**Status:** PENDING (proposed)
**Depends on:** T-FIX-27, T-FIX-23
**Deliverables:** readers for Bag frame, XP bar, Character frame, Enemy cast
bar, and Lootable-corpse indicator exactly as `docs/PERCEPTION.md` §2.1
specifies, plus their calibration anchors.

**Contract:** each reader obeys its documented confidence rule — below
threshold the field is `None`, never partial. A partial `inventory_count`
silently suppresses full-bag handling.

**Acceptance:**
- [ ] A synthesised-frame unit test per reader.
- [ ] Below-threshold behaviour matches that channel's §2.1 rule.
- [ ] The §6 precision/recall gate is measured against a corpus or recorded
      as still unmeasurable — not faked.

**Out of scope:** pose/distance (T-FIX-30); `resource_max` provenance
(ADR-002 unresolved question 2).

---

## T-FIX-32 — `RealPerceptionBackend`, behind a config gate

**Status:** PENDING (proposed)
**Depends on:** T-FIX-28, T-FIX-04, T-FIX-20, T-FIX-22
**Deliverables:**
- `src/wow_bot/perception/real_backend.py` — `RealPerceptionBackend(
  PerceptionBackend)`, composing capture → readers → builder, exposing
  `async def snapshot() -> GameState`.
- An integration test over a frozen frame corpus.

**Contract:**
- It is **never the configured default producer**; `MockPerception` remains
  the producer in `MOCK_MODE` and `LAB_MODE`, so `LAB_PHASE_ROADMAP.md`
  Global Rule 1 is not violated.
- Stale or missing frames are explicit, never silently reused.
- The fast loop is not blocked; T-FIX-20 owns scheduling.

**Acceptance:**
- [ ] `isinstance(RealPerceptionBackend(...), PerceptionBackend)` holds.
- [ ] End-to-end integration test asserts the eight projection outcomes.
- [ ] `MockPerception` is still the producer in both modes; no default config
      selects the real backend.
- [ ] Full suite green; `ruff` and `mypy` clean.

**Out of scope:** enabling it. That is Phase 13, and requires amending
`LAB_PHASE_ROADMAP.md` Global Rule 1 first — an operator decision recorded in
plan doc §6.1.

---

# Task ID reconciliation

`docs/decisions/ADR-002-gamestate-extension.md` §Decision 6 proposed task
ids that partly predate execution. The mapping below is normative for this
document.

| ADR-002 id | This document | Notes |
|---|---|---|
| `T-FIX-23` | `T-FIX-23` | Kept as-is. PERCEPTION.md vision channels, doc-only. |
| `T-FIX-02.5` | `T-FIX-03.6` | **Executed under the substituted id.** PR_DESCRIPTION.md §"Divergence from ADR-002's verbatim text" records the substitution; `AGENTS.md` §5.6 records the actual id. |
| `T-FIX-21` | `T-FIX-21` | Kept as-is. Adapter derivation and runtime context. |
| `T-FIX-24` | `T-FIX-24` | Kept as-is. xfail split into per-view expectations. |
| `T-FIX-22` | `T-FIX-22` | Kept as-is. `perception_confidence` plumbing. |
| `T-FIX-04` | `T-FIX-04` | Kept as-is. Reference MockAdapter. |
| — | `T-FIX-20` | **New here.** The async perception port and scheduling design. ADR-002 does not assign it; the readiness review Q1/Q3/Q7 requires it. |
| — | `T-FIX-25` | **New here.** Strategist plan refresh and lifetime. |
| — | `T-FIX-26` | **New here.** Farm profile as an executed cycle plan. |
| — | `T-FIX-27` … `T-FIX-32` | **New here.** The `hamberger` port. See `docs/lab_phase/HAMBERGER_PORT_PLAN.md`. `hamberger@9b968a4` is an **orphan root commit with no merge base**, so the work is a file port, not a branch merge. |

Two corrections to ADR-002's presentation, neither of which changes its
substance:

1. ADR-002 §Decision 6 numbers its steps `1..6`, then repeats `6`, `7`, `8`,
   and finishes at `9`, so its step numbers are not a stable sequence. The
   ordering is normalized in the dependency graph above.
2. ADR-002 lists `entities` as `Optional` in its preamble but specifies
   `field(default_factory=list)` in its concrete field list. T-FIX-03.6
   chose `tuple[EnemyInfo, ...]` with an empty factory and documented the
   divergence. Nothing in the remaining tasks depends on that choice.

# Traceability matrix

Every B-class finding from
`docs/reviews/PRE_PHASE_13_REVIEW.md` maps to exactly one owning task, or
is explicitly recorded as already handled.

| Review finding (§5 table and §4 Q-table) | Owning task |
|---|---|
| Lab state contract diverges from the frozen schema (Q1) | T-FIX-03 (done), T-FIX-03.6 (done), T-FIX-21, T-FIX-22 |
| No unified perception backend / async-vs-sync source (Q1) | T-FIX-20 |
| No ordinary IDLE-to-action progression (Q2) | T-FIX-05 |
| Reflex and watchdog construction without lifecycle (Q3) | T-FIX-06 |
| Reflex signals dropped by `default_rules` (Q3) | T-FIX-16 |
| Quiet ticks omitted from logging; fabricated tick counts (Q3) | T-FIX-12 |
| Accepted goals prevent further strategic refresh | T-FIX-25 |
| Strategy targets discarded; travel goes to origin; nearest-vendor selection | T-FIX-17 |
| Farm profile is not an executed cycle plan | T-FIX-26 |
| `ProgressSample` callback semantics incompatible; mixed delta aggregation | T-FIX-11 |
| False health and loop-detector evidence (Q6) | T-FIX-13 |
| Live safety/cleanup not integrated; no universal dry-run guarantee (Q8) | T-FIX-07 |
| Gameplay primitives are placeholders | T-FIX-08 |
| Recovery attempts cannot represent a full retry lifecycle | *(not assigned here; belongs to the recovery/FSM lifecycle work — T-FIX-05 updates the runner's behaviour layer, but a dedicated recovery-lifecycle task is still missing and should be raised as a separate finding)* |
| World updates do not update routes or strategist entity context | T-FIX-15 |
| Resource/log metrics do not establish cross-platform equivalence | T-FIX-14 |
| Soak harness does not implement all protocol nouns literally | *(telemetry/soak harness hardening; no dedicated task — see T-FIX-14 for the log-metric part)* |
| Aggregate omits promised stability summaries | T-FIX-18 |
| Aggregate percentiles are proxies, not pooled percentiles | T-FIX-19 |
| Non-claims wording differed across protected files | T-FIX-02 (done) |
| Static debt (Ruff/mypy) | T-FIX-01 (done) |
| Spectral model outside its stated target | T-FIX-09, T-FIX-10 |
| Oscillator output not coupled into drives | T-FIX-09 |

The two rows marked *(not assigned here)* are deliberately surfaced rather
than silently dropped. If they are to be fixed before Phase 13, they need
their own task ids ratified in this file.

---

# Phase 13 Entry Gate

`RealPerception` work (Phase 13 in `docs/lab_phase/LAB_PHASE_ROADMAP.md`)
MUST NOT start until **every** item below holds. This gate is the
definition of "ready to merge a real perception layer".

**Gate A — Baseline green**
1. `pytest` reports zero failures. At the snapshot above, two failures
   remain: `test_combat_view_structural_and_type_compatibility`
   (T-FIX-24) and `test_project_spectral_acceptance` (T-FIX-10).
2. `ruff check src tests scripts` is clean and `mypy src` is clean.
3. No new `skip`, `xfail`, `noqa`, or `type: ignore` was used to reach
   either state.

**Gate B — Perception contract complete**
4. T-FIX-23, T-FIX-21, T-FIX-24, T-FIX-22, and T-FIX-04 are merged.
5. Every one of the eight consumer views either projects successfully or
   raises `AdapterIncompleteError` naming a documented, still-unavailable
   field. Nothing is implicit.
6. Every `GameState` observed field that a real backend must fill has a
   named channel in `docs/PERCEPTION.md` with stated confidence semantics.
7. `perception_confidence` flows from producer to adapter, and a
   low-confidence observation degrades to `None` rather than a guess.

**Gate C — Perception is reachable from the pipeline**
8. T-FIX-20 is merged: an asynchronous `PerceptionBackend` can drive the
   lab runner without the fast loop being blocked, and without any
   downstream layer branching on producer identity.
9. A stale or missing snapshot is an explicit condition, never a silent
   reuse or fabrication.

**Gate D — Execution path is real**
10. T-FIX-05 is merged: a MOCK cycle progresses `IDLE → SCANNING` and
    emits at least one real intent.
11. T-FIX-06 is merged: the reflex loop and watchdog have real
    start/stop lifecycle and report their own tick counts.
12. T-FIX-07 is merged: `SafetyLayer` is armed before the actuator,
    isolation is checked at LAB_MODE startup, and input release is
    unconditional on every exit path.
13. T-FIX-08 is merged: cast, loot, vendor, and jump are distinct
    primitives, not degenerate `MoveTo`/`Turn` calls.

**Gate E — Telemetry and data are honest**
14. T-FIX-11, T-FIX-12, T-FIX-13, and T-FIX-14 are merged: no counter
    increments without a real event, and cross-platform metrics are not
    silently pooled.

**Gate F — World, navigation, and reporting**
15. T-FIX-15, T-FIX-16, and T-FIX-17 are merged.
16. T-FIX-18 and T-FIX-19 are merged, with the Phase 12 artifacts still
    frozen and `NON_CLAIMS` unchanged.

**Gate G — Process**
17. Every task that edited a frozen pre-lab module has its own bounded
    exception recorded verbatim in `AGENTS.md` §5, and that exception is
    closed after merge.
18. `T-FIX-20`, `T-FIX-25`, and `T-FIX-26` have been either ratified as
    tasks or explicitly rejected in this file.
19. No Phase 12 artifact was regenerated, and `docs/non_claims.json`,
    `docs/SOAK_PROTOCOL.md`, and the `NON_CLAIMS` tuple are byte-identical
    to their post-T-FIX-02 state.

**Gate H — Real perception only**
20. Phase 13 then implements `RealPerception(PerceptionBackend)` emitting
    the extended `GameState` per `docs/PERCEPTION.md`. Perception stays
    the only new producer; no consumer gains a producer-identity branch.

---

# Freeze policy and AGENTS.md exceptions

STRATEGY A keeps every pre-lab module frozen. Based on the deliverables
above, the following exceptions must be written into `AGENTS.md` §5 before
their tasks merge. Each must be bounded to the named files, state what it
does NOT authorize, and be closed after merge.

| Task | Frozen module(s) it must edit | Exception needed |
|---|---|---|
| T-FIX-03.6 | `shared/interfaces.py`, `mocks/mock_perception.py` | **Already recorded as §5.6 (closed).** |
| T-FIX-04 | — (new file under `perception/`) | No exception; `perception/` is not frozen. |
| T-FIX-22 | `mocks/mock_perception.py` | Yes — reopens a file §5.6 closed; a fresh §5 entry is required. |
| T-FIX-20 | `lab/runner_v2.py` | Yes. |
| T-FIX-05 | `executor/fsm_v2.py`, `lab/runner_v2.py` | Yes. |
| T-FIX-06 | `lab/runner_v2.py` | Yes. |
| T-FIX-07 | `lab/runner_v2.py`, `actuation/driver.py` | Yes. |
| T-FIX-08 | `actuation/mapper.py`, `combat/loop.py`, `farm/loot.py`, `farm/vendor.py`, `executor/recovery.py` | Yes. |
| T-FIX-11 | `watchdog/metrics.py`, `shared/interfaces.py` (additive), `scripts/lab/full_soak.py` | Yes. |
| T-FIX-12 | `reflex/loop.py`, `lab/runner_v2.py` | Yes. |
| T-FIX-13 | `lab/runner_v2.py`, `watchdog/health.py` | Yes. |
| T-FIX-14 | `analysis/lab_soak_v2.py`, `analysis/windows_sampler.py`, `scripts/lab/full_soak.py` | Yes. |
| T-FIX-15 | `world/sync.py`, `world/store.py`, `world/schema.py` | Yes. |
| T-FIX-16 | `reflex/rules.py`, `reflex/sources.py`, `reflex/sinks.py` | Yes. |
| T-FIX-17 | `lab/runner_v2.py`, `nav/navigator.py` | Yes. |
| T-FIX-18 | `analysis/aggregate.py`, `analysis/lab_soak_v2.py` | Yes — and it must not touch schema field names/types/defaults or the T12.0 split. |
| T-FIX-25 | `lab/runner_v2.py`, `strategist/orchestrator_v2.py` | Yes. |
| T-FIX-26 | `lab/runner_v2.py`, `farm/profile.py` | Yes. |

`perception/*`, `tests/*`, `docs/*`, and `config/lab.example.toml` are not
frozen and need no exception.

---

# Verification harness

Reproduce the snapshot claims in this document with:

```
.venv\Scripts\python.exe -m ruff check src tests scripts
.venv\Scripts\python.exe -m mypy src
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m pytest tests/test_perception_adapter.py -q
```

The full suite does not complete inside a 30-second command window on the
reference machine. Run it to a file and read the summary:

```
.venv\Scripts\python.exe -m pytest -q > reports/pytest_pre_real_perception.txt 2>&1
```

Environment observed at the snapshot: Windows, Python 3.12.4, repository
virtualenv at `.venv`. `uv` was not on `PATH` in that environment, so the
venv interpreter was invoked directly; both forms run the same tools.

---

# Document history

- **Original revision:** tier skeleton with `T-FIX-01` through `T-FIX-19`
  and no per-task detail.
- **This revision:** complete per-task specification (status, dependencies,
  deliverables, contract, acceptance, out-of-scope) for every task,
  including the ADR-002 tasks `T-FIX-21`–`T-FIX-24` that the original
  document omitted; the new `T-FIX-20`, `T-FIX-25`, and `T-FIX-26`
  proposals; a dependency graph; a task-ID reconciliation against ADR-002;
  a review traceability matrix; the Phase 13 entry gate; and the freeze
  and exception table.

Statuses reflect a re-verification at HEAD `49b1510`. This revision changes
documentation only: no source file, config key, test, or artifact was
modified.

**Task progress since that revision:**

- **T-FIX-23** — DONE. `docs/PERCEPTION.md` gained five vision-channel rows,
  a new §2.1 with per-channel extraction/accuracy/confidence rules, five
  calibration anchors, and an honest §6 note that the frozen frame corpus is
  absent so no precision/recall target is measurable yet. No source changed.
- **T-FIX-21** — DONE. New `perception/context.py` (runtime context seam for
  the internal quantities) and `perception/resource_table.py` (the derived
  `resource_max` lookup, with an empty default because the game data is not
  in this repository). `perception/adapter.py` now derives
  `target_in_range`, `target_is_alive`, `target_hp_percent`, `adds_count`,
  and `resource_max`, projects the entity channel into `TargetView`s, and
  accepts injected config/context. `perception/views.py`'s three
  `NotImplementedError` stubs now delegate to the context. Projectable views
  rose from one of eight to four. `tests/test_perception_derivation.py` is
  new; one stale assertion in `tests/test_perception_adapter.py` was
  corrected (see T-FIX-21 notes).
