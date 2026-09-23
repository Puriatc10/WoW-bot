# T-FIX-03.6: GameState Extension for Real Perception (implements ADR-002)

## Scope

Adds **11 additive `Optional` fields to `GameState` (5 observed + 6
channel-pending) and **10 additive `Optional` fields to `EnemyInfo**, per
ADR-002 Decisions 2, 3, and 5. The change is **additive-only**: no existing
field was renamed, removed, retyped, reordered, or given a new default. All new
`GameState` fields are appended strictly after the existing `events` field, in
the order ADR-002 lists them. A small frozen helper dataclass `IncomingCast`
was added to the same module because `incoming_casts` references it
(ADR-002 line 297-298).

Final field count: **11 GameState fields** (5 observed, 6 channel-pending) +
**10 EnemyInfo fields**, matching ADR-002's final inventory exactly.

Only 3 files changed: `src/wow_bot/shared/interfaces.py`,
`src/wow_bot/mocks/mock_perception.py`, `AGENTS.md`.

## Strategy A exception

New AGENTS.md subsection **5.6**, inserted immediately after 5.1 (existing
5.0–5.5 numbering untouched; no existing subsection renumbered):

> ### 5.6 Bounded exception: T-FIX-03.6 GameState extension (implements ADR-002)
>
> The task T-FIX-03.6 implements
> `docs/decisions/ADR-002-gamestate-extension.md` Decisions 2, 3, and 5 and
> authorizes a single, bounded exception to STRATEGY A. Its sole purpose is to
> add the eleven observed fields enumerated in ADR-002 (five observed today, six
> channel-pending) to `GameState` and `EnemyInfo` in
> `src/wow_bot/shared/interfaces.py` as `Optional` fields with `None` or
> empty-factory defaults, appended after all existing fields, together with the
> matching `to_dict`/`from_dict` round trip and the population of those fields by
> `MockPerception` in `src/wow_bot/mocks/mock_perception.py`. This is the only
> file scope of the exception: `shared/interfaces.py` and
> `mocks/mock_perception.py`. `resource_max` and `target_is_alive` are NOT added:
> they are derived by the perception adapter and never stored on `GameState`.
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
> After T-FIX-03.6 merges, this exception is closed. Any future edit to a
> module that STRATEGY A treats as FROZEN requires its own explicit
> exception recorded in this file.

**Divergence from ADR-002's verbatim text:** the ADR's proposed entry names the
task `T-FIX-02.5`; this task is `T-FIX-03.6`, so the id and the Decisions it
implements were substituted to match reality. Everything else is quoted as
proposed.

## Field inventory match

Appended to `GameState`, in this order, immediately after `events`:

| # | Field | Type | Category | MockPerception | ADR-002 line |
|---|---|---|---|---|---|
| 1 | `player_z` | `float \| None = None` | observed | `None` (mock is 2-D) | 282 |
| 2 | `target_x` | `float \| None = None` | observed | yes — pos + cos(facing)·dist | 283 |
| 3 | `target_y` | `float \| None = None` | observed | yes — pos + sin(facing)·dist | 284 |
| 4 | `entities` | `tuple[EnemyInfo, ...] = field(default_factory=tuple)` | observed | yes — `tuple(enemies)` | 285-286 |
| 5 | `perception_confidence` | `dict[str, float] = field(default_factory=dict)` | observed | `{}` — T-FIX-22 | 287 |
| 6 | `inventory_count` | `int \| None = None` | channel-pending ("Bag frame") | `None` | 291 |
| 7 | `inventory_max` | `int \| None = None` | channel-pending ("Bag frame") | `None` | 292 |
| 8 | `level_or_xp` | `float \| None = None` | channel-pending ("XP bar") | `None` | 293 |
| 9 | `durability_fraction` | `float \| None = None` | channel-pending ("Character frame") | `None` | 294-295 |
| 10 | `target_is_lootable` | `bool \| None = None` | channel-pending ("Lootable-corpse indicator") | `None` | 296 |
| 11 | `incoming_casts` | `tuple[IncomingCast, ...] = field(default_factory=tuple)` | channel-pending ("Enemy cast bar") | `()` — honest "none observed" | 297-298 |

Per the task's immutability constraint, the two collection-typed fields use
`tuple[...]` (with an empty-factory default, matching ADR-002's "empty factory"
wording) rather than `list[...] | None`.

`resource_max` and `target_is_alive` are deliberately **not** added — ADR-002
removed both from the schema in pass 2 (lines 300-302) as adapter-derived.

## EnemyInfo extension

Extended in place into a superset satisfying both `world.sync.EntityLike` and
`combat.targeting.TargetEntityLike` simultaneously, per ADR-002 Decision 3
(lines 181-186). Existing `bbox`, `confidence`, `distance_estimate` are
unchanged and first; the 10 new fields are appended after them:

| Field | Type | Category |
|---|---|---|
| `entity_id` | `str \| None` | assigned by perception, stable per mob |
| `kind` | `str \| None` | observed; must be in `VALID_NODE_KINDS` |
| `x` | `float \| None` | observed (world-space) |
| `y` | `float \| None` | observed (world-space) |
| `z` | `float \| None` | observed (world-space) |
| `hp_fraction` | `float \| None` | observed |
| `threat` | `float \| None` | observed |
| `is_attackable` | `bool \| None` | observed |
| `is_alive` | `bool \| None` | observed |
| `is_in_combat_with_self` | `bool \| None` | observed |

One canonical entity list (`entities`) now serves both views, so the two
consumer channels cannot drift, which was ADR-002's reason for rejecting
duplicate lists.

## Constructor audit

`GameState(` construction sites (via `grep -rn "GameState(" src tests scripts`):

| File | Line | Binding | Changed? |
|---|---|---|---|
| `src/wow_bot/mocks/mock_perception.py` | 313 | keyword | updated (in-scope; new fields added) |
| `tests/unit/test_review_gate.py` | 51 | **positional, 9 args** | **unchanged — verified binding preserved** |
| `tests/test_perception_adapter.py` | 46, 681 | keyword | unchanged (test file must not change) |
| `tests/unit/test_fsm.py` | 35, 591 | keyword | unchanged |
| `tests/unit/test_interfaces.py` | 25, 84, 157 | keyword | unchanged |
| `tests/unit/test_main.py` | 108 | keyword | unchanged |
| `tests/unit/test_meta_state.py` | 25 | keyword | unchanged |
| `tests/unit/test_scenario_runner.py` | 35 | keyword | unchanged |
| `tests/integration/test_fsm_perception.py` | 98 | keyword | unchanged |
| `tests/integration/test_spectrum.py` | 116 | keyword | unchanged |

`EnemyInfo(` construction sites:

| File | Line | Binding | Changed? |
|---|---|---|---|
| `src/wow_bot/mocks/mock_perception.py` | 458 | keyword | updated (in-scope) |
| `tests/unit/test_interfaces.py` | 39-40, 124, 129 | keyword | unchanged |

(The other `GameState`-suffixed identifiers in the grep — `DummyGameState`,
`FakeGameState`, `SyntheticGameState` — are unrelated test doubles, not
`GameState` itself.)

The append-after-`events` rule preserved every positional binding, exactly as
ADR-002 guaranteed (line 309-312). **No test file was modified.**

## Verification

```
$ ruff check src tests scripts
All checks passed!

$ mypy src
Success: no issues found in 115 source files

$ pytest -q
2 failed, 2228 passed, 7 skipped, 1 xfailed
```

Baseline was `1 failed, 2229 passed, 7 skipped, 1 xfailed`.

**Strict xfail outcome: remained `xfail` — did NOT XPASS.** All eight canonical
projections still raise `AdapterIncompleteError` on the canonical fixture
(`player_z` → `resource_max` → … → `inventory_count`), because the six
channel-pending fields are `None` and the derived fields are still owned by the
adapter. ADR-002's xfail-impact table predicted exactly this: only
`WorldSyncView` is unblocked in principle, and it still raises here because
`player_z` is `None` in a strictly canonical state.

**The one delta is a single pre-existing adapter test, not an XPASS:**
`tests/test_perception_adapter.py::test_combat_view_structural_and_type_compatibility`
now reports `len(view.entities) == 0` instead of `1`.

Root cause, traced to frozen code that this task must not modify:
`adapter.to_combat_view` reads `getattr(self.snapshot, "entities", None)` and
treats a non-`None` value as the entity channel, only falling back to
`to_targeting_views()` when the attribute is absent. Before this task
`GameState` had no `entities` attribute at all, so that fallback ran and
produced one `TargetView`. Now that `entities` exists (empty tuple by default,
per ADR-002's "empty factory"), the fallback no longer runs for a canonical
state, and `to_combat_view` honestly reports the empty entity channel.

The two alternative defaults for `entities` were both tested and both are
worse:

- `None` default: `to_combat_view` regains its fallback, but
  `to_world_sync_view` (`tuple(getattr(..., "entities", ()))` →
  `tuple(None)`) raises `TypeError`, breaking more tests — and it contradicts
  ADR-002's stated `field(default_factory=...)` shape.
- empty tuple (chosen): one test asserts a length that no longer holds;
  every other projection is unaffected.

The ADR's own empty-tuple semantics for `entities` (ADR-001 decision 4, quoted
at ADR-002 line 518: "empty collection states 'no observations of this kind are
available' and invents no values") are preserved. This assertion belongs to
T-FIX-24 (the xfail split / per-view expectation tests) or T-FIX-21 (adapter
derivation), which own `tests/test_perception_adapter.py` and
`perception/adapter.py` respectively — both frozen to this task.

The other failure, `tests/integration/test_spectrum.py::test_project_spectral_acceptance`,
is the pre-existing baseline failure (unrelated drive PSD-slope assertion).

## Deviations

1. **`entities` default.** ADR-002 lists `entities` as `Optional` in the
   preamble (line 277-278) but specifies `field(default_factory=list)` in the
   concrete list (line 285). Per the task's immutability constraint the type is
   `tuple[EnemyInfo, ...]`; per the concrete list the default is an empty
   factory, not `None`. This follows the concrete type and keeps
   `to_world_sync_view` iterable-safe.
2. **AGENTS.md task id.** ADR-002's proposed 5.6 text names `T-FIX-02.5`; the
   actual task is `T-FIX-03.6`. The id and the Decisions reference were updated
   to match reality; all other wording is quoted verbatim.
3. **`entity_id` stability.** ADR-002 Decision 3 requires ids "stable across
   frames for the same mob". `MockPerception` respawns combat entities each
   frame and tracks no persistent mob identity, so it cannot offer true
   cross-frame stability without inventing state. It derives the id
   deterministically from the detection's own bbox rather than from an
   unrelated random, and the mock comment records this limitation. True
   stability is a perception-implementation concern (ADR-002 Unresolved
   question 1) for the real backend.
4. **`perception_confidence` population.** Left `{}` by `MockPerception`.
   ADR-002 assigns mock population and thresholding of the map to T-FIX-22
   (Decision 6 step 6); populating it here would cross that task's scope.
5. **No new test added, xfail untouched.** Per task constraints.
