# ADR-001: Unified Perception Contract

## Status
Accepted

## Context
The repository currently lacks a unified perception interface. The canonical `GameState` schema (`wow_bot.shared.interfaces.GameState`) is consumed exclusively by `main.perception_loop` and `MockPerception`. The lab runner and related subsystems instead rely on eight divergent structural views:
- `world.sync.GameStateLike`
- `strategist.prompts_v2.GameStateView`
- `combat.loop.CombatStateView`
- `combat.targeting.TargetEntityLike`
- `combat.reactive.ReactiveStateView`
- `combat.flee.FleeStateView`
- `farm.loot.LootStateView`
- `farm.vendor.VendorStateView`

Additionally, the lab runner currently expects a synchronous state source (`Callable[[], object]`), whereas perception backends are naturally asynchronous. A real perception backend implementing only the canonical `GameState` schema cannot drop in to drive the lab pipeline without an intermediate adaptation layer.

## Decision
1. **PerceptionBackend Abstract Base Class**: Define `PerceptionBackend` as an ABC with a single asynchronous method `async def snapshot(self) -> GameState: ...`.
2. **GameStateAdapter Projections**: Define a frozen dataclass `GameStateAdapter` that wraps one `GameState` snapshot and exposes eight projection methods corresponding to each consumer view:
   - `to_world_sync_view(self) -> WorldSyncView`
   - `to_strategist_view(self, *, fsm_state: FSMState | str) -> StrategistView`
   - `to_combat_view(self) -> CombatView`
   - `to_targeting_views(self) -> tuple[TargetView, ...]`
   - `to_reactive_view(self) -> ReactiveView`
   - `to_flee_view(self) -> FleeView`
   - `to_loot_view(self) -> LootView`
   - `to_vendor_view(self) -> VendorView`
3. **Preserve Canonical Units with Adapter Conversion**: Canonical `GameState` preserves its fractional `[0.0, 1.0]` representations for HP and mana (`hp_pct`, `mana_pct`). The adapter explicitly converts fractions to percentage values `[0.0, 100.0]` expected by lab views using round-half-to-even rounding at 1 decimal place.
4. **Type-Compatible Optionality and Fail-Loud Missing Values**:
   - The view dataclasses agree with the target Protocols on exact types and Optionality: every field that the target Protocol declares as non-Optional is declared non-Optional in the corresponding view dataclass.
   - Fields that are declared Optional (`| None`) in the target Protocols remain Optional in the views and adapter:
     - `GameStateLike.target_entity_id` (`str | None`)
     - `GameStateView.current_target_id` (`str | None`), `GameStateView.target_hp_percent` (`float | None`)
     - `CombatStateView.current_target_id` (`str | None`), `CombatStateView.target_x` (`float | None`), `CombatStateView.target_y` (`float | None`)
     - `ReactiveStateView.current_target_id` (`str | None`)
     - `FleeStateView.current_target_id` (`str | None`), `FleeStateView.target_x` (`float | None`), `FleeStateView.target_y` (`float | None`)
     - `LootStateView.target_entity_id` (`str | None`), `LootStateView.target_distance` (`float | None`), `LootStateView.target_x` (`float | None`), `LootStateView.target_y` (`float | None`), `LootStateView.inventory_max` (`int | None`)
     - `VendorStateView.durability_fraction` (`float | None`)
   - Whenever any non-Optional **scalar** field required by a target Protocol cannot be supplied from the snapshot, the adapter raises `AdapterIncompleteError`. The adapter never fabricates plausible scalar defaults (no `(0,0)` coordinate fallbacks, no zero `player_z`, no zero headings, no synthetic entity IDs).
   - Collection-typed non-Optional fields are the one deliberate, bounded deviation from the rule above: `entities` (`GameStateLike`, `CombatStateView`) and `incoming_casts` (`ReactiveStateView`) are supplied as an empty tuple when the snapshot provides no channel for them. An empty collection states "no observations of this kind are available" and invents no values, the same way `to_targeting_views` returns `()` when no target is selected. `entities` cannot be made strict within this task because a single snapshot attribute would have to satisfy both `EntityLike` (world sync) and `TargetEntityLike` (combat), whose required member sets are disjoint; making it strict requires a design change (separate per-consumer channels) that is out of scope for T-FIX-03 and is recorded under Unresolved Blockers.
5. **Decoupled Lifecycle**: This ADR does not address runner wiring, scheduling, threading, or lifecycle management; those concerns belong to subsequent tasks.

## Consequences

**Primary consequence:** against a `GameState` built strictly from the canonical schema in `wow_bot.shared.interfaces` — the only shape `MockPerception` emits today — all eight projection methods raise `AdapterIncompleteError`. The adapter is currently a *detector of `GameState` insufficiency*, not a working bridge to the lab consumers: it cannot drive any of the eight subsystems until `GameState` is extended or a supplementary data channel is designed. The precise per-view table is recorded in "The adapter cannot produce a successful projection today" under Unresolved Blockers.

- **Positive**:
  - Establishes a single, authoritative async backend contract (`PerceptionBackend`).
  - Isolates downstream consumers from changes to perception implementations: a single adapter maps canonical state into consumer expectations.
  - View dataclasses are type-compatible in the strict mypy sense with consumer Protocols, preventing runtime `TypeError` from unexpected `None` values.
  - The fail-loud behaviour makes the gap between the canonical schema and the consumer Protocols machine-checkable instead of latent: every missing non-Optional field surfaces as a named `AdapterIncompleteError` at projection time.
  - Consumers retain their existing structural contracts without requiring modifications to frozen modules.
- **Negative**:
  - The adapter must maintain eight projection methods; any change to existing consumer Protocols will require an adapter update.
- **Neutral**:
  - The adapter is not wired into any consumer, pipeline, or runner, so no existing runtime path changes. Pass 3 corrected the adapter itself so that its behaviour matches this ADR: `player_z` is no longer fabricated as `0.0` in `to_world_sync_view`/`to_strategist_view`, and `to_combat_view` no longer swallows `AdapterIncompleteError` from the targeting projection to substitute an empty entity tuple.

## Unresolved Blockers for Phase 13
1. **Canonical Schema Deficits**: The canonical `GameState` schema does not provide several fields required as non-Optional by existing consumer Protocols:
   - `GameStateLike`: `player_z` (no canonical z coordinate; the adapter raises), `entities` (no canonical `EntityLike` channel — `GameState.enemies` is `EnemyInfo`, whose members `bbox`/`confidence`/`distance_estimate` are disjoint from `entity_id`/`kind`/`x`/`y`/`z`)
   - `GameStateView`: `inventory_count`, `level_or_xp`, `resource_max`, `player_z`
   - `CombatStateView`: `target_in_range`, `gcd_ready`, `resource_max`, `target_hp_percent` (when no target is selected)
   - `TargetEntityLike`: `threat`, `is_attackable`, `is_alive`, `is_in_combat_with_self`
   - `ReactiveStateView`: `target_in_range`, `incoming_casts` (no canonical cast channel; supplied as an empty tuple today)
   - `FleeStateView`: `adds_count`
   - `LootStateView`: `target_is_alive`, `target_is_lootable`, `inventory_count`
   - `VendorStateView`: `inventory_count`
   - Target coordinates: `target_x`, `target_y`
   Until `GameState` is extended or a supplementary data channel is designed, no real perception backend producing only canonical `GameState` can drive the full lab pipeline without raising `AdapterIncompleteError`.
2. **Unsupported Callable Methods**: Two callable methods in the Combat and Reactive views (`target_has_debuff`, `spell_cooldown_ready`) cannot be answered from `GameState` and raise `NotImplementedError`. A real backend that cannot answer them will crash any code path that executes them.
3. **Lifecycle, Scheduling, and Telemetry**: Lifecycle ownership, tick scheduling, and thread management between fast reflex loops and slower perception passes are outside this ADR and remain addressed in separate roadmap tasks.
4. **Frozen Boundary**: This ADR does not authorize any change to `GameState` or to the eight existing consumer Protocols under STRATEGY A.

### The adapter cannot produce a successful projection today

The adapter is currently a detector of `GameState` insufficiency: no projection succeeds against a canonical `GameState`. Verified in pass 3 by constructing a `GameState` from only the fields defined in `wow_bot/shared/interfaces.py` (no extra attributes, no fakes, no monkeypatched attributes) and calling each of the eight projection methods. The fixture carries a `TargetInfo` target, matching the shape `MockPerception` emits during combat (`mocks/mock_perception.py`, `_make_combat_entities`):

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

Two details that do not change the headline:

- If the canonical `GameState` has no selected target (`target=None`, what `MockPerception` emits out of combat), `to_targeting_views` returns `()` instead of raising. This is an accurate empty result — there is no target to project, so no `TargetEntityLike` field is left unsupplied — not a fabricated view. As soon as a target is present, it raises on `threat`, as shown above.
- `to_world_sync_view` and `to_combat_view` still supply `entities` as an empty tuple when the snapshot provides no `entities` attribute, and `to_reactive_view` does the same for `incoming_casts`. These are collection-typed non-Optional fields; the empty tuple states "no observations available" and is the bounded deviation recorded in decision 4. The projections above still raise before or instead of relying on those defaults (`WorldSyncView` raises on `player_z`, which is checked first), so no projection succeeds.

Consequence for follow-up work: **T-FIX-04 (`MockPerceptionAdapter`) is not meaningful until `GameState` is extended or a supplementary data channel exists**, because there is nothing for a mock adapter to project successfully. A mock adapter wrapping `MockPerception` output would raise `AdapterIncompleteError` on every consumer view, exactly as this table shows.

## Alternatives considered
- **`typing.Protocol` instead of ABC for `PerceptionBackend`**: Rejected. Future backends and consumers may require `isinstance` checks and shared default behavior, which ABC supports directly.
- **Change the eight existing views to consume `GameState` directly**: Rejected. Strategy A freezes those pre-lab modules to protect validated baseline behavior.
- **Modify canonical `GameState` schema to incorporate all lab fields**: Rejected. Canonical schema is protected and shared across the entire repository. Changing it ripples through frozen components and invalidates historical reports.
- **Type-only import of `FSMState` in `perception/views.py`**: Accepted, because it is unavoidable. `strategist.prompts_v2.GameStateView` declares `fsm_state: FSMState`, so `StrategistView` must carry that exact enum type to stay structurally compatible and mypy-strict; duplicating the enum locally while the Protocol still references the real one would create two divergent sources of truth for FSM state values. The import is for typing only: `FSMState` is a `str`-`Enum` used as a value type, the perception layer overrides no runtime behaviour from `wow_bot.executor`, and no consumer Protocol is imported or inherited by the views. This is why the view modules are described as free of *inheritance* coupling rather than free of all imports.

## References
- `docs/reviews/PRE_PHASE_13_REVIEW.md`, Section 4 Q1 and Q6.
- `src/wow_bot/shared/interfaces.py`.
