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
   - Whenever any non-Optional field required by a target Protocol cannot be supplied from the snapshot, the adapter raises `AdapterIncompleteError`. The adapter never fabricates plausible defaults (no `(0,0)` coordinate fallbacks, no zero headings).
5. **Decoupled Lifecycle**: This ADR does not address runner wiring, scheduling, threading, or lifecycle management; those concerns belong to subsequent tasks.

## Consequences
- **Positive**:
  - Establishes a single, authoritative async backend contract (`PerceptionBackend`).
  - Isolates downstream consumers from changes to perception implementations: a single adapter maps canonical state into consumer expectations.
  - View dataclasses are type-compatible in the strict mypy sense with consumer Protocols, preventing runtime `TypeError` from unexpected `None` values.
  - Consumers retain their existing structural contracts without requiring modifications to frozen modules.
- **Negative**:
  - The adapter must maintain eight projection methods; any change to existing consumer Protocols will require an adapter update.
  - Calling adapter projection methods against a pure canonical `GameState` lacking extended attributes will fail loud with `AdapterIncompleteError`.
- **Neutral**:
  - No runtime behavioral changes are introduced in this task. No existing pipelines or runners are modified.

## Unresolved Blockers for Phase 13
1. **Canonical Schema Deficits**: The canonical `GameState` schema does not provide several fields required as non-Optional by existing consumer Protocols:
   - `GameStateView`: `inventory_count`, `level_or_xp`, `resource_max`, `player_z`
   - `CombatStateView`: `target_in_range`, `gcd_ready`, `resource_max`, `target_hp_percent` (when no target is selected)
   - `TargetEntityLike`: `threat`, `is_attackable`, `is_alive`, `is_in_combat_with_self`
   - `ReactiveStateView`: `target_in_range`
   - `FleeStateView`: `adds_count`
   - `LootStateView`: `target_is_alive`, `target_is_lootable`, `inventory_count`
   - `VendorStateView`: `inventory_count`
   - Target coordinates: `target_x`, `target_y`
   Until `GameState` is extended or a supplementary data channel is designed, no real perception backend producing only canonical `GameState` can drive the full lab pipeline without raising `AdapterIncompleteError`.
2. **Unsupported Callable Methods**: Two callable methods in the Combat and Reactive views (`target_has_debuff`, `spell_cooldown_ready`) cannot be answered from `GameState` and raise `NotImplementedError`. A real backend that cannot answer them will crash any code path that executes them.
3. **Lifecycle, Scheduling, and Telemetry**: Lifecycle ownership, tick scheduling, and thread management between fast reflex loops and slower perception passes are outside this ADR and remain addressed in separate roadmap tasks.
4. **Frozen Boundary**: This ADR does not authorize any change to `GameState` or to the eight existing consumer Protocols under STRATEGY A.

## Alternatives considered
- **`typing.Protocol` instead of ABC for `PerceptionBackend`**: Rejected. Future backends and consumers may require `isinstance` checks and shared default behavior, which ABC supports directly.
- **Change the eight existing views to consume `GameState` directly**: Rejected. Strategy A freezes those pre-lab modules to protect validated baseline behavior.
- **Modify canonical `GameState` schema to incorporate all lab fields**: Rejected. Canonical schema is protected and shared across the entire repository. Changing it ripples through frozen components and invalidates historical reports.

## References
- `docs/reviews/PRE_PHASE_13_REVIEW.md`, Section 4 Q1 and Q6.
- `src/wow_bot/shared/interfaces.py`.
