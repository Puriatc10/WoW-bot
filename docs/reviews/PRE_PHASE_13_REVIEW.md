# Pre-Phase-13 Readiness Review

## 1. Executive summary

**Verdict: not ready to enable real perception as a drop-in replacement.** The Phase 12 stability PASS is accepted as a stability result, with functional acceptance deferred as specified; it does not establish that the live control path works. The archived soak exercises world synchronization and repeated rejected planning requests, but no actuator intents. This review fixes the Windows TOML fixture and the soak CLI's invalid scripted goal, with a regression that exercises the actual CLI LLM stub through the orchestrator. State-contract incompatibilities, missing FSM progression, unscheduled reflex/watchdog components, incomplete live safety wiring, and placeholder gameplay actions still require separately scoped work before live execution. No pre-lab module, GameState schema, protected document, NON_CLAIMS text, or T12.0 reporting split was changed. The final suite has **1 failed, 2204 passed, 7 skipped**; the remaining spectral acceptance failure is preserved. Ruff and mypy also retain pre-existing failures, enumerated in section 7.

Review baseline: `be97ed9fb755c8a6e450f6bb50d6fa4ecf56dba6`. Line references below refer to the reviewed working tree. The initial working tree was clean. AGENTS.md section 5 now explicitly permits this user-requested cross-task review without adding or completing roadmap tasks.

Evidence read includes the required available documents, all source files in `perception/`, `reflex/`, `strategist/`, `executor/`, and `world/`, the aggregate and soak harness, Config, the named failing tests, and relevant runtime, actuation, combat, dynamics, navigation, and validation code. Required document locations and missing documents are detailed in section 7.

Local evidence files, retained outside the tracked patch under the repository's ignored artifact directories:

- `reports/pre_phase_13_baseline_pytest.txt`: initial complete suite.
- `reports/pre_phase_13_regression_before.txt`: strengthened smoke regression failing before the scripted-goal fix.
- `reports/pre_phase_13_pytest.txt`: final complete suite.
- `reports/pre_phase_13_ruff.txt`, `reports/pre_phase_13_mypy.txt`: final static gates.
- `reports/pre_phase_13_soak_evidence.json`: event inventory and independent RSS calculation from the original archived soak.
- Original inputs: `runs/lab/soak-1h/events.jsonl`, `runs/lab/soak-1h/soak_report.json`, and `runs/lab/aggregate-1h/aggregate_v1.json`. These were not regenerated or altered.

## 2. Test failure triage

### Windows TOML fixture: six failures fixed in tests

The fixture interpolated a native Windows path into a TOML basic string. In a path beginning `C:\Users`, TOML interprets `\U` as a Unicode escape and rejects the following characters. Config correctly wraps that parser failure in `ConfigError` (`src/wow_bot/config.py:72`). This is a fixture defect, not a reason to relax configuration parsing.

Minimal fix: serialize `tmp_path.as_posix()` in `tests/test_full_soak.py:155`. Forward slashes are accepted by Windows paths and do not introduce TOML escapes; Linux path semantics are unchanged. These initially failing tests now pass:

| Test in `tests/test_full_soak.py` | Root cause / fix location |
|---|---|
| `test_run_soak_async_smoke_run` | Shared invalid TOML fixture; tests only |
| `test_run_soak_async_runner_error_writes_crash_report` | Same |
| `test_run_soak_async_empty_samples_non_crash_raises` | Same |
| `test_run_soak_async_empty_samples_crash_writes_valid_report` | Same |
| `test_run_soak_async_uncaught_exception_writes_crash_json` | Same |
| `test_run_soak_async_determinism` | Same |

Side-effect review: only the fixture's spelling of a filesystem path changes; no loader, runtime, or production configuration behavior changes. This addresses backslash escapes, not arbitrary quote characters in a TOML string. The generated pytest temporary path is the intended input. Linux execution was not available here; cross-platform suitability follows from the path representation, not a claimed Linux test run.

### Spectral acceptance: unresolved scientific/model mismatch, preserved

`tests/integration/test_spectrum.py::test_project_spectral_acceptance` still fails its intended target. The final run records:

| Drive | Fitted slope | R-squared |
|---|---:|---:|
| hunger | -6.1176 | 0.9982 |
| fatigue | -6.2234 | 0.9951 |
| curiosity | -6.0928 | 0.9989 |
| aggression | -6.0928 | 0.9989 |
| social | -6.0844 | 0.9991 |

Source: `reports/pre_phase_13_pytest.txt`, the diagnostic emitted by the unchanged test at `tests/integration/test_spectrum.py:287`. Configuration remains 10,000 samples, 1,000 warmup steps, dt 0.1 seconds, seed 42, and fitting band 0.005 to 2.5 Hz (`tests/integration/test_spectrum.py:236`).

Findings:

1. **Drive filtering is intentional.** `Drives.step` adds drift and a common chaos input, and `decay` relaxes toward baseline (`src/wow_bot/internal_dynamics/drives.py:113`, `:151`). Its default decay rate is 0.005 (`:49`). Away from clamping and events, their composition is the recurrence `d[n+1] - b = a * (d[n] - b + dt * (r + 0.001*c[n]))`, with `a = exp(-decay_rate*dt)`. Thus the response to the input is a first-order smoothing filter. `tests/unit/test_drives.py:63` explicitly requires relaxation after an event. A smoothing filter alone does **not** mathematically require a universal slope of approximately -6; the input spectrum and finite-window estimator also matter.
2. **The advertised composition is incomplete.** `MetaStateGenerator.step` computes the oscillator output and stores it in `_last_oscillator_value`, but does not add it to the drive input or output (`src/wow_bot/internal_dynamics/meta_state.py:137`). The drive vector only receives normalized chaos, drift, and events. In this test there are no events. Changing oscillator seeds cannot repair that missing coupling. This is evidence of an integration/design gap, not proof that any particular new coupling would meet the scientific target.
3. **Two time scales are used.** Each generator step calls Lorenz once (`meta_state.py:133`), using its fixed dt 0.001 (`src/wow_bot/internal_dynamics/chaos.py:34`), while drive/oscillator time advances by test dt 0.1. The full collection therefore spans 10 units of Lorenz integration time over 1,000 seconds of simulated collection time, after warmup. The code establishes this ratio; the intended scientific interpretation is not documented sufficiently to declare a replacement ratio correct.
4. **There is no evidence-based replacement fitting band.** The helper uses a positive, finite Welch PSD and fits against the specified sampling rate; its white/Brownian reference tests pass (`tests/integration/test_spectrum.py:146`). The oscillator frequencies in `src/wow_bot/internal_dynamics/oscillators.py:31` are slow, and most lie below this fit band, but that does not justify tuning the band after seeing this result, especially when their output is unused.
5. **The target remains authoritative.** `docs/ROADMAP.md:397` specifies `[-1.5, -0.5]`. The local validation roadmap separates analysis execution from scientific acceptance and prohibits rescuing an unfavorable result by changing band, seed, or dynamics (`docs/local_validation_docs/LOCAL_VALIDATION_ROADMAP.md`, sections 6.2-6.3).

Conclusion on alternatives (a), (b), and (c): the observed smoothing follows the implemented drive equations; there are real composition/time-scale concerns in the frozen implementation (a), but no demonstrated estimator correction or defensible replacement band (b), and no authorization to demote the target to an irrelevant aspiration (c). The strongest supported conclusion is **scientific target outside range with unresolved frozen-model integration concerns**. A justified future model correction belongs in `src/wow_bot/internal_dynamics/` following review; no minimal in-scope patch has been established. The test, its assertions, its band, and the target are unchanged. No skip or xfail was added.

### Additional regression for observation 1

The smoke test now uses `_FakeLlmClient` from the CLI harness rather than an independently correct test fake. It asserts `strategist_success` and `vocab_accepted`, and rejects any `vocab_rejected` event. With the old scripted `farm` response, this stronger test failed at the success assertion; the captured failure is in `reports/pre_phase_13_regression_before.txt`. The implementation fix is in `scripts/lab/full_soak.py:93`: emit the existing targetless `explore` goal. This requires neither inventing a target nor widening the vocabulary.

Side-effect review: acceptance now lets the runner leave its internal `farm` sentinel, which changes planning-event frequency. The current runner does not request another strategy once the goal becomes `explore`; this existing persistence issue is reported below. The fix does not imply actions or exploration occur, and does not retroactively change the old soak. Existing test fakes remain available for the other tests.

No other pytest failures were found in the full baseline or final run. Static-gate failures are separately enumerated in section 7 rather than being represented as passing tests.

## 3. Phase 12 observations triage

Here A means a bounded defect fixed in this review, B means a Phase 13 design concern to document without implementation here, and C means an intentional MOCK behavior or an expected measurement definition. A C classification does not validate gameplay outcomes. The existing integration gaps associated with B still gate live execution.

| Observation | Classification | Evidence and disposition |
|---|---|---|
| Repeated IDLE cooldown followed by `unknown_goal: farm` | **A** | The old CLI fake emitted `farm`; `prompts_v2.ALLOWED_GOALS` excludes it (`src/wow_bot/strategist/prompts_v2.py:117`), and the guard correctly rejects it (`vocab_v2.py:246`). `CooldownConfig.min_interval_s` is 15 seconds (`cooldown_v2.py:38`). Original soak lines 3-4 show the first pair; the event inventory contains 240 of each. Fix: emit `explore` with null target and test the production fake through the orchestrator. The guard stays strict. |
| `target_seen=false` throughout | **C** | `_SyntheticGameState` supplies empty entities and no target (`scripts/lab/full_soak.py:96`). WorldSync reports true only if `target_entity_id` matches an entity in that snapshot (`src/wow_bot/world/sync.py:193`). This is expected for that source. Permanent inability to observe targets is a separate **B** concern for combat/loot; it is not a navigation-enable flag. |
| `reflex_ticks_total=0` | **C** | The supported soak disables reflex construction and supplies no progress callback (`scripts/lab/full_soak.py:198`, `:278`; `docs/SOAK_PROTOCOL.md`, procedure). No reflex tick events appear in the archived event inventory. A real source alone cannot schedule the loop; missing runtime start/stop ownership and unreliable tick reporting are **B** concerns described in Q3. |
| Alternating `player_node_created` | **C** | The synthetic source increments both coordinates on every read (`scripts/lab/full_soak.py:453`). The runner reads it once for the cycle and again for health (`runner_v2.py:721`, `:954`). Consecutive sync positions therefore differ by `(2,2)`: distance is about 2.83 units, while every second sync is about 5.66 units from the last new node. The sync reuse radius is 5 units (`world/sync.py:70`, `:134`). Alternation follows spatial deduplication. Existing nodes explain why the archived run starts with several false entries; its first event records an existing world database. No node is being toggled or deleted. Unbounded exploration growth and missing graph updates remain B concerns. |
| Positive RSS slope despite lower final RSS | **C** | Not internally inconsistent. Start/end cover the full sample series, while regression uses the final 300 samples (`analysis/lab_soak_v2.py:150`, `:469`, `:474`). The archived report starts at 141537280 bytes and ends at 78352384 bytes. Its final window runs from timestamp 284048.453 to 284352.906, with endpoint RSS 78278656 and 78352384 bytes. Independent least-squares recomputation yields exactly the stored 895226.5908454325 bytes/hour. Source: archived `soak_report.json` and `reports/pre_phase_13_soak_evidence.json`. No calculation patch is warranted. This is an expected metric definition in either mode, not evidence by itself of a leak or stability. |
| Successful actions, position delta, inventory delta all zero | **C** | These report values default to zero when `progress_source` is absent (`scripts/lab/full_soak.py:193`); the CLI never supplies it. They do not measure the synthetic coordinate/inventory changes. Separately, the original event stream really has no actuator intents/results. There is a B telemetry contract problem: the callback is annotated `ProgressSample`, which has `position` and `inventory_count`, not the `position_delta`/`inventory_delta` attributes read by the sampler (`watchdog/metrics.py:57`). The runner's health counters are also fabricated from cycle count (`runner_v2.py:961`), so copying those into research reports would be wrong. Keep the aggregate outcome fields null. |

## 4. Real Perception Readiness Checklist

### Q1. Exact backend interface and required methods

There is **no implemented RealPerception class or unified Perception protocol** in `src/wow_bot/perception/`; its only file is the placeholder `__init__.py:1`.

For the frozen pre-lab pipeline, the behavioral API is `async def get_state(self) -> wow_bot.shared.interfaces.GameState`, implemented by `MockPerception` (`src/wow_bot/mocks/mock_perception.py:301`) and awaited by `main.perception_loop` (`src/wow_bot/main.py:205`). No start/stop/capture method is required by that consumer. `RuntimeComponents.perception` is still annotated as the concrete `MockPerception` (`main.py:100`). The immutable contract requirement concerns the schema, although the actual dataclass is mutable: `GameState` has `timestamp`, `hp_pct`, `mana_pct`, `position`, `facing`, `in_combat`, `target`, `enemies`, and `events` (`shared/interfaces.py:144`). HP and mana use fractions in `[0,1]`.

The lab runner instead accepts a **synchronous** `game_state_source: Callable[[], object]` (`src/wow_bot/lab/runner_v2.py:390`). Passing an async `get_state` directly produces a coroutine, not the expected state. Its consumers require the following incompatible structural views:

| Consumer protocol | Required data/methods | Definition |
|---|---|---|
| `world.sync.GameStateLike` | `player_x`, `player_y`, `player_z`, `entities`, `target_entity_id`; each entity has `entity_id`, `kind`, `x`, `y`, `z` | `src/wow_bot/world/sync.py:23`, `:43` |
| `strategist.prompts_v2.GameStateView` | `player_x/y/z`, `self_hp_percent`, `resource`, `resource_max`, `current_target_id`, `target_hp_percent`, `inventory_count`, `level_or_xp`, `fsm_state` | `src/wow_bot/strategist/prompts_v2.py:31` |
| `combat.loop.CombatStateView` | Current target, entities, range/HP/resource fields, `gcd_ready`, self/target coordinates; `target_has_debuff(debuff_id)` and `spell_cooldown_ready(spell_id)` | `src/wow_bot/combat/loop.py:20` |
| `combat.targeting.TargetEntityLike` | `entity_id`, `distance`, `threat`, `hp_percent`, `is_attackable`, `is_alive`, `is_in_combat_with_self` | `src/wow_bot/combat/targeting.py:15` |
| `combat.reactive.ReactiveStateView` | Self HP/combat/position, current target, range, `incoming_casts`, callable `spell_cooldown_ready`; casts need caster/spell IDs, remaining time, interruptibility | `src/wow_bot/combat/reactive.py:22`, `:31` |
| `combat.flee.FleeStateView` | Self HP/position, current target, adds count, target coordinates | `src/wow_bot/combat/flee.py:31` |
| `farm.loot.LootStateView` | Target identity/alive/lootable/distance, self/target coordinates, inventory count/capacity | `src/wow_bot/farm/loot.py:38` |
| `farm.vendor.VendorStateView` | Self coordinates, inventory count, durability fraction | `src/wow_bot/farm/vendor.py:46` |

`executor.fsm_v2.GameStateLike` is only a marker protocol (`fsm_v2.py:27`); it does not validate those requirements. The lab HP views use percentages up to 100, not the frozen fraction units. The prompt builder also expects `MetaStateLike.drives` as a mapping and `memory_summary` (`prompts_v2.py:21`), whereas canonical `MetaState.drives` is a method (`shared/interfaces.py:258`). A backend conforming only to the documented GameState schema cannot currently drive this lab pipeline. Phase 13 needs an explicitly reviewed adapter/contract design and unknown-value policy; this review does not change the schema or synthesize missing observations.

### Q2. Does MOCK reach an action intent?

The archived **Phase 12 soak does not**. Its event stream has only `world_loaded`, `world_sync`, `cooldown_allowed`, `vocab_rejected`, and `lab_run_finished` events (inventory in `reports/pre_phase_13_soak_evidence.json`). The initial goal sentinel is `farm` (`runner_v2.py:897`); rejection preserves it. Nevertheless, rejection does not itself skip the FSM: `_run_cycle` reaches its final tick branch (`runner_v2.py:823`). The FSM starts in IDLE (`executor/fsm_v2.py:120`), and `_CompositeBehavior.decide` handles only recovery, combat, and looting, returning None for IDLE (`runner_v2.py:262`). `FSM.tick` calls behavior but does not progress IDLE into SCANNING (`fsm_v2.py:166`). Thus there is no intent and no action-result feedback to advance the system. The accepted `explore` response fixes vocabulary exercise but still takes this inert branch.

This is not a claim that all MOCK components are incapable of intents: `ConstantTargetBehavior` can return MoveTo/Turn (`fsm_v2.py:84`), and navigator tests execute synthetic routes through fake actuators. The frozen pre-lab main also emits symbolic idle intents (`main.py:410`) and its separate `ExecutorFSM` changes states and records safety stop commands (`executor/fsm.py:115`). Those paths do not prove the integrated lab farm loop works. In particular, `tests/test_lab_runner_v2.py:341` checks iteration count and lack of failures, not a completed farm/action cycle.

### Q3. What makes reflex tick, and will perception change it?

`ReflexLoop.tick()` performs one poll/rules/dispatch pass (`src/wow_bot/reflex/loop.py:134`). Repeated execution requires calling `start()` for its thread (`:227`) or an explicit tick scheduler/run_for (`:197`). The constructor needs a rate, sources, sinks, rules, clock, and seed; it does not require a target or a strategist success.

Phase 12 intentionally passes `include_reflex_loop=False` (`full_soak.py:278`). Even setting that true only constructs a loop in the lab builder (`runner_v2.py:564`); `run_lab_loop_async` never starts or ticks it (`:875`). Construction without startup is explicitly tested (`tests/test_lab_runner_v2.py:267`). Replacing perception cannot change either condition.

Additional B concerns: default reflex rules cover kill/safety/timeout/focus but ignore `position_stuck`, `position_clear`, and the `combat_*` signals produced by registered sources (`src/wow_bot/reflex/rules.py:9`, `src/wow_bot/combat/reactive.py:265`, `runner_v2.py:569`). Quiet ticks are omitted from session logging (`reflex/loop.py:178`), so counting logged tick events is not a total-tick counter or an unbiased jitter series. The runner fabricates health tick counts instead of reading `tick_index` (`runner_v2.py:962`). A shared clock, lifecycle ownership, complete signal routing, and genuine telemetry are required before live use.

### Q4. Emitted and accepted goals; mismatch location

The old soak CLI scripted exactly `farm`; this patch scripts `explore` (`scripts/lab/full_soak.py:93`). The general v2 orchestrator has no fixed emitted goal: it parses the injected LLM response and guards it (`strategist/orchestrator_v2.py:369`). Its prompt enumerates `farm_herbs`, `grind_humans`, `explore`, `flee`, `sell_vendor`, `repair`, and `travel_to` (`strategist/prompts_v2.py:117`). The vocabulary guard uses the same list, so the genuine prompt and accepted vocabulary are aligned.

Target rules matter as well as names: herb/grind goals require nonempty free text, explore/flee require no target, vendor/repair require entity IDs, and travel requires a numeric waypoint ID (`strategist/vocab_v2.py:100`). The old frozen strategist supports only the first four goals (`shared/interfaces.py:213`, `strategist/parser.py:155`). Internal runner labels `farm`, `loot`, `combat`, and `go_to_vendor` are dispatch labels, not additional accepted LLM goals (`runner_v2.py:740`). The observed mismatch was in the scripted soak response, not a reason to add `farm` to the guard.

### Q5. Navigation and `target_seen`

`target_seen` is an output statistic of WorldSync, not a field consumed by Navigator (`src/wow_bot/world/sync.py:193`). Searches across `src/` find it only in that module. Navigator uses graph connectivity, the supplied destination, and live position from `position_source` (`src/wow_bot/nav/navigator.py:175`, `:200`, `:225`). It can navigate to a known waypoint even with no visible selected target.

If `target_seen` stays false because no entity matches the target, combat lookup returns `target_gone`/`no_target` (`combat/loop.py:160`) and loot may have no viable target (`farm/loot.py:38`); those operations need valid observations. There is no direct navigation failure caused by the boolean itself. Separate route failures include a missing snapped goal/start, an empty or disconnected graph, and stale graph contents. The runner builds the graph only at startup (`runner_v2.py:430`), while WorldSync later adds player nodes but no graph edges (`world/sync.py:148`). The `travel_to` branch also discards the strategy target and navigates toward `(0,0)` (`runner_v2.py:808`). These are existing B integration concerns.

### Q6. Mock-only assumptions and silent misbehavior

Yes. The critical distinction is between the canonical GameState and the richer synthetic lab views; the following assumptions can hide invalid or missing real data:

- `_extract_position` falls back to `(0,0)` for an unrecognized shape, and `_extract_heading` falls back to zero and does not read canonical `facing` (`runner_v2.py:290`, `:305`). Unknown real pose must not become a plausible origin/heading.
- Missing inventory defaults to zero, and progress uses `level` in preference to `xp` while ignoring the prompt's `level_or_xp` (`runner_v2.py:833`, `:954`). Constant level can hide XP progress. Missing observations are not equivalent to empty inventory or no progress.
- The runner fabricates successful-action/reflex totals from loop iterations and inserts the cycle index into the loop-detector signature (`runner_v2.py:961`, `:983`). Those quantities are not derived from actuation or perception; real frames do not make them valid.
- The prompt cooldown trusts `state.fsm_state` rather than the runtime FSM's state (`strategist/orchestrator_v2.py:264`). A synthetic snapshot hardcodes IDLE (`full_soak.py:108`); vision cannot observe the Python controller's actual FSM state.
- The same state callback is read independently by the cycle, health, position, heading, and reactive sources (`runner_v2.py:455`, `:721`, `:954`; `combat/reactive.py:296`). The soak's shared mutable object advances on every read, independent of actions. Real observations need consistent snapshots, timestamps, freshness, and cross-thread ownership rather than inheriting that fixture behavior.
- Canonical-to-lab HP aliases need unit conversion; copying a fraction into a percent field would silently trigger inappropriate low-HP responses (`shared/interfaces.py:148`, `combat/reactive.py:34`). Other missing lab fields fail loudly, as Q1 describes.

The actual synthetic lab state also lacks some fields required for reachable combat paths, such as `gcd_ready` and `target_has_debuff`; its inert IDLE trajectory does not exercise them (`full_soak.py:96`, `combat/loop.py:33`).

### Q7. Perception exceptions: graceful degradation?

Not consistently. The frozen pre-lab `perception_loop` directly awaits `get_state()` (`main.py:214`). A backend exception terminates the task; `run_pipeline` detects it, cancels siblings, stops controller inputs, closes resources, then re-raises (`main.py:693`, `:709`, `:757`). This is coordinated termination, not frame-level retry/degradation.

In the lab loop, exceptions from the cycle's state read/world sync become `LabRunStatus.RUNTIME_ERROR` (`runner_v2.py:932`). The second state read and health observation occur **outside** that try block (`:954`), so their exceptions escape. Neither the normal stop nor that error return automatically closes the actuator or aborts safety. `full_soak.run_soak_async` writes a crash file for escaped exceptions, but its finally closes the session/world only (`full_soak.py:319`). A returned RUNTIME_ERROR produces a crash-marked soak report without necessarily producing `crash.json`.

Reflex source errors are caught and retained in `last_source_error`, with no required abort signal (`reflex/loop.py:143`); this can continue control processing after perception fails. Rules or session-write errors can instead escape its thread. WorldSync's periodic source callback also propagates exceptions (`world/sync.py:246`). Phase 13 must distinguish transient dropped frames from stale/unknown state and critical failure. Blindly catching everything and continuing would violate the fail-closed requirements in `docs/LAB_CONSTRAINTS.md`.

### Q8. Dry-run guarantee and backend replacement

The frozen `Controller(dry_run=True)` rejects false and only records commands (`src/wow_bot/executor/controller.py:40`). Swapping its perception source cannot make that controller inject OS input.

The Phase 12 harness explicitly selects `driver_name="null"` and `NullFocusBackend` (`full_soak.py:272`), so its physical-output boundary stays synthetic with this patch. However, it forcibly sets `lab_mode=True` on a configuration copy and builds the LAB actuator (`:263`). The generic lab builder does not enforce `dry_run`, call `enter_mode`, check isolation, arm SafetyLayer, or install a physical kill switch before creating the actuator (`runner_v2.py:411`, `:434`). `make_driver` accepts but does not use `lab_mode` to restrict real backends (`actuation/driver.py:124`). `RealActuator.execute` checks abort/focus but has no independent dry-run gate (`actuation/actuator.py:116`). The correct mode gate exists separately (`mode.py:24`) and is not invoked by that builder.

Therefore a perception replacement does not itself change the driver, but there is **no universal enforced dry-run guarantee for arbitrary lab-builder callers**. Safe live startup and cleanup are unresolved pre-existing integration requirements. No real backend, network isolation probe, screenshot capture, or live client was used in this review.

## 5. Pre-existing issues outside the scope of this review

These findings are reported, not repaired. B concerns below must be resolved in explicit follow-up scope before they are relied on in Phase 13; they are not reclassified as harmless because this review has a narrow patch budget.

| Finding | Evidence / implication |
|---|---|
| Requested `wowbot` versus `wow_bot` inconsistency | Not reproduced in this checkout's `tests/test_nav_telemetry.py`. Imports at lines 12-24 and AST path at line 616 correctly use `wow_bot`. The opt-in environment variable at line 602 is `WOWBOT_LAB_TELEMETRY`; it is not a Python package import. The file is unchanged. Any actual `wowbot` import in another revision would be a bug, but inventing one here would be inaccurate. |
| Lab state contract diverges from the frozen schema | Q1. `docs/PERCEPTION.md` asks for unknown values and a confidence map while the current required numeric fields cannot uniformly represent them. No adapter or schema amendment was introduced. |
| No ordinary IDLE-to-action progression | Q2. Merely ticking, accepting a goal, or returning `MAX_CYCLES_REACHED` does not prove a farm cycle completed. |
| Accepted goals prevent further strategic refresh | `_run_cycle` only calls the orchestrator while `prev_goal == "farm"` (`runner_v2.py:740`); successful explore/herb/grind goals persist without expiry/replanning. The new accepted fake exposes this existing behavior. Strategy targets are discarded; travel goes to the origin (`:809`) and vendor selection uses the nearest vendor rather than the requested entity (`:786`). |
| Farm profile is not an executed cycle plan | The builder stores the profile (`runner_v2.py:643`), but `_run_cycle` does not select its configured nodes/routes or implement its stop-after-cycles policy. Its iteration counter measures loop passes, not farm outcomes. |
| Reflex and watchdog construction without lifecycle | Q3. The lab runner does not start/stop either component; combat runs in the same cycle as synchronous strategist calls. `time.sleep` and navigation calls block the async loop (`runner_v2.py:999`; `nav/navigator.py:225`). Real perception needs a scheduling design that preserves the fast loop and allows cancellation. |
| Stuck and reactive combat signals are dropped | Registered sources emit signals omitted by `default_rules`. RecoverySink and FSMSink both target the bridge for recovery, but a rule must first emit it. Source errors are not fail-closed (Q3/Q7). |
| Live safety/cleanup not integrated | Q8. The runner creates an unarmed SafetyLayer and has no unconditional input release on success, stop, or runtime error. Fixing this is wider startup/lifecycle work, not a perception implementation or a reason to change SafetyLayer in this review. |
| Gameplay primitives are placeholders | Mapper's `Intent` is only `MoveTo | Turn` (`actuation/mapper.py:38`). Combat casts use a zero-distance MoveTo (`combat/loop.py:136`); loot and sell/repair explicitly use the same no-op convention (`farm/loot.py:6`, `farm/vendor.py:6`). Recovery jump is a zero-angle Turn (`executor/recovery.py:139`). Real observations cannot turn these into actual cast/loot/sell/jump operations. Movement mapping also chooses world-axis keys without heading (`actuation/mapper.py:139`). |
| False health and loop-detector evidence | Q6. Counter increments happen even when no actions or reflex ticks occur. The cycle index makes each action signature different, suppressing repetition detection. The one-hour lack of watchdog shutdown cannot validate these paths. |
| Recovery attempts cannot represent a full retry lifecycle | Repeated feedback while already in STUCK_RECOVERY returns STAY (`executor/feedback.py:197`), while RecoveryBehavior has its own lifetime attempt counter and requires an explicit reset (`executor/recovery.py:245`). The composite owner does not reset it. Hard-stuck and repeat-recovery behavior need integrated tests. |
| World updates do not update routes or strategist entity context | WorldSync stores entities in `wm_entities_seen`, but WorldSummary queries map nodes and combat history (`world/sync.py:181`; `world/summary.py:147`). No promotion into mob/vendor map nodes or stale-entity expiry is shown. Graph construction precedes subsequent world sync. |
| Soak progress callback has incompatible semantics | Q6 and observation 6. Also, summary inventory delta subtracts first from last while position delta sums per-sample deltas (`analysis/lab_soak_v2.py:499`); the intended cumulative-versus-incremental callback contract needs clarification before wiring real telemetry. |
| Resource/log metrics do not establish cross-platform equivalence | POSIX sampler uses `ru_maxrss` (peak resident usage), Windows uses `WorkingSetSize` (current working set): `analysis/lab_soak_v2.py:79`, `analysis/windows_sampler.py:115`. These are different quantities. The CLI samples `app.log` (`full_soak.py:443`), but this harness does not set up its file logger; the archived log-size samples are zero while events.jsonl grows. |
| Soak harness does not implement all protocol nouns literally | It uses `_SyntheticGameState`, not `mocks.MockPerception`, and the builder supplies `RealDelay`, not NullDelay (`full_soak.py:96`; `runner_v2.py:438`, `:450`). The actionless run masks the delay difference. Its exists-then-create path uses O_TRUNC (`full_soak.py:232`), which is unsafe under competing creation; no logs were altered in this review. |
| Aggregate omits promised stability summaries | `aggregate.py` loads soak reports but only uses their count in aggregate metrics (`analysis/aggregate.py:431`, `:482`, `:666`); it does not aggregate crash/RSS/log trends or humanizer PIT/KS fits. The archived aggregate has `report_count=0` and `has_report_v2=false`, so its zero event counters do not mean the underlying event stream was empty. No T12.2 rewrite was made. |
| Aggregate percentiles are proxies, not pooled percentiles | The implementation documents weighted means of per-session medians and maxima of per-session percentiles (`analysis/aggregate.py:103`, `:615`). Duplicate session inputs can also add metrics twice while session IDs are deduplicated (`:433`, `:678`). Consumers must not interpret these as exact pooled distributions. |
| Non-claims wording already differs across protected files | `analysis/aggregate.py:28` uses full-sentence NON_CLAIMS; `docs/SOAK_PROTOCOL.md` uses shorter bullet phrases. They are not verbatim identical at baseline. Both were preserved exactly, as requested. |
| Static debt and scientific acceptance remain | Section 7 enumerates the exact Ruff/mypy failures. The frozen spectral model remains outside its stated scientific target; this is not repaired through lint suppression or relaxed tests. |

## 6. Files changed by this review

Exact tracked/new deliverable list:

1. `AGENTS.md` - section 5 only: explicitly record the review-specific workflow exception before cross-task edits. If committed, the commit body must state **AGENTS.md section 5 changed**.
2. `scripts/lab/full_soak.py` - valid scripted `explore` goal.
3. `tests/test_full_soak.py` - portable TOML fixture and stronger smoke assertions using the actual CLI fake.
4. `docs/reviews/PRE_PHASE_13_REVIEW.md` - this report, including the PR description below.

No `src/` file was modified. No dependencies, config keys, skip/xfail marks, exception catches, type-ignore comments, or noqa comments were added. The existing output artifacts under `reports/` are validation evidence, not additional source deliverables. The baseline one-hour session remains unchanged.

Side effect of the AGENTS edit: the normal one-task workflow exception is explicitly limited to this named review and retains the schema/safety/frozen-module restrictions. It is not standing authorization for future cross-phase implementation.

### Prepared PR description

**Scope:** Review pre-Phase-13 readiness across the perception contracts, lab runner, reflex/FSM, strategist, world model, and reporting. Fix six Windows soak-test failures by using a portable TOML path and fix the soak CLI's rejected scripted goal. Strengthen the smoke test to verify the actual CLI response reaches vocabulary acceptance and strategist success. Preserve the unresolved spectral acceptance assertion.

**Files Changed:** `AGENTS.md`; `scripts/lab/full_soak.py`; `tests/test_full_soak.py`; `docs/reviews/PRE_PHASE_13_REVIEW.md`.

**Verdict:** No, the system is not ready for a perception-only Phase 13 enablement. Phase 12 remains a stability PASS; the live contract, FSM progression, scheduling, safety lifecycle, and gameplay primitives need separately scoped implementation and validation.

**Unresolved:** Spectral target failure; 98 existing Ruff diagnostics; 8 existing mypy errors; missing ARCHITECTURE.md and RESULTS.md; live/long-duration and Linux validation; the reported lowercase package typo is absent in this checkout. The report lists the exact evidence and limitations.

**Strategy A compliance:** No pre-lab module was touched. No source module under `src/` changed. Protected documents, NON_CLAIMS wording, and the T12.0 scope split were preserved.

**Validation:** Full suite: 1 failed, 2204 passed, 7 pre-existing skips. The single failing test is the unchanged scientific spectral acceptance gate. The stronger smoke regression failed before the CLI goal fix and passes in the final full suite. Ruff and mypy remain failing as documented in section 7. This description is prepared for review; no remote PR was opened.

## 7. Anything not determined, and exact remaining gates

### Commands and environment

Validation used Windows, Python 3.12.4, and uv 0.12.17. The restricted tool account could not launch the repository virtualenv's base Python; the installed user toolchain was then used with reviewed execution permission. The uv scripts directory was prepended to PATH for the requested commands. No dependencies were added.

| Command | Result | Evidence |
|---|---|---|
| Baseline `.venv/Scripts/python.exe -m pytest -q` | 7 failed, 2198 passed, 7 skipped | `reports/pre_phase_13_baseline_pytest.txt` |
| `uv run ruff check src tests scripts` | 98 errors, pre-existing | `reports/pre_phase_13_ruff.txt` |
| `uv run mypy src` | 8 errors in 2 files; 112 source files checked | `reports/pre_phase_13_mypy.txt` |
| `uv run pytest -q` | 1 failed, 2204 passed, 7 skipped | `reports/pre_phase_13_pytest.txt` |

The final pytest failure is exactly `tests/integration/test_spectrum.py::test_project_spectral_acceptance`, at line 287. It cannot be responsibly fixed within the freeze without either a reviewed model change or weakening the existing scientific acceptance. Section 2 explains why neither a band adjustment nor a changed target is justified.

The seven skips already existed: three POSIX resource sampler tests and one POSIX backend-selection test on Windows (`tests/test_full_soak.py:784`, `:792`, `:803`, `:818`); two Unix permission-mode tests (`tests/test_session.py:203`, `:253`); and opt-in real CPU telemetry (`tests/test_nav_telemetry.py:601`). The patch adds no skip or xfail. The final passing count includes the six repaired tests, not six new tests.

### Exact Ruff failures retained

All locations below are existing diagnostics, not findings introduced by this patch. Notation is `rule:line`; repeated positions can indicate different columns on one line. They are listed rather than automatically fixed because the requested implementation scope is failing-test fixes and classified A observations; other pre-existing issues are report-only. Some also touch explicitly frozen modules, including `main.py`. This is a scope restriction, not a claim that lint repairs are technically impossible.

| File | Diagnostics |
|---|---|
| `src/wow_bot/combat/reactive.py` | SIM102:134, SIM102:166 |
| `src/wow_bot/farm/profile.py` | I001:7, UP035:12, SIM103:80, TRY004:105, TRY004:108, TRY004:139, TRY004:141, TRY004:154, TRY004:175, TRY004:177, TRY004:179, TRY004:181, TRY004:213, TRY004:215, TRY004:217, TRY004:219, TRY004:222 |
| `src/wow_bot/humanize/imperfections.py` | TRY004:56, TRY004:107, TRY004:142 |
| `src/wow_bot/lab/runner_v2.py` | I001:20, TRY004:166, B009:293, B009:297 (two columns), B009:299 (two columns), B009:301 (two columns), B009:308, B009:310, B009:312, SIM114:759, B009:1025 |
| `src/wow_bot/main.py` | B010:564 |
| `src/wow_bot/reporting/schema_v2.py` | TRY004:55, TRY004:62, TRY004:74, TRY004:114, TRY004:142, TRY004:145, TRY004:174, TRY004:202, TRY004:205, TRY004:292, TRY004:301, TRY004:329, TRY004:332, TRY004:340, TRY004:348, TRY004:376, TRY004:379, TRY004:387, TRY004:394, TRY004:460 |
| `src/wow_bot/strategist/prompts_v2.py` | UP035:9 |
| `src/wow_bot/watchdog/loops.py` | TRY004:82, TRY004:144, TRY004:240 |
| `src/wow_bot/watchdog/shutdown.py` | TRY004:77, TRY004:79, TRY004:88, TRY004:114, TRY004:121, TRY004:198 |
| `src/wow_bot/watchdog/watchdog.py` | UP035:35, S110:462, S110:603 |
| `tests/test_astar.py` | SIM102:491 |
| `tests/test_combat_reactive.py` | UP035:7, F401:18, F401:24, F841:553, SIM102:611, SIM102:612 |
| `tests/test_farm_profile.py` | I001:3, SIM102:567 |
| `tests/test_lab_runner_v2.py` | I001:3, PLW1510:587, PLW1510:599 |
| `tests/test_prompts_v2.py` | UP035:9, SIM102:412, SIM102:413 |
| `tests/test_report_schema_v2.py` | F401:10, F401:37, F401:38, SIM102:714, SIM102:728, SIM102:729 |
| `tests/unit/test_watchdog_reconciliation.py` | I001:3, F401:9, UP035:11, F401:11, F401:13, F401:20, B009:43 |

Rule meanings as reported by Ruff: import ordering (I001), modern import location (UP035), unused imports/locals (F401/F841), unnecessary getattr/setattr (B009/B010), simplifiable branching (SIM102/SIM103/SIM114), inappropriate exception type for type checks (TRY004), subprocess check argument missing (PLW1510), and swallowed exceptions without logging (S110). Exact messages and columns are retained in the gate output.

### Exact mypy failures retained

| Location | Error / root cause |
|---|---|
| `src/wow_bot/analysis/lab_soak_v2.py:79` | Two `attr-defined` errors: Windows typing for module `resource` does not provide `getrusage` or `RUSAGE_SELF`. The runtime rejects this backend when resource is unavailable, but the import/type declaration does not give mypy a portable interface. |
| `src/wow_bot/analysis/windows_sampler.py:62`, `:63` | Callable annotation for `_get_current_process` does not include ctypes `restype`/`argtypes`. |
| `src/wow_bot/analysis/windows_sampler.py:66`, `:67` | Same for `_get_process_times`. |
| `src/wow_bot/analysis/windows_sampler.py:76`, `:77` | Same for `_get_process_memory_info`. |

These pre-existing typing defects were present before the behavioral patch and are not the source of the six TOML failures. Correct repairs would require platform-aware resource interfaces and a typed ctypes-function interface; adding blanket ignores would violate the request. They are outside the authorized failing-test/A-observation implementation scope and are explicitly left for follow-up.

### Evidence limitations and unresolved decisions

- `ARCHITECTURE.md` is absent at the root and under docs. `docs/lab_phase/RESULTS.md` and the roadmap's alternative `docs/RESULTS.md` are absent. Required files otherwise live under `docs/`, with the lab roadmap under `docs/lab_phase/` and the local validation roadmap under `docs/local_validation_docs/`. Architecture conclusions above come from code and README, not an invented missing document. The user's Phase 12 stability verdict is accepted; its unavailable write-up cannot be audited.
- The archived session is available, and its stored slope was independently reproduced. No new one-hour or live soak was run. The smoke test uses a fake clock for its simulated duration and does not establish real elapsed stability or gameplay success.
- No recorded-frame corpus or real backend was supplied. Accuracy, calibration, confidence thresholds, stale-frame policy, capture timing, target identity continuity, coordinate calibration, and the real-input lifecycle cannot be validated by these tests. Unknown values must not be fabricated to satisfy today's incompatible views.
- The low-pass recurrence and unused oscillator output are established from code; the exact allocation of the measured steep spectral slope between input smoothness, transients, and window effects was not determined. No reviewed scientific design specifies a correct replacement, and no source supports changing the target.
- The supplied `wowbot` typo and cross-file verbatim NON_CLAIMS equality are not true of the checked-out files as described. The former was reported as not reproduced; the latter is an existing discrepancy preserved because both texts are protected. Neither was silently changed.
- The prepared PR description is included in section 6. Validation applies to this working tree based on the baseline commit above; it is not represented as a new committed or published revision.
