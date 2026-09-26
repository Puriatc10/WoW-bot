# ADR-003: World-Pose, Target-Distance, and Target-Reaction Channels

## Status

**Accepted.** This ADR records the decision T-FIX-30 requires before its
implementation starts. It is the "Decision required before implementation"
item in
`docs/lab_phase/PRE_REAL_PERCEPTION_FIX_ROADMAP.md` §T-FIX-30 and in
`docs/lab_phase/HAMBERGER_PORT_PLAN.md` §7 (Stage 3).

The decision is recorded here, restated in the three new
`docs/PERCEPTION.md` §2.1 rows, and implemented by the T-FIX-30 readers under
`src/wow_bot/perception/`.

## Context

`docs/lab_phase/HAMBERGER_PORT_PLAN.md` finding **F-1** established that the
ported `hamberger` perception code observes **no world position**: the
`MinimapTracker` returns the centre of the matched arrow *inside the minimap
ROI* — screen pixels, not world coordinates — and in the live client that
arrow is pinned to the minimap centre, so the match is nearly constant and
encodes *facing*, not *position*.

The consequence, restated by the plan doc, is that **seven of the eight
consumer views** require `player_x`/`player_y` (or `self_x`/`self_y`) as
non-Optional, so a naive port projects **0 of 8 views**. World position is the
critical path.

Three candidate sources were enumerated (plan doc §F-1, port plan §T-FIX-30):

| Candidate | Source | Assessment in the plan doc |
|---|---|---|
| **A** | On-screen coordinate frame from an addon, OCR'd | Most honest; per-frame; needs a small stable OCR region |
| **B** | Dead reckoning from actuation | Needs T-FIX-08/T-FIX-20; drifts; never self-corrects |
| **C** | Minimap scroll offset against a map anchor | Highest accuracy, highest effort; requires map imagery and a calibration step |
| **D** | Feed minimap pixels as world coordinates | **Rejected by the plan doc**: silently corrupts the world model |

Two further gaps travel with the same task (finding **F-2**):

- `TargetInfo.distance_estimate` has no source, so `_derive_target_in_range`
  has no input and `ReactiveView` / `CombatView` stay blocked.
- `TargetInfo.reaction` has no source, and `TargetInfo.__post_init__` rejects
  an empty reaction, so a target cannot be built at all without it.

## Decision

### Decision 1: World pose uses candidate **A** — OCR of an on-screen addon coordinate frame

**We choose A.** A coordinate addon (TomTom/MapCoords style) renders the
player's current zone coordinates as a short, high-contrast text line at a
fixed screen position. That line is a genuine **observation** of a world
position, in the game's own world units, and it carries no integration error.

Reasons, in order of weight:

1. **It is an observation, not an inference.** The repository's governing
   rule (ADR-002 Decision 4, `docs/PERCEPTION.md` §2.1) is that a channel
   emits what it saw and otherwise emits `None`. OCR of a rendered coordinate
   reads a value the client already computed. Dead reckoning (B) computes a
   value from *our* model of movement and therefore cannot be falsified by the
   next frame.
2. **A wrong pose is the most expensive error in the system.** `position`
   is the input to `world/sync.py`, which deduplicates player nodes at a
   `player_node_min_distance_units = 5.0` radius. A drifting pose (B) writes a
   chain of wrong nodes and corrupts the graph the navigator plans over, and it
   never self-corrects. A frame that OCRs badly emits `None` instead, which is
   recoverable by the next frame.
3. **The failure mode is bounded and visible.** The coordinate text is a
   fixed screen region; the reader either parses exactly two unambiguous
   numbers or it does not. Both the parse rule and the confidence rule are
   testable offline with a faked Tesseract, exactly like T-FIX-27's target and
   event readers.
4. **It does not pre-empt a later, better channel.** Choosing A adds one
   reader and one config section. Candidate C can replace it later without
   changing any downstream layer, because the contract is "world units or
   `None`".

**Rejected: B (dead reckoning).** It needs two unfinished tasks
(T-FIX-08 gameplay primitives, T-FIX-20 scheduling) before it can even be
written, it drifts without bound, and its output cannot be distinguished from
a correct pose by any consumer. It would be the *only* channel in the system
whose confidence is not derived from a measurement.

**Rejected: C (minimap scroll offset against a map anchor).** It is the most
accurate option in principle, but it requires map imagery, a per-zone
calibration step, and anchor identification that the repository does not
contain. It is strictly more work than A for a precision the world model does
not currently need (the sync radius is `5.0` world units). It stays available
as a future replacement.

**Rejected again: D (minimap pixels as world coordinates).** The plan doc
already rejects it; this ADR restates the rejection because it is the cheapest
wrong answer. Screen pixels inside a ~206×217 ROI would be read as world
units, and world sync would rebuild the graph around a 200-unit rectangle.

**Units and error characteristics (required by the T-FIX-30 contract):**

- The emitted `position` is in **world coordinate units**, the unit the addon
  frame prints and the unit `world/sync.py` consumes. No conversion is
  applied between the OCR result and `GameState.position`.
- The addon frame is quantised (typically to `0.1` of a zone coordinate),
  which is far coarser than the `5.0`-unit player-node dedup radius and is
  therefore not the limiting error term.
- The dominant error class is **OCR digit error**, and its dangerous form is a
  digit transposition or a dropped decimal point (for example `57.3` read as
  `573`). The reader's confidence gate is the only defence, which is why a
  failed parse and a low-confidence parse both produce `position = None`
  rather than a "best effort" number.
- The reader is **frame-sampled** (`[pose].sampling_hz`) and re-serves its last
  sample while throttled, carrying that sample's monotonic timestamp in
  `PoseReading.sampled_at`. A consumer that needs a fresher pose than the
  sampling period can gate on that timestamp; T-FIX-32 owns the staleness
  gate, per its own contract ("stale or missing frames are explicit, never
  silently reused").
- **Comma-decimal locales are not accepted.** The parser treats `.` as the
  decimal separator and `,` / whitespace / `;` as the coordinate separator.
  A locale that prints `57,3 42,1` must configure the addon (or the client's
  number format) to print a dot; otherwise the reader reports `None` rather
  than guessing which separator is which. This is a recorded limitation, not
  a silent one.

### Decision 2: `player_z` stays `None` permanently, and two views stay blocked by design

Candidate A observes a **two**-dimensional coordinate. The addon frame does
not print elevation, and no other channel in the four available extraction
methods (colour segmentation, template match, template-match-plus-centroid,
OCR) observes world height.

Therefore, per the T-FIX-30 deliverable:

- `player_z` is documented as the **permanent `None`** value in LAB_MODE.
- **Do not invent a z.** A fabricated elevation writes a node at a height the
  world model will then plan through.
- `WorldSyncView` and `StrategistView` **remain blocked by design**, because
  their protocols declare `player_z` non-Optional (`perception/views.py`).
  Unblocking them requires a separate decision that either changes those
  contracts or supplies a validated height channel.
- `WorldSyncView` is nevertheless proven projectable **once a `player_z` is
  supplied from outside the channel** (a test injection), so the pose channel
  itself is verified end to end and only the height input is missing.

### Decision 3: Target distance is inferred from the observed nameplate HP-bar width

`TargetInfo.distance_estimate` needs a number in yards. The only per-frame
observable that scales with distance in the ported reader set is the rendered
size of the target nameplate. T-FIX-27's `TargetReader._extract_hp` already
locates both ends of the nameplate HP bar by colour segmentation; the observed
width in pixels is therefore a **measurement the port already makes** and is
exposed additively as `TargetReading.bar_width_px`.

The conversion is an inverse-proportional (pinhole) model, calibrated by two
constants:

```
distance_yd = reference_distance_yd * reference_width_px / bar_width_px
```

- Units: `bar_width_px` is screen pixels; `distance_yd` is yards, the unit
  `TargetInfo.distance_estimate` documents.
- Accuracy class: **low to medium**, and the contract says so. Absolute error
  grows with distance (a one-pixel error is a large relative error on a small
  plate), the model is exact only at the calibration pair, and the client may
  clamp nameplate scale at short and long range. It is an *inference from an
  observation*, which is why it is confidence-gated and never fabricated.
- A width is only reported when the bar's **right edge was actually observed**.
  The existing fallback (`right_x = left_x + template_w + 30`) keeps the HP
  fraction working but is not a measurement, so the width is `None` in that
  case.
- Confidence: the nameplate template-match score of the same read. When it is
  below `[proximity].min_confidence`, or the width is unobserved, the distance
  is `None`; `build_target_info` then returns `target = None`, keeping
  `ReactiveView` blocked rather than letting it act on a guessed range.

**Rejected: a per-whitelist-member reference table.** It is a constant per mob,
not a per-frame observation, and it would make `target_in_range` a property of
the config rather than of the frame. It is the same mistake ADR-002 rejected
for `resource_max`.

### Decision 4: Target reaction is read from the nameplate text colour

The client already encodes reaction in the nameplate text colour: red for
hostile, yellow for neutral, green for friendly. A colour classifier over
`[reaction].nameplate_roi` therefore reads a value the client rendered, in the
same way the bar readers read rendered bars.

- Confidence: the winning class's share of the coloured pixels in the ROI.
- `reaction = None` when fewer than `[reaction].min_pixels` coloured pixels are
  present, or when no class holds `[reaction].dominance_thresh` of them.
- An unobserved reaction leaves `TargetInfo` absent. That is deliberate:
  `TargetInfo.__post_init__` rejects an empty reaction, and a fabricated
  reaction would silently re-gate targeting and combat.

### Decision 5: `target_x` / `target_y` stay unobserved

The builder's `InjectedObservations` also carries `target_x` / `target_y`. A
world-space target position would have to be derived as *pose + bearing +
distance*, which is a second-order inference with no validated error model
(bearing comes from the minimap arrow, distance from Decision 3). Writing a
derived target position would create target nodes in the world model with
compounded error. They therefore stay `None`, and nothing in this task
fabricates them. Only the selected target's **identity, HP, reaction, and
distance** are supplied.

## Consequences

**Positive**

- The critical-path gap is closed with an observation rather than an
  integration: `position` has a real source, and its failure mode is `None`.
- `ReactiveView` becomes projectable through the same path once a distance and
  a reaction exist, with no change to any consumer protocol or to
  `GameState`.
- Both new pixel-derived channels reuse extraction methods already documented
  in `docs/PERCEPTION.md` (OCR; colour segmentation), so no new dependency,
  no model weight, and no network path is introduced.
- Every new value is confidence-gated and absent-by-default, so the existing
  fail-loud adapter behaviour is preserved: a missing observation still
  produces `AdapterIncompleteError`, never a plausible number.

**Negative / recorded uncertainties**

- `player_z` remains unavailable, so two views stay blocked **by design**. This
  ADR does not pretend otherwise.
- The pose channel's precision/recall and the distance model's error curve are
  **not measured**: the frozen frame corpus required by `docs/PERCEPTION.md`
  §6 does not exist in the repository. Both channels are recorded in the §6
  table as unmeasurable today rather than reported as validated.
- Coordinate OCR is locale-sensitive (Decision 1) and depends on the operator
  running a coordinate addon. Without the addon, the channel reports `None` and
  the world model receives no player node.
- `TargetReader._extract_hp` gains one additive output (`bar_width_px`). Its
  existing HP fraction behaviour is unchanged, including the right-edge
  fallback.

## Scope

Files this decision's implementation touches:

- `src/wow_bot/perception/pose.py` (new)
- `src/wow_bot/perception/reaction.py` (new)
- `src/wow_bot/perception/proximity.py` (new)
- `src/wow_bot/perception/observations.py` (new)
- `src/wow_bot/perception/target.py` (additive field only)
- `src/wow_bot/perception/builder.py` (additive: an optional confidence
  carrier on `InjectedObservations`, so the three channels' measured scores
  reach `GameState.perception_confidence` instead of being discarded)
- `src/wow_bot/perception/perception_config.py` and
  `config/perception.example.toml` (the new channel keys)
- `docs/PERCEPTION.md`, `docs/lab_phase/PRE_REAL_PERCEPTION_FIX_ROADMAP.md`
- tests for the above

Out of scope: `GameState` (no schema change), `SafetyLayer`, `Session`,
`Config`, the eight consumer Protocols, the adapter's derivation rules, the
UI-panel channels (T-FIX-31), navigation, live actuation, and enabling a real
producer (T-FIX-32 / Phase 13).

`src/wow_bot/perception/` is a T-FIX-03 module and is **not** frozen under
STRATEGY A (roadmap Global Rule 12), so no `AGENTS.md` §5 exception is
required for this task.
