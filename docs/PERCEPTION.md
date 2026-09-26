# RealPerception Design (LAB_MODE)

## Pipeline

    [Capture] -> [Vision] -> [State Builder] -> GameState

The output MUST conform to the existing GameState schema used by
MockPerception. No downstream layer may branch on producer identity.

## 1. Capture

- Library: `dxcam` (Windows) or `mss` (cross-platform fallback).
- Target: client window region, resolved once at startup and re-resolved
  on window move/resize events.
- Rate: configurable, default 20 Hz. Frames dropped rather than queued.
- Every frame carries a monotonic timestamp from the same clock as the
  reflex loop.

## 2. Vision

Sub-components, each independently testable on recorded frames:

| Component        | Method                     | Output                          |
|------------------|----------------------------|---------------------------------|
| Health/Mana bars | Color segmentation + Hough | normalized [0,1] floats         |
| Target frame     | Template match             | presence + entity id slot       |
| Nameplates       | OCR (Tesseract)            | text + confidence               |
| Minimap          | Template match + centroid  | player pose on map              |
| Action bar       | Template match             | cooldown state per slot         |
| Chat / events    | OCR on event region        | text lines with timestamps      |
| Bag frame        | Template match per slot    | occupied-slot count + capacity  |
| XP bar           | Segmentation + Hough + OCR | level + xp fraction             |
| Character frame  | OCR per-slot durability    | mean durability fraction        |
| Enemy cast bar   | OCR + template match       | one entry per cast              |
| Lootable corpse  | Color segmentation         | lootable status or unknown      |
| World pose       | OCR on addon coordinate frame | world (x, y) + confidence    |
| Target distance  | Color segmentation (nameplate HP-bar width) | yards + confidence |
| Target reaction  | Color segmentation (nameplate text colour) | reaction + confidence |

All vision outputs carry confidence. Low-confidence fields are marked
`unknown` rather than guessed.

The World pose, Target distance, and Target reaction rows are the T-FIX-30
channels and are specified in full in §2.1, together with the source,
extraction method, accuracy class, and confidence semantics the T-FIX-30
contract requires. The choice of an addon coordinate frame for World pose —
and the rejection of dead reckoning, minimap scroll offset, and raw minimap
pixels — is recorded in `docs/decisions/ADR-003-world-pose-channel.md`.

### 2.1 Extended observability channels

ADR-002 extends `GameState` with fields that need channels beyond the six
rows above, and T-FIX-30 adds the world-pose, target-distance, and
target-reaction channels the consumers need before any projection can
succeed. The channels below are specified here so that a real backend has a
documented extraction rule for every observed field. Implementing them is a
Phase 13 concern, not this document's. Each entry states its source channel,
extraction method, expected accuracy class, confidence semantics, and the
fields it feeds.

**Bag frame** — feeds `inventory_count` and `inventory_max`.

- Source channel: the bag window's slot grid, captured only while the
  window is open. This is an opportunistic observation, not a per-frame
  one.
- Extraction method: template match per slot against empty-slot and
  occupied-slot templates. `inventory_count` is the number of occupied
  matches; `inventory_max` is the number of slot positions in the grid, a
  layout property that is stable for a session and can be cached.
- Accuracy class: medium for the count, high for the capacity. Slot
  templates are small and the grid shifts with bag size, but the decision
  is binary per slot; the grid count is a small integer with a clear
  template footprint.
- Confidence semantics: mean per-slot match score. Below threshold the
  whole count is `None` rather than a partial count, because the consumer
  gates a full/not-full decision on it. `inventory_max` needs only a
  single high-confidence reading; otherwise it is `None`, and the consumer
  falls back to its configured absolute threshold, which is the existing
  safe path.
- Rejected alternative: accumulating "You receive item" lines from the
  Chat/events row. Chat lines scroll out of the OCR region while the region
  is occluded, so the accumulator silently drops lines and drifts. A
  per-frame absolute count does not drift.

**XP bar** — feeds `level_or_xp`.

- Source channel: the experience bar, a thin fill bar at the bottom edge of
  the viewport.
- Extraction method: color segmentation plus Hough on the bar — the same
  method the Health/Mana row uses — yielding a `[0,1]` fill fraction. The
  integer level is OCR'd from the bar's level label with the same Tesseract
  method as the Nameplates row.
- Emitted value: `level + xp_fraction`, a single monotone scalar, so the
  runner's `level`-then-`xp` read and the strategist's `level_or_xp` field
  agree on one number.
- Accuracy class: medium for the fraction, high for the small integer.
- Confidence semantics: bar-occlusion confidence. Below threshold the field
  is `None`, never a guessed level, because a constant level hides XP
  progress.

**Character frame** — feeds `durability_fraction`.

- Source channel: the character sheet's per-slot durability readouts.
- Extraction method: OCR of the per-slot percentage text (the Tesseract
  method of the Nameplates row), then a mean over the slots the state
  builder read confidently.
- Accuracy class: low to medium. The values are small text on a busy frame,
  so OCR is the limiting factor; the mean smooths one misread but not many.
- Confidence semantics: the mean is weighted by per-slot OCR confidence,
  and the field is emitted only when a quorum of slots read above
  threshold. Below quorum it is `None`, which routes to the existing safe
  path: the vendor consumer treats `None` as `"durability_unknown"` and
  skips repairing.
- Emitted scale: `[0,1]`.

**Enemy cast bar** — feeds `incoming_casts`.

- Source channel: the cast bar that appears on the target frame and above
  enemy nameplates while a mob is casting.
- Extraction method: OCR for the spell name plus template match for the
  interruptibility indicator and the bar's fill fraction. `spell_id` is the
  OCR'd name mapped through the spell-id table; `remaining_cast_time_s` is
  the fill fraction times the spell's known cast duration.
- Accuracy class: low to medium. Spell names are short and the cast bar is
  small, so OCR precision is the limiting factor.
- Confidence semantics: per-cast confidence. A cast below threshold is
  omitted from the list rather than included with a guessed spell id,
  because the reactive layer iterates the list for interrupt decisions and
  a wrong spell id produces a wrong interrupt.
- Unknown sub-field: `is_interruptible` is a template match on a border
  color. If that match fails, the whole cast entry is dropped, since the
  interrupt is gated on it directly.
- Empty list is honest: with no confident casts, `()` means "no casts
  observed", which is a legitimate observation, not a fabricated absence.

**Lootable-corpse indicator** (table row: "Lootable corpse") — feeds
`target_is_lootable`.

- Source channel: the loot sparkle on a corpse in the game world. This is
  the weakest channel in this document.
- Extraction method: color segmentation for the sparkle hue in the region
  around a dead unit.
- Accuracy class: low. The sparkle is small, transient, and camera-angle
  dependent, so false negatives dominate.
- Confidence semantics: a high threshold is required to assert `True`, and
  there is no confident negative. Absent a confident positive the field is
  `None`, never `False`, because the loot consumer skips looting on a
  not-lootable answer and the bot would walk away from loot it could have
  taken.
- Open validation gate: this channel has no measured precision/recall yet,
  because the frozen frame corpus required by §6 is not present in the
  repository. Until that measurement exists, the field is emitted as
  `None` and the loot view keeps raising `AdapterIncompleteError` on it.
  This is a recorded uncertainty, not an assumption.

**World pose** (table row: "World pose") — feeds `position`.

- Source channel: an on-screen **addon coordinate frame** (TomTom/MapCoords
  style) that renders the player's current world coordinates as a short
  text line at a fixed screen position, inside `[pose].coordinate_roi`.
  This is decision **A** of the three candidates in
  `docs/lab_phase/HAMBERGER_PORT_PLAN.md` finding F-1; the decision and the
  rejections of dead reckoning, minimap scroll offset, and raw minimap
  pixels are recorded in `docs/decisions/ADR-003-world-pose-channel.md`.
  The minimap arrow is explicitly **not** this channel: it returns screen
  pixels inside the minimap ROI and is nearly constant in the live client.
- Extraction method: OCR (the Tesseract method of the Nameplates row) of the
  coordinate region at `--psm 7`, using the word-confidence API; the joined
  tokens must parse to **exactly two** decimal numbers with `.` as the
  decimal separator and `,`/whitespace/`;` as separators. The emitted
  `position` is in **world coordinate units**, the unit of the addon frame
  and of `world/sync.py`; no unit conversion is applied. A comma-decimal
  locale that prints `57,3 42,1` is **not** accepted — the reader reports
  `None` rather than guessing which separator is which.
- Accuracy class: medium. The addon frame is quantised (typically `0.1` of a
  zone coordinate), far below the `5.0`-unit player-node dedup radius, so the
  limiting error is **OCR digit error** — a transposition or a dropped
  decimal point (`57.3` read as `573`) is a wrong pose, not an imprecise one.
- Confidence semantics: the mean Tesseract word confidence of the tokens
  that carry digits. Below `[pose].min_confidence`, or when the text does not
  parse to exactly two unambiguous numbers, `position` is `None` and no
  world node is created. A wrong pose writes a wrong node into the world
  model permanently, so a low-confidence pose must never be guessed. The
  reader is frame-sampled (`[pose].sampling_hz`) and exposes the sample's
  monotonic timestamp as `PoseReading.sampled_at`; a consumer that needs a
  fresher pose than one sampling period gates on it. That staleness gate
  belongs to T-FIX-32.
- Not observed by this channel: `player_z` (see below) and `target_x` /
  `target_y`.

**Target distance** (table row: "Target distance") — feeds
`TargetInfo.distance_estimate` (and therefore the derived `target_in_range`).

- Source channel: the rendered **width of the target nameplate's HP bar**,
  the per-frame observable that scales with distance in the ported reader
  set. It is already measured by the T-FIX-27 target reader; the observed
  pixel width is exposed additively as `TargetReading.bar_width_px`.
- Extraction method: colour segmentation — the same scan the Health/Mana row
  uses, as implemented by `TargetReader._extract_hp` — followed by an
  inverse-proportional (pinhole) conversion in world units:
  `distance_yd = reference_distance_yd * reference_width_px /
  bar_width_px`, calibrated by `[proximity].reference_width_px` and
  `[proximity].reference_distance_yd`. A width is reported only when the
  bar's right edge was actually observed; the reader's right-edge fallback
  keeps the HP fraction working but is not a measurement, so the width is
  `None` in that case.
- Accuracy class: low to medium. Absolute error grows with distance — a
  one-pixel error is a large relative error on a small plate — the model is
  exact only at its calibration pair, and the client may clamp nameplate
  scale at short and long range. This is an inference from an observation,
  not a direct read, and it is documented as such.
- Confidence semantics: the nameplate template-match score of the same
  read. Below `[proximity].min_confidence`, or when the width is
  unobserved, the distance is `None`; `build_target_info` then returns
  `target = None` and `ReactiveView` stays blocked rather than deriving
  `target_in_range` from a guessed distance.
- Rejected alternative: a per-whitelist-member reference distance. It is a
  constant per mob, not a per-frame observation, and would make
  `target_in_range` a property of the config rather than of the frame.

**Target reaction** (table row: "Target reaction") — feeds
`TargetInfo.reaction`.

- Source channel: the **colour of the nameplate text**, which the client
  already renders as red (hostile), yellow (neutral), or green (friendly),
  inside `[reaction].nameplate_roi`.
- Extraction method: colour classification of the ROI's pixels against
  hostile / neutral / friendly predicates; the dominant class must hold
  `[reaction].dominance_thresh` of the coloured pixels and at least
  `[reaction].min_pixels` coloured pixels must be present.
- Accuracy class: medium to high when the text is legible; it fails low when
  the ROI holds no nameplate or the frame is heavily tinted, both of which
  produce `None` rather than a wrong label.
- Confidence semantics: the winning class's share of the coloured pixels.
  Below `dominance_thresh`, or below `min_pixels`, `reaction` is `None`, and
  `TargetInfo` stays absent: `TargetInfo.__post_init__` rejects an empty
  reaction, and a fabricated reaction would silently re-gate targeting and
  combat.

**Player elevation (`player_z`) — permanently unobserved.** The world-pose
channel observes two coordinates; the addon frame does not print elevation,
and none of the four available extraction methods observes world height.
`player_z` is therefore documented as the **permanent `None`** value in
LAB_MODE. **Do not invent a z**: a fabricated elevation writes a node at a
height the world model will then plan through. `WorldSyncView` and
`StrategistView` stay blocked **by design** until a separate decision either
changes their contracts or supplies a validated height channel; injecting a
`player_z` from outside the channel is the only way to project them today.

## 3. State Builder

- Assembles a `GameState` from vision outputs.
- Fills missing fields with `None` / `unknown`; never fabricates.
- Attaches `perception_confidence` map to the state for downstream use.
- Emits a compact `perception_trace` per frame for offline analysis.

## 4. Determinism Boundary

Vision is inherently noisy. To keep downstream determinism:

- Reflex layer consumes vision with confidence thresholds.
- Strategist receives only aggregated, thresholded state.
- All randomness downstream of perception uses per-component seeds,
  never global.

## 5. Calibration

A calibration step runs at session start:
- Locate UI anchors (minimap corner, action bar, health globe, bag-window
  grid origin, XP bar, enemy cast-bar origin, loot-sparkle region,
  addon coordinate frame, nameplate-text region, nameplate HP-bar scan
  origin).
- Record per-resolution offsets.
- Fail closed if anchors cannot be located within N seconds.

The bag frame, XP bar, character frame, enemy cast bar, and loot-sparkle
region added in §2.1 are *opportunistic* anchors: they exist only while
their UI element is on screen. Calibration records their offsets the first
time they are seen and does **not** fail the session when they are absent.
While an anchor is unavailable, its fields stay `None` (see §2.1).

The three T-FIX-30 anchors are treated differently, and deliberately:
the addon coordinate frame is a **required** anchor for a LAB_MODE session
that intends to sync the world model, because without it every pose is
`None` and no player node is ever written; the nameplate-text region and the
nameplate HP-bar scan origin are per-frame *opportunistic* anchors like the
§2.1 UI panels, since a target nameplate exists only while a target is in
view.

## 6. Testing

- Unit tests run on a frozen corpus of frames under `tests/fixtures/`.
- Each vision component has precision/recall targets documented here.
- No test may require a live client.

The frozen corpus does **not** exist in the repository yet, so none of the
targets below is currently measurable. Until a corpus is supplied, every
channel in §2.1 reports `None` for its fields rather than a guess, and the
fail-loud adapter keeps raising on the ones the consumers require.

| Channel | Target to record | Measurable today |
|---|---|---|
| Lootable-corpse indicator | Precision and recall, recorded before the field may ever be emitted `True` | No — corpus absent |
| Bag frame | Occupied/empty classification accuracy, per slot | No — corpus absent |
| XP bar | Level-label OCR accuracy and bar-fill error | No — corpus absent |
| Character frame | Per-slot durability OCR accuracy | No — corpus absent |
| Enemy cast bar | Spell-name OCR accuracy | No — corpus absent |
| World pose | Coordinate-OCR character accuracy, and the rate of dangerous misreads (digit transposition / dropped decimal point) | No — corpus absent |
| Target distance | Yards error against a measured range, per distance band | No — corpus absent |
| Target reaction | Per-class precision and recall | No — corpus absent |

The three T-FIX-30 channels are therefore **unvalidated by measurement**.
They are gated by confidence and absent by default, so an unvalidated channel
degrades to `None` rather than to a plausible value; the numbers above must
still be recorded before Phase 13 may trust them.

Supplying the corpus and recording these numbers is out of scope for this
document; it is a prerequisite for Phase 13 perception validation.