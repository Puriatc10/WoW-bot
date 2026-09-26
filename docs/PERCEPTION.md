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

All vision outputs carry confidence. Low-confidence fields are marked
`unknown` rather than guessed.

### 2.1 Extended observability channels

ADR-002 extends `GameState` with fields that need channels beyond the six
rows above. The five channels below are specified here so that a real
backend has a documented extraction rule for every observed field.
Implementing them is a Phase 13 concern, not this document's. Each entry
states its source channel, extraction method, expected accuracy class,
confidence semantics, and the fields it feeds.

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
  grid origin, XP bar, enemy cast-bar origin, loot-sparkle region).
- Record per-resolution offsets.
- Fail closed if anchors cannot be located within N seconds.

The bag frame, XP bar, character frame, enemy cast bar, and loot-sparkle
region added in §2.1 are *opportunistic* anchors: they exist only while
their UI element is on screen. Calibration records their offsets the first
time they are seen and does **not** fail the session when they are absent.
While an anchor is unavailable, its fields stay `None` (see §2.1).

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

Supplying the corpus and recording these numbers is out of scope for this
document; it is a prerequisite for Phase 13 perception validation.