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

All vision outputs carry confidence. Low-confidence fields are marked
`unknown` rather than guessed.

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
- Locate UI anchors (minimap corner, action bar, health globe).
- Record per-resolution offsets.
- Fail closed if anchors cannot be located within N seconds.

## 6. Testing

- Unit tests run on a frozen corpus of frames under `tests/fixtures/`.
- Each vision component has precision/recall targets documented here.
- No test may require a live client.