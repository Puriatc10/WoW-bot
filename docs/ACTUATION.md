# RealActuator Design (LAB_MODE)

## Pipeline

    Strategy/FSM intent -> Action Mapper -> Humanizer -> Input Driver -> OS

Intents use the existing symbolic vocabulary
(`move_to`, `target`, `cast`, `loot`, `interact`, `flee`, ...).
The mapper is the only place that knows about keys and mouse.

## 1. Input Driver

- Library: `interception` (kernel-level, Windows) preferred; `pynput`
  as fallback.
- Only the game window receives events; focus is asserted before each
  event batch.
- Input events are timestamped and logged.

## 2. Focus Manager

- Resolves the client window handle at startup.
- Before each action batch, verifies foreground window == client.
- If focus lost, emits a `focus_lost` event to the reflex layer and
  pauses actuation.

## 3. Action Mapper

Deterministic mapping table:

| Intent        | Primitive sequence                                  |
|---------------|-----------------------------------------------------|
| move_to(p)    | turn camera toward p, press forward, release at p   |
| target(id)    | move cursor to nameplate, click, verify target frame|
| cast(spell)   | press bound key, wait for GCD, verify cooldown      |
| loot          | move cursor to corpse, right-click, wait for window |
| interact(npc) | move cursor, right-click, wait for dialog           |
| flee(dest)    | turn away, press forward, sprint if available       |

Every primitive returns success/failure; failures feed stuck recovery.

## 4. Humanizer

Extends the existing `humanize.py` lognormal model. Now applied to
*inter-event intervals* and *cursor paths*, not just symbolic delays.

- Inter-key delay: clipped lognormal, parameters from Task 8.2.
- Cursor path: Bezier with jittered control points, duration lognormal.
- Micro-pauses: injected every N actions with configurable probability.
- Occasional miss-click: probability p, distance d, both configurable
  and both logged for analysis.
- Camera turn: piecewise, with small overshoot and correction.

## 5. Reflex Coupling

The actuator exposes an `abort()` that the reflex layer can call at any
time. Abort MUST complete within 100 ms and leave the client in a safe
state (no keys held, cursor released).

## 6. Testing

- All actuation tests run against a `NullInputSink` in MOCK_MODE.
- LAB_MODE actuation is validated manually per session; the session log
  is the evidence.
- No automated test may synthesize real OS input.