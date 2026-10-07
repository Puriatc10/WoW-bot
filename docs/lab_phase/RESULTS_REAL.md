# Phase 13 Results — Real Perception Live Soak & Functional Outcomes

Perception mode: REAL (`RealPerceptionBackend` active via `ScreenCapture`, Tesseract OCR, OpenCV template matching).
Execution mode: LAB (`RealActuator`, real reflex loop, physical kill switch active).

---

## 1. Title and Phase Identifier

This document presents the functional evaluation, real-world perception performance, and gameplay outcome findings for Phase 13 of the WoW-bot research prototype.

- **Phase Identifier:** Phase 13 (Real Perception Live Soak)
- **Perception Mode:** REAL (`RealPerceptionBackend` processing live video frames, OCR coordinates, and HUD panels).
- **Actuation Mode:** LAB (`RealActuator` executing OS inputs with humanized timing against a private game client).
- **Target Environment:** Private, self-owned game server on an isolated network matching allowlist.

---

## 2. Scope and Non-Claims

Phase 13 evaluates real perception accuracy, end-to-end cycle progression, and gameplay outcomes in an isolated laboratory setting.

All results in this report are published alongside the following explicit non-claims:

- We do NOT claim any connection to retail WoW, Blizzard services, or third-party servers.
- We do NOT claim anti-cheat evasion, detection avoidance, or warden bypassing.
- We do NOT claim humanizer timing would evade commercial anti-cheat heuristics.
- All testing was performed strictly on a private, self-hosted emulator on an isolated local network.

---

## 3. Live Session Configuration

The Phase 13 live run was executed using `scripts/lab/live_soak.py` / `scripts/lab/run_farm_v2.py --mode LAB --real-perception`.

- **Session Identifier:** `[OPERATOR: session_id]`
- **Target Window:** `[OPERATOR: window_title (e.g. WoW)]`
- **Client Resolution:** `[OPERATOR: e.g. 1920x1080 Windowed, UI Scale 1.0]`
- **Driver Backend:** `[OPERATOR: pynput / interception]`
- **Kill Switch Key:** `[OPERATOR: e.g. F12]`
- **Configured Duration:** `[OPERATOR: wall-clock seconds]`

---

## 4. Real Perception Performance Metrics

| Metric | Measured Value | Target Envelope | Status |
|---|---|---|---|
| Screen Capture FPS | `[PLACEHOLDER]` | 10.0 – 20.0 Hz | `[PENDING]` |
| Frame Capture Latency | `[PLACEHOLDER] ms` | < 50.0 ms | `[PENDING]` |
| World Pose OCR Accuracy | `[PLACEHOLDER] %` | > 95.0 % | `[PENDING]` |
| Target HP Bar Extraction Latency | `[PLACEHOLDER] ms` | < 15.0 ms | `[PENDING]` |
| Minimap Heading Match Score | `[PLACEHOLDER]` | > 0.70 | `[PENDING]` |
| Bag Slot Grid Quorum Rate | `[PLACEHOLDER] %` | > 90.0 % | `[PENDING]` |

---

## 5. In-Game Gameplay Outcome Findings

| Outcome Dimension | Measured Value | Notes |
|---|---|---|
| Farm Cycles Completed | `[PLACEHOLDER]` | Distinct completed route traversals |
| Waypoint Nodes Reached | `[PLACEHOLDER]` | Navigated via world graph |
| Combat Encounters Initiated | `[PLACEHOLDER]` | Triggered upon hostile target detection |
| Combat Encounters Won | `[PLACEHOLDER]` | Target defeated & combat dropped |
| Loot Attempts Executed | `[PLACEHOLDER]` | Loot primitive triggered |
| Vendor / Repair Cycles | `[PLACEHOLDER]` | Bags emptied upon full inventory |
| Stuck Recovery Incidents | `[PLACEHOLDER]` | Handled via RecoverySink |

---

## 6. Safety & Humanizer Findings

| Safety / Control Check | Result | Verification |
|---|---|---|
| Network Isolation Sentinel Check | `PASS` | No external network routing permitted |
| Allowlist Verification | `PASS` | Local private IP allowlist enforced |
| Kill Switch Response Latency | `[PLACEHOLDER] ms` | Ceases OS input in < 100 ms |
| Unconditional Key Release on Abort | `PASS` | Verified in driver tear-down |
| Humanizer Keydown Timing Distribution | `[PLACEHOLDER]` | Lognormal fit KS-test p > 0.05 |

---

## 7. Observations & Engineering Notes

`[OPERATOR: Record any visual anomalies, OCR misrecognitions under dynamic lighting, or navigation pathing issues observed during the live session.]`
