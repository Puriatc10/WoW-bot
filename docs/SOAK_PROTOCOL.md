# SOAK PROTOCOL

## Scope

This protocol defines how the Phase 12 soak is executed and what it establishes. It distinguishes a MOCK_MODE soak (executed in Phase 12) from a real-perception soak (deferred to Phase 13).

## Phase 12 Soak — MOCK_MODE (1 hour)

Purpose:
  Verify software stability of the control stack over an extended run. NOT verify in-game outcomes.

Environment:
  - MOCK_MODE
  - NullDriver, NullFocusBackend, NullDelay
  - FakeLlmClient (scripted responses)
  - MockPerception
  - No game client. No Ollama. No OS input.

Procedure:
  1. Build the runtime via lab.runner_v2.build_lab_runtime_async with the mock stack.
  2. Run for 1 hour wall-clock.
  3. Soak telemetry written to soak_report.json in the session directory.
  4. On completion, archive the session directory.

Metrics reported:
  - crash count (target: zero)
  - RSS trend over the trailing window
  - log size trend
  - reflex tick rate and jitter
  - humanizer interval distribution (PIT/KS p-values)
  - watchdog transition counts
  - loop detection counts

Metrics NOT reported as research findings:
  - cycle success rate
  - distance to target
  - in-game farm outcome
  - anything requiring real perception

## Phase 13 Soak — Real Perception (deferred, 24-72 hours)

Purpose:
  Verify in-game outcomes with real perception, real actuation, and real LLM.

Environment:
  - LAB_MODE
  - Real InputDriver
  - Real FocusManager with a real window
  - Real Ollama (Qwen 2.5 7B)
  - RealPerception (Phase 13)
  - Private server on an isolated network

Procedure:
  To be defined in Phase 13.

Metrics reported:
  To be defined in Phase 13.

## Non-Claims for the Phase 12 Soak

The Phase 12 soak does NOT establish:
  - in-game farm success
  - navigation robustness against real obstacles
  - human-likeness versus real gameplay
  - any anti-cheat-related claim

Results of the Phase 12 soak MUST be reported with these non-claims alongside them.
