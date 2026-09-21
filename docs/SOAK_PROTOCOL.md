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
  1. Build the runtime via lab.runner_v2.build_lab_runtime_async with:
       - mode="MOCK"
       - driver_name="null"
       - focus_backend=NullFocusBackend()
       - sleep=<soak_sleep wrapper>
       - llm_client=<FakeLlmClient>
       - include_reflex_loop=False
       - include_watchdog=False
  2. Run run_lab_loop_async(runtime, max_cycles=LARGE, stop_event=<asyncio.Event>).
  3. The soak_sleep wrapper sets stop_event when runtime.clock() >= started_at + duration_s.
  4. Samples are collected at sample_interval_s in the same wrapper.
  5. After the loop returns, build the SoakReport via lab_soak_v2.build_soak_report and write it via lab_soak_v2.write_soak_report to session_dir / "soak_report.json".
  6. Exit codes:
       - 0: soak completed (STOP_EVENT_SET or MAX_CYCLES_REACHED)
       - 1: soak ended with an abnormal runner status (HEALTH_CRITICAL, LOOP_DETECTED, MAX_FAILURES_REACHED, RUNTIME_ERROR, BUILD_ERROR)
       - 2: argument or setup error
       - 3: unhandled exception during the run

Note: This procedure is the ONLY supported Phase 12 soak. Any LAB-mode soak requires Phase 13 infrastructure and is out of scope.

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
