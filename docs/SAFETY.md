# Safety and Recovery

## 1. Kill Switch

- Bound to a physical key (configurable) and to `SIGTERM`/`SIGINT`.
- On trigger: stop actuation, release all inputs, flush logs, exit 0.
- Reflex layer checks the kill flag every tick.

## 2. Allowlist Enforcement

- `LAB_SERVER_ALLOWLIST` is a config file, read once at startup.
- Every outbound game connection is checked against it.
- On mismatch: abort within 1 second, log the offending address.

## 3. Network Isolation Check

At LAB_MODE startup:
- Attempt connection to a sentinel external address with 1s timeout.
- If it succeeds, abort with `IsolationViolation`.
- This guarantees the lab cannot reach retail.

## 4. Disconnect Handling

- If the client loses connection to the lab server, actuation pauses.
- Reflex layer waits for reconnection up to a configurable window.
- On timeout, session ends gracefully with a `disconnect` event.

## 5. Stuck Detection

- Position sampled at 1 Hz from perception.
- If displacement < threshold for N seconds, enter `STUCK_RECOVERY`.
- Recovery behaviors: small random walk, camera sweep, jump, re-target.
- If recovery fails K times, emit `hard_stuck` and pause session.

## 6. Log Integrity

- Session log directory is created with append-only semantics.
- No code path may truncate or delete a log file on error.
- On crash, the process writes a `crash.json` with the last known state.

## 7. Out-of-Scope Events

The following MUST halt the session:
- Focus lost to a window that is not the client.
- Any connection attempt to a non-allowlisted address.
- Isolation check failure.
- Kill switch activation.