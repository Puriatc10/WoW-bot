# Lab Constraints

This document defines the hard boundaries of the lab execution mode.
It is normative. Any code path that violates these constraints MUST fail
closed (raise, abort, kill switch) rather than degrade.

## 1. Scope

The project supports two execution modes:

| Mode        | Perception          | Actuation           | Target                    |
|-------------|---------------------|---------------------|---------------------------|
| `MOCK_MODE` | MockPerception      | SimulationController| Synthetic GameState       |
| `LAB_MODE`  | RealPerception      | RealActuator        | Private server on LAN/VM  |

`MOCK_MODE` is the default. It is the only mode that runs in CI.
`LAB_MODE` is opt-in, manual, and never runs unattended.

## 2. Hard Constraints (MUST)

- The target server MUST be listed in `LAB_SERVER_ALLOWLIST` (config file,
  not env var). Any connection attempt to an address outside the allowlist
  MUST abort the process within 1 second.
- The allowlist MUST NOT contain any Blizzard-operated endpoint, retail
  realm, or public server. Enforced by a static check on the string.
- `LAB_MODE` MUST run inside a network namespace with no route to the
  public internet. Verified at startup by attempting a connection to a
  sentinel external address and asserting failure.
- Process memory of the game client MUST NOT be read or written.
  No `ReadProcessMemory`, no `WriteProcessMemory`, no DLL injection,
  no hooking.
- The game binary MUST NOT be modified on disk. Only the unmodified
  client, launched by the user, is observed via screen capture and
  controlled via synthetic input.
- Every lab session MUST produce an immutable, append-only log under
  `runs/lab/<session_id>/`. Logs MUST NOT be truncated or deleted on error.

## 3. Soft Constraints (SHOULD)

- Prefer running the client inside a dedicated VM or container.
- Prefer a dedicated user account with no access to personal files.
- Prefer a kill switch bound to a physical key and to `SIGTERM`.
- Prefer recording the full screen for post-hoc audit, not just the
  cropped client region.

## 4. Out of Scope (MUST NOT be implemented)

- Any form of anti-cheat bypass, evasion, or detection avoidance targeting
  a third-party service.
- Any memory inspection of the retail client.
- Any automation intended to run against a server the operator does not own.
- Any distribution of lab artifacts that could be repurposed for retail use.

## 5. Rationale

The research claim of this project is about the *architecture* of an
LLM-driven agent operating under human-like timing and reflex constraints
in a controlled environment. The claim is falsifiable only if the
environment is reproducible, isolated, and ethically bounded. Retail
execution would invalidate reproducibility and violate third-party rights,
and is therefore excluded by design.