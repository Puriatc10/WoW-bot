"""Loop detection module for the behavioral watchdog (Task 8.3).

Identifies repeated action patterns without world state progress using a bounded
observation buffer. Pure, deterministic, and isolated from OS, I/O, LLMs, and wall-clock time.
"""

import hashlib
import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from wow_bot.session import Session


class LoopError(Exception):
    """Raised when loop detector operations or observation updates violate constraints."""


@dataclass(frozen=True)
class LoopConfig:
    """Configuration parameters for loop detection sliding window.

    Validation in __post_init__:
      * window_s > 0.0, else ValueError.
      * max_signatures >= 32, else ValueError.
      * cycle_length >= 2, else ValueError.
      * min_repeats >= 2, else ValueError.
      * stagnation_epsilon >= 0.0, else ValueError.
      * All floats finite, else ValueError.
    """

    window_s: float = 600.0
    max_signatures: int = 2_000
    cycle_length: int = 4
    min_repeats: int = 3
    stagnation_epsilon: float = 0.0
    emit_on_stagnation_only: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.window_s, bool)
            or not isinstance(self.window_s, (int, float))
            or not math.isfinite(self.window_s)
        ):
            raise ValueError("window_s must be a finite float")
        if self.window_s <= 0.0:
            raise ValueError(f"window_s must be > 0.0, got {self.window_s}")

        if (
            isinstance(self.max_signatures, bool)
            or not isinstance(self.max_signatures, int)
            or self.max_signatures < 32
        ):
            raise ValueError(f"max_signatures must be an integer >= 32, got {self.max_signatures}")

        if (
            isinstance(self.cycle_length, bool)
            or not isinstance(self.cycle_length, int)
            or self.cycle_length < 2
        ):
            raise ValueError(f"cycle_length must be an integer >= 2, got {self.cycle_length}")

        if (
            isinstance(self.min_repeats, bool)
            or not isinstance(self.min_repeats, int)
            or self.min_repeats < 2
        ):
            raise ValueError(f"min_repeats must be an integer >= 2, got {self.min_repeats}")

        if (
            isinstance(self.stagnation_epsilon, bool)
            or not isinstance(self.stagnation_epsilon, (int, float))
            or not math.isfinite(self.stagnation_epsilon)
        ):
            raise ValueError("stagnation_epsilon must be a finite float")
        if self.stagnation_epsilon < 0.0:
            raise ValueError(f"stagnation_epsilon must be >= 0.0, got {self.stagnation_epsilon}")

        if not isinstance(self.emit_on_stagnation_only, bool):
            raise TypeError("emit_on_stagnation_only must be a boolean")


@dataclass(frozen=True)
class ActionObservation:
    """A single action observation snapshot.

    Invariants (checked in __post_init__):
      * ts finite, >= 0.0.
      * signature is a non-empty string without whitespace at the ends.
      * level_or_xp finite, >= 0.0.
    """

    ts: float
    signature: str
    level_or_xp: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.ts, bool)
            or not isinstance(self.ts, (int, float))
            or not math.isfinite(self.ts)
            or self.ts < 0.0
        ):
            raise ValueError(f"ts must be a finite float >= 0.0, got {self.ts}")

        if not isinstance(self.signature, str) or len(self.signature) == 0:
            raise ValueError("signature must be a non-empty string")
        if self.signature != self.signature.strip():
            raise ValueError("signature must not have leading or trailing whitespace")

        if (
            isinstance(self.level_or_xp, bool)
            or not isinstance(self.level_or_xp, (int, float))
            or not math.isfinite(self.level_or_xp)
            or self.level_or_xp < 0.0
        ):
            raise ValueError(f"level_or_xp must be a finite float >= 0.0, got {self.level_or_xp}")


@dataclass(frozen=True)
class LoopDetection:
    """Result of loop evaluation for an observation.

    Invariants (checked in __post_init__):
      * detected is False implies repeats == 0 and reason != "".
      * detected is True implies repeats >= 1 and reason != "".
      * ts finite, >= 0.0.
      * cycle_length >= 0.
      * progress_since_window_start >= 0.0.
    """

    detected: bool
    cycle_length: int
    repeats: int
    ts: float
    window_start_ts: float | None
    progress_since_window_start: float
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.detected, bool):
            raise TypeError("detected must be a boolean")

        if (
            isinstance(self.ts, bool)
            or not isinstance(self.ts, (int, float))
            or not math.isfinite(self.ts)
            or self.ts < 0.0
        ):
            raise ValueError(f"ts must be a finite float >= 0.0, got {self.ts}")

        if (
            isinstance(self.cycle_length, bool)
            or not isinstance(self.cycle_length, int)
            or self.cycle_length < 0
        ):
            raise ValueError(f"cycle_length must be an integer >= 0, got {self.cycle_length}")

        if (
            isinstance(self.progress_since_window_start, bool)
            or not isinstance(self.progress_since_window_start, (int, float))
            or not math.isfinite(self.progress_since_window_start)
            or self.progress_since_window_start < 0.0
        ):
            raise ValueError(
                f"progress_since_window_start must be a finite float >= 0.0, got {self.progress_since_window_start}"
            )

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")

        if self.window_start_ts is not None and (
            isinstance(self.window_start_ts, bool)
            or not isinstance(self.window_start_ts, (int, float))
            or not math.isfinite(self.window_start_ts)
            or self.window_start_ts < 0.0
        ):
            raise ValueError(
                f"window_start_ts must be a finite float >= 0.0 or None, got {self.window_start_ts}"
            )

        if not self.detected:
            if self.repeats != 0:
                raise ValueError(f"repeats must be 0 when detected is False, got {self.repeats}")
        else:
            if (
                isinstance(self.repeats, bool)
                or not isinstance(self.repeats, int)
                or self.repeats < 1
            ):
                raise ValueError(
                    f"repeats must be an integer >= 1 when detected is True, got {self.repeats}"
                )


@dataclass(frozen=True)
class LoopEvent:
    """Immutable event payload generated when a loop is detected.

    Invariants (checked in __post_init__):
      * len(signature_hashes) == cycle_length.
      * cycle_length >= 2.
      * repeats >= 2.
      * reason is a non-empty string.
      * ts finite, >= 0.0.
    """

    ts: float
    cycle_length: int
    repeats: int
    signature_hashes: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.ts, bool)
            or not isinstance(self.ts, (int, float))
            or not math.isfinite(self.ts)
            or self.ts < 0.0
        ):
            raise ValueError(f"ts must be a finite float >= 0.0, got {self.ts}")

        if (
            isinstance(self.cycle_length, bool)
            or not isinstance(self.cycle_length, int)
            or self.cycle_length < 2
        ):
            raise ValueError(f"cycle_length must be an integer >= 2, got {self.cycle_length}")

        if (
            isinstance(self.repeats, bool)
            or not isinstance(self.repeats, int)
            or self.repeats < 2
        ):
            raise ValueError(f"repeats must be an integer >= 2, got {self.repeats}")

        if not isinstance(self.signature_hashes, tuple):
            raise TypeError("signature_hashes must be a tuple")

        if len(self.signature_hashes) != self.cycle_length:
            raise ValueError(
                f"len(signature_hashes) ({len(self.signature_hashes)}) must equal cycle_length ({self.cycle_length})"
            )

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")

    def to_json(self) -> dict[str, Any]:
        """Return event dictionary representation suitable for session logging."""
        return {
            "event": "watchdog_loop_detected",
            "ts": float(self.ts),
            "cycle_length": int(self.cycle_length),
            "repeats": int(self.repeats),
            "signature_hashes": list(self.signature_hashes),
            "reason": self.reason,
        }


class LoopDetector:
    """Bounded observation buffer identifying repeated action patterns without progress.

    Single-thread ownership note:
    This class is NOT thread-safe. It must be used from a single sampling thread.
    """

    def __init__(
        self,
        *,
        config: LoopConfig | None = None,
        session: "Session | None" = None,
    ) -> None:
        self._config: LoopConfig = config if config is not None else LoopConfig()
        self._session: Any | None = session
        self._deque: deque[tuple[float, str, float]] = deque(maxlen=self._config.max_signatures)

    def observe(
        self,
        observation: ActionObservation,
    ) -> LoopDetection | None:
        """Observe an action snapshot and evaluate whether an action loop is occurring.

        Semantics:
          1. Enforce monotonic ts.
          2. Enforce monotonic level_or_xp.
          3. Compute signature_hash via blake2b.
          4. Append (ts, signature_hash, level_or_xp) to deque.
          5. Drop entries older than observation.ts - window_s.
          6. If deque has < 2 * cycle_length entries, return None.
          7. Verify most recent 2 * cycle_length form a repeating cycle.
          8. Count total consecutive matching blocks (repeats). Return None if < min_repeats.
          9. Compute progress_since_window_start.
         10. Handle emit_on_stagnation_only condition if progress > stagnation_epsilon.
         11. Construct LoopEvent.
         12. Emit event to session if attached.
         13. Return detected LoopDetection.
        """
        if self._deque:
            last_ts, _, last_xp = self._deque[-1]
            if observation.ts <= last_ts:
                raise LoopError(
                    f"Observation timestamp {observation.ts} is not strictly greater than previous timestamp {last_ts}"
                )
            if observation.level_or_xp < last_xp:
                raise LoopError(
                    f"level_or_xp decreased from {last_xp} to {observation.level_or_xp}"
                )

        sig_hash = hashlib.blake2b(
            observation.signature.encode("utf-8"),
            digest_size=8,
        ).hexdigest()

        self._deque.append((observation.ts, sig_hash, observation.level_or_xp))

        cutoff = observation.ts - self._config.window_s
        while self._deque and self._deque[0][0] < cutoff:
            self._deque.popleft()

        cycle_len = self._config.cycle_length
        if len(self._deque) < 2 * cycle_len:
            return None

        recent = [h for (_, h, _) in self._deque][-2 * cycle_len :]
        half = len(recent) // 2
        first_half = recent[:half]
        second_half = recent[half:]
        if first_half != second_half:
            return None

        all_signatures = [h for (_, h, _) in self._deque]
        target_block = all_signatures[-cycle_len:]
        repeats = 0
        pos = len(all_signatures)
        while pos >= cycle_len:
            block = all_signatures[pos - cycle_len : pos]
            if block == target_block:
                repeats += 1
                pos -= cycle_len
            else:
                break

        if repeats < self._config.min_repeats:
            return None

        earliest_ts, _, earliest_xp = self._deque[0]
        window_start_ts: float | None = earliest_ts
        progress = observation.level_or_xp - earliest_xp

        if self._config.emit_on_stagnation_only and progress > self._config.stagnation_epsilon:
            return LoopDetection(
                detected=False,
                cycle_length=0,
                repeats=0,
                ts=observation.ts,
                window_start_ts=window_start_ts,
                progress_since_window_start=progress,
                reason="progress_within_window",
            )

        event = LoopEvent(
            ts=observation.ts,
            cycle_length=cycle_len,
            repeats=repeats,
            signature_hashes=tuple(recent[-cycle_len:]),
            reason="cycle_repeated_without_progress",
        )

        if self._session is not None:
            self._session.write_event(event.to_json())

        return LoopDetection(
            detected=True,
            cycle_length=cycle_len,
            repeats=repeats,
            ts=observation.ts,
            window_start_ts=window_start_ts,
            progress_since_window_start=progress,
            reason="cycle_repeated_without_progress",
        )

    def reset(self) -> None:
        """Clear the observation deque. Idempotent."""
        self._deque.clear()

    @property
    def size(self) -> int:
        """Return current size of observation buffer."""
        return len(self._deque)

    @property
    def window_start_ts(self) -> float | None:
        """Return timestamp of earliest surviving entry in window, or None if empty."""
        return self._deque[0][0] if self._deque else None

    @property
    def window_end_ts(self) -> float | None:
        """Return timestamp of latest surviving entry in window, or None if empty."""
        return self._deque[-1][0] if self._deque else None
