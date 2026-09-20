"""Signal sources and signal types for the reflex loop."""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol


class SignalError(Exception):
    """Raised when signal validation or signal source operations fail."""


@dataclass(frozen=True)
class Signal:
    """Represents an input signal to the reflex loop.

    Payload MUST be JSON-serializable.
    """

    name: str
    payload: dict[str, Any]
    ts: float

    def __post_init__(self) -> None:
        if not isinstance(self.payload, dict):
            raise SignalError("Signal payload must be a dictionary")
        try:
            json.dumps(self.payload)
        except (TypeError, ValueError) as err:
            raise SignalError(f"Signal payload must be JSON-serializable: {err}") from err


class SignalSource(Protocol):
    """Protocol for sources that yield signals to the reflex loop."""

    def poll(self, now: float) -> list[Signal]:
        """Poll the signal source for new signals at timestamp `now`."""
        ...


class NullSignalSource:
    """A no-op SignalSource that always returns an empty list."""

    def poll(self, now: float) -> list[Signal]:
        return []


class ListSignalSource:
    """A SignalSource backed by an in-memory list of signals.

    Used in tests to inject deterministic signal sequences.
    """

    def __init__(self, signals: Iterable[Signal]) -> None:
        self._signals: list[Signal] = list(signals)

    def poll(self, now: float) -> list[Signal]:
        batch = self._signals
        self._signals = []
        return batch

    def remaining(self) -> int:
        return len(self._signals)
