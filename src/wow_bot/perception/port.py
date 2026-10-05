"""Async perception port and scheduling bridge (Task T-FIX-20).

Provides PerceptionPort, bridging an asynchronous PerceptionBackend to
synchronous consumers (such as the lab runner or reflex fast loop)
via a background snapshot pump and a synchronous sample() accessor.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from typing import Self

from wow_bot.perception.protocol import PerceptionBackend
from wow_bot.shared.interfaces import GameState


class PerceptionPortError(Exception):
    """Base error raised by PerceptionPort."""


class PerceptionStaleError(PerceptionPortError):
    """Raised when the latest snapshot is missing, uninitialized, or exceeds staleness bound."""


class PerceptionBackendError(PerceptionPortError):
    """Raised when the underlying perception backend failed during snapshot acquisition."""


class PerceptionPort:
    """Bridge between asynchronous PerceptionBackend and synchronous consumers.

    Owns the latest snapshot via a background snapshot() pump and exposes a
    synchronous sample() accessor.

    Invariants:
    - Never fabricates a snapshot.
    - Never silently reuses or substitutes a stale snapshot.
    - Emits GameState without branding or branching on backend identity.
    - Thread-safe and asyncio-safe.
    """

    def __init__(
        self,
        backend: PerceptionBackend,
        *,
        staleness_bound_s: float = 0.5,
        pump_interval_s: float = 0.05,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Initialize PerceptionPort with a backend and timing parameters.

        Parameters
        ----------
        backend:
            The underlying async PerceptionBackend instance.
        staleness_bound_s:
            Maximum allowable snapshot age in seconds before sample() declares it stale.
        pump_interval_s:
            Target interval in seconds between consecutive background snapshot() captures.
        clock:
            Monotonic time provider, default is time.monotonic.
        """
        if not isinstance(backend, PerceptionBackend):
            raise TypeError(f"backend must be a PerceptionBackend, got {type(backend).__name__}")
        if (
            isinstance(staleness_bound_s, bool)
            or not isinstance(staleness_bound_s, (int, float))
            or staleness_bound_s <= 0.0
        ):
            raise ValueError(f"staleness_bound_s must be > 0.0, got {staleness_bound_s!r}")
        if (
            isinstance(pump_interval_s, bool)
            or not isinstance(pump_interval_s, (int, float))
            or pump_interval_s <= 0.0
        ):
            raise ValueError(f"pump_interval_s must be > 0.0, got {pump_interval_s!r}")

        self._backend = backend
        self._staleness_bound_s = float(staleness_bound_s)
        self._pump_interval_s = float(pump_interval_s)
        self._clock = clock if clock is not None else time.monotonic

        self._latest_snapshot: GameState | None = None
        self._latest_timestamp: float | None = None
        self._last_error: Exception | None = None

        self._lock = threading.Lock()
        self._pump_task: asyncio.Task[None] | None = None
        self._thread: threading.Thread | None = None
        self._thread_loop: asyncio.AbstractEventLoop | None = None
        self._is_running = False
        self._stop_event = threading.Event()

    @property
    def backend(self) -> PerceptionBackend:
        """The underlying PerceptionBackend instance."""
        return self._backend

    @property
    def is_running(self) -> bool:
        """Whether the background pump is currently running."""
        return self._is_running

    @property
    def staleness_bound_s(self) -> float:
        """The snapshot staleness bound in seconds."""
        return self._staleness_bound_s

    @property
    def pump_interval_s(self) -> float:
        """The pump loop interval in seconds."""
        return self._pump_interval_s

    @property
    def latest_timestamp(self) -> float | None:
        """Monotonic timestamp of the latest snapshot, or None if uninitialized."""
        with self._lock:
            return self._latest_timestamp

    @property
    def latest_snapshot(self) -> GameState | None:
        """The latest snapshot without staleness validation, or None if uninitialized."""
        with self._lock:
            return self._latest_snapshot

    async def pump_once(self) -> GameState:
        """Execute a single snapshot capture cycle and store the result.

        Returns
        -------
        GameState
            The newly captured GameState.

        Raises
        ------
        Exception
            Any exception raised by the backend snapshot() is recorded and re-raised.
        """
        try:
            snapshot = await self._backend.snapshot()
            now = self._clock()
            with self._lock:
                self._latest_snapshot = snapshot
                self._latest_timestamp = now
                self._last_error = None
            return snapshot
        except Exception as exc:
            with self._lock:
                self._last_error = exc
            raise

    async def run_pump_loop(self) -> None:
        """Coroutine that repeatedly runs pump_once at pump_interval_s until stopped."""
        while not self._stop_event.is_set():
            try:
                await self.pump_once()
            except asyncio.CancelledError:
                break
            except Exception:  # noqa: BLE001, S110
                # Backend error is recorded in self._last_error; loop continues to attempt next frame
                pass

            try:
                await asyncio.sleep(self._pump_interval_s)
            except asyncio.CancelledError:
                break

    def start_pump(
        self,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
        threaded: bool = False,
    ) -> None:
        """Start the background snapshot pump.

        Parameters
        ----------
        loop:
            Optional asyncio event loop to run the pump task in.
        threaded:
            If True, spawns a dedicated background thread with its own event loop
            so the pump is not blocked by synchronous calls on the main thread.
        """
        if self._is_running:
            return

        self._stop_event.clear()
        self._is_running = True

        if threaded:

            def _thread_worker() -> None:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                self._thread_loop = new_loop
                try:
                    new_loop.run_until_complete(self.run_pump_loop())
                finally:
                    new_loop.close()

            self._thread = threading.Thread(
                target=_thread_worker,
                name="PerceptionPortPumpThread",
                daemon=True,
            )
            self._thread.start()
        else:
            target_loop = loop
            if target_loop is None:
                try:
                    target_loop = asyncio.get_running_loop()
                except RuntimeError:
                    # No active loop on current thread, fall back to threaded worker
                    self._is_running = False
                    self.start_pump(threaded=True)
                    return
            self._pump_task = target_loop.create_task(self.run_pump_loop())

    def stop_pump(self) -> None:
        """Stop the background snapshot pump and release resources."""
        if not self._is_running:
            return

        self._stop_event.set()

        if self._pump_task is not None:
            self._pump_task.cancel()
            self._pump_task = None

        if self._thread_loop is not None:
            self._thread_loop.call_soon_threadsafe(self._stop_event.set)
            self._thread_loop = None

        if self._thread is not None and self._thread.is_alive():
            if self._thread != threading.current_thread():
                self._thread.join(timeout=2.0)
            self._thread = None

        self._is_running = False

    def close(self) -> None:
        """Alias for stop_pump()."""
        self.stop_pump()

    async def __aenter__(self) -> Self:
        self.start_pump()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.stop_pump()

    def sample(self) -> GameState:
        """Synchronously retrieve the latest GameState snapshot.

        Returns
        -------
        GameState
            The authoritative GameState snapshot.

        Raises
        ------
        PerceptionBackendError:
            If the latest snapshot capture failed.
        PerceptionStaleError:
            If no snapshot has been captured yet, or if the snapshot's age
            exceeds staleness_bound_s.
        """
        with self._lock:
            if self._last_error is not None:
                raise PerceptionBackendError(
                    f"Backend snapshot failed: {self._last_error}"
                ) from self._last_error

            if self._latest_snapshot is None or self._latest_timestamp is None:
                raise PerceptionStaleError("No perception snapshot available (uninitialized)")

            now = self._clock()
            age = now - self._latest_timestamp
            if age > self._staleness_bound_s:
                raise PerceptionStaleError(
                    f"Perception snapshot is stale: age {age:.4f}s exceeds bound {self._staleness_bound_s:.4f}s"
                )

            return self._latest_snapshot
