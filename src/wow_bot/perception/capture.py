"""Screen capture and frame budget helpers (T-FIX-27).

Ported from ``hamberger@9b968a4`` ``capture/screen_capture.py`` (author
Amirm227) and reshaped for the master architecture:

* import-time safety — ``mss`` is imported lazily inside
  :meth:`ScreenCapture.start`, so MOCK_MODE imports this module without
  the capture stack installed;
* no global capture singleton — a reader never reaches for a capture
  device, composition happens in the T-FIX-28 builder;
* per-channel frame budgets — :class:`Throttle` implements plan doc
  §6.6: cheap channels run every frame, OCR and YOLO are sampled on
  their own schedule and cached in between;
* no ``time.time()`` inside readers — every time source is injected or
  defaults to :func:`time.monotonic`, and the clock is a parameter so
  tests can advance it deterministically.

Paths are never hardcoded here; everything comes from
:class:`wow_bot.perception.perception_config.PerceptionConfig`.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from types import TracebackType
from typing import Any

import numpy as np

from wow_bot.perception import deps

__all__ = [
    "PER_FRAME_HZ",
    "FrameLike",
    "ScreenCapture",
    "Throttle",
    "normalize_bgr",
]

#: The "run on every frame" sampling budget used by the cheap channels.
PER_FRAME_HZ: float = float("inf")

#: Any real frame: ``(h, w, 3)`` BGR or ``(h, w, 4)`` BGRA, ``uint8``.
FrameLike = np.ndarray


class Throttle:
    """Per-channel sampling budget with cached-result semantics.

    :meth:`allow` is a pure predicate: it reports whether the channel may
    run at ``now`` without mutating state. The owner calls :meth:`record`
    only when it actually performs the expensive work, so a budget is
    never consumed by an early return.

    A non-finite ``hz`` means "run every frame" (plan doc §6.6).
    """

    __slots__ = ("_clock", "_hz", "_interval_s", "_last_s")

    def __init__(
        self,
        hz: float = PER_FRAME_HZ,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(hz, (int, float)) or isinstance(hz, bool):
            raise TypeError("hz must be a number")
        self._hz = float(hz)
        if self._hz <= 0.0 or self._hz != self._hz:
            raise ValueError("hz must be > 0 and not NaN")
        self._clock = clock
        self._interval_s = 0.0 if self._hz == PER_FRAME_HZ else 1.0 / self._hz
        self._last_s: float | None = None

    @property
    def hz(self) -> float:
        """Configured sampling budget in Hz (``inf`` means every frame)."""
        return self._hz

    @property
    def interval_s(self) -> float:
        """Minimum seconds between two accepted samples (0 for per-frame)."""
        return self._interval_s

    def allow(self, now: float | None = None) -> bool:
        """Return True when this channel may sample now."""
        if self._interval_s <= 0.0:
            return True
        if self._last_s is None:
            return True
        current = self._now() if now is None else now
        return (current - self._last_s) >= self._interval_s

    def record(self, now: float | None = None) -> None:
        """Commit a sample at ``now`` (defaults to the injected clock)."""
        self._last_s = self._now() if now is None else now

    def _now(self) -> float:
        return float(self._clock())


def normalize_bgr(frame: FrameLike) -> FrameLike:
    """Return a 3-channel BGR ``uint8`` view of ``frame``.

    ``mss`` hands back BGRA while synthesised test frames are usually BGR
    or grey; every reader funnels through here so channel handling is not
    duplicated per module. ``None`` is rejected rather than silently
    coerced, because a missing frame must stay explicit (T-FIX-32).
    """
    if frame is None:
        raise TypeError("frame must not be None")
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a numpy array")
    if frame.dtype != np.uint8:
        raise ValueError(f"frame dtype must be uint8, got {frame.dtype}")
    if frame.ndim == 2:
        return frame[:, :, np.newaxis].repeat(3, axis=2)
    if frame.ndim != 3:
        raise ValueError(f"frame must be 2D or 3D, got {frame.ndim}D")
    if frame.shape[2] == 4:
        return frame[:, :, :3]
    if frame.shape[2] == 3:
        return frame
    if frame.shape[2] == 1:
        return frame.repeat(3, axis=2)
    raise ValueError(f"frame must have 1, 3, or 4 channels, got {frame.shape[2]}")


class ScreenCapture:
    """``mss`` screen grabber on a daemon thread with a mode-driven rate.

    The capture device owns no reader logic and no reader holds a
    reference to it (T-FIX-27 contract). ``mss`` is imported in
    :meth:`start`, never at import time, so importing this module stays
    free of the capture stack.
    """

    def __init__(
        self,
        idle_fps: int = 10,
        combat_fps: int = 30,
        *,
        monitor_idx: int = 1,
        pool_size: int = 5,
    ) -> None:
        if idle_fps <= 0 or combat_fps <= 0:
            raise ValueError("idle_fps and combat_fps must be > 0")
        if pool_size <= 0:
            raise ValueError("pool_size must be > 0")
        self.idle_fps = int(idle_fps)
        self.combat_fps = int(combat_fps)
        self.current_fps = self.idle_fps
        self.monitor_idx = int(monitor_idx)
        self.pool_size = int(pool_size)
        self.running = False
        self.pool: list[FrameLike] = []
        self.pool_idx = 0
        self.latest_frame: FrameLike | None = None
        self._lock = threading.Lock()
        self._sct: Any | None = None
        self._monitor: dict[str, int] | None = None
        self._thread: threading.Thread | None = None

    @property
    def monitor(self) -> dict[str, int] | None:
        """The resolved monitor bounds, or ``None`` before :meth:`start`."""
        return self._monitor

    def set_mode(self, mode: str) -> int:
        """Select ``"combat"`` or ``"idle"`` rate and return the new FPS."""
        self.current_fps = self.combat_fps if mode == "combat" else self.idle_fps
        return self.current_fps

    def get_frame(self) -> FrameLike | None:
        """Return a copy of the latest frame, or ``None`` if none yet."""
        with self._lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def _open(self) -> None:
        mss = deps.require_mss()
        self._sct = mss.mss()
        monitors = self._sct.monitors
        if not 0 <= self.monitor_idx < len(monitors):
            raise ValueError(
                f"monitor_idx {self.monitor_idx} out of range; "
                f"{len(monitors)} monitors reported by mss"
            )
        self._monitor = dict(monitors[self.monitor_idx])
        height = int(self._monitor["height"])
        width = int(self._monitor["width"])
        self.pool = [
            np.empty((height, width, 4), dtype=np.uint8) for _ in range(self.pool_size)
        ]

    def grab(self) -> FrameLike | None:
        """Grab one frame synchronously and store it as the latest."""
        if self._sct is None or self._monitor is None:
            self._open()
        assert self._sct is not None and self._monitor is not None
        grabbed = np.array(self._sct.grab(self._monitor), dtype=np.uint8)
        with self._lock:
            buffer = self.pool[self.pool_idx] if self.pool else None
            if buffer is not None and buffer.shape == grabbed.shape:
                buffer[:] = grabbed
                self.latest_frame = buffer.copy()
            else:
                self.latest_frame = grabbed
            if self.pool:
                self.pool_idx = (self.pool_idx + 1) % len(self.pool)
        return self.latest_frame

    def start(self) -> None:
        """Open the device and start the daemon capture thread."""
        if self.running:
            return
        self._open()
        self.running = True
        thread = threading.Thread(target=self._loop, daemon=True)
        self._thread = thread
        thread.start()

    def _loop(self) -> None:
        while self.running:
            started = time.monotonic()
            self.grab()
            remaining = 1.0 / self.current_fps - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)

    def stop(self) -> None:
        """Stop the capture thread and release the device."""
        self.running = False
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
            self._thread = None
        sct = self._sct
        self._sct = None
        self._monitor = None
        if sct is not None and hasattr(sct, "close"):
            sct.close()

    def __enter__(self) -> ScreenCapture:  # noqa: PYI034 - explicit, not Self
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
