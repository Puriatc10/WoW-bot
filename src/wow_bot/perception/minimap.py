"""Minimap arrow tracker: position, tip angle, circular smoothing (T-FIX-27).

Ported from ``hamberger@9b968a4`` ``perception/minimap.py`` (author
Amirm227). The template match and the circular-mean smoothing
(sin/cos mean rather than naive averaging) are kept as written; the
reshaping is:

* the template path and the smoothing window come from config, and the
  template is loaded lazily on first use so constructing the tracker never
  touches the filesystem at import time;
* the rotation is exposed in **degrees** (hamberger's unit) through
  :meth:`angle_degrees` and in **radians** — the unit the canonical
  ``GameState.facing`` uses — through :attr:`facing_radians`. That
  property is the **single** degrees→radians site in the package, so
  there is exactly one place to get the conversion wrong (plan doc §3.1
  trap 2);
* ``[minimap].sampling_hz`` throttles the template match with cached
  results in between (plan doc §6.6).

**Not a world position.** :meth:`update` returns the arrow's centre
*inside the minimap ROI* — screen pixels. hamberger has no world-pose
channel, so these coordinates must never be written to
``GameState.position`` (plan doc finding F-1). The T-FIX-28 builder
consumes only :attr:`facing_radians`; a real pose source is T-FIX-30.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wow_bot.perception import deps
from wow_bot.perception.capture import FrameLike, Throttle, normalize_bgr

__all__ = ["MinimapReading", "MinimapTracker", "degrees_to_facing_radians"]

#: White-arrow HSV gate (low saturation, high value).
_WHITE_LOW = (0, 0, 180)
_WHITE_HIGH = (180, 60, 255)

#: Minimum white pixels before a tip estimate is attempted.
_MIN_TIP_PIXELS = 5

#: Default template-match score for accepting the arrow.
_DEFAULT_MATCH_THRESH = 0.55


def degrees_to_facing_radians(angle_degrees: float) -> float:
    """Convert a screen arrow angle in ``[0, 360)`` to ``[-pi, pi)`` radians.

    The single conversion site for facing (T-FIX-27 contract). The
    minimap angle is degrees counter-clockwise from the screen's +x axis,
    with ``y`` growing downward; the canonical ``GameState.facing`` is
    the mathematical angle with ``y`` growing upward. The two conventions
    therefore agree on 0 and differ only by the direction of ``y``, so no
    sign flip is applied. The result is mapped into the half-open range
    ``[-pi, pi)``, so ``180`` degrees becomes ``-pi`` rather than ``+pi``.
    """
    wrapped = math.fmod(math.radians(angle_degrees), 2.0 * math.pi)
    if wrapped >= math.pi:
        wrapped -= 2.0 * math.pi
    elif wrapped < -math.pi:
        wrapped += 2.0 * math.pi
    return wrapped


@dataclass(frozen=True)
class MinimapReading:
    """One frame's minimap observation.

    ``position_px`` is a **screen-pixel** point inside the minimap ROI,
    not a world coordinate. ``angle_degrees`` is in ``[0, 360)``.
    """

    position_px: tuple[float, float] | None
    angle_degrees: float | None
    confidence: float

    @property
    def facing_radians(self) -> float | None:
        """The arrow heading in radians, normalised to ``[-pi, pi)``.

        This is the only degrees→radians conversion in the perception
        package. The minimap angle is measured counter-clockwise with
        screen ``y`` growing downward, so negating it restores the
        mathematical convention ``MockPerception`` uses for
        ``GameState.facing`` (0 = east, pi/2 = north, -pi/2 = south).
        """
        if self.angle_degrees is None:
            return None
        return degrees_to_facing_radians(self.angle_degrees)


class MinimapTracker:
    """Matches the player arrow on the minimap and tracks its heading."""

    def __init__(
        self,
        minimap_roi: tuple[int, int, int, int],
        arrow_template: str | Path,
        *,
        match_thresh: float = _DEFAULT_MATCH_THRESH,
        smoothing_frames: int = 5,
        sampling_hz: float = 5.0,
        base_dir: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if smoothing_frames <= 0:
            raise ValueError("smoothing_frames must be > 0")
        self.roi = minimap_roi
        self.match_thresh = float(match_thresh)
        self.smoothing_frames = int(smoothing_frames)
        self._template_path = self._resolve(arrow_template, base_dir)
        self._template: FrameLike | None = None
        self.angle_history: list[float] = []
        self.last_pos: tuple[float, float] | None = None
        self.last_angle: float | None = None
        self._throttle = Throttle(sampling_hz, clock=clock)
        self._last = MinimapReading(position_px=None, angle_degrees=None, confidence=0.0)

    @staticmethod
    def _resolve(path: str | Path, base_dir: Path | None) -> Path:
        candidate = Path(path)
        if base_dir is not None and not candidate.is_absolute():
            candidate = base_dir / candidate
        return candidate

    @property
    def sampling_hz(self) -> float:
        """Configured sampling budget in Hz."""
        return self._throttle.hz

    @property
    def template_path(self) -> Path:
        """Resolved path of the arrow template (read on first use)."""
        return self._template_path

    @property
    def last_reading(self) -> MinimapReading:
        """The most recent reading (also the cached value while throttled)."""
        return self._last

    @property
    def facing_radians(self) -> float | None:
        """Most recent heading in radians, so callers need no conversion."""
        return self._last.facing_radians

    def _load_template(self) -> FrameLike:
        if self._template is not None:
            return self._template
        cv2 = deps.require_cv2()
        template = cv2.imread(str(self._template_path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise deps.PerceptionDependencyError(
                f"Minimap arrow template not found or unreadable: {self._template_path}. "
                "Point [minimap].arrow_template at a readable PNG; MOCK_MODE never "
                "loads templates."
            )
        self._template = np.asarray(template)
        return self._template

    def _find_arrow(
        self, crop: FrameLike
    ) -> tuple[tuple[float, float] | None, tuple[int, int] | None, float]:
        cv2 = deps.require_cv2()
        template = self._load_template()
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        confidence = float(max_val)
        if confidence < self.match_thresh:
            return None, None, confidence
        template_h, template_w = template.shape[:2]
        center = (max_loc[0] + template_w / 2.0, max_loc[1] + template_h / 2.0)
        return center, (int(max_loc[0]), int(max_loc[1])), confidence

    def _estimate_facing(self, crop: FrameLike, top_left: tuple[int, int]) -> float | None:
        """Estimate the arrow tip angle in degrees ``[0, 360)``.

        The tip is the white pixel farthest from the white centroid; the
        screen ``y`` axis is flipped so the angle reads as a mathematical
        rotation.
        """
        cv2 = deps.require_cv2()
        template = self._load_template()
        template_h, template_w = template.shape[:2]
        x, y = top_left
        region = crop[y : y + template_h, x : x + template_w]
        if region.size == 0:
            return None

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, _WHITE_LOW, _WHITE_HIGH)
        ys, xs = np.nonzero(white_mask > 0)
        if xs.size < _MIN_TIP_PIXELS:
            return None

        mean_x = float(xs.mean())
        mean_y = float(ys.mean())
        distances = np.sqrt((xs - mean_x) ** 2 + (ys - mean_y) ** 2)
        tip = int(np.argmax(distances))
        tip_x = float(xs[tip])
        tip_y = float(ys[tip])

        angle = math.degrees(math.atan2(-(tip_y - mean_y), tip_x - mean_x))
        if angle < 0:
            angle += 360.0
        return angle

    def _smooth_angle(self, angle: float | None) -> float | None:
        """Circular mean over the last ``smoothing_frames`` angles."""
        if angle is None:
            return self.last_angle
        self.angle_history.append(angle)
        if len(self.angle_history) > self.smoothing_frames:
            self.angle_history.pop(0)
        radians = np.radians(np.asarray(self.angle_history, dtype=float))
        mean_sin = float(np.mean(np.sin(radians)))
        mean_cos = float(np.mean(np.cos(radians)))
        smoothed = math.degrees(math.atan2(mean_sin, mean_cos))
        if smoothed < 0:
            smoothed += 360.0
        return smoothed

    def update(self, frame: FrameLike, *, now: float | None = None) -> MinimapReading:
        """Return the arrow position/heading, or the cache while throttled."""
        if not self._throttle.allow(now):
            return self._last

        normalized = normalize_bgr(frame)
        x, y, width, height = self.roi
        crop = normalized[y : y + height, x : x + width]
        if crop.size == 0:
            self._throttle.record(now)
            return self._last

        position, top_left, confidence = self._find_arrow(crop)
        if position is None or top_left is None:
            # No observation this frame: the arrow is not where it was, so
            # there is no position to trust. The last *heading* is kept
            # because orientation is a smoothed quantity, and confidence
            # drops to what this frame actually matched.
            self._last = MinimapReading(
                position_px=None,
                angle_degrees=self.last_angle,
                confidence=confidence,
            )
            self._throttle.record(now)
            return self._last

        angle = self._smooth_angle(self._estimate_facing(crop, top_left))
        self.last_pos = position
        if angle is not None:
            self.last_angle = angle

        self._last = MinimapReading(
            position_px=self.last_pos,
            angle_degrees=self.last_angle,
            confidence=confidence,
        )
        self._throttle.record(now)
        return self._last
