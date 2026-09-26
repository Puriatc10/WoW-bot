"""YOLO enemy detector (T-FIX-27).

Ported from ``hamberger@9b968a4`` ``perception/enemies.py`` (author
Amirm227) and reshaped:

* the default ``runs/detect/train-2/weights/best.pt`` path is gone — the
  weight path comes from config and is resolved relative to the repo
  root, and a missing file fails closed with a named remedy instead of
  loading a wrong model;
* ``ultralytics`` is imported **lazily inside the constructor**, so
  importing this module needs neither the optional ``yolo`` extra nor the
  ``YOLO_OFFLINE`` environment, and MOCK_MODE never touches torch;
* ``click_position`` is **not ported** — it is an actuation concern, and
  AGENTS.md §7 keeps OS input out of this package;
* the detector is throttled by ``[enemies].sampling_hz`` with cached
  detections in between (plan doc §6.6).

**Output convention:** bounding boxes stay ``(x1, y1, x2, y2)`` corner
pairs exactly as hamberger emits them. Converting to ``EnemyInfo.bbox``'s
``(x, y, w, h)`` origin+size form is the T-FIX-28 builder's job, so there
is one conversion site (plan doc §3.1 trap 3).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from wow_bot.perception import deps
from wow_bot.perception.capture import FrameLike, Throttle, normalize_bgr

__all__ = ["EnemyDetection", "EnemyDetector"]


@dataclass(frozen=True)
class EnemyDetection:
    """One detected enemy in frame pixels.

    ``bbox`` is ``(x1, y1, x2, y2)`` — a corner pair, not origin+size.
    """

    name: str
    bbox: tuple[int, int, int, int]
    confidence: float

    @property
    def center(self) -> tuple[int, int]:
        """Box centre in frame pixels."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def width(self) -> int:
        """Box width in pixels."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        """Box height in pixels."""
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        """Box area in square pixels (hamberger's proximity proxy)."""
        return self.width * self.height


class EnemyDetector:
    """Runs a config-supplied YOLO model over a frame.

    Construction is where the optional dependency and the weights file are
    both required, so a misconfiguration fails at wiring time rather than
    on the first frame.
    """

    def __init__(
        self,
        yolo_weights: str | Path,
        *,
        confidence: float = 0.5,
        sampling_hz: float = 2.0,
        base_dir: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        self.confidence = float(confidence)
        self.weights_path = self._resolve(yolo_weights, base_dir)
        if not self.weights_path.is_file():
            raise deps.PerceptionDependencyError(
                f"YOLO weights not found: {self.weights_path}. "
                "Point [enemies].yolo_weights at a local .pt file; weights are "
                "gitignored (plan doc §6.5) and MOCK_MODE never loads them."
            )
        self._throttle = Throttle(sampling_hz, clock=clock)
        self._last: list[EnemyDetection] = []
        # ultralytics pulls torch and must not be imported at module import
        # time; force the offline env before the lazy import (plan doc R-6).
        deps.apply_yolo_offline_env()
        ultralytics = deps.require_ultralytics()
        self.model: Any = ultralytics.YOLO(str(self.weights_path))

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
    def last_detections(self) -> list[EnemyDetection]:
        """The most recent detections (also the cached value while throttled)."""
        return list(self._last)

    def class_names(self) -> Sequence[str]:
        """Model class names, if the backend exposes them."""
        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            return [str(names[key]) for key in sorted(names)]
        return [str(name) for name in names]

    def detect(self, frame: FrameLike, *, now: float | None = None) -> list[EnemyDetection]:
        """Return detections for ``frame``, or the cache while throttled."""
        if not self._throttle.allow(now):
            return list(self._last)

        normalized = normalize_bgr(frame)
        results = self.model(normalized, conf=self.confidence, verbose=False)
        detections: list[EnemyDetection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                # ``xyxy[0]`` is normally flat, but a backend may hand back a
                # (1, 4) tensor; flatten so the corner pair unpacks either way.
                coordinates = np.asarray(box.xyxy[0].cpu().numpy()).reshape(-1)
                x1, y1, x2, y2 = (int(value) for value in coordinates[:4])
                class_id = int(box.cls[0])
                name = str(self.model.names[class_id])
                detections.append(
                    EnemyDetection(
                        name=name,
                        bbox=(x1, y1, x2, y2),
                        confidence=float(box.conf[0]),
                    )
                )

        self._last = detections
        self._throttle.record(now)
        return list(detections)

    def closest_enemy(self, frame: FrameLike, *, now: float | None = None) -> EnemyDetection | None:
        """Largest-box detection — hamberger's proximity proxy.

        This is *not* a distance estimate and must not be reported as one
        (plan doc §3.2: ``distance_estimate`` has no source).
        """
        detections = self.detect(frame, now=now)
        if not detections:
            return None
        return max(detections, key=lambda detection: detection.area)
