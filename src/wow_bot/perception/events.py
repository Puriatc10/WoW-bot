"""Combat-log and chat event detector (T-FIX-27).

Ported from ``hamberger@9b968a4`` ``perception/events.py`` (author
Amirm227) and reshaped:

* regions, keyword tables and the OCR budget are injected — nothing is a
  module constant and no ``tesseract_cmd`` is assigned at import time;
* the OCR fan-out is explicit: a combat-log line is read through the
  ``None`` and ``purple`` filters, a chat line through ``None``,
  ``green`` and ``yellow`` (5 Tesseract calls per sampled frame, down
  from hamberger's 6, and the whole channel is throttled by
  ``[events].sampling_hz`` with cached events in between);
* the output is an :class:`EventCandidate` with the frame's text and
  source but **no timestamp** — the monotonic timestamp is attached by
  the T-FIX-28 builder so readers stay deterministic given one frame
  (plan doc §3.1 trap 6).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from loguru import logger

from wow_bot.perception import deps
from wow_bot.perception.capture import FrameLike, Throttle, normalize_bgr

__all__ = [
    "CHAT_KEYWORDS",
    "COMBAT_KEYWORDS",
    "EVENT_PRIORITY",
    "EventCandidate",
    "EventDetector",
]

#: Combat-log keyword table, ported verbatim from hamberger.
COMBAT_KEYWORDS: Mapping[str, tuple[str, ...]] = {
    "death": ("dies", "slain", "died", "you killed", "experience"),
    "damage_dealt": ("you hit", "auto shot", "arcane shot"),
    "damage_taken": ("hits you", "hit you", "you for"),
    "miss": ("misses", "dodged", "parried"),
}

#: Chat keyword table, ported verbatim from hamberger.
CHAT_KEYWORDS: Mapping[str, tuple[str, ...]] = {
    "loot": ("receive", "loot", "received", "item"),
    "quest": ("quest", "accepted", "completed"),
    "level_up": ("reached level", "level up"),
    "warning": ("warning", "you are", "cannot"),
}

#: Classification precedence, ported verbatim from hamberger.
EVENT_PRIORITY: tuple[str, ...] = (
    "death",
    "loot",
    "level_up",
    "quest",
    "damage_dealt",
    "damage_taken",
    "miss",
    "warning",
)

#: Colour filters per region: (combat log, chat).
_COMBAT_FILTERS: tuple[str | None, ...] = (None, "purple")
_CHAT_FILTERS: tuple[str | None, ...] = (None, "green", "yellow")

_COLOR_FILTER_RANGES: Mapping[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "green": ((30, 80, 100), (90, 255, 255)),
    "yellow": ((20, 80, 100), (35, 255, 255)),
    "purple": ((120, 50, 100), (160, 255, 255)),
}

_GRAY_THRESHOLD = 90
_OCR_UPSCALE = 4

#: Dedup memory is dropped once it grows past this many lines.
_DEFAULT_DEDUP_LIMIT = 100


@dataclass(frozen=True)
class EventCandidate:
    """One classified log line, timestamped later by the builder."""

    type: str
    text: str
    source: str


class EventDetector:
    """Colour-filtered OCR of the combat-log and chat regions.

    The detector is stateful only for deduplication: a line already
    reported for its region is not reported again until the dedup set is
    cleared, which is what stops a static chat line from emitting an
    event on every frame.
    """

    def __init__(
        self,
        combat_region: tuple[int, int, int, int],
        chat_region: tuple[int, int, int, int],
        *,
        sampling_hz: float = 1.0,
        dedup_limit: int = _DEFAULT_DEDUP_LIMIT,
        combat_keywords: Mapping[str, tuple[str, ...]] = COMBAT_KEYWORDS,
        chat_keywords: Mapping[str, tuple[str, ...]] = CHAT_KEYWORDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if dedup_limit <= 0:
            raise ValueError("dedup_limit must be > 0")
        self.combat_region = combat_region
        self.chat_region = chat_region
        self.combat_keywords = dict(combat_keywords)
        self.chat_keywords = dict(chat_keywords)
        self.dedup_limit = int(dedup_limit)
        self._combat_seen: set[str] = set()
        self._chat_seen: set[str] = set()
        self._throttle = Throttle(sampling_hz, clock=clock)
        self._last: list[EventCandidate] = []
        self._tesseract_configured = False

    @property
    def sampling_hz(self) -> float:
        """Configured sampling budget in Hz."""
        return self._throttle.hz

    @property
    def last_events(self) -> list[EventCandidate]:
        """The most recent events (also the cached value while throttled)."""
        return list(self._last)

    @property
    def ocr_calls_per_sample(self) -> int:
        """Tesseract calls issued per accepted sample (5 in this revision)."""
        return len(_COMBAT_FILTERS) + len(_CHAT_FILTERS)

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------
    def _configure_tesseract(self, tesseract_cmd: str) -> None:
        if self._tesseract_configured:
            return
        pytesseract = deps.require_pytesseract()
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self._tesseract_configured = True

    def _read_region(
        self,
        frame: FrameLike,
        region: tuple[int, int, int, int],
        color_filter: str | None = None,
    ) -> str:
        cv2 = deps.require_cv2()
        pytesseract = deps.require_pytesseract()
        x, y, width, height = region
        frame_h, frame_w = frame.shape[:2]
        x2 = min(x + width, frame_w)
        y2 = min(y + height, frame_h)
        if x < 0 or y < 0 or x2 <= x or y2 <= y:
            return ""

        crop = frame[y:y2, x:x2]
        if crop.size == 0:
            return ""

        big = cv2.resize(
            crop, (0, 0), fx=_OCR_UPSCALE, fy=_OCR_UPSCALE, interpolation=cv2.INTER_CUBIC
        )
        if color_filter is not None:
            lower, upper = _COLOR_FILTER_RANGES[color_filter]
            hsv = cv2.cvtColor(big, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
        else:
            gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, _GRAY_THRESHOLD, 255, cv2.THRESH_BINARY)

        text: str = pytesseract.image_to_string(mask, config="--psm 6")
        return text

    def _read_region_multi(
        self,
        frame: FrameLike,
        region: tuple[int, int, int, int],
        filters: Iterable[str | None],
    ) -> str:
        texts: list[str] = []
        for color in filters:
            try:
                text = self._read_region(frame, region, color_filter=color)
            except deps.PerceptionDependencyError:
                # A missing dependency is never a silent skip.
                raise
            except Exception as exc:  # noqa: BLE001 - one bad filter must not kill the channel
                logger.debug(
                    "event OCR filter {!r} failed on region {}: {}",
                    color,
                    region,
                    exc,
                )
                continue
            if text:
                texts.append(text)
        return "\n".join(texts)

    # ------------------------------------------------------------------
    # classification
    # ------------------------------------------------------------------
    @staticmethod
    def _is_valid_line(line: str) -> bool:
        line = line.strip()
        if len(line) < 6:
            return False
        if line.count("[DAMAGE") > 1 or line.count("[") > 2:
            return False
        alnum = sum(
            character.isalnum() or character in " .,:;'[]()-" for character in line
        )
        return alnum / max(len(line), 1) >= 0.4

    @staticmethod
    def _classify(text: str, keywords: Mapping[str, tuple[str, ...]]) -> str | None:
        text_lower = text.lower()
        for event_type in EVENT_PRIORITY:
            if event_type not in keywords:
                continue
            for keyword in keywords[event_type]:
                if keyword in text_lower:
                    return event_type
        return None

    def _process_region(
        self,
        frame: FrameLike,
        region: tuple[int, int, int, int],
        keywords: Mapping[str, tuple[str, ...]],
        seen: set[str],
        tag: str,
        filters: Iterable[str | None],
    ) -> list[EventCandidate]:
        text = self._read_region_multi(frame, region, filters)
        events: list[EventCandidate] = []
        if not text:
            return events

        for raw_line in text.split("\n"):
            line = raw_line.strip()
            if not self._is_valid_line(line):
                continue
            if line in seen:
                continue
            event_type = self._classify(line, keywords)
            if event_type is None:
                continue
            events.append(EventCandidate(type=event_type, text=line, source=tag))
            seen.add(line)

        if len(seen) > self.dedup_limit:
            seen.clear()
        return events

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------
    def detect(
        self,
        frame: FrameLike,
        *,
        now: float | None = None,
        tesseract_cmd: str | None = None,
    ) -> list[EventCandidate]:
        """Return newly observed events, or the cache while throttled."""
        if not self._throttle.allow(now):
            return list(self._last)
        if tesseract_cmd is not None:
            self._configure_tesseract(tesseract_cmd)

        normalized = normalize_bgr(frame)
        events: list[EventCandidate] = []
        events += self._process_region(
            normalized,
            self.combat_region,
            self.combat_keywords,
            self._combat_seen,
            "combat",
            _COMBAT_FILTERS,
        )
        events += self._process_region(
            normalized,
            self.chat_region,
            self.chat_keywords,
            self._chat_seen,
            "chat",
            _CHAT_FILTERS,
        )
        self._last = events
        self._throttle.record(now)
        return list(events)
