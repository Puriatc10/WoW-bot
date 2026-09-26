"""Target nameplate reader: template match + OCR + whitelist (T-FIX-27).

Ported from ``hamberger@9b968a4`` ``perception/target.py`` (author
Amirm227) and reshaped:

* the module-level ``KNOWN_ENEMIES`` list and the hardcoded Tesseract path
  are gone — both come from
  :class:`~wow_bot.perception.perception_config.PerceptionConfig`;
* the OCR thresholds that were magic numbers inside ``_find_target`` are
  config keys (``ocr_thresh`` / ``ocr_confirm_thresh``) and are enforced,
  not decorative;
* the position lock is unchanged in spirit: a locked nameplate is only
  re-matched locally instead of searched for again, and the name is
  re-OCR'd once every ``name_refresh_frames`` accepted reads;
* **unit contract:** :attr:`TargetReading.hp_pct` is an ``int 0..100``,
  matching hamberger. The ``/100.0`` normalisation to the canonical
  ``TargetInfo.hp_pct`` fraction belongs to the T-FIX-28 builder and is
  deliberately *not* done here (plan doc §3.1 trap 1);
* this is an OCR channel, so it is throttled by ``[target].sampling_hz``
  with cached results between samples (plan doc §6.6).
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wow_bot.perception import deps
from wow_bot.perception.capture import FrameLike, Throttle, normalize_bgr

__all__ = ["TargetReader", "TargetReading"]

#: Fraction of the frame searched for the nameplate (hamberger's window).
_SEARCH_X0 = 0.35
_SEARCH_X1 = 0.80
_SEARCH_Y0 = 0.50
_SEARCH_Y1 = 0.85

#: How far below the matched nameplate the target HP bar starts.
_HP_BAR_OFFSET_PX = 3
_HP_BAR_SCAN_PX = 30
_HP_BAR_MAX_WIDTH_PX = 250

#: Upscale factor before OCR — the nameplate text is tiny.
_OCR_UPSCALE = 6

#: Accepted reads before the locked nameplate is re-OCR'd.
_DEFAULT_NAME_REFRESH_FRAMES = 5

#: Consecutive misses before the cached name is dropped.
_DEFAULT_MISS_LIMIT = 3


@dataclass(frozen=True)
class TargetReading:
    """One frame's target observation.

    ``hp_pct`` is an ``int 0..100`` (hamberger convention) and is only
    meaningful when ``name`` is not ``None``. ``frame_confidence`` is the
    template-match score; ``ocr_confidence`` is set only on reads that
    ran OCR, and is ``None`` on locked frames served from the cache.
    """

    name: str | None
    hp_pct: int
    frame_confidence: float
    ocr_confidence: float | None = None

    @property
    def found(self) -> bool:
        """True when a whitelisted target name is currently observed."""
        return self.name is not None


class TargetReader:
    """Finds the target nameplate, reads its name and HP, and locks onto it."""

    def __init__(
        self,
        name_template: str,
        frame_template: str,
        known_enemies: tuple[str, ...],
        *,
        match_thresh: float = 0.35,
        confirm_thresh: float = 0.80,
        ocr_thresh: float = 0.50,
        ocr_confirm_thresh: float = 0.85,
        sampling_hz: float = 2.0,
        name_refresh_frames: int = _DEFAULT_NAME_REFRESH_FRAMES,
        miss_limit: int = _DEFAULT_MISS_LIMIT,
        base_dir: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not known_enemies:
            raise ValueError("known_enemies must not be empty")
        if name_refresh_frames <= 0:
            raise ValueError("name_refresh_frames must be > 0")
        if miss_limit <= 0:
            raise ValueError("miss_limit must be > 0")

        self.known_enemies = tuple(known_enemies)
        self.match_thresh = float(match_thresh)
        self.confirm_thresh = float(confirm_thresh)
        self.ocr_thresh = float(ocr_thresh)
        self.ocr_confirm_thresh = float(ocr_confirm_thresh)
        self.name_refresh_frames = int(name_refresh_frames)
        self.miss_limit = int(miss_limit)

        self._frame_template = self._load_template(frame_template, base_dir)
        self._name_template = self._load_template(name_template, base_dir)
        self._name_template_gray = self._to_gray(self._name_template)
        self._frame_template_gray = self._to_gray(self._frame_template)

        self.locked_position: tuple[int, int] | None = None
        self.last_name: str | None = None
        self.last_ocr_confidence: float | None = None
        self._miss_count = 0
        self._refresh_count = 0
        self._throttle = Throttle(sampling_hz, clock=clock)
        self._tesseract_configured = False
        self._last = TargetReading(name=None, hp_pct=0, frame_confidence=0.0)

    @property
    def sampling_hz(self) -> float:
        """Configured sampling budget in Hz."""
        return self._throttle.hz

    @property
    def last_reading(self) -> TargetReading:
        """The most recent reading (also the cached value while throttled)."""
        return self._last

    # ------------------------------------------------------------------
    # template loading
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve(path: str | Path, base_dir: Path | None) -> Path:
        candidate = Path(path)
        if base_dir is not None and not candidate.is_absolute():
            candidate = base_dir / candidate
        return candidate

    @classmethod
    def _load_template(cls, path: str | Path, base_dir: Path | None) -> FrameLike:
        cv2 = deps.require_cv2()
        resolved = cls._resolve(path, base_dir)
        template = cv2.imread(str(resolved))
        if template is None:
            raise deps.PerceptionDependencyError(
                f"Target template not found or unreadable: {resolved}. "
                "Point [target].name_template / [target].frame_template at a "
                "readable PNG; MOCK_MODE never loads templates."
            )
        return np.asarray(template)

    @staticmethod
    def _to_gray(image: FrameLike) -> FrameLike:
        cv2 = deps.require_cv2()
        gray: FrameLike = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    # ------------------------------------------------------------------
    # colour heuristics (unchanged from hamberger)
    # ------------------------------------------------------------------
    @staticmethod
    def _is_green(b: int, g: int, r: int) -> bool:
        return g > 90 and g > r + 20 and g > b + 20

    @staticmethod
    def _is_golden(b: int, g: int, r: int) -> bool:
        return r > 170 and g > 120 and b < 130

    # ------------------------------------------------------------------
    # nameplate search and lock confirmation
    # ------------------------------------------------------------------
    def _find_target(self, frame: FrameLike) -> tuple[tuple[int, int] | None, float]:
        cv2 = deps.require_cv2()
        height, width = frame.shape[:2]
        x1 = int(width * _SEARCH_X0)
        x2 = int(width * _SEARCH_X1)
        y1 = int(height * _SEARCH_Y0)
        y2 = int(height * _SEARCH_Y1)
        roi = frame[y1:y2, x1:x2]
        threshold = self._effective_action_thresh()
        if roi.size == 0 or threshold <= 0:
            return None, 0.0

        gray = self._to_gray(roi)
        result = cv2.matchTemplate(gray, self._name_template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        confidence = float(max_val)
        if confidence < threshold:
            return None, confidence
        return (x1 + int(max_loc[0]), y1 + int(max_loc[1])), confidence

    def _effective_action_thresh(self) -> float:
        """First-pass threshold for locating a nameplate candidate.

        ``ocr_thresh`` tightens the config ``match_thresh`` when it is set
        higher, so all four hamberger constants (0.35/0.80/0.50/0.85) act
        as real gates rather than documentation.
        """
        return max(self.match_thresh, self.ocr_thresh)

    def _confirms(self, confidence: float) -> bool:
        """Whether a candidate is strong enough to take the position lock.

        ``confirm_thresh`` is hamberger's *confirmation* constant: a weak
        first match is not enough to commit to a nameplate, because a
        false lock then blocks the real search.
        """
        return confidence >= max(self.confirm_thresh, self._effective_action_thresh())

    def _is_target_at(self, frame: FrameLike, position: tuple[int, int]) -> float:
        cv2 = deps.require_cv2()
        template_h, template_w = self._name_template.shape[:2]
        x, y = position
        height, width = frame.shape[:2]
        if x < 0 or y < 0 or x + template_w > width or y + template_h > height:
            return 0.0
        roi = frame[y : y + template_h, x : x + template_w]
        if roi.size == 0:
            return 0.0
        gray = self._to_gray(roi)
        result = cv2.matchTemplate(gray, self._name_template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return float(max_val)

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------
    def _configure_tesseract(self, tesseract_cmd: str) -> None:
        if self._tesseract_configured:
            return
        pytesseract = deps.require_pytesseract()
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self._tesseract_configured = True

    def _ocr_name_candidates(self, frame: FrameLike, position: tuple[int, int]) -> list[str]:
        cv2 = deps.require_cv2()
        pytesseract = deps.require_pytesseract()
        template_h, template_w = self._name_template.shape[:2]
        x, y = position
        height, width = frame.shape[:2]
        if x + template_w > width or y + template_h > height:
            return []
        crop = frame[y : y + template_h, x : x + template_w]
        if crop.size == 0:
            return []

        big = cv2.resize(
            crop,
            (0, 0),
            fx=_OCR_UPSCALE,
            fy=_OCR_UPSCALE,
            interpolation=cv2.INTER_CUBIC,
        )
        gray = self._to_gray(big)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, fixed = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
        inverted = cv2.bitwise_not(otsu)

        candidates: list[str] = []
        for image in (otsu, fixed, inverted):
            text = pytesseract.image_to_string(image, config="--psm 7").strip()
            candidates.append(text)
        return candidates

    def _read_name(self, frame: FrameLike, position: tuple[int, int]) -> tuple[str | None, float]:
        """OCR the locked nameplate; return the whitelist match and confidence."""
        candidates = self._ocr_name_candidates(frame, position)
        best_name: str | None = None
        best_score = 0
        best_confidence = 0.0
        fallback = ""
        for text in candidates:
            if len(text) > len(fallback):
                fallback = text
            matched, score = self._match_whitelist(text)
            if matched is not None and score > best_score:
                best_name = matched
                best_score = score
                best_confidence = self._ocr_confidence(text, matched)

        if best_name is None:
            # No whitelist hit: report the best raw text explicitly instead
            # of inventing a name or reusing a stale one.
            return None, self._text_confidence(fallback)
        return best_name, best_confidence

    def _text_confidence(self, text: str) -> float:
        """Placeholder-free confidence for a non-whitelisted OCR string."""
        stripped = text.strip()
        if not stripped:
            return 0.0
        readable = sum(character.isalnum() or character.isspace() for character in stripped)
        return min(readable / len(stripped), 1.0)

    def _ocr_confidence(self, text: str, matched: str) -> float:
        """Confidence for an OCR string that matched the whitelist.

        Precision beyond a lexical match is not available from
        ``image_to_string`` without the word-confidence API, so this is
        the match quality plus the OCR readability. A real per-word
        probability needs ``image_to_data`` and a recorded corpus
        (plan doc R-7).
        """
        _, score = self._match_whitelist(text)
        base = min(0.5 + 0.1 * score, 0.9)
        return base * self._text_confidence(text)

    def _match_whitelist(self, raw_name: str) -> tuple[str | None, int]:
        """Return ``(whitelisted name, score)`` for a raw OCR string."""
        if not raw_name:
            return None, 0
        found = re.search(r"[A-Za-z]", raw_name)
        if found is None:
            return None, 0
        text = raw_name[found.start() :].strip()
        text_lower = text.lower()

        for known in self.known_enemies:
            if known.lower() in text_lower:
                return known, 10

        text_words = re.findall(r"[a-z]+", text_lower)
        best_match: str | None = None
        best_score = 0
        for known in self.known_enemies:
            known_words = re.findall(r"[a-z]+", known.lower())
            score = 0
            for index in range(1, len(known_words) - 1):
                if index < len(text_words) and known_words[index] == text_words[index]:
                    score += 2
            if (
                len(known_words) >= 2
                and len(text_words) >= 2
                and text_words[-1][:1] == known_words[-1][:1]
            ):
                score += 3
            if score > best_score:
                best_score = score
                best_match = known
        return (best_match, best_score) if best_score >= 2 else (None, 0)

    # ------------------------------------------------------------------
    # HP
    # ------------------------------------------------------------------
    def _extract_hp(self, frame: FrameLike, name_position: tuple[int, int]) -> float:
        """Return the target HP ratio in ``[0, 1]`` from the bar below the name."""
        template_h, template_w = self._name_template.shape[:2]
        x, y = name_position
        height, width = frame.shape[:2]

        start_y = y + template_h + _HP_BAR_OFFSET_PX
        end_y = min(y + template_h + _HP_BAR_SCAN_PX, height)

        left_x: int | None = None
        left_y: int | None = None
        for offset in (15, 20, 25, 10, 30, 35, 40, 5):
            scan_x = x + offset
            if scan_x >= width or scan_x < 0:
                continue
            for scan_y in range(start_y, end_y):
                b, g, r = frame[scan_y, scan_x, :3]
                if self._is_green(int(b), int(g), int(r)):
                    left_x = scan_x
                    left_y = scan_y
                    break
            if left_x is not None:
                break

        if left_x is None or left_y is None:
            return 0.0

        right_x: int | None = None
        limit = min(left_x + _HP_BAR_MAX_WIDTH_PX, width)
        for scan_x in range(left_x, limit):
            b, g, r = frame[left_y, scan_x, :3]
            if self._is_golden(int(b), int(g), int(r)):
                right_x = scan_x
                break
        if right_x is None:
            right_x = min(left_x + template_w + 30, width - 1)

        last_green_x = left_x
        for scan_x in range(left_x, right_x):
            b, g, r = frame[left_y, scan_x, :3]
            if self._is_green(int(b), int(g), int(r)):
                last_green_x = scan_x

        total = right_x - left_x
        if total <= 0:
            return 0.0
        filled = last_green_x - left_x
        return min(max(filled / total, 0.0), 1.0)

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------
    def read(
        self,
        frame: FrameLike,
        *,
        now: float | None = None,
        tesseract_cmd: str | None = None,
    ) -> TargetReading:
        """Read the current target, returning the cached reading while throttled.

        ``tesseract_cmd`` is applied to ``pytesseract`` on the first OCR
        call only — never at import time and never globally at
        construction.
        """
        if not self._throttle.allow(now):
            return self._last
        if tesseract_cmd is not None:
            self._configure_tesseract(tesseract_cmd)

        normalized = normalize_bgr(frame)
        reading = self._read_locked(normalized)
        if reading is None:
            reading = self._search_new(normalized)
        self._last = reading
        self._throttle.record(now)
        return reading

    def _read_locked(self, frame: FrameLike) -> TargetReading | None:
        """Handle a frame while a nameplate is locked; None means re-search."""
        position = self.locked_position
        if position is None:
            return None

        lock_confidence = self._is_target_at(frame, position)
        if lock_confidence < self._effective_action_thresh():
            self._break_lock()
            return TargetReading(
                name=None, hp_pct=0, frame_confidence=lock_confidence
            )

        name = self.last_name
        ocr_confidence: float | None = None
        self._refresh_count += 1
        if self._refresh_count >= self.name_refresh_frames:
            self._refresh_count = 0
            refreshed, ocr_confidence = self._read_name(frame, position)
            if refreshed is not None:
                self.last_name = refreshed
                self.last_ocr_confidence = ocr_confidence
                name = refreshed
            else:
                ocr_confidence = self.last_ocr_confidence

        hp_ratio = self._extract_hp(frame, position)
        return TargetReading(
            name=name,
            hp_pct=round(hp_ratio * 100),
            frame_confidence=lock_confidence,
            ocr_confidence=ocr_confidence,
        )

    def _search_new(self, frame: FrameLike) -> TargetReading:
        """Search the frame for a nameplate and acquire the lock."""
        position, confidence = self._find_target(frame)
        if position is None:
            self._miss_count += 1
            if self._miss_count >= self.miss_limit:
                self.last_name = None
                self.last_ocr_confidence = None
            return TargetReading(name=None, hp_pct=0, frame_confidence=confidence)

        if not self._confirms(confidence):
            # Above the search threshold but below confirmation: report the
            # candidate without committing the lock, so the next read is
            # still free to search instead of defending a weak guess.
            return TargetReading(name=None, hp_pct=0, frame_confidence=confidence)

        self._miss_count = 0
        self.locked_position = position
        self._refresh_count = 0

        name, ocr_confidence = self._read_name(frame, position)
        if name is not None:
            self.last_name = name
            self.last_ocr_confidence = ocr_confidence
        elif self.last_ocr_confidence is not None:
            ocr_confidence = self.last_ocr_confidence
            name = self.last_name

        hp_ratio = self._extract_hp(frame, position)
        return TargetReading(
            name=name,
            hp_pct=round(hp_ratio * 100),
            frame_confidence=confidence,
            ocr_confidence=ocr_confidence,
        )

    def _break_lock(self) -> None:
        self.locked_position = None
        self.last_name = None
        self.last_ocr_confidence = None
        self._miss_count = 0
        self._refresh_count = 0
