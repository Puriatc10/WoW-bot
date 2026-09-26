"""Shared OCR plumbing for the T-FIX-31 UI panel channels.

This is a **private** module (leading underscore): it is not part of the
public perception API and is imported only by the panel readers. It exists so
that the Tesseract thresholding and word-confidence extraction the panel
channels need live in exactly one place instead of being copied into four
readers.

The three-way thresholding (Otsu, fixed, inverted) mirrors the T-FIX-27
target reader and the T-FIX-30 pose reader: a UI panel may render
light-on-dark or dark-on-light depending on the operator's client, and a
single binarisation would silently fail on one of the two.

No import here is eager: ``cv2`` and ``pytesseract`` both arrive through
:mod:`wow_bot.perception.deps`, so MOCK_MODE imports this module without the
vision stack installed. No clock, no RNG, and no global state live here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from wow_bot.perception import deps
from wow_bot.perception.capture import FrameLike, normalize_bgr

__all__ = [
    "ocr_line_candidates",
    "ocr_words",
    "threshold_variants",
    "upscale_gray",
]

#: Fixed-threshold binarisation cutoff used alongside Otsu by every channel.
_FIXED_THRESHOLD = 130


def upscale_gray(crop: FrameLike, factor: int) -> FrameLike:
    """Return a grayscale, ``factor``-times upscaled copy of ``crop``.

    UI text is small; Tesseract is markedly more reliable on an upscaled
    image. The crop is normalised to BGR first so a grey or BGRA examination
    region behaves the same as a colour one.
    """
    cv2 = deps.require_cv2()
    big = cv2.resize(
        normalize_bgr(crop),
        (0, 0),
        fx=factor,
        fy=factor,
        interpolation=cv2.INTER_CUBIC,
    )
    gray: FrameLike = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    return gray


def threshold_variants(gray: FrameLike) -> tuple[FrameLike, FrameLike, FrameLike]:
    """Return the Otsu, fixed, and inverted-Otsu binarisations of ``gray``."""
    cv2 = deps.require_cv2()
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, fixed = cv2.threshold(gray, _FIXED_THRESHOLD, 255, cv2.THRESH_BINARY)
    inverted: FrameLike = cv2.bitwise_not(otsu)
    return otsu, fixed, inverted


def ocr_line_candidates(crop: FrameLike, *, upscale: int, psm: int = 7) -> list[str]:
    """OCR ``crop`` under the three thresholdings; return one text per pass.

    ``psm`` defaults to ``7`` (single text line), which is what a UI label
    is; a caller reading a multi-line region passes its own page-segmentation
    mode explicitly rather than relying on a default that would silently
    merge lines.
    """
    pytesseract = deps.require_pytesseract()
    gray = upscale_gray(crop, upscale)
    config = f"--psm {psm}"
    return [
        pytesseract.image_to_string(image, config=config).strip()
        for image in threshold_variants(gray)
    ]


def ocr_words(
    data: Mapping[str, Sequence[object]], *, numeric_only: bool = True
) -> tuple[str, float]:
    """Join OCR word tokens and average their measured confidences.

    ``pytesseract.image_to_data`` reports a ``[0, 100]`` confidence per word
    and ``-1`` for rows that are not words. A ``-1`` row contributes nothing,
    so "not a word" is never averaged in as a zero.

    ``numeric_only`` selects which tokens the average covers:

    * ``True`` (default) averages only the tokens carrying a digit. A numeric
      parse consumes exactly those tokens, so an average over punctuation
      noise would be a different quantity.
    * ``False`` averages every non-empty token. A channel whose vocabulary is
      *words* — a spell name, for instance — gets no confidence at all under
      the numeric rule, because none of its tokens carry a digit.

    When no token qualifies the result is ``0.0`` — "not measured" — never an
    invented confidence.
    """
    texts = data.get("text", ())
    confidences = data.get("conf", ())
    words: list[str] = []
    scores: list[float] = []
    for index, raw_text in enumerate(texts):
        token = str(raw_text).strip()
        if not token:
            continue
        words.append(token)
        if numeric_only and not any(character.isdigit() for character in token):
            continue
        raw_confidence = confidences[index] if index < len(confidences) else -1
        if not isinstance(raw_confidence, (int, float, str)):
            value = -1.0
        else:
            try:
                value = float(raw_confidence)
            except (TypeError, ValueError):
                value = -1.0
        if value >= 0.0:
            scores.append(min(value, 100.0) / 100.0)
    confidence = sum(scores) / len(scores) if scores else 0.0
    return " ".join(words), confidence
