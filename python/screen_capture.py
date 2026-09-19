"""
screen_capture.py
-------------------
یه گیرنده‌ی فریم مشترک برای همه‌ی ماژول‌های بینایی (enemy_vision و هر چیزی که بعداً
اضافه بشه، مثل gathering_detector). به‌جای این‌که هر ماژول جدا mss.grab() صدا بزنه،
یه‌بار کل صفحه (یا یه ناحیه‌ی بزرگ شامل همه‌ی نواحی موردنیاز) گرفته می‌شه و بقیه فقط
ازش برش می‌زنن.

همچنین throttle داره: اگه توی یه بازه‌ی زمانی خیلی کوتاه (کمتر از min_interval) دوباره
درخواست فریم بشه، همون فریم قبلی (کش‌شده) رو برمی‌گردونه به‌جای گرفتن فریم تازه -
چون گرفتن اسکرین‌شات هزینه داره و اکثر ماژول‌ها با هم توی یه تیک صدا زده می‌شن.
"""

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

# مختصات مطلق نواحی (روی رزولوشن 1920x1080) - همون کالیبراسیون‌های قبلی پروژه
REGIONS = {
    "hp_bar": {"top": 771, "left": 514, "width": 118, "height": 15},
    "energy_bar": {"top": 792, "left": 514, "width": 118, "height": 9},
    "minimap": {"top": 47, "left": 1698, "width": 212, "height": 212},
    "viewport": {"top": 120, "left": 350, "width": 1220, "height": 440},
}


def _bounding_box(regions: dict) -> dict:
    """کوچیک‌ترین مستطیلی که همه‌ی نواحی داده‌شده رو در بر می‌گیره."""
    lefts = [r["left"] for r in regions.values()]
    tops = [r["top"] for r in regions.values()]
    rights = [r["left"] + r["width"] for r in regions.values()]
    bottoms = [r["top"] + r["height"] for r in regions.values()]
    left, top = min(lefts), min(tops)
    return {
        "left": left,
        "top": top,
        "width": max(rights) - left,
        "height": max(bottoms) - top,
    }


@dataclass
class CachedFrame:
    full_frame: np.ndarray   # BGR، نسبت به bounding_box (نه کل مانیتور)
    bbox: dict               # مختصات مطلق bounding_box روی صفحه
    captured_at: float


class ScreenCapture:
    def __init__(self, regions: dict = None, min_interval: float = 0.03):
        self.regions = regions or REGIONS
        self.bbox = _bounding_box(self.regions)
        self.min_interval = min_interval
        self._cache: Optional[CachedFrame] = None
        self._sct = None

    def _get_sct(self):
        if self._sct is None:
            import mss  # فقط روی ویندوز واقعی لازمه؛ lazy import برای قابل‌تست بودن
            self._sct = mss.mss()
        return self._sct

    def get_frame(self, force: bool = False) -> CachedFrame:
        """
        یه فریم تازه (یا کش‌شده اگه هنوز داخل min_interval هستیم) برمی‌گردونه.
        force=True یعنی حتماً فریم تازه بگیر، حتی اگه کش تازه باشه.
        """
        now = time.monotonic()
        if not force and self._cache is not None and (now - self._cache.captured_at) < self.min_interval:
            return self._cache

        sct = self._get_sct()
        shot = sct.grab(self.bbox)
        frame = np.array(shot)[:, :, :3]  # BGRA -> BGR

        self._cache = CachedFrame(full_frame=frame, bbox=self.bbox, captured_at=now)
        return self._cache

    def crop_region(self, cached: CachedFrame, region_name: str) -> np.ndarray:
        """یه ناحیه‌ی نام‌گذاری‌شده (مثل 'viewport') رو از فریم بزرگ برش می‌زنه."""
        if region_name not in self.regions:
            raise KeyError(f"ناحیه‌ی '{region_name}' توی REGIONS تعریف نشده.")

        r = self.regions[region_name]
        # مختصات ناحیه رو نسبت به bbox تبدیل می‌کنیم (چون فریم فقط bbox رو داره، نه کل مانیتور)
        x0 = r["left"] - cached.bbox["left"]
        y0 = r["top"] - cached.bbox["top"]
        x1 = x0 + r["width"]
        y1 = y0 + r["height"]

        h, w = cached.full_frame.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)

        return cached.full_frame[y0:y1, x0:x1]

    def get_region(self, region_name: str, force: bool = False) -> np.ndarray:
        """میان‌بر: get_frame + crop_region با هم."""
        cached = self.get_frame(force=force)
        return self.crop_region(cached, region_name)
