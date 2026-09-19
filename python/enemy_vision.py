"""
enemy_vision.py
----------------
جایگزین minimap_detector برای جهت‌یابی: به‌جای فهمیدن موقعیت مطلق دشمن روی نقشه،
فقط می‌فهمیم نیم‌پلیت دشمن روی صفحه (Viewport) کجاست نسبت به مرکز -> چپ/راست/وسط.

این روش نیازی به تشخیص واضح روی مینی‌مپ نداره و از همون کالیبراسیون قبلی enemy_detector
(کادر ویوپورت + بازه‌ی HSV قرمز نیم‌پلیت + فیلتر ابعاد) استفاده می‌کنه.

خروجی اصلی: get_nearest_enemy_offset() -> (dx_normalized, found)
    dx_normalized بین -1 (کاملاً چپ) تا +1 (کاملاً راست)، 0 یعنی وسط صفحه.
    found=False یعنی هیچ نیم‌پلیتی توی ویوپورت پیدا نشد.
"""

import cv2
import numpy as np

VIEWPORT = {"top": 120, "left": 350, "width": 1220, "height": 440}

# بازه‌ی قرمز نوار نیم‌پلیت (دو رنج چون قرمز توی HSV دور hue=0 می‌پیچه)
RED_LOWER_1 = np.array([0, 160, 120])
RED_UPPER_1 = np.array([6, 255, 255])
RED_LOWER_2 = np.array([174, 160, 120])
RED_UPPER_2 = np.array([180, 255, 255])

MIN_W, MAX_W = 30, 220
MIN_H, MAX_H = 6, 35
MIN_ASPECT_RATIO = 1.8

_sct = None


def _get_sct():
    global _sct
    if _sct is None:
        import mss  # فقط وقتی واقعاً لازمه (روی ویندوز واقعی) ایمپورت می‌شه
        _sct = mss.mss()
    return _sct


def _find_nameplate_candidates(frame_bgr):
    """کانتورهایی که با فیلتر ابعاد/نسبت نیم‌پلیت مطابقت دارن رو برمی‌گردونه."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
    mask2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if not (MIN_W <= w <= MAX_W and MIN_H <= h <= MAX_H):
            continue
        if h == 0 or (w / h) < MIN_ASPECT_RATIO:
            continue
        center_x = x + w / 2
        center_y = y + h / 2
        candidates.append({"x": center_x, "y": center_y, "w": w, "h": h})

    return candidates


def get_all_enemies(frame_bgr=None):
    """
    همه‌ی نیم‌پلیت‌های قابل‌تشخیص توی ویوپورت رو برمی‌گردونه (نه فقط نزدیک‌ترین به مرکز).

    frame_bgr: می‌تونی فریم ویوپورت رو خودت بدی (مثلاً از یه ScreenCapture مشترک -
               screen_capture.py - که چند ماژول ازش استفاده می‌کنن)، یا None بذار
               تا خودش مستقل یه اسکرین‌شات بگیره (رفتار قبلی، هنوز کار می‌کنه).

    خروجی: لیستی از دیکشنری‌ها:
        {
            "dx_normalized": -1..+1  (چپ..راست نسبت به مرکز صفحه)
            "dy_normalized": -1..+1  (بالا..پایین نسبت به مرکز صفحه)
            "screen_x", "screen_y": مختصات پیکسلی مرکز نیم‌پلیت (نسبت به ویوپورت)
            "width", "height": ابعاد نیم‌پلیت به پیکسل
            "proximity_score": 0..1  (تقریبی از نزدیکی؛ هرچی بزرگ‌تر یعنی نزدیک‌تره،
                                      چون نیم‌پلیت دورتر روی صفحه کوچیک‌تر دیده می‌شه)
        }
    مرتب‌شده از نزدیک‌ترین (proximity_score بیشتر) به دورترین.

    نکته: این فقط یه *تخمین نسبی* فاصله‌ست بر اساس اندازه‌ی ظاهری روی صفحه، نه فاصله‌ی
    دقیق به یارد؛ برای اولویت‌بندی ساده (کدوم نزدیک‌تره) کافیه.
    """
    if frame_bgr is None:
        sct = _get_sct()
        shot = sct.grab(VIEWPORT)
        frame_bgr = np.array(shot)[:, :, :3]

    candidates = _find_nameplate_candidates(frame_bgr)
    if not candidates:
        return []

    viewport_center_x = VIEWPORT["width"] / 2
    viewport_center_y = VIEWPORT["height"] / 2

    enemies = []
    for c in candidates:
        dx = (c["x"] - viewport_center_x) / viewport_center_x
        dy = (c["y"] - viewport_center_y) / viewport_center_y
        # proximity_score: نسبت عرض نیم‌پلیت به حداکثر عرض مجاز (MAX_W) - ساده و کافیه
        proximity_score = min(1.0, c["w"] / MAX_W)
        enemies.append({
            "dx_normalized": max(-1.0, min(1.0, dx)),
            "dy_normalized": max(-1.0, min(1.0, dy)),
            "screen_x": c["x"],
            "screen_y": c["y"],
            "width": c["w"],
            "height": c["h"],
            "proximity_score": proximity_score,
        })

    enemies.sort(key=lambda e: e["proximity_score"], reverse=True)
    return enemies


def get_all_enemies_from_capture(screen_capture, force: bool = False):
    """
    نسخه‌ای که از یه ScreenCapture مشترک (screen_capture.py) استفاده می‌کنه به‌جای
    این‌که خودش جدا اسکرین‌شات بگیره. برای وقتی چند ماژول هم‌زمان دارن از یه فریم
    استفاده می‌کنن (کارآمدتر از این‌که هرکدوم جدا mss.grab() بزنن).
    """
    frame = screen_capture.get_region("viewport", force=force)
    return get_all_enemies(frame)


def pick_target(enemies, strategy: str = "nearest_to_center"):
    """
    یه استراتژی ساده‌ی انتخاب هدف از بین لیست enemies (خروجی get_all_enemies).
    برمی‌گردونه یه enemy dict یا None اگه لیست خالی باشه.

    strategy:
        "nearest_to_center" -> نزدیک‌ترین به وسط صفحه (یعنی چیزی که تقریباً روبروته،
                                معمولاً همون چیزیه که با Tab هم انتخاب می‌شه)
        "closest_by_distance" -> بزرگ‌ترین نیم‌پلیت (یعنی از نظر فاصله‌ی واقعی نزدیک‌تره)
    """
    if not enemies:
        return None

    if strategy == "closest_by_distance":
        return max(enemies, key=lambda e: e["proximity_score"])

    # پیش‌فرض: nearest_to_center
    return min(enemies, key=lambda e: abs(e["dx_normalized"]))


def get_nearest_enemy_offset(frame_bgr=None):
    """
    (نگه‌داشته‌شده برای سازگاری با کد قبلی - fsm.py از همین استفاده می‌کنه)
    برمی‌گردونه: (dx_normalized, found)
    """
    enemies = get_all_enemies(frame_bgr)
    target = pick_target(enemies, strategy="nearest_to_center")
    if target is None:
        return 0.0, False
    return target["dx_normalized"], True


def has_any_enemy_visible(frame_bgr=None) -> bool:
    _, found = get_nearest_enemy_offset(frame_bgr)
    return found
