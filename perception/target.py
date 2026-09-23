import cv2
import pytesseract
import re
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ⚠️ لیست سفید اسم دشمن‌ها
KNOWN_ENEMIES = [
    "Primal Proto-Drake",
    "Primal Proto-Whelp",
]


class TargetReader:
    def __init__(self, name_template_path, match_thresh=0.65):
        """
        name_template_path: مسیر target_name_template.png
        """
        self.template = cv2.imread(name_template_path)
        if self.template is None:
            raise FileNotFoundError(f"Template not found: {name_template_path}")
        self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        self.match_thresh = match_thresh

        # Position Lock + Cache
        self.locked_position = None
        self.last_name = None
        self.miss_count = 0
        self.name_miss_count = 0

    # ============================================================
    # تشخیص رنگ
    # ============================================================
    def _is_green(self, b, g, r):
        b, g, r = int(b), int(g), int(r)
        return g > 90 and g > r + 20 and g > b + 20

    def _is_golden(self, b, g, r):
        b, g, r = int(b), int(g), int(r)
        return r > 170 and g > 120 and b < 130

    # ============================================================
    # پیدا کردن ناحیه اسم TargetFrame
    # ============================================================
    def _find_target(self, frame):
        h_frame, w_frame = frame.shape[:2]

        # جستجو توی ناحیه وسط-پایین
        x1 = int(w_frame * 0.35)
        x2 = int(w_frame * 0.80)
        y1 = int(h_frame * 0.50)
        y2 = int(h_frame * 0.85)

        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        res = cv2.matchTemplate(gray, self.template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val < self.match_thresh:
            return None, max_val

        abs_x = x1 + max_loc[0]
        abs_y = y1 + max_loc[1]
        return (abs_x, abs_y), max_val

    # ============================================================
    # بررسی: آیا هنوز سر جاشه؟
    # ============================================================
    def _is_target_at(self, frame, position):
        tw, th = self.template.shape[1], self.template.shape[0]
        x, y = position
        h_frame, w_frame = frame.shape[:2]

        if x + tw > w_frame or y + th > h_frame or x < 0 or y < 0:
            return 0.0

        roi = frame[y:y+th, x:x+tw]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray, self.template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        return max_val

    # ============================================================
    # استخراج اسم از ناحیه‌ای که پیدا شده
    # ============================================================
    def _extract_name(self, frame, position):
        tw, th = self.template.shape[1], self.template.shape[0]
        x, y = position
        h_frame, w_frame = frame.shape[:2]

        # چک کن ناحیه توی فریم هست
        if x + tw > w_frame or y + th > h_frame:
            return ""

        name_crop = frame[y:y+th, x:x+tw, :3]

        if name_crop.size == 0:
            return ""

        big = cv2.resize(name_crop, (0, 0), fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

        candidates = []

        # روش ۱: Otsu
        _, t1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates.append(pytesseract.image_to_string(t1, config="--psm 7").strip())

        # روش ۲: آستانه ثابت
        _, t2 = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
        candidates.append(pytesseract.image_to_string(t2, config="--psm 7").strip())

        # روش ۳: invert
        t3 = cv2.bitwise_not(t1)
        candidates.append(pytesseract.image_to_string(t3, config="--psm 7").strip())

        # بهترین: اونی که whitelist match کنه
        for c in candidates:
            if self._match_whitelist(c) is not None:
                return c
        return max(candidates, key=len) if candidates else ""

    # ============================================================
    # استخراج HP — نوار HP دقیقاً زیر ناحیه اسم هست
    # ============================================================
    def _extract_hp(self, frame, name_position):
        """
        ناحیه اسم رو گرفتیم، حالا نوار HP دقیقاً زیرشه.
        """
        tw, th = self.template.shape[1], self.template.shape[0]
        x, y = name_position
        h_frame, w_frame = frame.shape[:2]

        # نوار HP حدود ۵ تا ۲۰ پیکسل پایین‌تر از ناحیه اسم شروع می‌شه
        hp_start_y = y + th + 3
        hp_end_y = min(y + th + 30, h_frame)

        # اسکن از چند نقطه برای پیدا کردن اولین پیکسل سبز
        left_x = None
        left_y = None
        for offset in [15, 20, 25, 10, 30, 35, 40, 5]:
            scan_x = x + offset
            if scan_x >= w_frame:
                continue
            for yy in range(hp_start_y, hp_end_y):
                b, g, r = frame[yy, scan_x, :3]
                if self._is_green(b, g, r):
                    left_x = scan_x
                    left_y = yy
                    break
            if left_x is not None:
                break

        if left_x is None:
            return 0.0

        # راست برو تا پیکسل طلایی
        right_x = None
        for xx in range(left_x, min(left_x + 250, w_frame)):
            b, g, r = frame[left_y, xx, :3]
            if self._is_golden(b, g, r):
                right_x = xx
                break

        if right_x is None:
            right_x = left_x + tw + 30

        # آخرین پیکسل سبز
        last_green_x = left_x
        for xx in range(left_x, right_x):
            b, g, r = frame[left_y, xx, :3]
            if self._is_green(b, g, r):
                last_green_x = xx

        total = right_x - left_x
        if total <= 0:
            return 0.0

        filled = last_green_x - left_x
        return min(max(filled / total, 0.0), 1.0)

    # ============================================================
    # تطبیق اسم با لیست سفید
    # ============================================================
    def _match_whitelist(self, raw_name):
        if not raw_name:
            return None

        m = re.search(r"[A-Za-z]", raw_name)
        if not m:
            return None
        text = raw_name[m.start():].strip()
        text_lower = text.lower()

        for known in KNOWN_ENEMIES:
            if known.lower() in text_lower:
                return known

        text_words = re.findall(r"[a-z]+", text_lower)
        best_match = None
        best_score = 0

        for known in KNOWN_ENEMIES:
            known_words = re.findall(r"[a-z]+", known.lower())
            score = 0

            for i in range(1, len(known_words) - 1):
                if i < len(text_words) and known_words[i] == text_words[i]:
                    score += 2

            if len(known_words) >= 2 and len(text_words) >= 2:
                if text_words[-1][:1] == known_words[-1][:1]:
                    score += 3

            if score > best_score:
                best_score = score
                best_match = known

        return best_match if best_score >= 2 else None

    # ============================================================
    # تابع اصلی
    # ============================================================
    def read(self, frame):
        # ---------- Position Lock ----------
        if self.locked_position is not None:
            lock_conf = self._is_target_at(frame, self.locked_position)

            if lock_conf >= self.match_thresh:
                # هنوز سر جاشه
                hp = self._extract_hp(frame, self.locked_position)
                hp_pct = int(round(hp * 100))
                name = self.last_name

                # هر چند فریم یه بار اسم رو دوباره بخون
                self.name_miss_count += 1
                if self.name_miss_count >= 5:
                    self.name_miss_count = 0
                    raw_name = self._extract_name(frame, self.locked_position)
                    new_name = self._match_whitelist(raw_name)
                    if new_name is not None:
                        self.last_name = new_name
                        name = new_name

                return name, hp_pct, lock_conf
            else:
                # قفل شکسته
                self.locked_position = None
                self.last_name = None
                self.miss_count = 0
                return None, 0, lock_conf

        # ---------- جستجوی جدید ----------
        top_left, conf = self._find_target(frame)

        if top_left is None:
            self.miss_count += 1
            if self.miss_count >= 3:
                self.last_name = None
            return None, 0, conf

        self.miss_count = 0
        self.locked_position = top_left

        # اسم
        raw_name = self._extract_name(frame, top_left)
        name = self._match_whitelist(raw_name)
        if name is not None:
            self.last_name = name
        else:
            name = self.last_name

        # HP
        hp = self._extract_hp(frame, top_left)
        hp_pct = int(round(hp * 100))

        return name, hp_pct, conf