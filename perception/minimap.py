import cv2
import numpy as np


class MinimapTracker:
    def __init__(self, minimap_roi, template_path, match_thresh=0.55):
        """
        minimap_roi: (x, y, w, h) مستطیل دور Minimap
        template_path: مسیر فایل arrow_template.png
        match_thresh: حداقل شباهت برای قبول Template
        """
        self.roi = minimap_roi
        self.template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if self.template is None:
            raise FileNotFoundError(f"Template not found: {template_path}")
        self.match_thresh = match_thresh
        self.last_pos = None
        self.last_angle = None
        # smoothing
        self.angle_history = []
        self.history_size = 5

    def _find_arrow(self, crop):
        """فلش رو با Template Matching پیدا می‌کنه"""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val < self.match_thresh:
            return None, None, 0.0

        # مرکز Template پیدا شده
        th, tw = self.template.shape[:2]
        cx = max_loc[0] + tw / 2.0
        cy = max_loc[1] + th / 2.0
        return (cx, cy), max_loc, max_val

    def _estimate_facing(self, crop, top_left):
        """
        زاویه‌ی فلش رو از داخل خود Template پیدا می‌کنه.
        نوک فلش سفید = دورترین پیکسل روشن از مرکز Template.
        """
        th, tw = self.template.shape[:2]
        x, y = top_left
        # ناحیه‌ای که Template توش پیدا شده
        match_region = crop[y:y+th, x:x+tw, :3]

        # تبدیل به HSV و ماسک سفید (فلش)
        hsv = cv2.cvtColor(match_region, cv2.COLOR_BGR2HSV)
        # سفید: اشباع کم، روشنایی زیاد
        white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 60, 255))

        ys, xs = np.where(white_mask > 0)
        if len(xs) < 5:
            return None

        # مرکز جرم
        mx, my = xs.mean(), ys.mean()
        # دورترین پیکسل = نوک فلش
        dx = xs - mx
        dy = ys - my
        dists = np.sqrt(dx**2 + dy**2)
        tip = np.argmax(dists)
        tip_x, tip_y = xs[tip], ys[tip]

        # زاویه از مرکز جرم به نوک
        # در تصویر، y به سمت پایین مثبته → معکوس می‌کنیم برای ریاضیات استاندارد
        angle = np.degrees(np.arctan2(-(tip_y - my), tip_x - mx))
        if angle < 0:
            angle += 360
        return angle

    def _smooth_angle(self, angle):
        if angle is None:
            return self.last_angle
        self.angle_history.append(angle)
        if len(self.angle_history) > self.history_size:
            self.angle_history.pop(0)
        rads = np.radians(self.angle_history)
        mean_sin = np.mean(np.sin(rads))
        mean_cos = np.mean(np.cos(rads))
        smoothed = np.degrees(np.arctan2(mean_sin, mean_cos))
        if smoothed < 0:
            smoothed += 360
        return smoothed

    def update(self, frame):
        x, y, w, h = self.roi
        crop = frame[y:y+h, x:x+w, :3]

        pos, top_left, conf = self._find_arrow(crop)

        if pos is None:
            return self.last_pos, self.last_angle, 0.0

        angle = self._estimate_facing(crop, top_left)
        angle = self._smooth_angle(angle)

        self.last_pos = pos
        if angle is not None:
            self.last_angle = angle

        return self.last_pos, self.last_angle, conf