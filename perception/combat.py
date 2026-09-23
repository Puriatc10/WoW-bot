import cv2
import numpy as np
import time


class CombatDetector:
    def __init__(self, edge_size=150, red_ratio_thresh=0.02, cooldown=1.0):
        """
        edge_size: ضخامت حاشیه‌ای که بررسی می‌شه (پیکسل)
        red_ratio_thresh: اگه نسبت قرمز از این بیشتر شد → combat
        cooldown: چند ثانیه بعد از آخرین combat، هنوز combat حساب شه
        """
        self.edge_size = edge_size
        self.red_ratio_thresh = red_ratio_thresh
        self.cooldown = cooldown
        self.last_combat_time = 0.0

    def _red_mask(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0, 100, 80), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 100, 80), (180, 255, 255))
        return mask1 | mask2

    def detect(self, frame):
        h, w = frame.shape[:2]
        e = self.edge_size

        top    = frame[0:e, :, :3]
        bottom = frame[h-e:h, :, :3]
        left   = frame[:, 0:e, :3]
        right  = frame[:, w-e:w, :3]

        masks = [
            self._red_mask(top),
            self._red_mask(bottom),
            self._red_mask(left),
            self._red_mask(right),
        ]

        total_red = sum(m.sum() // 255 for m in masks)
        total_px = sum(m.size for m in masks)
        ratio = total_red / total_px if total_px > 0 else 0.0

        now = time.time()
        instant_combat = ratio > self.red_ratio_thresh

        if instant_combat:
            self.last_combat_time = now

        # اگه الان combat هست یا هنوز cooldown تموم نشده → combat
        in_combat = instant_combat or (now - self.last_combat_time) < self.cooldown

        return in_combat, ratio