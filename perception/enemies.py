from ultralytics import YOLO
import numpy as np


class EnemyDetector:
    def __init__(self, model_path="runs/detect/train-2/weights/best.pt", conf=0.5):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)
        enemies = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                name = self.model.names[cls_id]

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                w = x2 - x1
                h = y2 - y1

                enemies.append({
                    "name": name,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "conf": conf,
                    "center": (int(cx), int(cy)),
                    "width": int(w),
                    "height": int(h),
                    "area": int(w * h),  # ← اضافه شد
                })

        return enemies

    def closest_enemy(self, frame):
        """نزدیک‌ترین دشمن = بزرگ‌ترین Box (بر اساس مساحت)"""
        enemies = self.detect(frame)
        if not enemies:
            return None
        # ← بر اساس area (نه فقط height)
        return max(enemies, key=lambda e: e["area"])

    def click_position(self, enemy):
        """نقطه کلیک = پایین Box (پای دشمن)"""
        x1, y1, x2, y2 = enemy["bbox"]
        cx = (x1 + x2) // 2
        cy = y2 - int((y2 - y1) * 0.2)  # ۲۰٪ بالاتر از پایین
        return (cx, cy)