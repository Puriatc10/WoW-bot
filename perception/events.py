import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class EventDetector:
    def __init__(self):
        self.combat_region = (31, 681, 418, 161)
        self.chat_region = (34, 873, 446, 176)

        self.last_combat_lines = set()
        self.last_chat_lines = set()

        self.combat_keywords = {
            "death": ["dies", "slain", "died", "you killed", "experience"],
            "damage_dealt": ["you hit", "auto shot", "arcane shot"],
            "damage_taken": ["hits you", "hit you", "you for"],
            "miss": ["misses", "dodged", "parried"],
        }

        self.chat_keywords = {
            "loot": ["receive", "loot", "received", "item"],
            "quest": ["quest", "accepted", "completed"],
            "level_up": ["reached level", "level up"],
            "warning": ["warning", "you are", "cannot"],
        }

        self.priority = [
            "death", "loot", "level_up", "quest",
            "damage_dealt", "damage_taken", "miss", "warning"
        ]

    def _read_region(self, frame, region, color_filter=None):
        x, y, w, h = region
        h_frame, w_frame = frame.shape[:2]
        x2, y2 = min(x + w, w_frame), min(y + h, h_frame)

        crop = frame[y:y2, x:x2]
        if crop.size == 0:
            return ""

        big = cv2.resize(crop, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

        if color_filter == "green":
            hsv = cv2.cvtColor(big, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (30, 80, 100), (90, 255, 255))
        elif color_filter == "yellow":
            hsv = cv2.cvtColor(big, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (20, 80, 100), (35, 255, 255))
        elif color_filter == "purple":
            # ⚠️ بنفش/ارغوانی برای پیام مرگ
            hsv = cv2.cvtColor(big, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (120, 50, 100), (160, 255, 255))
        else:
            # ⚠️ آستانه‌گذاری کم‌تر (90 به جای 130)
            gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)

        return pytesseract.image_to_string(mask, config="--psm 6")

    def _read_region_multi(self, frame, region):
        """چند فیلتر رنگی"""
        texts = []
        for color in [None, "green", "yellow", "purple"]:
            try:
                t = self._read_region(frame, region, color_filter=color)
                if t:
                    texts.append(t)
            except Exception:
                pass
        return "\n".join(texts)

    def _read_combat_multi(self, frame, region):
        """Combat Log: default + purple"""
        texts = []
        for color in [None, "purple"]:
            try:
                t = self._read_region(frame, region, color_filter=color)
                if t:
                    texts.append(t)
            except Exception:
                pass
        return "\n".join(texts)

    def _is_valid_line(self, line):
        line = line.strip()
        if len(line) < 6:
            return False
        if line.count("[DAMAGE") > 1 or line.count("[") > 2:
            return False
        alnum = sum(c.isalnum() or c in " .,:;'[]()-" for c in line)
        if alnum / max(len(line), 1) < 0.4:
            return False
        return True

    def _classify(self, text, keywords):
        text_lower = text.lower()
        for event_type in self.priority:
            if event_type not in keywords:
                continue
            for kw in keywords[event_type]:
                if kw in text_lower:
                    return event_type
        return None

    def _process_region(self, frame, region, keywords, last_lines, tag, multi_color=False, combat_multi=False):
        events = []

        if combat_multi:
            text = self._read_combat_multi(frame, region)
        elif multi_color:
            text = self._read_region_multi(frame, region)
        else:
            text = self._read_region(frame, region)

        if not text:
            return events

        for line in text.split("\n"):
            line = line.strip()
            if not self._is_valid_line(line):
                continue
            if line in last_lines:
                continue

            event_type = self._classify(line, keywords)
            if event_type:
                events.append({
                    "type": event_type,
                    "text": line,
                    "source": tag,
                })
                last_lines.add(line)

        if len(last_lines) > 100:
            last_lines.clear()

        return events

    def detect(self, frame):
        events = []

        # Combat Log - با فیلتر بنفش
        events += self._process_region(
            frame, self.combat_region,
            self.combat_keywords, self.last_combat_lines, "combat",
            combat_multi=True,
        )

        # General Chat - با فیلتر سبز/زرد
        events += self._process_region(
            frame, self.chat_region,
            self.chat_keywords, self.last_chat_lines, "chat",
            multi_color=True,
        )

        return events