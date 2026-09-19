"""
controller.py
--------------
لایه‌ی اکشن: تبدیل تصمیمات FSM به فشردن کلید / حرکت موس واقعی با pydirectinput.

نکته: pydirectinput از SendInput ویندوز استفاده می‌کنه که اکثر بازی‌ها (از جمله WoW) اونو
به‌عنوان ورودی واقعی قبول می‌کنن (برخلاف بعضی کتابخونه‌های شبیه‌سازی سطح بالاتر).
"""

import math
import time
import pydirectinput

import humanize

pydirectinput.PAUSE = 0.0  # خودمون تایمینگ رو کنترل می‌کنیم


class Controller:
    def __init__(self, key_bindings: dict):
        """
        key_bindings مثال:
        {
            "target_nearest_enemy": "tab",
            "interact_target": "t",
            "loot": "e",  -- یا هر کلیدی که خودت روی /loot بایند کردی
            "auto_run": None,  -- استفاده نمی‌کنیم، خودمون W نگه می‌داریم
            "rotation": ["1", "2", "3"],  -- ترتیب اسکیل‌ها
        }
        """
        self.keys = key_bindings
        self._moving_forward = False
        self._turning = None  # "left" | "right" | None

    # ---------- حرکت ----------

    def move_forward_start(self):
        if not self._moving_forward:
            pydirectinput.keyDown("w")
            self._moving_forward = True

    def move_forward_stop(self):
        if self._moving_forward:
            pydirectinput.keyUp("w")
            self._moving_forward = False

    def turn_stop(self):
        if self._turning == "left":
            pydirectinput.keyUp("a")
        elif self._turning == "right":
            pydirectinput.keyUp("d")
        self._turning = None

    def turn_toward(self, current_facing: float, target_facing: float, tolerance: float = 0.08):
        """
        current_facing و target_facing برحسب رادیان (0..2pi).
        بر اساس کوتاه‌ترین مسیر زاویه‌ای، چرخش چپ/راست رو استارت یا استاپ می‌کنه.
        خروجی: True اگه دیگه توی tolerance هست (نیازی به چرخش نیست).
        """
        diff = (target_facing - current_facing + math.pi) % (2 * math.pi) - math.pi
        # diff مثبت یعنی هدف سمت چپه (در سیستم رادیان WoW)، منفی یعنی راست -- بسته به کالیبراسیون خودت چک کن

        if abs(diff) <= tolerance:
            self.turn_stop()
            return True

        wanted = "left" if diff > 0 else "right"
        if self._turning != wanted:
            self.turn_stop()
            key = "a" if wanted == "left" else "d"
            pydirectinput.keyDown(key)
            self._turning = wanted
        return False

    def turn_relative(self, dx_normalized: float, tolerance: float = None):
        """
        دشمن رو با افست نسبی روی صفحه می‌چرخونه سمتش (-1 چپ ... +1 راست).
        خروجی: True یعنی دیگه توی tolerance هست (نیازی به چرخش نیست).
        """
        if tolerance is None:
            tolerance = humanize.humanized_turn_tolerance()
        if abs(dx_normalized) <= tolerance:
            self.turn_stop()
            return True

        wanted = "right" if dx_normalized > 0 else "left"
        if self._turning != wanted:
            self.turn_stop()
            key = "d" if wanted == "right" else "a"
            pydirectinput.keyDown(key)
            self._turning = wanted
        return False

    def stop_all_movement(self):
        self.move_forward_stop()
        self.turn_stop()

    # ---------- اکشن‌های گسسته ----------

    def press(self, key: str, hold_sec: float = None):
        if hold_sec is None:
            hold_sec = humanize.humanized_key_hold()
        pydirectinput.keyDown(key)
        time.sleep(hold_sec)
        pydirectinput.keyUp(key)

    def target_nearest_enemy(self):
        self.press(self.keys.get("target_nearest_enemy", "tab"))

    def interact_target(self):
        """برای شروع کشتن Node منابع یا تعامل با NPC/جسد."""
        self.press(self.keys.get("interact_target", "t"))

    def loot(self):
        self.press(self.keys.get("loot", "e"))

    def cast_rotation_step(self, step_index: int):
        rotation = self.keys.get("rotation", ["1", "2", "3"])
        key = rotation[step_index % len(rotation)]
        self.press(key)  # hold_sec خودش از humanize.humanized_key_hold() میاد
