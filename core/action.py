import pydirectinput
import time

class ActionController:
    def __init__(self):
        pydirectinput.PAUSE = 0.01
        pydirectinput.FAILSAFE = False

    def click(self, x, y):
        pydirectinput.moveTo(x, y)
        time.sleep(0.03)
        pydirectinput.click()
        time.sleep(0.03)

    def press_key(self, key, duration=0.05):
        pydirectinput.keyDown(key)
        time.sleep(duration)
        pydirectinput.keyUp(key)

    def click_enemy(self, enemy):
        x, y = enemy["center"]
        self.click(x, y)

    def attack(self, attack_key="1"):
        self.press_key(attack_key, duration=0.03)

    def move_forward_short(self):
        """حرکت کوتاه جلو (نرم)"""
        pydirectinput.keyDown("w")
        time.sleep(0.15)
        pydirectinput.keyUp("w")

    def move_left_short(self):
        pydirectinput.keyDown("a")
        time.sleep(0.1)
        pydirectinput.keyUp("a")

    def move_right_short(self):
        pydirectinput.keyDown("d")
        time.sleep(0.1)
        pydirectinput.keyUp("d")

    def turn_toward_enemy(self, enemy, screen_center_x=960):
        """چرخوندن دوربین سمت دشمن با موس"""
        enemy_x, _ = enemy["center"]
        diff = enemy_x - screen_center_x

        if abs(diff) > 30:
            # نسبت به فاصله، موس رو بچرخون (نه یهو، نرم)
            move_x = int(diff * 0.5)
            pydirectinput.moveRel(move_x, 0, relative=True)

    def move_toward_enemy(self, enemy, screen_center_x=960):
        """
        حرکت نرم: چرخش + حرکت
        - اول دوربین رو بچرخون
        - بعد حرکت کن (کوتاه و پیوسته)
        """
        enemy_x, _ = enemy["center"]
        diff = enemy_x - screen_center_x

        # ۱. چرخش نرم دوربین
        if abs(diff) > 50:
            self.turn_toward_enemy(enemy, screen_center_x)
            time.sleep(0.02)

        # ۲. حرکت کوتاه
        if abs(diff) < 50:
            # دشمن مستقیم جلوئه
            self.move_forward_short()
        elif diff < 0:
            # دشمن چپ‌تره
            pydirectinput.keyDown("w")
            pydirectinput.keyDown("a")
            time.sleep(0.12)
            pydirectinput.keyUp("a")
            pydirectinput.keyUp("w")
        else:
            # دشمن راست‌تره
            pydirectinput.keyDown("w")
            pydirectinput.keyDown("d")
            time.sleep(0.12)
            pydirectinput.keyUp("d")
            pydirectinput.keyUp("w")