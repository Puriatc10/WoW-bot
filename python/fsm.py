"""
fsm.py
------
مغز بات: حالت فعلی رو بر اساس BotState (از bridge_reader) تصمیم می‌گیره و
از Controller برای اجرای اکشن استفاده می‌کنه.

چرخه‌ی حالت‌ها:
    SCAN        -> پیدا کردن نزدیک‌ترین دشمن (Tab-target)
    MOVE_TO     -> نزدیک شدن تا وارد رنج مبارزه بشیم
    COMBAT      -> اجرای Rotation تا کشتن هدف
    LOOT        -> رفتن سراغ جسد و لوت
    FLEEING     -> اگه HP بحرانی شد (اولویت روی همه‌چی)

نکته‌ی مهم: fsm فعلاً برای «جهتِ حرکت به سمت هدف» به یه منبع جهت نیاز داره چون
API آدرس بازی مختصات target رو نمی‌ده. دو گزینه:
  ۱) minimap_detector.py که قبلاً ساختی (نقطه‌ی قرمز روی مینی‌مپ -> بردار dx,dy)
  ۲) اگه موجود نبود، فعلاً یه heuristic ساده (حرکت مستقیم + تنظیم دوره‌ای) استفاده می‌شه.

این فایل import اختیاری از minimap_detector داره؛ اگه فایل نبود، بدون کرش با heuristic ادامه می‌ده.
"""

import time
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from bridge_reader import AutoBridge, BotState
from controller import Controller
from watchdog import Watchdog, WatchdogConfig
import humanize

try:
    import enemy_vision
    HAS_ENEMY_VISION = True
except ImportError:
    HAS_ENEMY_VISION = False


class State(Enum):
    SCAN = auto()
    MOVE_TO = auto()
    COMBAT = auto()
    LOOT = auto()
    FLEEING = auto()


@dataclass
class FSMConfig:
    flee_hp_pct: int = 20          # زیر این درصد جون، فرار کن
    resume_hp_pct: int = 60        # بالای این درصد، برگرد به کار
    scan_retry_sec: float = 1.5    # هر چند وقت یه‌بار Tab بزنه اگه هدفی پیدا نشد
    move_timeout_sec: float = 12.0 # اگه این‌قدر طول کشید برسیم ولی combat شروع نشد، برگرد به SCAN
    rotation_gcd_sec: float = 1.5  # فاصله‌ی زمانی بین اسکیل‌ها (Global Cooldown)
    loot_wait_sec: float = 1.2     # چقدر صبر کنه بعد از مرگ هدف قبل از لوت
    target_dead_hp_threshold: int = 5  # زیر این درصد رو "مرده" حساب کن؛ برای گم‌شدن ناگهانی هم مصونیت داره


class GrindFSM:
    def __init__(self, bridge, controller: Controller, config: FSMConfig = None):
        self.bridge = bridge
        self.controller = controller
        self.config = config or FSMConfig()

        self.state = State.SCAN
        self._state_entered_at = time.monotonic()
        self._last_scan_attempt = 0.0
        self._last_rotation_step_at = 0.0
        self._next_rotation_delay = self.config.rotation_gcd_sec
        self._rotation_index = 0
        self._pre_flee_state = State.SCAN

        self._session_clock = humanize.SessionClock()
        self._paused_until = 0.0

        self._last_known_target_hp_pct: Optional[int] = None
        self._targets_lost_count = 0   # فقط برای دیباگ/لاگ، اختیاری

    # ---------- کمکی ----------

    def _enter(self, new_state: State):
        if new_state != self.state:
            self.controller.stop_all_movement()
            print(f"[FSM] {self.state.name} -> {new_state.name}")
            self.state = new_state
            self._state_entered_at = time.monotonic()

    def _time_in_state(self) -> float:
        return time.monotonic() - self._state_entered_at

    def _get_target_dx(self):
        """
        افست نسبی نیم‌پلیت دشمن از مرکز صفحه رو برمی‌گردونه (-1 چپ ... +1 راست)
        یا None اگه دیده نمی‌شه.
        """
        if HAS_ENEMY_VISION:
            try:
                dx, found = enemy_vision.get_nearest_enemy_offset()
                return dx if found else None
            except Exception as e:
                print(f"[FSM] enemy_vision error: {e}")
                return None
        return None

    # ---------- تیک اصلی ----------

    def tick(self):
        now = time.monotonic()

        # مکث‌های انسانی فقط وقتی معناداره که خطر فوری در کار نباشه (مثلاً HP بحرانی نیست)
        if now < self._paused_until:
            self.controller.stop_all_movement()
            return

        state = self.bridge.get_state()

        if state.is_stale(3.0):
            # اگه دیتا از ادان قطع شده (مثلاً لودینگ‌اسکرین/دیسکانکت)، همه‌چی رو متوقف کن
            self.controller.stop_all_movement()
            return

        is_critical = 0 < state.hp_pct <= self.config.flee_hp_pct
        safe_to_pause = (not is_critical) and (not state.in_combat) and self.state in (State.SCAN, State.LOOT)

        if state.has_target:
            self._last_known_target_hp_pct = state.target_hp_pct

        if safe_to_pause:
            if humanize.should_take_long_break(self._session_clock.elapsed()):
                dur = humanize.long_break_duration()
                print(f"[FSM] استراحت شبیه‌سازی‌شده ({dur:.1f} ثانیه)")
                self._paused_until = now + dur
                self.controller.stop_all_movement()
                return

            if humanize.should_take_micro_break():
                dur = humanize.micro_break_duration()
                self._paused_until = now + dur
                self.controller.stop_all_movement()
                return

        # --- اولویت اول: فرار ---
        if self.state != State.FLEEING and state.hp_pct <= self.config.flee_hp_pct and state.hp_pct > 0:
            self._pre_flee_state = self.state
            self._enter(State.FLEEING)

        if self.state == State.FLEEING:
            self._handle_fleeing(state)
            return

        if self.state == State.SCAN:
            self._handle_scan(state)
        elif self.state == State.MOVE_TO:
            self._handle_move_to(state)
        elif self.state == State.COMBAT:
            self._handle_combat(state)
        elif self.state == State.LOOT:
            self._handle_loot(state)

    # ---------- حالت‌ها ----------

    def _handle_scan(self, state: BotState):
        if state.has_target and state.target_reaction == "hostile" and state.target_hp_pct > 0:
            self._enter(State.MOVE_TO)
            return

        now = time.monotonic()
        if now - self._last_scan_attempt >= self.config.scan_retry_sec:
            self._last_scan_attempt = now
            self.controller.target_nearest_enemy()

    def _handle_move_to(self, state: BotState):
        # اگه تارگت رو از دست دادیم یا دیگه hostile نیست، برگرد به SCAN
        if not state.has_target or state.target_reaction != "hostile" or state.target_hp_pct <= 0:
            self._enter(State.SCAN)
            return

        # اگه در حال حاضر داخل کامباتیم، دیگه نیازی به حرکت نیست
        if state.in_combat:
            self._enter(State.COMBAT)
            return

        if self._time_in_state() > self.config.move_timeout_sec:
            print("[FSM] MOVE_TO timeout - برمی‌گردیم به SCAN")
            self._enter(State.SCAN)
            return

        target_dx = self._get_target_dx()
        if target_dx is not None:
            aligned = self.controller.turn_relative(target_dx)
            if aligned:
                self.controller.move_forward_start()
        else:
            # نیم‌پلیت دیده نمی‌شه (مثلاً پشت مانع یا خیلی دوره) -> فعلاً مستقیم برو جلو
            self.controller.move_forward_start()

    def _handle_combat(self, state: BotState):
        self.controller.stop_all_movement()

        if not state.has_target:
            # هدف ناپدید شده - باید بفهمیم مرده یا فرار کرده/گم شده
            was_dying = (
                self._last_known_target_hp_pct is not None
                and self._last_known_target_hp_pct <= self.config.target_dead_hp_threshold
            )
            self._rotation_index = 0
            if was_dying:
                self._enter(State.LOOT)
            else:
                self._targets_lost_count += 1
                print(f"[FSM] هدف گم شد (نه مرد) - آخرین HP شناخته‌شده: "
                      f"{self._last_known_target_hp_pct}% - برمی‌گردیم به SCAN")
                self._last_known_target_hp_pct = None
                self._enter(State.SCAN)
            return

        if state.target_hp_pct <= 0:
            self._rotation_index = 0
            self._enter(State.LOOT)
            return

        if not state.in_combat and state.target_reaction != "hostile":
            self._enter(State.SCAN)
            return

        now = time.monotonic()
        if now - self._last_rotation_step_at >= self._next_rotation_delay:
            self._last_rotation_step_at = now
            self._next_rotation_delay = humanize.humanized_rotation_gcd(self.config.rotation_gcd_sec)
            self.controller.cast_rotation_step(self._rotation_index)
            self._rotation_index += 1

    def _handle_loot(self, state: BotState):
        if self._time_in_state() < self.config.loot_wait_sec:
            return

        self.controller.interact_target()
        self.controller.loot()

        if self._time_in_state() > self.config.loot_wait_sec + 2.0:
            self._last_known_target_hp_pct = None
            self._enter(State.SCAN)

    def _handle_fleeing(self, state: BotState):
        # ساده: بچرخ ۱۸۰ درجه از دشمن و برو جلو تا HP برگرده
        if state.hp_pct >= self.config.resume_hp_pct or state.hp_pct == 0:
            self._enter(self._pre_flee_state)
            return

        away_angle = (state.facing + math.pi) % (2 * math.pi)
        aligned = self.controller.turn_toward(state.facing, away_angle)
        if aligned:
            self.controller.move_forward_start()


def run(wow_root: str, key_bindings: dict, tick_hz: float = 10.0):
    bridge = AutoBridge(wow_root)
    bridge.start()

    controller = Controller(key_bindings)
    fsm = GrindFSM(bridge, controller)
    watchdog = Watchdog(WatchdogConfig())

    print("FSM شروع شد. Ctrl+C یا کلید Kill Switch (پیش‌فرض F10) برای توقف.")
    interval = 1.0 / tick_hz
    try:
        while True:
            fsm.tick()

            watchdog.notify_fsm_state(fsm.state.name)
            is_moving_state = fsm.state in (State.MOVE_TO, State.FLEEING)
            status = watchdog.check(bridge.get_state(), is_moving_state=is_moving_state)

            if status.halted:
                controller.stop_all_movement()
                print(f"[run] بات متوقف شد: {status.reason.name} - {status.message}")
                print("[run] برای ادامه، برنامه رو دستی ری‌استارت کن (بعد از رفع مشکل).")
                break

            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop_all_movement()
        bridge.stop()
        print("متوقف شد.")


if __name__ == "__main__":
    import sys

    wow_root = sys.argv[1] if len(sys.argv) > 1 else r"C:\World of Warcraft"

    bindings = {
        "target_nearest_enemy": "tab",
        "interact_target": "t",
        "loot": "e",
        "rotation": ["1", "2", "3"],
    }

    run(wow_root, bindings)
