"""
watchdog.py
------------
لایه‌ی ایمنی مستقل از FSM: مشکلات رایج (گیر کردن، مرگ کاراکتر، قطع ارتباط، حلقه‌ی
بی‌نتیجه) رو تشخیص می‌ده و می‌تونه بات رو کامل متوقف کنه (STOP اضطراری).

طراحی عمدی: این ماژول از FSM جدا نگه داشته شده، چون Watchdog باید حتی وقتی FSM
خودش دچار باگ/گیر شده هم بتونه تشخیص بده و جلوش رو بگیره - نباید به سلامت همون
چیزی که داره چکش می‌کنه وابسته باشه.
"""

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

try:
    import keyboard  # برای Kill Switch سراسری (حتی وقتی فوکوس روی بازی نیست)
    HAS_KEYBOARD_LIB = True
except ImportError:
    HAS_KEYBOARD_LIB = False


class HaltReason(Enum):
    NONE = auto()
    KILL_SWITCH = auto()
    CHARACTER_DEAD = auto()
    BRIDGE_DISCONNECTED = auto()
    STUCK_NO_MOVEMENT = auto()
    STATE_LOOP_TIMEOUT = auto()


@dataclass
class WatchdogConfig:
    kill_switch_key: str = "f10"

    stuck_check_window_sec: float = 20.0      # توی این بازه، انتظار جابه‌جایی داریم
    stuck_min_position_delta: float = 0.01    # کمتر از این تغییر mx/my یعنی "گیر کرده"
    stuck_requires_moving_state: bool = True  # فقط وقتی که قرار بوده حرکت کنه چک کن

    state_loop_max_sec: float = 90.0          # بیش از این توی یه state خاص بمونه یعنی گیره

    bridge_stale_sec: float = 5.0


@dataclass
class WatchdogStatus:
    halted: bool = False
    reason: HaltReason = HaltReason.NONE
    message: str = ""


class Watchdog:
    def __init__(self, config: WatchdogConfig = None):
        self.config = config or WatchdogConfig()
        self._status = WatchdogStatus()

        self._position_history: list[tuple[float, float, float]] = []  # (t, mx, my)
        self._current_fsm_state_name: Optional[str] = None
        self._fsm_state_entered_at: float = time.monotonic()

        self._kill_switch_triggered = False
        if HAS_KEYBOARD_LIB:
            try:
                keyboard.add_hotkey(self.config.kill_switch_key, self._on_kill_switch)
            except Exception as e:
                print(f"[Watchdog] نتونستم kill switch رو ثبت کنم: {e}")

    # ---------- Kill Switch ----------

    def _on_kill_switch(self):
        self._kill_switch_triggered = True
        print(f"[Watchdog] Kill Switch ({self.config.kill_switch_key}) فعال شد!")

    def reset_kill_switch(self):
        """بعد از این‌که خودت دستی متوقفش کردی و می‌خوای دوباره ران کنی."""
        self._kill_switch_triggered = False
        self._status = WatchdogStatus()

    # ---------- ورودی وضعیت ----------

    def notify_fsm_state(self, state_name: str):
        """FSM هر تیک این رو صدا بزنه تا Watchdog بفهمه الان کجاست."""
        if state_name != self._current_fsm_state_name:
            self._current_fsm_state_name = state_name
            self._fsm_state_entered_at = time.monotonic()

    def notify_position(self, mx: float, my: float):
        now = time.monotonic()
        self._position_history.append((now, mx, my))
        cutoff = now - self.config.stuck_check_window_sec
        self._position_history = [p for p in self._position_history if p[0] >= cutoff]

    # ---------- بررسی اصلی ----------

    def check(self, bot_state, is_moving_state: bool) -> WatchdogStatus:
        """
        bot_state: همون BotState از bridge_reader
        is_moving_state: True اگه FSM الان توی حالتیه که انتظار داریم شخصیت حرکت کنه
                         (مثلاً MOVE_TO یا FLEEING) - برای اینکه توی COMBAT ایستاده
                         رو اشتباهی "گیر کرده" تشخیص ندیم.
        """
        if self._status.halted:
            return self._status  # یه‌بار halt شد، تا reset دستی نشه همون‌جا می‌مونه

        if self._kill_switch_triggered:
            return self._halt(HaltReason.KILL_SWITCH, "کاربر Kill Switch رو فعال کرد.")

        if bot_state.is_stale(self.config.bridge_stale_sec):
            return self._halt(HaltReason.BRIDGE_DISCONNECTED,
                               "دیتا از ادان قطع شده (لودینگ‌اسکرین/دیسکانکت/کرش؟).")

        if bot_state.hp_pct <= 0:
            return self._halt(HaltReason.CHARACTER_DEAD,
                               "کاراکتر مرده (hp=0). نیاز به بررسی دستی / Release+Corpse Run.")

        self.notify_position(bot_state.mx, bot_state.my)

        if self.config.stuck_requires_moving_state and is_moving_state:
            stuck = self._is_stuck()
            if stuck:
                return self._halt(HaltReason.STUCK_NO_MOVEMENT,
                                   "موقعیت (mx,my) برای مدت طولانی تغییر نکرده - احتمالاً گیر کرده.")

        if self._current_fsm_state_name is not None:
            time_in_state = time.monotonic() - self._fsm_state_entered_at
            if time_in_state > self.config.state_loop_max_sec:
                return self._halt(HaltReason.STATE_LOOP_TIMEOUT,
                                   f"بیش از {self.config.state_loop_max_sec:.0f} ثانیه توی حالت "
                                   f"'{self._current_fsm_state_name}' گیر کرده.")

        return self._status

    def _is_stuck(self) -> bool:
        if len(self._position_history) < 2:
            return False
        # فقط وقتی که پنجره‌ی زمانی کامل پر شده باشه قضاوت کن (نه همون اول اجرا)
        oldest_t = self._position_history[0][0]
        if (time.monotonic() - oldest_t) < self.config.stuck_check_window_sec * 0.9:
            return False

        xs = [p[1] for p in self._position_history]
        ys = [p[2] for p in self._position_history]
        max_delta = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        return max_delta < self.config.stuck_min_position_delta

    def _halt(self, reason: HaltReason, message: str) -> WatchdogStatus:
        self._status = WatchdogStatus(halted=True, reason=reason, message=message)
        print(f"[Watchdog] *** HALT *** reason={reason.name} | {message}")
        return self._status

    @property
    def status(self) -> WatchdogStatus:
        return self._status
