"""
bridge_reader.py
-----------------
Console.log رو Tail می‌کنه (مثل tail -f)، خطوط WBB| رو پارس می‌کنه و آخرین state رو نگه می‌داره.

استفاده:
    bridge = BridgeReader(r"C:\\Path\\To\\WoW\\Logs\\Console.log")
    bridge.start()
    ...
    state = bridge.get_state()
    print(state.hp_pct, state.mx, state.my, state.target_name)

نکته‌ی مهم Setup:
    داخل بازی یک‌بار اجرا کن:  /console consoleLog 1   سپس  /reload
    مسیر فایل معمولاً: <WoW Folder>/Logs/Console.log
"""

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BotState:
    timestamp: int = 0
    hp_pct: int = 0
    power_pct: int = 0
    mx: float = 0.0          # موقعیت نرمال‌شده روی نقشه (0..1)
    my: float = 0.0
    facing: float = 0.0      # رادیان، 0..2pi
    in_combat: bool = False
    has_target: bool = False
    target_name: str = ""
    target_hp_pct: int = 0
    target_dist: int = -1
    target_reaction: str = "unknown"   # hostile / neutral / friendly / unknown
    last_update_monotonic: float = field(default_factory=lambda: float("-inf"))

    def is_stale(self, max_age_sec: float = 2.0) -> bool:
        """اگه مدتیه خط جدید نیومده (مثلاً ادان لود نشده یا لاگ خاموشه)."""
        return (time.monotonic() - self.last_update_monotonic) > max_age_sec


_LINE_RE = re.compile(
    r"WBB\|t:(?P<t>\d+)\|hp:(?P<hp>\d+)\|power:(?P<power>\d+)\|"
    r"mx:(?P<mx>[\d.]+)\|my:(?P<my>[\d.]+)\|facing:(?P<facing>[\d.]+)\|"
    r"combat:(?P<combat>\d)\|tgt:(?P<tgt>\d)"
    r"(?:\|tname:(?P<tname>[^|]*)\|thp:(?P<thp>\d+)\|tdist:(?P<tdist>-?\d+)\|treact:(?P<treact>\w+))?"
)


def parse_line(line: str) -> Optional[BotState]:
    m = _LINE_RE.search(line)
    if not m:
        return None
    d = m.groupdict()
    state = BotState(
        timestamp=int(d["t"]),
        hp_pct=int(d["hp"]),
        power_pct=int(d["power"]),
        mx=float(d["mx"]),
        my=float(d["my"]),
        facing=float(d["facing"]),
        in_combat=(d["combat"] == "1"),
        has_target=(d["tgt"] == "1"),
    )
    if state.has_target:
        state.target_name = (d.get("tname") or "").replace("_", " ")
        state.target_hp_pct = int(d.get("thp") or 0)
        state.target_dist = int(d.get("tdist") or -1)
        state.target_reaction = d.get("treact") or "unknown"
    return state


class BridgeReader:
    def __init__(self, log_path: str, poll_interval: float = 0.05):
        self.log_path = log_path
        self.poll_interval = poll_interval
        self._state = BotState()
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=2)

    def get_state(self) -> BotState:
        with self._lock:
            return self._state

    def _run(self):
        f = None
        while not self._stop_flag.is_set():
            try:
                if f is None:
                    f = open(self.log_path, "r", encoding="utf-8", errors="ignore")
                    f.seek(0, 2)  # برو انتهای فایل (نمی‌خوایم تاریخچه‌ی قدیمی رو پردازش کنیم)

                line = f.readline()
                if not line:
                    time.sleep(self.poll_interval)
                    continue

                parsed = parse_line(line)
                if parsed is not None:
                    parsed.last_update_monotonic = time.monotonic()
                    with self._lock:
                        self._state = parsed

            except FileNotFoundError:
                # فایل هنوز ساخته نشده (consoleLog فعال نیست یا بازی هنوز اجرا نشده)
                time.sleep(1.0)
            except Exception as e:
                print(f"[BridgeReader] error: {e}")
                time.sleep(1.0)

        if f:
            f.close()


class FileStateReader:
    """
    برای حالت 'file' ادان: هر تیک یه فایل کوچیک بازنویسی می‌شه (نه append).
    اینجا فقط فایل رو می‌خونیم و پارس می‌کنیم - خیلی ساده‌تر و بدون ریسک رشد بی‌نهایت فایل.
    """

    def __init__(self, file_path: str, poll_interval: float = 0.05):
        self.file_path = file_path
        self.poll_interval = poll_interval
        self._state = BotState()
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=2)

    def get_state(self) -> BotState:
        with self._lock:
            return self._state

    def _run(self):
        while not self._stop_flag.is_set():
            try:
                with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().strip()
                if content and content != "WBB_TEST":
                    last_line = content.splitlines()[-1]
                    parsed = parse_line(last_line)
                    if parsed is not None:
                        parsed.last_update_monotonic = time.monotonic()
                        with self._lock:
                            self._state = parsed
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"[FileStateReader] error: {e}")
            time.sleep(self.poll_interval)


class AutoBridge:
    """
    خودش تشخیص می‌ده ادان از کدوم حالت داره استفاده می‌کنه (file یا console log تیل)
    و همون‌جوری داده رو می‌خونه. کافیه مسیر ریشه‌ی پوشه‌ی نصب WoW رو بدی.

    مثال:
        bridge = AutoBridge(r"C:\\Games\\MyPrivateServerWoW")
        bridge.start()
        state = bridge.get_state()
    """

    def __init__(self, wow_root: str, detect_after_sec: float = 3.0):
        import os as _os

        self.wow_root = wow_root.rstrip("\\/")
        self.state_file_path = _os.path.join(
            self.wow_root, "Interface", "AddOns", "WoWBotBridge", "wbb_state.txt"
        )
        self.console_log_path = _os.path.join(self.wow_root, "Logs", "Console.log")
        self.detect_after_sec = detect_after_sec

        self._file_reader = FileStateReader(self.state_file_path)
        self._log_reader = BridgeReader(self.console_log_path)
        self._active: Optional[str] = None  # "file" | "chatlog"
        self._started_at = 0.0

    def start(self):
        self._file_reader.start()
        self._log_reader.start()
        self._started_at = time.monotonic()

    def stop(self):
        self._file_reader.stop()
        self._log_reader.stop()

    def get_state(self) -> BotState:
        file_state = self._file_reader.get_state()
        log_state = self._log_reader.get_state()

        # هر کدوم که تازه‌تره (یعنی همونیه که ادان واقعاً داره ازش استفاده می‌کنه) رو انتخاب کن
        if not file_state.is_stale(1.0):
            self._active = "file"
            return file_state
        if not log_state.is_stale(1.0):
            self._active = "chatlog"
            return log_state

        # هیچ‌کدوم تازه نیستن - آخرین چیزی که داشتیم رو برگردون (FSM خودش stale بودن رو چک می‌کنه)
        return file_state if file_state.timestamp >= log_state.timestamp else log_state

    @property
    def active_mode(self) -> Optional[str]:
        return self._active


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else r"C:\World of Warcraft\Logs\Console.log"
    reader = BridgeReader(path)
    reader.start()
    print(f"در حال خواندن از: {path}")
    print("منتظر داده... (اگه چیزی نیومد، /console consoleLog 1 و /reload رو داخل بازی بزن)")
    try:
        while True:
            s = reader.get_state()
            status = "STALE" if s.is_stale() else "OK"
            print(
                f"[{status}] hp={s.hp_pct}% power={s.power_pct}% "
                f"pos=({s.mx:.3f},{s.my:.3f}) facing={s.facing:.2f} "
                f"combat={s.in_combat} target={s.target_name or '-'} "
                f"thp={s.target_hp_pct}% react={s.target_reaction}"
            )
            time.sleep(0.5)
    except KeyboardInterrupt:
        reader.stop()
