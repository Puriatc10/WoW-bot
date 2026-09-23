import mss
import numpy as np
import time
import threading

class ScreenCapture:
    def __init__(self, monitor_idx=1, idle_fps=10, combat_fps=30, pool_size=5):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[monitor_idx]
        self.idle_fps = idle_fps
        self.combat_fps = combat_fps
        self.current_fps = idle_fps
        self.running = False
        self._lock = threading.Lock()

        h = self.monitor["height"]
        w = self.monitor["width"]
        self.pool = [np.empty((h, w, 4), dtype=np.uint8) for _ in range(pool_size)]
        self.pool_idx = 0
        self.latest_frame = None

    def _grab(self):
        img = np.array(self.sct.grab(self.monitor), dtype=np.uint8)
        with self._lock:
            buf = self.pool[self.pool_idx]
            if buf.shape == img.shape:
                buf[:] = img
                self.latest_frame = buf.copy()
            else:
                self.latest_frame = img
            self.pool_idx = (self.pool_idx + 1) % len(self.pool)

    def set_mode(self, mode: str):
        self.current_fps = self.combat_fps if mode == "combat" else self.idle_fps

    def get_frame(self):
        with self._lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def start(self):
        self.running = True
        def loop():
            while self.running:
                t0 = time.time()
                self._grab()
                dt = 1.0 / self.current_fps - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)