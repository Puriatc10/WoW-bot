"""
WoW Bot - Final GameState Monitor
همه‌ی فیلدهای GameState رو زنده نشون می‌ده
"""

from capture.screen_capture import ScreenCapture
from perception.bars import BarReader
from perception.combat import CombatDetector
from perception.target import TargetReader
from perception.enemies import EnemyDetector
from perception.events import EventDetector
from perception.minimap import MinimapTracker
from core.extractor import StateExtractor
import time


# ============================================================
# تنظیمات
# ============================================================
HP_ROI = (514, 771, 123, 18)
MANA_ROI = (514, 793, 123, 17)
TEMPLATE_PATH = "models/target_name_template.png"
YOLO_PATH = "runs/detect/train-2/weights/best.pt"
MINIMAP_ROI = (1705, 33, 206, 217)
ARROW_TEMPLATE = "models/arrow_template.png"

# ============================================================
# راه‌اندازی
# ============================================================
print("=" * 80)
print("  WoW Bot - Full GameState Monitor")
print("=" * 80)
print("\nInitializing...")

cap = ScreenCapture(idle_fps=10, combat_fps=30)
cap.start()
time.sleep(2)

bar_reader = BarReader(HP_ROI, MANA_ROI)
combat_detector = CombatDetector(edge_size=150, red_ratio_thresh=0.02, cooldown=1.0)
target_reader = TargetReader(TEMPLATE_PATH, match_thresh=0.65)
enemy_detector = EnemyDetector(YOLO_PATH, conf=0.5)
event_detector = EventDetector()

# Minimap (اختیاری — اگه فایل‌ها موجود باشن)
minimap = None
try:
    import os
    if os.path.exists(ARROW_TEMPLATE):
        minimap = MinimapTracker(MINIMAP_ROI, ARROW_TEMPLATE, match_thresh=0.55)
        print("   ✅ Minimap loaded")
    else:
        print("   ⚠️ Minimap template not found, skipping")
except Exception as e:
    print(f"   ⚠️ Minimap error: {e}")

extractor = StateExtractor(
    capture=cap,
    bar_reader=bar_reader,
    combat_detector=combat_detector,
    target_reader=target_reader,
    enemy_detector=enemy_detector,
    event_detector=event_detector,
    minimap_tracker=minimap,
)

print("✅ All modules loaded.\n")
print("=" * 80)
print(f"{'TIME':<10} {'HP':>4} {'MP':>4} {'CMB':>5} {'POS':>12} {'FACE':>6} "
      f"{'TRG':<18} {'ENM':>4} {'EVT':>4}")
print("=" * 80)

# ============================================================
# حلقه اصلی
# ============================================================
try:
    while True:
        state = extractor.extract()
        if state is None:
            time.sleep(0.1)
            continue

        # خط اصلی
        t = time.strftime("%H:%M:%S")
        cmb = "YES" if state.in_combat else "no"
        pos = (f"({state.position[0]:.0f},{state.position[1]:.0f})"
               if state.position else "-")
        face = f"{state.facing:.0f}°" if state.facing is not None else "-"
        trg = (state.target or "-")[:16]

        print(f"{t:<10} {state.hp_pct:>3}% {state.mana_pct:>3}% {cmb:>5} "
              f"{pos:>12} {face:>6} {trg:<18} "
              f"{len(state.enemies):>4} {len(state.events):>4}")

        # رویدادها
        for e in state.events:
            src = e.get("source", "?")[:3]
            print(f"           [{src}] {e['type'].upper()}: {e['text'][:55]}")

        # نزدیک‌ترین دشمن
        if state.enemies:
            closest = max(state.enemies,
                          key=lambda e: e["width"] * e["height"])
            print(f"           [ENM] conf={closest['conf']:.2f} "
                  f"at {closest['center']} "
                  f"size={closest['width']}x{closest['height']}")

        time.sleep(1.0)

except KeyboardInterrupt:
    pass

cap.stop()
print("\n" + "=" * 80)
print("Stopped.")