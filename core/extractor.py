from core.state import GameState
import time


class StateExtractor:
    def __init__(self, capture, bar_reader, combat_detector,
                 target_reader, enemy_detector, event_detector,
                 minimap_tracker=None):
        self.cap = capture
        self.bars = bar_reader
        self.combat = combat_detector
        self.target = target_reader
        self.enemies = enemy_detector
        self.events = event_detector
        self.minimap = minimap_tracker
        self.last_state = None

    def extract(self) -> GameState:
        frame = self.cap.get_frame()
        if frame is None:
            return self.last_state

        # HP / Mana
        _, hp_pct = self.bars.read_hp(frame)
        _, mp_pct = self.bars.read_mana(frame)

        # Combat
        in_combat, _ = self.combat.detect(frame)

        # Target
        target_name, _, _ = self.target.read(frame)

        # Enemies
        enemies = self.enemies.detect(frame)

        # Events
        events = self.events.detect(frame)

        # Minimap (position + facing)
        position = None
        facing = None
        if self.minimap is not None:
            pos, ang, _ = self.minimap.update(frame)
            if pos is not None:
                position = (float(pos[0]), float(pos[1]))
            if ang is not None:
                facing = float(ang)

        # نرخ FPS
        self.cap.set_mode("combat" if in_combat else "idle")

        state = GameState(
            timestamp=time.time(),
            hp_pct=hp_pct,
            mana_pct=mp_pct,
            in_combat=in_combat,
            position=position,
            facing=facing,
            target=target_name,
            enemies=enemies,
            events=events,
        )

        self.last_state = state
        return state