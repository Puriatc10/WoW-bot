from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import time


@dataclass
class GameState:
    timestamp: float
    hp_pct: int
    mana_pct: int
    in_combat: bool
    position: Optional[Tuple[float, float]] = None
    facing: Optional[float] = None
    target: Optional[str] = None
    enemies: List = field(default_factory=list)
    events: List = field(default_factory=list)

    def __str__(self):
        pos = f"({self.position[0]:.0f},{self.position[1]:.0f})" if self.position else "-"
        face = f"{self.facing:.0f}°" if self.facing is not None else "-"
        combat = "⚔️" if self.in_combat else "🟢"
        return (f"[{self.timestamp:.1f}] "
                f"HP={self.hp_pct}% MP={self.mana_pct}% "
                f"{combat} "
                f"pos={pos} face={face} "
                f"trg={self.target or '-'} "
                f"enm={len(self.enemies)} evt={len(self.events)}")