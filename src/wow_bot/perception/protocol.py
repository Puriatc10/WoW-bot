"""Abstract base class defining the unified perception backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from wow_bot.shared.interfaces import GameState


class PerceptionBackend(ABC):
    """Abstract base class defining the async perception backend contract."""

    @abstractmethod
    async def snapshot(self) -> GameState:
        """Capture and return a single authoritative GameState snapshot."""
        ...
