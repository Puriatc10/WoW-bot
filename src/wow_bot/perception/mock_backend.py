"""Reference mock backend for T-FIX-04."""

from __future__ import annotations

from wow_bot.mocks.mock_perception import MockPerception
from wow_bot.perception.protocol import PerceptionBackend
from wow_bot.shared.interfaces import GameState


class MockPerceptionBackend(PerceptionBackend):
    """Reference adapter wrapping MockPerception to satisfy PerceptionBackend."""

    def __init__(self, mock: MockPerception) -> None:
        """Initialize the mock backend.

        Parameters
        ----------
        mock:
            The MockPerception instance to draw state from.
        """
        self._mock = mock

    async def snapshot(self) -> GameState:
        """Capture and return a single authoritative GameState snapshot."""
        return await self._mock.get_state()
