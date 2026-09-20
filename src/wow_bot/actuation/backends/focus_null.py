"""Null focus backend implementation for MOCK_MODE and testing."""


class _NullWindowHandle:
    def __repr__(self) -> str:
        return "null-window"


class NullFocusBackend:
    """Synthetic focus backend that always reports the target window is focused.

    Used in MOCK_MODE and CI tests.
    """

    def __init__(self) -> None:
        self._handle = _NullWindowHandle()

    def find_window(self, title_substring: str) -> object | None:
        """Returns a synthetic handle representing the requested window."""
        return self._handle

    def get_foreground_handle(self) -> object | None:
        """Returns the synthetic foreground window handle."""
        return self._handle

    def same_window(self, a: object, b: object) -> bool:
        """Compares two handles for object identity."""
        return a is b

    def close(self) -> None:
        """No-op cleanup for NullFocusBackend."""
