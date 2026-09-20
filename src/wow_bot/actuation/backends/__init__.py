"""Focus backend implementations."""

from wow_bot.actuation.backends.focus_null import NullFocusBackend
from wow_bot.actuation.backends.focus_win32 import Win32FocusBackend

__all__ = [
    "NullFocusBackend",
    "Win32FocusBackend",
]
