"""Win32 focus backend implementation for LAB_MODE on Windows."""

from typing import Any, cast


class Win32FocusBackend:
    """Read-only Win32 focus backend for tracking game window focus.

    Imports ctypes and Win32 APIs lazily inside __init__.
    MUST NOT invoke any mutating Win32 APIs (e.g., SetForegroundWindow, ShowWindow).
    """

    def __init__(self) -> None:
        from wow_bot.actuation.focus import FocusBackendError

        try:
            import ctypes
            import sys

            if sys.platform != "win32":
                raise FocusBackendError("Win32FocusBackend is only supported on Windows")

            import ctypes.wintypes

            self._ctypes = ctypes
            self._wintypes = ctypes.wintypes
            windll = getattr(ctypes, "windll", None)
            if windll is None:
                raise FocusBackendError("ctypes.windll not available")
            self._user32: Any = windll.user32

            # Setup prototypes
            self._WINFUNCTYPE: Any = ctypes.WINFUNCTYPE
            self._EnumWindowsProc: Any = self._WINFUNCTYPE(
                ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM
            )

            self._user32.EnumWindows.argtypes = [self._EnumWindowsProc, ctypes.wintypes.LPARAM]
            self._user32.EnumWindows.restype = ctypes.c_bool

            self._user32.GetWindowTextLengthW.argtypes = [ctypes.wintypes.HWND]
            self._user32.GetWindowTextLengthW.restype = ctypes.c_int

            self._user32.GetWindowTextW.argtypes = [
                ctypes.wintypes.HWND,
                ctypes.wintypes.LPWSTR,
                ctypes.c_int,
            ]
            self._user32.GetWindowTextW.restype = ctypes.c_int

            self._user32.GetForegroundWindow.argtypes = []
            self._user32.GetForegroundWindow.restype = ctypes.wintypes.HWND

        except Exception as exc:
            if isinstance(exc, FocusBackendError):
                raise
            raise FocusBackendError(f"Failed to initialize Win32 focus backend: {exc}") from exc

        self._cached_handle: object | None = None

    def find_window(self, title_substring: str) -> object | None:
        """Enumerates top-level windows and returns the first matching handle."""
        found_hwnd: object | None = None

        def enum_windows_callback(hwnd: Any, lparam: Any) -> bool:
            nonlocal found_hwnd
            length = self._user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buffer = self._ctypes.create_unicode_buffer(length + 1)
                self._user32.GetWindowTextW(hwnd, buffer, length + 1)
                title = buffer.value
                if title_substring in title:
                    found_hwnd = hwnd
                    return False  # Stop enumeration
            return True  # Continue enumeration

        callback = self._EnumWindowsProc(enum_windows_callback)
        self._user32.EnumWindows(callback, 0)
        self._cached_handle = found_hwnd
        return found_hwnd

    def get_foreground_handle(self) -> object | None:
        """Returns the current foreground window handle."""
        hwnd = self._user32.GetForegroundWindow()
        if not hwnd:
            return None
        return hwnd  # type: ignore[no-any-return]

    def same_window(self, a: object, b: object) -> bool:
        """Compares two HWND values."""
        if a is None or b is None:
            return a is b
        try:
            return bool(int(cast(Any, a)) == int(cast(Any, b)))
        except (TypeError, ValueError, AttributeError):
            return bool(a == b)

    def close(self) -> None:
        """Releases cached handle reference; idempotent."""
        self._cached_handle = None
