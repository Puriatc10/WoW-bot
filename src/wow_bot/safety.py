"""Safety layer skeleton for WoW-bot lab execution mode.

Provides exception definitions, signal handler registration (arm/disarm),
address allowlist verification, isolation sentinel checking, and thread/signal-safe
abort sequence execution.
"""

import signal
import socket
import threading
from collections.abc import Callable
from typing import Any

from wow_bot.config import Config
from wow_bot.session import Session


class SafetyError(Exception):
    """Base exception for safety layer errors."""


class AllowlistViolation(SafetyError):
    """Raised when an address is not present in the server allowlist."""


class IsolationViolation(SafetyError):
    """Raised when the isolation sentinel address is reachable."""


def _parse_host_port(addr: str, *, require_port: bool = False) -> tuple[str, int | None]:
    """Parse address into (host, port) tuple.

    Raises SafetyError if addr is malformed.
    """
    if not isinstance(addr, str) or not addr.strip():
        raise SafetyError("Address must be a non-empty string.")

    addr_clean = addr.strip()
    if ":" in addr_clean:
        parts = addr_clean.split(":")
        if len(parts) != 2:
            raise SafetyError(f"Malformed address format: '{addr}'.")
        host, port_str = parts[0].strip(), parts[1].strip()
        if not host:
            raise SafetyError(f"Malformed host in address: '{addr}'.")
        if not port_str.isdigit():
            raise SafetyError(f"Malformed port in address: '{addr}'.")
        port = int(port_str)
        if not (1 <= port <= 65535):
            raise SafetyError(f"Port out of range (1-65535) in address: '{addr}'.")
        return host, port
    else:
        if require_port:
            raise SafetyError(f"Expected 'host:port' format, got: '{addr}'.")
        if not addr_clean:
            raise SafetyError("Malformed host in address.")
        return addr_clean, None


class SafetyLayer:
    """Safety supervisor managing signal handlers, allowlist, network isolation, and abort procedures."""

    def __init__(self, config: Config, session: Session | None = None) -> None:
        self._config = config
        self._session = session
        self._abort_event = threading.Event()
        self._abort_reason: str | None = None
        self._callbacks: list[Callable[[], None]] = []
        self._lock = threading.Lock()
        self._armed = False
        self._prev_sigterm: Any = None
        self._prev_sigint: Any = None

    def arm(self) -> None:
        """Register signal handlers for SIGTERM and SIGINT.

        Must be called from the main thread. Idempotent.
        """
        if threading.current_thread() is not threading.main_thread():
            raise SafetyError("arm() must be called from the main thread")

        if self._armed:
            return

        def _handle_sigterm(signum: int, frame: Any) -> None:
            try:
                self.abort("signal:SIGTERM")
            except Exception:  # noqa: BLE001, S110
                pass

        def _handle_sigint(signum: int, frame: Any) -> None:
            try:
                self.abort("signal:SIGINT")
            except Exception:  # noqa: BLE001, S110
                pass

        self._prev_sigterm = signal.signal(signal.SIGTERM, _handle_sigterm)
        self._prev_sigint = signal.signal(signal.SIGINT, _handle_sigint)
        self._armed = True

    def disarm(self) -> None:
        """Restore previous signal handlers. Idempotent."""
        if not self._armed:
            return

        if self._prev_sigterm is not None:
            signal.signal(signal.SIGTERM, self._prev_sigterm)
            self._prev_sigterm = None

        if self._prev_sigint is not None:
            signal.signal(signal.SIGINT, self._prev_sigint)
            self._prev_sigint = None

        self._armed = False

    def check_allowlist(self, addr: str) -> None:
        """Verify addr against config server allowlist.

        Raises SafetyError if addr is malformed.
        Raises AllowlistViolation if addr is not allowlisted.
        Emits no session events. Must not acquire locks or allocate heavily.
        """
        host, port = _parse_host_port(addr, require_port=False)
        host_lower = host.lower()

        for entry in self._config.server_allowlist:
            entry_host, entry_port = _parse_host_port(entry, require_port=False)
            if host_lower == entry_host.lower():
                if entry_port is None:
                    return
                if port == entry_port:
                    return

        raise AllowlistViolation(f"Address '{addr}' is not in server allowlist.")

    def check_isolation(self) -> None:
        """Verify network isolation by attempting to connect to isolation sentinel.

        Raises SafetyError if sentinel format is malformed.
        Raises IsolationViolation if sentinel is reachable.
        Returns None if connection fails for any reason.
        """
        host, port = _parse_host_port(self._config.isolation_sentinel, require_port=True)
        assert port is not None

        try:
            sock = socket.create_connection((host, port), timeout=1.0)
        except Exception:  # noqa: BLE001
            return
        else:
            sock.close()
            raise IsolationViolation(
                f"isolation sentinel reachable: {self._config.isolation_sentinel}"
            )

    def abort(self, reason: str) -> None:
        """Abort execution, running kill switch callbacks and emitting session events. Idempotent."""
        if self._abort_event.is_set():
            return

        self._abort_event.set()

        with self._lock:
            if self._abort_reason is not None:
                return
            self._abort_reason = reason
            callbacks = list(self._callbacks)

        for callback in callbacks:
            try:
                callback()
            except Exception as exc:  # noqa: BLE001
                if self._session is not None:
                    try:
                        self._session.write_event({
                            "event": "safety_callback_failed",
                            "payload": {"error": repr(exc)},
                        })
                    except Exception:  # noqa: BLE001, S110
                        pass

        if self._session is not None:
            try:
                self._session.write_event({
                    "event": "safety_abort",
                    "payload": {"reason": reason},
                })
            except Exception:  # noqa: BLE001, S110
                pass

    def is_aborted(self) -> bool:
        """Return True if abort event is set, False otherwise."""
        return self._abort_event.is_set()

    def abort_reason(self) -> str | None:
        """Return stored abort reason or None."""
        with self._lock:
            return self._abort_reason

    def register_kill_switch(self, callback: Callable[[], None]) -> None:
        """Register a callback to execute on abort.

        Registration after abort is allowed but is a no-op for the callback itself. MUST NOT raise.
        """
        with self._lock:
            self._callbacks.append(callback)
