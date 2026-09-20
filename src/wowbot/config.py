"""Configuration loader and schema for lab execution mode."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

REQUIRED_KEYS = {
    "lab_mode",
    "server_allowlist",
    "isolation_sentinel",
    "kill_switch_key",
    "session_root",
    "dry_run",
    "max_session_seconds",
    "log_level",
}

VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
FORBIDDEN_DOMAINS = ("blizzard.com", "battle.net", "worldofwarcraft.com")


class ConfigError(Exception):
    """Exception raised when configuration loading or validation fails."""


@dataclass(frozen=True)
class Config:
    """Immutable lab execution configuration."""

    lab_mode: bool
    server_allowlist: tuple[str, ...]
    isolation_sentinel: str
    kill_switch_key: str
    session_root: Path
    dry_run: bool
    max_session_seconds: int
    log_level: str


def _validate_host_or_host_port(addr: str, *, require_port: bool) -> None:
    """Validate format of host or host:port string."""
    if not isinstance(addr, str) or not addr.strip():
        raise ConfigError("Host address must be a non-empty string.")

    if ":" in addr:
        parts = addr.split(":")
        if len(parts) != 2:
            raise ConfigError(f"Invalid host:port format: '{addr}'.")
        host, port_str = parts
        if not host.strip():
            raise ConfigError(f"Invalid host in '{addr}'.")
        if not port_str.isdigit():
            raise ConfigError(f"Invalid port in '{addr}'.")
        port = int(port_str)
        if not (1 <= port <= 65535):
            raise ConfigError(f"Port out of range (1-65535) in '{addr}'.")
    else:
        if require_port:
            raise ConfigError(f"Expected 'host:port' format, got: '{addr}'.")
        if not addr.strip():
            raise ConfigError(f"Invalid host format: '{addr}'.")


def load_config(path: Path) -> Config:
    """Load TOML configuration from file and return a validated Config instance."""
    if not isinstance(path, Path):
        path = Path(path)

    if not path.exists() or not path.is_file():
        raise ConfigError(f"Config file does not exist: '{path}'.")

    try:
        with path.open("rb") as f:
            raw_data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Config file is not valid TOML: {e}.") from e
    except OSError as e:
        raise ConfigError(f"Error opening config file '{path}': {e}.") from e

    if not isinstance(raw_data, dict):
        raise ConfigError("TOML root must be a table.")

    keys = set(raw_data.keys())

    missing = REQUIRED_KEYS - keys
    if missing:
        raise ConfigError(f"Missing required config key(s): {sorted(missing)}.")

    unknown = keys - REQUIRED_KEYS
    if unknown:
        raise ConfigError(f"Unknown config key(s) present: {sorted(unknown)}.")

    # lab_mode validation
    lab_mode = raw_data["lab_mode"]
    if not isinstance(lab_mode, bool):
        raise ConfigError(f"'lab_mode' must be a boolean, got {type(lab_mode).__name__}.")

    # dry_run validation
    dry_run = raw_data["dry_run"]
    if not isinstance(dry_run, bool):
        raise ConfigError(f"'dry_run' must be a boolean, got {type(dry_run).__name__}.")

    # log_level validation
    log_level = raw_data["log_level"]
    if not isinstance(log_level, str) or log_level not in VALID_LOG_LEVELS:
        raise ConfigError(f"'log_level' must be one of {sorted(VALID_LOG_LEVELS)}, got {log_level!r}.")

    # max_session_seconds validation
    max_session_seconds = raw_data["max_session_seconds"]
    if (
        isinstance(max_session_seconds, bool)
        or not isinstance(max_session_seconds, int)
        or max_session_seconds <= 0
    ):
        raise ConfigError(
            f"'max_session_seconds' must be a positive integer, got {max_session_seconds!r}."
        )

    # session_root validation
    session_root_raw = raw_data["session_root"]
    if not isinstance(session_root_raw, str) or not session_root_raw.strip():
        raise ConfigError(f"'session_root' must be a non-empty string, got {session_root_raw!r}.")
    session_root = Path(session_root_raw)

    # kill_switch_key validation
    kill_switch_key = raw_data["kill_switch_key"]
    if not isinstance(kill_switch_key, str) or not kill_switch_key.strip():
        raise ConfigError(
            f"'kill_switch_key' must be a non-empty string, got {kill_switch_key!r}."
        )

    # isolation_sentinel validation
    isolation_sentinel = raw_data["isolation_sentinel"]
    if not isinstance(isolation_sentinel, str):
        raise ConfigError(
            f"'isolation_sentinel' must be a string, got {type(isolation_sentinel).__name__}."
        )
    _validate_host_or_host_port(isolation_sentinel, require_port=True)

    # server_allowlist validation
    server_allowlist_raw = raw_data["server_allowlist"]
    if not isinstance(server_allowlist_raw, list):
        raise ConfigError(
            f"'server_allowlist' must be a list of strings, got {type(server_allowlist_raw).__name__}."
        )

    allowlist_items: list[str] = []
    for item in server_allowlist_raw:
        if not isinstance(item, str):
            raise ConfigError(
                f"All entries in 'server_allowlist' must be strings, got {type(item).__name__}."
            )

        item_lower = item.lower()
        for forbidden in FORBIDDEN_DOMAINS:
            if forbidden in item_lower:
                raise ConfigError(
                    f"'server_allowlist' entry {item!r} contains forbidden domain {forbidden!r}."
                )

        _validate_host_or_host_port(item, require_port=False)
        allowlist_items.append(item)

    server_allowlist = tuple(allowlist_items)

    return Config(
        lab_mode=lab_mode,
        server_allowlist=server_allowlist,
        isolation_sentinel=isolation_sentinel,
        kill_switch_key=kill_switch_key,
        session_root=session_root,
        dry_run=dry_run,
        max_session_seconds=max_session_seconds,
        log_level=log_level,
    )
