"""Execution mode gate for WoW-bot system startup."""

from dataclasses import dataclass

from wow_bot.config import Config
from wow_bot.logging_setup import setup_logging
from wow_bot.safety import SafetyLayer
from wow_bot.session import Session


class ModeError(Exception):
    """Raised when execution mode configuration or validation fails."""


@dataclass(frozen=True)
class ModeContext:
    """Immutable context containing system state for the active execution mode."""

    mode: str
    config: Config
    session: Session
    safety: SafetyLayer
    dry_run: bool


def enter_mode(config: Config, session: Session) -> ModeContext:
    """Validate execution mode prerequisites and initialize safety and logging.

    Args:
        config: Loaded and validated system configuration.
        session: Active execution session.

    Returns:
        ModeContext representing either MOCK or LAB execution context.

    Raises:
        ModeError: If mode-specific requirements (such as dry_run flag or allowlist) are violated.
        IsolationViolation: In LAB mode, if network isolation sentinel is reachable.
        SafetyError: If safety layer arming fails.
    """
    if config.lab_mode:
        if config.dry_run:
            raise ModeError("LAB_MODE requires dry_run=False.")

        safety = SafetyLayer(config, session)
        safety.check_isolation()

        for entry in config.server_allowlist:
            if not isinstance(entry, str) or not entry.strip():
                raise ModeError("All server allowlist entries must be non-empty strings.")

        safety.arm()
        setup_logging(session, config.log_level)
        session.write_event({
            "event": "mode_entered",
            "mode": "LAB",
            "allowlist": list(config.server_allowlist),
        })
        return ModeContext(
            mode="LAB",
            config=config,
            session=session,
            safety=safety,
            dry_run=False,
        )
    else:
        if not config.dry_run:
            raise ModeError("MOCK_MODE requires dry_run=True.")

        safety = SafetyLayer(config, session)
        safety.arm()
        setup_logging(session, config.log_level)
        session.write_event({
            "event": "mode_entered",
            "mode": "MOCK",
        })
        return ModeContext(
            mode="MOCK",
            config=config,
            session=session,
            safety=safety,
            dry_run=True,
        )
