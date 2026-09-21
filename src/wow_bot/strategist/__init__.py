"""Strategist layer: local LLM client, prompts, parsing, and orchestration."""

from wow_bot.strategist.prompts import (
    DynamicContext,  # DEPRECATED: see STRATEGIST_RECONCILIATION.md
    build_user_prompt,  # DEPRECATED: see STRATEGIST_RECONCILIATION.md
    load_system_prompt,
)

__all__ = [
    "DynamicContext",  # DEPRECATED: see STRATEGIST_RECONCILIATION.md
    "build_user_prompt",  # DEPRECATED: see STRATEGIST_RECONCILIATION.md
    "load_system_prompt",
]
