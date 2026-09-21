"""Strategist layer: local LLM client, prompts, parsing, and orchestration."""

from wow_bot.strategist.cooldown_v2 import (
    CooldownCheck,
    CooldownConfig,
    CooldownDecision,
    CooldownError,
    CooldownGate,
)
from wow_bot.strategist.prompts import (
    DynamicContext,  # DEPRECATED: see STRATEGIST_RECONCILIATION.md
    build_user_prompt,  # DEPRECATED: see STRATEGIST_RECONCILIATION.md
    load_system_prompt,
)
from wow_bot.strategist.prompts_v2 import (
    ALLOWED_GOALS,
    GameStateView,
    MetaStateLike,
    PromptBundle,
    PromptConfig,
    PromptError,
    build_prompt,
    render_schema,
)

__all__ = [
    "ALLOWED_GOALS",
    "CooldownCheck",
    "CooldownConfig",
    "CooldownDecision",
    "CooldownError",
    "CooldownGate",
    "DynamicContext",  # DEPRECATED: see STRATEGIST_RECONCILIATION.md
    "GameStateView",
    "MetaStateLike",
    "PromptBundle",
    "PromptConfig",
    "PromptError",
    "build_prompt",
    "build_user_prompt",  # DEPRECATED: see STRATEGIST_RECONCILIATION.md
    "load_system_prompt",
    "render_schema",
]
