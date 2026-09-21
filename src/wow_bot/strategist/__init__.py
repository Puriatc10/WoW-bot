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
from wow_bot.strategist.vocab_v2 import (
    DEFAULT_GOAL_RULES,
    GoalRule,
    RejectionReason,
    StrategyCandidate,
    TargetKind,
    ValidatedStrategy,
    VocabConfig,
    VocabDecision,
    VocabError,
    VocabularyGuard,
)

__all__ = [
    "ALLOWED_GOALS",
    "DEFAULT_GOAL_RULES",
    "CooldownCheck",
    "CooldownConfig",
    "CooldownDecision",
    "CooldownError",
    "CooldownGate",
    "DynamicContext",  # DEPRECATED: see STRATEGIST_RECONCILIATION.md
    "GameStateView",
    "GoalRule",
    "MetaStateLike",
    "PromptBundle",
    "PromptConfig",
    "PromptError",
    "RejectionReason",
    "StrategyCandidate",
    "TargetKind",
    "ValidatedStrategy",
    "VocabConfig",
    "VocabDecision",
    "VocabError",
    "VocabularyGuard",
    "build_prompt",
    "build_user_prompt",  # DEPRECATED: see STRATEGIST_RECONCILIATION.md
    "load_system_prompt",
    "render_schema",
]
