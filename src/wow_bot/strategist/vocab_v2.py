"""Vocabulary guard and shape validator for WoW-bot strategist v2 (Task 9.3).

Provides pure, deterministic goal vocabulary and target shape validation on parsed
LLM strategy outputs. Enforces GoalRule limits, emits session events on decision outcomes,
and tracks rejection counters for telemetry. Isolated from time reads, I/O, threads, and LLM modules.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from wow_bot.strategist.prompts_v2 import ALLOWED_GOALS

if TYPE_CHECKING:
    from wow_bot.session import Session


class VocabError(Exception):
    """Raised when an error or invalid configuration occurs in vocabulary guard operations."""


class TargetKind(str, Enum):
    """Categorizes the expected target shape for a given strategy goal."""

    NONE = "none"
    ENTITY_ID = "entity_id"
    WAYPOINT_ID = "waypoint_id"
    FREE_TEXT = "free_text"


class RejectionReason(str, Enum):
    """Reasons for strategy candidate rejection by VocabularyGuard."""

    UNKNOWN_GOAL = "unknown_goal"
    MISSING_TARGET = "missing_target"
    UNEXPECTED_TARGET = "unexpected_target"
    MALFORMED_ENTITY_ID = "malformed_entity_id"
    MALFORMED_WAYPOINT_ID = "malformed_waypoint_id"
    MALFORMED_FREE_TEXT = "malformed_free_text"
    RATIONALE_TOO_LONG = "rationale_too_long"
    RATIONALE_EMPTY = "rationale_empty"
    MALFORMED_GOAL_TYPE = "malformed_goal_type"


@dataclass(frozen=True)
class GoalRule:
    """Configurable shape and length rule for a single goal target."""

    target_kind: TargetKind
    min_target_length: int = 1
    max_target_length: int = 128
    allowed_target_prefixes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.target_kind, TargetKind):
            raise ValueError(f"target_kind must be a TargetKind, got {self.target_kind!r}")  # noqa: TRY004

        if (
            isinstance(self.min_target_length, bool)
            or not isinstance(self.min_target_length, int)
        ):
            raise ValueError("min_target_length must be an int")  # noqa: TRY004

        if (
            isinstance(self.max_target_length, bool)
            or not isinstance(self.max_target_length, int)
        ):
            raise ValueError("max_target_length must be an int")  # noqa: TRY004

        if self.target_kind == TargetKind.NONE:
            if self.min_target_length != 0:
                raise ValueError("min_target_length must equal 0 for TargetKind.NONE")
            if self.max_target_length != 0:
                raise ValueError("max_target_length must equal 0 for TargetKind.NONE")
            if len(self.allowed_target_prefixes) > 0:
                raise ValueError("allowed_target_prefixes must be empty for TargetKind.NONE")
        else:
            if self.min_target_length < 1:
                raise ValueError(f"min_target_length must be >= 1, got {self.min_target_length}")
            if self.max_target_length < self.min_target_length:
                raise ValueError(
                    f"max_target_length ({self.max_target_length}) must be >= min_target_length ({self.min_target_length})"
                )

        if not isinstance(self.allowed_target_prefixes, tuple):
            raise ValueError("allowed_target_prefixes must be a tuple")  # noqa: TRY004

        for p in self.allowed_target_prefixes:
            if not isinstance(p, str) or len(p) == 0:
                raise ValueError("allowed_target_prefixes must contain non-empty strings")


DEFAULT_GOAL_RULES: Mapping[str, GoalRule] = MappingProxyType({
    "farm_herbs": GoalRule(target_kind=TargetKind.FREE_TEXT),
    "grind_humans": GoalRule(target_kind=TargetKind.FREE_TEXT),
    "explore": GoalRule(target_kind=TargetKind.NONE, min_target_length=0, max_target_length=0),
    "flee": GoalRule(target_kind=TargetKind.NONE, min_target_length=0, max_target_length=0),
    "sell_vendor": GoalRule(target_kind=TargetKind.ENTITY_ID),
    "repair": GoalRule(target_kind=TargetKind.ENTITY_ID),
    "travel_to": GoalRule(target_kind=TargetKind.WAYPOINT_ID),
})


@dataclass(frozen=True)
class VocabConfig:
    """Configuration mapping allowed strategy goals to target rules and rationale bounds."""

    max_rationale_length: int = 200
    rules: Mapping[str, GoalRule] = field(
        default_factory=lambda: DEFAULT_GOAL_RULES
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_rationale_length, bool)
            or not isinstance(self.max_rationale_length, int)
        ):
            raise ValueError("max_rationale_length must be an int")  # noqa: TRY004
        if self.max_rationale_length < 1:
            raise ValueError(f"max_rationale_length must be >= 1, got {self.max_rationale_length}")

        if not isinstance(self.rules, Mapping):
            raise ValueError("rules must be a Mapping")  # noqa: TRY004

        missing = set(ALLOWED_GOALS) - set(self.rules.keys())
        if missing:
            raise ValueError(f"rules missing goals from ALLOWED_GOALS: {sorted(missing)}")

        unknown = set(self.rules.keys()) - set(ALLOWED_GOALS)
        if unknown:
            raise ValueError(f"rules contain unknown goals not in ALLOWED_GOALS: {sorted(unknown)}")

        if isinstance(self.rules, dict):
            object.__setattr__(self, "rules", MappingProxyType(dict(self.rules)))


@dataclass(frozen=True)
class StrategyCandidate:
    """Raw candidate strategy parsed from LLM JSON output."""

    goal: Any
    target: Any
    rationale: Any


@dataclass(frozen=True)
class ValidatedStrategy:
    """Validated strategy satisfying all vocabulary and shape constraints."""

    goal: str
    target: str | None
    rationale: str

    def __post_init__(self) -> None:
        if self.goal not in ALLOWED_GOALS:
            raise ValueError(f"goal '{self.goal}' is not in ALLOWED_GOALS")


@dataclass(frozen=True)
class VocabDecision:
    """Result of VocabularyGuard validation on a strategy candidate."""

    accepted: bool
    strategy: ValidatedStrategy | None
    reason: RejectionReason | None
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.detail, str) or len(self.detail) == 0:
            raise ValueError("detail must be a non-empty string")

        if self.accepted:
            if self.strategy is None:
                raise ValueError("accepted is True implies strategy is not None")
            if self.reason is not None:
                raise ValueError("accepted is True implies reason is None")
        else:
            if self.strategy is not None:
                raise ValueError("accepted is False implies strategy is None")
            if self.reason is None:
                raise ValueError("accepted is False implies reason is not None")


class VocabularyGuard:
    """Pure, deterministic vocabulary guard and shape validator for LLM strategy responses."""

    def __init__(
        self,
        *,
        config: VocabConfig | None = None,
        session: Session | None = None,
    ) -> None:
        self._config: VocabConfig = config if config is not None else VocabConfig()
        self._session: Session | None = session
        self._accepted_count: int = 0
        self._rejection_counts: dict[RejectionReason, int] = {}

    def allowed_goals(self) -> tuple[str, ...]:
        """Return ALLOWED_GOALS tuple from prompts_v2."""
        return ALLOWED_GOALS

    def rejection_counts(self) -> Mapping[RejectionReason, int]:
        """Return a MappingProxyType snapshot of counts per rejection reason."""
        return MappingProxyType(dict(self._rejection_counts))

    def reset_counts(self) -> None:
        """Clear all internal counters."""
        self._accepted_count = 0
        self._rejection_counts.clear()

    def validate(
        self,
        candidate: StrategyCandidate,
    ) -> VocabDecision:
        """Validate candidate goal, target, and rationale against VocabConfig rules in strict evaluation order.

        Semantics evaluated in strict order:
          1. If not isinstance(candidate.goal, str): reject MALFORMED_GOAL_TYPE.
          2. If candidate.goal not in config.rules: reject UNKNOWN_GOAL.
          3. If candidate.rationale is not a string or is empty after .strip(): reject RATIONALE_EMPTY.
          4. If len(candidate.rationale) > config.max_rationale_length: reject RATIONALE_TOO_LONG.
          5. rule = config.rules[candidate.goal].
          6. If rule.target_kind == TargetKind.NONE:
               if candidate.target is not None: reject UNEXPECTED_TARGET.
          7. If rule.target_kind != TargetKind.NONE:
               if candidate.target is None: reject MISSING_TARGET.
               if not isinstance(candidate.target, str): reject kind-specific MALFORMED_* reason.
               length check: reject kind-specific MALFORMED_* reason if outside min/max bounds.
               prefix check: reject kind-specific MALFORMED_* reason if no prefix matches.
               kind-specific formatting checks (WAYPOINT_ID digits, ENTITY_ID whitespace/slash, FREE_TEXT strip).
          8. Success: build ValidatedStrategy, increment accepted counter, emit vocab_accepted, return accepted VocabDecision.
          9. Rejection: increment per-reason counter, emit vocab_rejected, return rejected VocabDecision.
        """
        # 1. If not isinstance(candidate.goal, str): reject MALFORMED_GOAL_TYPE.
        if not isinstance(candidate.goal, str):
            return self._reject(
                candidate=candidate,
                reason=RejectionReason.MALFORMED_GOAL_TYPE,
                detail=f"candidate.goal must be a string, got {type(candidate.goal).__name__}",
            )

        # 2. If candidate.goal not in config.rules: reject UNKNOWN_GOAL.
        if candidate.goal not in self._config.rules:
            return self._reject(
                candidate=candidate,
                reason=RejectionReason.UNKNOWN_GOAL,
                detail=f"goal '{candidate.goal}' is not in allowed vocabulary rules",
            )

        # 3. If candidate.rationale is not a string or is empty after .strip(): reject RATIONALE_EMPTY.
        if not isinstance(candidate.rationale, str) or len(candidate.rationale.strip()) == 0:
            return self._reject(
                candidate=candidate,
                reason=RejectionReason.RATIONALE_EMPTY,
                detail="rationale must be a non-empty string",
            )

        # 4. If len(candidate.rationale) > config.max_rationale_length: reject RATIONALE_TOO_LONG.
        if len(candidate.rationale) > self._config.max_rationale_length:
            return self._reject(
                candidate=candidate,
                reason=RejectionReason.RATIONALE_TOO_LONG,
                detail=f"rationale length {len(candidate.rationale)} exceeds max {self._config.max_rationale_length}",
            )

        # 5. rule = config.rules[candidate.goal]
        rule = self._config.rules[candidate.goal]

        # Map target_kind to kind-specific rejection reason for step 7 failures
        if rule.target_kind == TargetKind.ENTITY_ID:
            kind_malformed_reason = RejectionReason.MALFORMED_ENTITY_ID
        elif rule.target_kind == TargetKind.WAYPOINT_ID:
            kind_malformed_reason = RejectionReason.MALFORMED_WAYPOINT_ID
        elif rule.target_kind == TargetKind.FREE_TEXT:
            kind_malformed_reason = RejectionReason.MALFORMED_FREE_TEXT
        else:
            kind_malformed_reason = None

        # 6. If rule.target_kind == TargetKind.NONE:
        if rule.target_kind == TargetKind.NONE:
            if candidate.target is not None:
                return self._reject(
                    candidate=candidate,
                    reason=RejectionReason.UNEXPECTED_TARGET,
                    detail=f"goal '{candidate.goal}' does not allow a target, got '{candidate.target}'",
                )

        # 7. If rule.target_kind != TargetKind.NONE:
        else:
            assert kind_malformed_reason is not None
            if candidate.target is None:
                return self._reject(
                    candidate=candidate,
                    reason=RejectionReason.MISSING_TARGET,
                    detail=f"goal '{candidate.goal}' requires a target of kind '{rule.target_kind.value}'",
                )

            if not isinstance(candidate.target, str):
                return self._reject(
                    candidate=candidate,
                    reason=kind_malformed_reason,
                    detail=f"target must be a string, got {type(candidate.target).__name__}",
                )

            target_str = candidate.target

            if len(target_str) < rule.min_target_length or len(target_str) > rule.max_target_length:
                return self._reject(
                    candidate=candidate,
                    reason=kind_malformed_reason,
                    detail=f"target length {len(target_str)} is outside allowed range [{rule.min_target_length}, {rule.max_target_length}]",
                )

            if rule.allowed_target_prefixes and not any(
                target_str.startswith(p) for p in rule.allowed_target_prefixes
            ):
                return self._reject(
                    candidate=candidate,
                    reason=kind_malformed_reason,
                    detail=f"target '{target_str}' does not start with any allowed prefix {rule.allowed_target_prefixes}",
                )

            if rule.target_kind == TargetKind.WAYPOINT_ID:
                if not target_str.isdigit():
                    return self._reject(
                        candidate=candidate,
                        reason=RejectionReason.MALFORMED_WAYPOINT_ID,
                        detail=f"waypoint_id target '{target_str}' must consist only of digits",
                    )
            elif rule.target_kind == TargetKind.ENTITY_ID:
                if any(c.isspace() for c in target_str) or "/" in target_str:
                    return self._reject(
                        candidate=candidate,
                        reason=RejectionReason.MALFORMED_ENTITY_ID,
                        detail=f"entity_id target '{target_str}' contains whitespace or '/'",
                    )
            elif rule.target_kind == TargetKind.FREE_TEXT and len(target_str.strip()) == 0:
                return self._reject(
                    candidate=candidate,
                    reason=RejectionReason.MALFORMED_FREE_TEXT,
                    detail="free_text target cannot be empty or whitespace-only",
                )

        # 8. Success
        validated = ValidatedStrategy(
            goal=candidate.goal,
            target=candidate.target,
            rationale=candidate.rationale,
        )
        self._accepted_count += 1
        if self._session is not None:
            self._session.write_event({
                "event": "vocab_accepted",
                "goal": candidate.goal,
                "has_target": candidate.target is not None,
            })
        return VocabDecision(
            accepted=True,
            strategy=validated,
            reason=None,
            detail="ok",
        )

    def _reject(
        self,
        candidate: StrategyCandidate,
        reason: RejectionReason,
        detail: str,
    ) -> VocabDecision:
        self._rejection_counts[reason] = self._rejection_counts.get(reason, 0) + 1
        goal_val = candidate.goal if isinstance(candidate.goal, str) else str(candidate.goal)
        if self._session is not None:
            self._session.write_event({
                "event": "vocab_rejected",
                "reason": reason.value,
                "goal": goal_val,
                "detail": detail,
            })
        return VocabDecision(
            accepted=False,
            strategy=None,
            reason=reason,
            detail=detail,
        )


__all__ = [
    "DEFAULT_GOAL_RULES",
    "GoalRule",
    "RejectionReason",
    "StrategyCandidate",
    "TargetKind",
    "ValidatedStrategy",
    "VocabConfig",
    "VocabDecision",
    "VocabError",
    "VocabularyGuard",
]
