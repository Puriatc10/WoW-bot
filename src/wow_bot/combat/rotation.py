"""
Data-driven, deterministic rotation decision table for combat spells.
"""

from dataclasses import dataclass
from typing import Any, Protocol


class RotationError(Exception):
    """Raised when loading or validating a rotation configuration fails."""


class CombatStateLike(Protocol):
    """
    Opaque view of combat state consumed by the rotation table.
    Implementations MUST provide these attributes (or properties):
    """

    target_in_range: bool
    target_hp_percent: float  # 0.0 .. 100.0
    self_hp_percent: float  # 0.0 .. 100.0
    resource: float  # class-specific, >= 0.0
    resource_max: float  # > 0.0
    gcd_ready: bool

    def target_has_debuff(self, debuff_id: str, /) -> bool: ...

    def spell_cooldown_ready(self, spell_id: str, /) -> bool: ...


_ALLOWED_KINDS: set[str] = {
    "target_in_range",
    "target_hp_below",
    "target_hp_above",
    "self_hp_below",
    "self_hp_above",
    "resource_at_least",
    "resource_below",
    "spell_ready",
    "gcd_ready",
    "target_has_debuff",
    "target_lacks_debuff",
}

_NUMERIC_KINDS: set[str] = {
    "target_hp_below",
    "target_hp_above",
    "self_hp_below",
    "self_hp_above",
    "resource_at_least",
    "resource_below",
}

_PERCENT_KINDS: set[str] = {
    "target_hp_below",
    "target_hp_above",
    "self_hp_below",
    "self_hp_above",
}

_RESOURCE_KINDS: set[str] = {
    "resource_at_least",
    "resource_below",
}

_BOOL_KINDS: set[str] = {
    "target_in_range",
    "gcd_ready",
}

_STR_KINDS: set[str] = {
    "target_has_debuff",
    "target_lacks_debuff",
}


@dataclass(frozen=True)
class Condition:
    kind: str
    value: float | str | bool
    spell_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in _ALLOWED_KINDS:
            raise ValueError(f"Invalid condition kind: {self.kind!r}")

        if self.kind in _NUMERIC_KINDS:
            if isinstance(self.value, bool) or not isinstance(
                self.value, (int, float)
            ):
                raise ValueError(
                    f"Condition kind {self.kind!r} requires int or float value"
                )
            num_val = float(self.value)
            if self.kind in _PERCENT_KINDS and not (0.0 <= num_val <= 100.0):
                raise ValueError(
                    f"Percent condition {self.kind!r} requires value in [0.0, 100.0], got {self.value}"
                )
            if self.kind in _RESOURCE_KINDS and num_val < 0.0:
                raise ValueError(
                    f"Resource condition {self.kind!r} requires value >= 0.0, got {self.value}"
                )

        if self.kind in _BOOL_KINDS and not isinstance(self.value, bool):
            raise ValueError(
                f"Condition kind {self.kind!r} requires bool value"
            )

        if self.kind in _STR_KINDS and (
            not isinstance(self.value, str) or len(self.value) == 0
        ):
            raise ValueError(
                f"Condition kind {self.kind!r} requires non-empty str value"
            )

        if self.kind == "spell_ready":
            if not isinstance(self.value, bool):
                raise ValueError("Condition 'spell_ready' requires bool value")
            if not isinstance(self.spell_id, str) or len(self.spell_id) == 0:
                raise ValueError(
                    "Condition 'spell_ready' requires non-empty str spell_id"
                )
        elif self.spell_id is not None:
            raise ValueError(
                f"Condition kind {self.kind!r} must have spell_id=None"
            )


@dataclass(frozen=True)
class RotationRule:
    spell_id: str
    conditions: tuple[Condition, ...]
    priority: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.spell_id, str) or len(self.spell_id) == 0:
            raise ValueError("RotationRule spell_id must be a non-empty string")
        if not isinstance(self.conditions, tuple) or len(self.conditions) == 0:
            raise ValueError(
                "RotationRule conditions must be a non-empty tuple"
            )
        if not all(isinstance(c, Condition) for c in self.conditions):
            raise ValueError(
                "All elements in conditions must be Condition instances"
            )
        if (
            isinstance(self.priority, bool)
            or not isinstance(self.priority, int)
            or self.priority < 0
        ):
            raise ValueError("RotationRule priority must be an integer >= 0")


@dataclass(frozen=True)
class RotationConfig:
    rules: tuple[RotationRule, ...]
    default_spell_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.rules) is not tuple:
            raise ValueError("RotationConfig rules must be a tuple")
        if not all(isinstance(r, RotationRule) for r in self.rules):
            raise ValueError("All elements in rules must be RotationRule instances")
        if self.default_spell_id is not None and (
            not isinstance(self.default_spell_id, str)
            or len(self.default_spell_id) == 0
        ):
            raise ValueError(
                "RotationConfig default_spell_id must be a non-empty string or None"
            )

        seen_pairs: set[tuple[int, str]] = set()
        for rule in self.rules:
            pair = (rule.priority, rule.spell_id)
            if pair in seen_pairs:
                raise ValueError(
                    f"Duplicate (priority, spell_id) pair found: {pair}"
                )
            seen_pairs.add(pair)


def _eval_condition(cond: Condition, state: CombatStateLike) -> bool:
    kind = cond.kind
    if kind == "target_in_range":
        return bool(state.target_in_range == cond.value)
    elif kind == "target_hp_below":
        return bool(state.target_hp_percent < float(cond.value))
    elif kind == "target_hp_above":
        return bool(state.target_hp_percent > float(cond.value))
    elif kind == "self_hp_below":
        return bool(state.self_hp_percent < float(cond.value))
    elif kind == "self_hp_above":
        return bool(state.self_hp_percent > float(cond.value))
    elif kind == "resource_at_least":
        return bool(state.resource >= float(cond.value))
    elif kind == "resource_below":
        return bool(state.resource < float(cond.value))
    elif kind == "spell_ready":
        assert isinstance(cond.spell_id, str)
        return bool(state.spell_cooldown_ready(cond.spell_id) == cond.value)
    elif kind == "gcd_ready":
        return bool(state.gcd_ready == cond.value)
    elif kind == "target_has_debuff":
        assert isinstance(cond.value, str)
        return state.target_has_debuff(cond.value) is True
    elif kind == "target_lacks_debuff":
        assert isinstance(cond.value, str)
        return state.target_has_debuff(cond.value) is False
    else:
        raise ValueError(f"Unknown condition kind: {kind}")


class RotationTable:
    def __init__(self, config: RotationConfig) -> None:
        if not isinstance(config, RotationConfig):
            raise TypeError("config must be a RotationConfig instance")
        self._config = config
        # Pre-sort rules by (priority ASC, index in original tuple ASC)
        self._rules: tuple[RotationRule, ...] = tuple(
            rule
            for _, rule in sorted(
                enumerate(config.rules),
                key=lambda pair: (pair[1].priority, pair[0]),
            )
        )

    @property
    def rule_count(self) -> int:
        return len(self._rules)

    @property
    def rule_order(self) -> tuple[str, ...]:
        return tuple(rule.spell_id for rule in self._rules)

    def select(self, state: CombatStateLike) -> str | None:
        for rule in self._rules:
            if all(_eval_condition(cond, state) for cond in rule.conditions):
                return rule.spell_id
        return self._config.default_spell_id

    def explain(self, state: CombatStateLike) -> list[tuple[str, bool]]:
        result: list[tuple[str, bool]] = []
        for rule in self._rules:
            matched = all(
                _eval_condition(cond, state) for cond in rule.conditions
            )
            result.append((rule.spell_id, matched))
        return result


def load_rotation_from_dict(data: dict[str, Any]) -> RotationConfig:
    if not isinstance(data, dict):
        raise RotationError("Data must be a dict")

    allowed_top_keys = {"rules", "default_spell_id"}
    unknown_top_keys = set(data.keys()) - allowed_top_keys
    if unknown_top_keys:
        raise RotationError(f"Unknown top-level key(s): {unknown_top_keys}")

    if "rules" not in data:
        raise RotationError("Missing required key 'rules'")

    default_spell_id = data.get("default_spell_id")
    if default_spell_id is not None and not isinstance(default_spell_id, str):
        raise RotationError("default_spell_id must be a string or None")

    raw_rules = data["rules"]
    if not isinstance(raw_rules, list):
        raise RotationError("'rules' must be a list")

    parsed_rules: list[RotationRule] = []
    allowed_rule_keys = {"spell_id", "conditions", "priority"}

    for r_idx, raw_rule in enumerate(raw_rules):
        if not isinstance(raw_rule, dict):
            raise RotationError(f"Rule at index {r_idx} must be a dict")

        unknown_rule_keys = set(raw_rule.keys()) - allowed_rule_keys
        if unknown_rule_keys:
            raise RotationError(
                f"Unknown key(s) in rule at index {r_idx}: {unknown_rule_keys}"
            )

        if "spell_id" not in raw_rule or "conditions" not in raw_rule:
            raise RotationError(
                f"Rule at index {r_idx} missing required key ('spell_id' or 'conditions')"
            )

        spell_id = raw_rule["spell_id"]
        if not isinstance(spell_id, str):
            raise RotationError(f"Rule spell_id at index {r_idx} must be a str")

        priority = raw_rule.get("priority", 0)
        if isinstance(priority, bool) or not isinstance(priority, int):
            raise RotationError(f"Rule priority at index {r_idx} must be an int")

        raw_conditions = raw_rule["conditions"]
        if not isinstance(raw_conditions, list):
            raise RotationError(f"Rule conditions at index {r_idx} must be a list")

        parsed_conditions: list[Condition] = []
        allowed_cond_keys = {"kind", "value", "spell_id"}

        for c_idx, raw_cond in enumerate(raw_conditions):
            if not isinstance(raw_cond, dict):
                raise RotationError(
                    f"Condition at rule {r_idx}, index {c_idx} must be a dict"
                )

            unknown_cond_keys = set(raw_cond.keys()) - allowed_cond_keys
            if unknown_cond_keys:
                raise RotationError(
                    f"Unknown key(s) in condition at rule {r_idx}, index {c_idx}: {unknown_cond_keys}"
                )

            if "kind" not in raw_cond or "value" not in raw_cond:
                raise RotationError(
                    f"Condition at rule {r_idx}, index {c_idx} missing required key ('kind' or 'value')"
                )

            kind = raw_cond["kind"]
            if not isinstance(kind, str):
                raise RotationError(
                    f"Condition kind at rule {r_idx}, index {c_idx} must be a str"
                )

            value = raw_cond["value"]
            cond_spell_id = raw_cond.get("spell_id")
            if cond_spell_id is not None and not isinstance(cond_spell_id, str):
                raise RotationError(
                    f"Condition spell_id at rule {r_idx}, index {c_idx} must be a str or None"
                )

            try:
                cond = Condition(
                    kind=kind, value=value, spell_id=cond_spell_id
                )
            except ValueError as e:
                raise RotationError(
                    f"Invalid condition at rule {r_idx}, index {c_idx}: {e}"
                ) from e

            parsed_conditions.append(cond)

        try:
            rule = RotationRule(
                spell_id=spell_id,
                conditions=tuple(parsed_conditions),
                priority=priority,
            )
        except ValueError as e:
            raise RotationError(f"Invalid rule at index {r_idx}: {e}") from e

        parsed_rules.append(rule)

    try:
        config = RotationConfig(
            rules=tuple(parsed_rules), default_spell_id=default_spell_id
        )
    except ValueError as e:
        raise RotationError(f"Invalid rotation config: {e}") from e

    return config
