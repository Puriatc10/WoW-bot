import ast
import json
import math
import random
from statistics import median

import pytest

from wow_bot.humanize.imperfections import (
    ImperfectionConfig,
    ImperfectionError,
    ImperfectionLog,
    ImperfectionSampler,
    MissClickDecision,
    PauseDecision,
)


def test_config_pause_probability_out_of_bounds() -> None:
    with pytest.raises(ValueError, match="pause_probability"):
        ImperfectionConfig(pause_probability=-0.1)
    with pytest.raises(ValueError, match="pause_probability"):
        ImperfectionConfig(pause_probability=1.1)


def test_config_pause_duration_sigma_non_positive() -> None:
    with pytest.raises(ValueError, match="pause_duration_sigma"):
        ImperfectionConfig(pause_duration_sigma=0.0)
    with pytest.raises(ValueError, match="pause_duration_sigma"):
        ImperfectionConfig(pause_duration_sigma=-0.5)


def test_config_invalid_clip_range() -> None:
    with pytest.raises(ValueError, match="pause_duration_clip_low"):
        ImperfectionConfig(pause_duration_clip_low=0.0)
    with pytest.raises(ValueError, match="pause_duration_clip_high"):
        ImperfectionConfig(
            pause_duration_clip_low=2.0, pause_duration_clip_high=1.0
        )
    with pytest.raises(ValueError, match="pause_duration_clip_high"):
        ImperfectionConfig(
            pause_duration_clip_low=2.0, pause_duration_clip_high=2.0
        )


def test_config_miss_click_probability_out_of_bounds() -> None:
    with pytest.raises(ValueError, match="miss_click_probability"):
        ImperfectionConfig(miss_click_probability=-0.01)
    with pytest.raises(ValueError, match="miss_click_probability"):
        ImperfectionConfig(miss_click_probability=1.01)


def test_config_non_positive_miss_click_max_offset_units() -> None:
    with pytest.raises(ValueError, match="miss_click_max_offset_units"):
        ImperfectionConfig(miss_click_max_offset_units=0.0)
    with pytest.raises(ValueError, match="miss_click_max_offset_units"):
        ImperfectionConfig(miss_click_max_offset_units=-1.0)


def test_config_miss_click_sigma_fraction_out_of_bounds() -> None:
    with pytest.raises(ValueError, match="miss_click_sigma_fraction"):
        ImperfectionConfig(miss_click_sigma_fraction=0.0)
    with pytest.raises(ValueError, match="miss_click_sigma_fraction"):
        ImperfectionConfig(miss_click_sigma_fraction=-0.1)
    with pytest.raises(ValueError, match="miss_click_sigma_fraction"):
        ImperfectionConfig(miss_click_sigma_fraction=1.1)


def test_config_max_pause_redraws_invalid() -> None:
    with pytest.raises(ValueError, match="max_pause_redraws"):
        ImperfectionConfig(max_pause_redraws=0)
    with pytest.raises(ValueError, match="max_pause_redraws"):
        ImperfectionConfig(max_pause_redraws=-5)


def test_config_non_finite_float() -> None:
    with pytest.raises(ValueError, match="pause_probability"):
        ImperfectionConfig(pause_probability=float("nan"))
    with pytest.raises(ValueError, match="pause_duration_mu"):
        ImperfectionConfig(pause_duration_mu=float("inf"))
    with pytest.raises(ValueError, match="miss_click_max_offset_units"):
        ImperfectionConfig(miss_click_max_offset_units=float("-inf"))


def test_pause_decision_invariants() -> None:
    # occurred=False with non-zero duration
    with pytest.raises(ValueError, match="occurred=False implies duration_s == 0.0"):
        PauseDecision(occurred=False, duration_s=0.5, reason="test")

    # occurred=True with zero duration
    with pytest.raises(ValueError, match="occurred=True implies duration_s > 0.0"):
        PauseDecision(occurred=True, duration_s=0.0, reason="test")

    # occurred=True with duration outside clip range
    with pytest.raises(ValueError, match="outside range"):
        PauseDecision(
            occurred=True,
            duration_s=0.05,
            reason="test",
            clip_low=0.1,
            clip_high=3.0,
        )
    with pytest.raises(ValueError, match="outside range"):
        PauseDecision(
            occurred=True,
            duration_s=3.5,
            reason="test",
            clip_low=0.1,
            clip_high=3.0,
        )

    # empty reason
    with pytest.raises(ValueError, match="non-empty string"):
        PauseDecision(occurred=False, duration_s=0.0, reason="")


def test_miss_click_decision_invariants() -> None:
    # occurred=False with non-zero offsets/magnitude
    with pytest.raises(ValueError, match="occurred=False implies"):
        MissClickDecision(
            occurred=False, offset_x=1.0, offset_y=0.0, magnitude=1.0, reason="test"
        )

    # occurred=True with zero magnitude
    with pytest.raises(ValueError, match="occurred=True implies magnitude > 0.0"):
        MissClickDecision(
            occurred=True, offset_x=0.0, offset_y=0.0, magnitude=0.0, reason="test"
        )

    # occurred=True with magnitude beyond max
    with pytest.raises(ValueError, match="exceeds max_offset_units"):
        MissClickDecision(
            occurred=True,
            offset_x=10.0,
            offset_y=0.0,
            magnitude=10.0,
            reason="test",
            max_offset_units=8.0,
        )

    # magnitude inconsistent with offsets
    with pytest.raises(ValueError, match="inconsistent with offsets"):
        MissClickDecision(
            occurred=True,
            offset_x=3.0,
            offset_y=4.0,
            magnitude=6.0,  # sqrt(3^2 + 4^2) = 5.0
            reason="test",
            max_offset_units=8.0,
        )

    # empty reason
    with pytest.raises(ValueError, match="non-empty string"):
        MissClickDecision(
            occurred=False, offset_x=0.0, offset_y=0.0, magnitude=0.0, reason=""
        )


def test_maybe_pause_zero_probability() -> None:
    cfg = ImperfectionConfig(pause_probability=0.0)
    sampler = ImperfectionSampler(cfg)
    rng = random.Random(42)

    for _ in range(100):
        decision = sampler.maybe_pause(rng)
        assert not decision.occurred
        assert decision.duration_s == 0.0
        assert decision.reason == "not_selected"


def test_maybe_pause_one_probability() -> None:
    cfg = ImperfectionConfig(
        pause_probability=1.0,
        pause_duration_clip_low=0.1,
        pause_duration_clip_high=3.0,
    )
    sampler = ImperfectionSampler(cfg)
    rng = random.Random(42)

    for _ in range(100):
        decision = sampler.maybe_pause(rng)
        assert decision.occurred
        assert 0.1 <= decision.duration_s <= 3.0
        assert decision.reason == "pause"


def test_maybe_pause_duration_bounds_across_samples() -> None:
    cfg = ImperfectionConfig(
        pause_probability=1.0,
        pause_duration_clip_low=0.1,
        pause_duration_clip_high=3.0,
    )
    sampler = ImperfectionSampler(cfg)
    rng = random.Random(12345)

    for _ in range(10_000):
        decision = sampler.maybe_pause(rng)
        assert decision.occurred
        assert 0.1 <= decision.duration_s <= 3.0


def test_maybe_pause_empirical_rate() -> None:
    cfg = ImperfectionConfig(pause_probability=0.05)
    sampler = ImperfectionSampler(cfg)
    rng = random.Random(999)

    occurred_count = 0
    total = 100_000
    for _ in range(total):
        if sampler.maybe_pause(rng).occurred:
            occurred_count += 1

    rate = occurred_count / total
    assert 0.04 <= rate <= 0.06


def test_maybe_pause_deterministic() -> None:
    cfg = ImperfectionConfig(pause_probability=0.05)
    sampler = ImperfectionSampler(cfg)

    rng1 = random.Random(0)
    rng2 = random.Random(0)

    seq1 = [sampler.maybe_pause(rng1) for _ in range(1000)]
    seq2 = [sampler.maybe_pause(rng2) for _ in range(1000)]

    assert seq1 == seq2


def test_maybe_pause_duration_distribution_median() -> None:
    mu = -0.7
    cfg = ImperfectionConfig(
        pause_probability=1.0,
        pause_duration_mu=mu,
        pause_duration_sigma=0.5,
        pause_duration_clip_low=0.1,
        pause_duration_clip_high=3.0,
    )
    sampler = ImperfectionSampler(cfg)
    rng = random.Random(777)

    durations = [sampler.maybe_pause(rng).duration_s for _ in range(20_000)]
    emp_median = median(durations)
    expected_median = math.exp(mu)  # exp(-0.7) ~= 0.49658

    # Empirical median within 30% of expected_median
    assert abs(emp_median - expected_median) / expected_median <= 0.30


def test_maybe_miss_click_zero_probability() -> None:
    cfg = ImperfectionConfig(miss_click_probability=0.0)
    sampler = ImperfectionSampler(cfg)
    rng = random.Random(42)

    for _ in range(100):
        decision = sampler.maybe_miss_click(rng)
        assert not decision.occurred
        assert decision.offset_x == 0.0
        assert decision.offset_y == 0.0
        assert decision.magnitude == 0.0
        assert decision.reason == "not_selected"


def test_maybe_miss_click_one_probability() -> None:
    cfg = ImperfectionConfig(
        miss_click_probability=1.0,
        miss_click_max_offset_units=8.0,
    )
    sampler = ImperfectionSampler(cfg)
    rng = random.Random(42)

    for _ in range(100):
        decision = sampler.maybe_miss_click(rng)
        assert decision.occurred
        assert 0.0 < decision.magnitude <= 8.0
        assert decision.reason == "miss_click"


def test_maybe_miss_click_magnitude_bounds_across_samples() -> None:
    max_offset = 8.0
    cfg = ImperfectionConfig(
        miss_click_probability=1.0,
        miss_click_max_offset_units=max_offset,
    )
    sampler = ImperfectionSampler(cfg)
    rng = random.Random(54321)

    for _ in range(10_000):
        decision = sampler.maybe_miss_click(rng)
        assert decision.occurred
        assert 0.0 < decision.magnitude <= max_offset


def test_maybe_miss_click_offset_consistency() -> None:
    cfg = ImperfectionConfig(
        miss_click_probability=1.0,
        miss_click_max_offset_units=8.0,
    )
    sampler = ImperfectionSampler(cfg)
    rng = random.Random(101)

    for _ in range(1000):
        decision = sampler.maybe_miss_click(rng)
        calc_mag = math.sqrt(decision.offset_x**2 + decision.offset_y**2)
        assert abs(calc_mag - decision.magnitude) <= 1e-6


def test_maybe_miss_click_empirical_rate() -> None:
    cfg = ImperfectionConfig(miss_click_probability=0.02)
    sampler = ImperfectionSampler(cfg)
    rng = random.Random(888)

    occurred_count = 0
    total = 100_000
    for _ in range(total):
        if sampler.maybe_miss_click(rng).occurred:
            occurred_count += 1

    rate = occurred_count / total
    assert 0.015 <= rate <= 0.025


def test_maybe_miss_click_deterministic() -> None:
    cfg = ImperfectionConfig(miss_click_probability=0.02)
    sampler = ImperfectionSampler(cfg)

    rng1 = random.Random(0)
    rng2 = random.Random(0)

    seq1 = [sampler.maybe_miss_click(rng1) for _ in range(1000)]
    seq2 = [sampler.maybe_miss_click(rng2) for _ in range(1000)]

    assert seq1 == seq2


def test_maybe_miss_click_angle_uniformity() -> None:
    cfg = ImperfectionConfig(
        miss_click_probability=1.0,
        miss_click_max_offset_units=8.0,
    )
    sampler = ImperfectionSampler(cfg)
    rng = random.Random(321)

    positive_x_count = 0
    total = 10_000
    for _ in range(total):
        decision = sampler.maybe_miss_click(rng)
        if decision.offset_x > 0:
            positive_x_count += 1

    fraction = positive_x_count / total
    assert 0.45 <= fraction <= 0.55


def test_describe_config() -> None:
    cfg = ImperfectionConfig()
    sampler = ImperfectionSampler(cfg)

    desc = sampler.describe_config()

    expected_fields = {
        "pause_probability",
        "pause_duration_mu",
        "pause_duration_sigma",
        "pause_duration_clip_low",
        "pause_duration_clip_high",
        "miss_click_probability",
        "miss_click_max_offset_units",
        "miss_click_sigma_fraction",
        "max_pause_redraws",
    }
    assert set(desc.keys()) == expected_fields
    assert json.dumps(desc)  # JSON-serializable test

    # Ensure config immutability/non-mutation
    assert cfg.pause_probability == 0.05
    assert cfg.miss_click_max_offset_units == 8.0


def test_imperfection_log_invalid_max_entries() -> None:
    with pytest.raises(ImperfectionError, match="max_entries"):
        ImperfectionLog(max_entries=0)
    with pytest.raises(ImperfectionError, match="max_entries"):
        ImperfectionLog(max_entries=-10)


def test_imperfection_log_record_occurring_only() -> None:
    log = ImperfectionLog(max_entries=10)

    # Non-occurring pause & miss click
    log.record_pause(PauseDecision(occurred=False, duration_s=0.0, reason="not_selected"))
    log.record_miss_click(
        MissClickDecision(
            occurred=False, offset_x=0.0, offset_y=0.0, magnitude=0.0, reason="not_selected"
        )
    )

    assert log.size == 0
    assert len(log.entries()) == 0

    # Occurring pause
    p_dec = PauseDecision(occurred=True, duration_s=0.5, reason="pause")
    log.record_pause(p_dec)
    assert log.size == 1

    # Occurring miss click
    m_dec = MissClickDecision(
        occurred=True,
        offset_x=3.0,
        offset_y=4.0,
        magnitude=5.0,
        reason="miss_click",
    )
    log.record_miss_click(m_dec)
    assert log.size == 2

    entries = log.entries()
    assert entries[0] == {"kind": "pause", "duration_s": 0.5}
    assert entries[1] == {
        "kind": "miss_click",
        "offset_x": 3.0,
        "offset_y": 4.0,
        "magnitude": 5.0,
    }


def test_imperfection_log_entries_copy_safety() -> None:
    log = ImperfectionLog(max_entries=10)
    p_dec = PauseDecision(occurred=True, duration_s=0.5, reason="pause")
    log.record_pause(p_dec)

    snapshot = log.entries()
    # Mutate returned dict in snapshot
    snapshot[0]["duration_s"] = 999.0

    fresh_snapshot = log.entries()
    assert fresh_snapshot[0]["duration_s"] == 0.5


def test_imperfection_log_bounded_fifo() -> None:
    max_e = 5
    log = ImperfectionLog(max_entries=max_e)

    for i in range(max_e + 5):
        log.record_pause(
            PauseDecision(
                occurred=True,
                duration_s=0.1 * (i + 1),
                reason="pause",
            )
        )

    assert log.size == max_e
    entries = log.entries()
    assert len(entries) == max_e

    # Oldest 5 entries (durations 0.1 .. 0.5) were dropped, surviving are 0.6 .. 1.0
    surviving_durations = [e["duration_s"] for e in entries]
    expected = [round(0.1 * (i + 1), 6) for i in range(5, 10)]
    assert [round(d, 6) for d in surviving_durations] == expected


def test_imperfection_log_clear_and_size() -> None:
    log = ImperfectionLog(max_entries=10)
    log.record_pause(PauseDecision(occurred=True, duration_s=0.5, reason="pause"))
    assert log.size == 1

    log.clear()
    assert log.size == 0
    assert len(log.entries()) == 0


def test_no_global_random_state_mutation() -> None:
    random.seed(12345)
    global_before = random.random()

    # Reset global seed to 12345 to re-produce global_before
    random.seed(12345)

    sampler = ImperfectionSampler()
    local_rng = random.Random(999)

    for _ in range(100):
        sampler.maybe_pause(local_rng)
        sampler.maybe_miss_click(local_rng)

    global_after = random.random()
    assert global_before == global_after


def test_static_ast_checks() -> None:
    with open("src/wow_bot/humanize/imperfections.py", "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)

    forbidden_imports = {
        "wow_bot.strategist",
        "wow_bot.perception",
        "wow_bot.reflex",
        "wow_bot.actuation",
        "wow_bot.executor",
        "wow_bot.world",
        "wow_bot.nav",
        "wow_bot.combat",
        "wow_bot.session",
        "aiosqlite",
        "asyncio",
        "threading",
        "numpy",
    }

    forbidden_substrings = ("ollama", "openai", "anthropic", "llm")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_name = alias.name
                assert mod_name not in forbidden_imports, f"Forbidden import: {mod_name}"
                for sub in forbidden_substrings:
                    assert sub not in mod_name.lower(), f"Forbidden import containing {sub}: {mod_name}"

        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            assert mod_name not in forbidden_imports, f"Forbidden import: {mod_name}"
            for sub in forbidden_substrings:
                assert sub not in mod_name.lower(), f"Forbidden import containing {sub}: {mod_name}"

        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "time":
            assert node.attr not in ("monotonic", "time", "perf_counter"), (
                f"Forbidden time function call: time.{node.attr}"
            )
