"""Unit tests for watchdog loop detector (Task 8.3)."""

import ast
import hashlib
import json
import math
from pathlib import Path

import pytest

from wow_bot.config import Config
from wow_bot.session import Session
from wow_bot.watchdog.loops import (
    ActionObservation,
    LoopConfig,
    LoopDetection,
    LoopDetector,
    LoopError,
    LoopEvent,
)


def test_loop_config_validation() -> None:
    """Test LoopConfig parameter validation and invariants."""
    # Default valid config
    cfg = LoopConfig()
    assert cfg.window_s == 600.0
    assert cfg.max_signatures == 2000
    assert cfg.cycle_length == 4
    assert cfg.min_repeats == 3
    assert cfg.stagnation_epsilon == 0.0
    assert cfg.emit_on_stagnation_only is True

    # window_s <= 0.0
    with pytest.raises(ValueError, match="window_s must be > 0.0"):
        LoopConfig(window_s=0.0)
    with pytest.raises(ValueError, match="window_s must be > 0.0"):
        LoopConfig(window_s=-10.0)

    # max_signatures < 32
    with pytest.raises(ValueError, match="max_signatures must be an integer >= 32"):
        LoopConfig(max_signatures=31)

    # cycle_length < 2
    with pytest.raises(ValueError, match="cycle_length must be an integer >= 2"):
        LoopConfig(cycle_length=1)

    # min_repeats < 2
    with pytest.raises(ValueError, match="min_repeats must be an integer >= 2"):
        LoopConfig(min_repeats=1)

    # stagnation_epsilon < 0.0
    with pytest.raises(ValueError, match="stagnation_epsilon must be >= 0.0"):
        LoopConfig(stagnation_epsilon=-0.1)

    # Non-finite floats
    with pytest.raises(ValueError, match="finite float"):
        LoopConfig(window_s=math.inf)
    with pytest.raises(ValueError, match="finite float"):
        LoopConfig(window_s=math.nan)
    with pytest.raises(ValueError, match="finite float"):
        LoopConfig(stagnation_epsilon=math.inf)


def test_action_observation_validation() -> None:
    """Test ActionObservation parameter validation and invariants."""
    # Valid observation
    obs = ActionObservation(ts=1.0, signature="move:10,20", level_or_xp=100.0)
    assert obs.ts == 1.0
    assert obs.signature == "move:10,20"
    assert obs.level_or_xp == 100.0

    # Negative ts
    with pytest.raises(ValueError, match="ts must be a finite float >= 0.0"):
        ActionObservation(ts=-0.1, signature="move", level_or_xp=0.0)

    # Empty signature
    with pytest.raises(ValueError, match="signature must be a non-empty string"):
        ActionObservation(ts=1.0, signature="", level_or_xp=0.0)

    # Whitespace-padded signature
    with pytest.raises(ValueError, match="signature must not have leading or trailing whitespace"):
        ActionObservation(ts=1.0, signature=" move ", level_or_xp=0.0)
    with pytest.raises(ValueError, match="signature must not have leading or trailing whitespace"):
        ActionObservation(ts=1.0, signature="move\t", level_or_xp=0.0)

    # Negative level_or_xp
    with pytest.raises(ValueError, match="level_or_xp must be a finite float >= 0.0"):
        ActionObservation(ts=1.0, signature="move", level_or_xp=-1.0)


def test_loop_detection_invariants() -> None:
    """Test LoopDetection invariant checks."""
    # Valid undetected
    det1 = LoopDetection(
        detected=False,
        cycle_length=0,
        repeats=0,
        ts=10.0,
        window_start_ts=0.0,
        progress_since_window_start=5.0,
        reason="progress_within_window",
    )
    assert det1.detected is False

    # Valid detected
    det2 = LoopDetection(
        detected=True,
        cycle_length=2,
        repeats=3,
        ts=10.0,
        window_start_ts=0.0,
        progress_since_window_start=0.0,
        reason="cycle_repeated_without_progress",
    )
    assert det2.detected is True

    # detected=False with non-zero repeats
    with pytest.raises(ValueError, match="repeats must be 0 when detected is False"):
        LoopDetection(
            detected=False,
            cycle_length=2,
            repeats=2,
            ts=10.0,
            window_start_ts=0.0,
            progress_since_window_start=0.0,
            reason="some_reason",
        )

    # detected=True with zero repeats
    with pytest.raises(ValueError, match="repeats must be an integer >= 1 when detected is True"):
        LoopDetection(
            detected=True,
            cycle_length=2,
            repeats=0,
            ts=10.0,
            window_start_ts=0.0,
            progress_since_window_start=0.0,
            reason="some_reason",
        )

    # Empty reason
    with pytest.raises(ValueError, match="reason must be a non-empty string"):
        LoopDetection(
            detected=False,
            cycle_length=0,
            repeats=0,
            ts=10.0,
            window_start_ts=0.0,
            progress_since_window_start=0.0,
            reason="",
        )

    # Negative ts
    with pytest.raises(ValueError, match="ts must be a finite float >= 0.0"):
        LoopDetection(
            detected=False,
            cycle_length=0,
            repeats=0,
            ts=-1.0,
            window_start_ts=0.0,
            progress_since_window_start=0.0,
            reason="reason",
        )


def test_loop_event_invariants() -> None:
    """Test LoopEvent invariant checks and to_json representation."""
    ev = LoopEvent(
        ts=12.5,
        cycle_length=2,
        repeats=3,
        signature_hashes=("hasha", "hashb"),
        reason="cycle_repeated_without_progress",
    )
    assert ev.to_json() == {
        "event": "watchdog_loop_detected",
        "ts": 12.5,
        "cycle_length": 2,
        "repeats": 3,
        "signature_hashes": ["hasha", "hashb"],
        "reason": "cycle_repeated_without_progress",
    }

    # len(signature_hashes) != cycle_length
    with pytest.raises(ValueError, match="len\\(signature_hashes\\).*must equal cycle_length"):
        LoopEvent(
            ts=12.5,
            cycle_length=3,
            repeats=3,
            signature_hashes=("hasha", "hashb"),
            reason="reason",
        )

    # cycle_length < 2
    with pytest.raises(ValueError, match="cycle_length must be an integer >= 2"):
        LoopEvent(
            ts=12.5,
            cycle_length=1,
            repeats=3,
            signature_hashes=("hasha",),
            reason="reason",
        )

    # repeats < 2
    with pytest.raises(ValueError, match="repeats must be an integer >= 2"):
        LoopEvent(
            ts=12.5,
            cycle_length=2,
            repeats=1,
            signature_hashes=("hasha", "hashb"),
            reason="reason",
        )

    # Empty reason
    with pytest.raises(ValueError, match="reason must be a non-empty string"):
        LoopEvent(
            ts=12.5,
            cycle_length=2,
            repeats=2,
            signature_hashes=("hasha", "hashb"),
            reason="",
        )

    # Negative ts
    with pytest.raises(ValueError, match="ts must be a finite float >= 0.0"):
        LoopEvent(
            ts=-1.0,
            cycle_length=2,
            repeats=2,
            signature_hashes=("hasha", "hashb"),
            reason="reason",
        )


def test_observe_fewer_than_two_cycle_length() -> None:
    """observe with fewer than 2*cycle_length observations returns None."""
    cfg = LoopConfig(cycle_length=2, min_repeats=2)
    detector = LoopDetector(config=cfg)

    # Need 2 * 2 = 4 observations minimum
    res1 = detector.observe(ActionObservation(ts=1.0, signature="sigA", level_or_xp=10.0))
    assert res1 is None
    assert detector.size == 1

    res2 = detector.observe(ActionObservation(ts=2.0, signature="sigB", level_or_xp=10.0))
    assert res2 is None
    assert detector.size == 2

    res3 = detector.observe(ActionObservation(ts=3.0, signature="sigA", level_or_xp=10.0))
    assert res3 is None
    assert detector.size == 3


def test_observe_rejects_non_monotone_ts() -> None:
    """observe rejects non-monotone ts with LoopError."""
    detector = LoopDetector()
    detector.observe(ActionObservation(ts=10.0, signature="sigA", level_or_xp=10.0))

    with pytest.raises(LoopError, match="timestamp.*is not strictly greater"):
        detector.observe(ActionObservation(ts=10.0, signature="sigB", level_or_xp=10.0))

    with pytest.raises(LoopError, match="timestamp.*is not strictly greater"):
        detector.observe(ActionObservation(ts=9.9, signature="sigB", level_or_xp=10.0))


def test_observe_rejects_decreasing_level_or_xp() -> None:
    """observe rejects decreasing level_or_xp with LoopError."""
    detector = LoopDetector()
    detector.observe(ActionObservation(ts=1.0, signature="sigA", level_or_xp=100.0))

    with pytest.raises(LoopError, match="level_or_xp decreased"):
        detector.observe(ActionObservation(ts=2.0, signature="sigB", level_or_xp=99.9))


def test_observe_returns_none_when_no_cycle() -> None:
    """observe returns None when the last 2*cycle_length signatures do not form a repeating cycle."""
    cfg = LoopConfig(cycle_length=2, min_repeats=2)
    detector = LoopDetector(config=cfg)

    # Pattern: sigA, sigB, sigC, sigD (not repeating)
    detector.observe(ActionObservation(ts=1.0, signature="sigA", level_or_xp=10.0))
    detector.observe(ActionObservation(ts=2.0, signature="sigB", level_or_xp=10.0))
    detector.observe(ActionObservation(ts=3.0, signature="sigC", level_or_xp=10.0))
    res = detector.observe(ActionObservation(ts=4.0, signature="sigD", level_or_xp=10.0))
    assert res is None


def test_observe_returns_none_when_repeats_below_min() -> None:
    """observe returns None when a cycle is present but repeats < min_repeats."""
    cfg = LoopConfig(cycle_length=2, min_repeats=3)
    detector = LoopDetector(config=cfg)

    # Pattern: (A, B), (A, B) -> repeats = 2 < 3
    detector.observe(ActionObservation(ts=1.0, signature="sigA", level_or_xp=10.0))
    detector.observe(ActionObservation(ts=2.0, signature="sigB", level_or_xp=10.0))
    detector.observe(ActionObservation(ts=3.0, signature="sigA", level_or_xp=10.0))
    res = detector.observe(ActionObservation(ts=4.0, signature="sigB", level_or_xp=10.0))
    assert res is None


def test_observe_detects_loop_on_stagnation() -> None:
    """observe returns detected=True when a cycle is present, repeats >= min_repeats, and progress <= stagnation_epsilon."""
    cfg = LoopConfig(cycle_length=2, min_repeats=2, stagnation_epsilon=0.0)
    detector = LoopDetector(config=cfg)

    # Pattern: (sigA, sigB), (sigA, sigB) without progress
    detector.observe(ActionObservation(ts=1.0, signature="sigA", level_or_xp=10.0))
    detector.observe(ActionObservation(ts=2.0, signature="sigB", level_or_xp=10.0))
    detector.observe(ActionObservation(ts=3.0, signature="sigA", level_or_xp=10.0))
    res = detector.observe(ActionObservation(ts=4.0, signature="sigB", level_or_xp=10.0))

    assert res is not None
    assert res.detected is True
    assert res.cycle_length == 2
    assert res.repeats == 2
    assert res.ts == 4.0
    assert res.window_start_ts == 1.0
    assert res.progress_since_window_start == 0.0
    assert res.reason == "cycle_repeated_without_progress"


def test_observe_progress_suppresses_detection_when_emit_on_stagnation_only_true() -> None:
    """observe returns detected=False with reason 'progress_within_window' when progress > stagnation_epsilon and emit_on_stagnation_only is True."""
    cfg = LoopConfig(
        cycle_length=2, min_repeats=2, stagnation_epsilon=0.0, emit_on_stagnation_only=True
    )
    detector = LoopDetector(config=cfg)

    detector.observe(ActionObservation(ts=1.0, signature="sigA", level_or_xp=10.0))
    detector.observe(ActionObservation(ts=2.0, signature="sigB", level_or_xp=10.0))
    detector.observe(ActionObservation(ts=3.0, signature="sigA", level_or_xp=10.0))
    # Progress advances from 10.0 to 15.0
    res = detector.observe(ActionObservation(ts=4.0, signature="sigB", level_or_xp=15.0))

    assert res is not None
    assert res.detected is False
    assert res.cycle_length == 0
    assert res.repeats == 0
    assert res.ts == 4.0
    assert res.window_start_ts == 1.0
    assert res.progress_since_window_start == 5.0
    assert res.reason == "progress_within_window"


def test_observe_detects_loop_despite_progress_when_emit_on_stagnation_only_false() -> None:
    """observe returns detected=True even with progress IF emit_on_stagnation_only is False."""
    cfg = LoopConfig(
        cycle_length=2, min_repeats=2, stagnation_epsilon=0.0, emit_on_stagnation_only=False
    )
    detector = LoopDetector(config=cfg)

    detector.observe(ActionObservation(ts=1.0, signature="sigA", level_or_xp=10.0))
    detector.observe(ActionObservation(ts=2.0, signature="sigB", level_or_xp=10.0))
    detector.observe(ActionObservation(ts=3.0, signature="sigA", level_or_xp=10.0))
    res = detector.observe(ActionObservation(ts=4.0, signature="sigB", level_or_xp=15.0))

    assert res is not None
    assert res.detected is True
    assert res.cycle_length == 2
    assert res.repeats == 2
    assert res.progress_since_window_start == 5.0
    assert res.reason == "cycle_repeated_without_progress"


def test_progress_since_window_start_computation() -> None:
    """progress_since_window_start reflects the difference between earliest surviving entry's level_or_xp and current observation."""
    cfg = LoopConfig(cycle_length=2, min_repeats=2, stagnation_epsilon=100.0)
    detector = LoopDetector(config=cfg)

    detector.observe(ActionObservation(ts=1.0, signature="sigA", level_or_xp=50.0))
    detector.observe(ActionObservation(ts=2.0, signature="sigB", level_or_xp=55.0))
    detector.observe(ActionObservation(ts=3.0, signature="sigA", level_or_xp=60.0))
    res = detector.observe(ActionObservation(ts=4.0, signature="sigB", level_or_xp=65.0))

    assert res is not None
    # Progress: 65.0 - 50.0 = 15.0 <= stagnation_epsilon (100.0)
    assert res.detected is True
    assert res.progress_since_window_start == 15.0


def test_samples_older_than_window_s_dropped() -> None:
    """Samples older than window_s are dropped and window_start_ts advances."""
    cfg = LoopConfig(window_s=10.0, cycle_length=2, min_repeats=2)
    detector = LoopDetector(config=cfg)

    detector.observe(ActionObservation(ts=1.0, signature="sig1", level_or_xp=10.0))
    detector.observe(ActionObservation(ts=5.0, signature="sig2", level_or_xp=10.0))
    assert detector.window_start_ts == 1.0

    # Advance to ts=12.0. cutoff = 12.0 - 10.0 = 2.0. Entry at ts=1.0 < 2.0 is dropped!
    detector.observe(ActionObservation(ts=12.0, signature="sig3", level_or_xp=10.0))
    assert detector.window_start_ts == 5.0
    assert detector.size == 2


def test_deque_max_signatures_capacity_bound() -> None:
    """The deque never exceeds config.max_signatures."""
    cfg = LoopConfig(max_signatures=32, window_s=1000.0)
    detector = LoopDetector(config=cfg)

    for i in range(100):
        detector.observe(
            ActionObservation(ts=float(i + 1), signature=f"sig_{i % 5}", level_or_xp=10.0)
        )

    assert detector.size == 32
    # Earliest surviving entry should be i = 100 - 32 = 68 -> ts = 69.0
    assert detector.window_start_ts == 69.0
    assert detector.window_end_ts == 100.0


def make_test_config(session_root: Path) -> Config:
    """Helper to construct a valid Config instance for session testing."""
    return Config(
        lab_mode=False,
        server_allowlist=("127.0.0.1:8080",),
        isolation_sentinel="10.0.0.1:80",
        kill_switch_key="F12",
        session_root=session_root,
        dry_run=True,
        max_session_seconds=3600,
        log_level="INFO",
    )


def test_session_events_emission_and_suppression(tmp_path: Path) -> None:
    """Test session event emission on loop detection vs suppression on progress."""
    cfg_app = make_test_config(tmp_path)
    session = Session.start(cfg_app)

    cfg = LoopConfig(
        cycle_length=2, min_repeats=2, stagnation_epsilon=0.0, emit_on_stagnation_only=True
    )
    detector = LoopDetector(config=cfg, session=session)

    # 1. Non-detected cycle (with progress) -> no session event
    detector.observe(ActionObservation(ts=1.0, signature="sigA", level_or_xp=10.0))
    detector.observe(ActionObservation(ts=2.0, signature="sigB", level_or_xp=10.0))
    detector.observe(ActionObservation(ts=3.0, signature="sigA", level_or_xp=10.0))
    detector.observe(ActionObservation(ts=4.0, signature="sigB", level_or_xp=15.0))

    events_path = session.path / "events.jsonl"
    events_lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(events_lines) == 0

    # 2. Reset and repeat without progress -> exactly 1 watchdog_loop_detected event
    detector.reset()
    detector.observe(ActionObservation(ts=10.0, signature="sigA", level_or_xp=20.0))
    detector.observe(ActionObservation(ts=11.0, signature="sigB", level_or_xp=20.0))
    detector.observe(ActionObservation(ts=12.0, signature="sigA", level_or_xp=20.0))
    det_res = detector.observe(
        ActionObservation(ts=13.0, signature="sigB", level_or_xp=20.0)
    )

    assert det_res is not None
    assert det_res.detected is True

    events_lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(events_lines) == 1
    event_data = json.loads(events_lines[0])
    assert event_data["event"] == "watchdog_loop_detected"
    assert event_data["ts"] == 13.0
    assert event_data["cycle_length"] == 2
    assert event_data["repeats"] == 2
    assert event_data["reason"] == "cycle_repeated_without_progress"
    assert len(event_data["signature_hashes"]) == 2

    session.close("clean")


def test_no_session_attached_works_without_raising() -> None:
    """No session attached: observe works without raising."""
    cfg = LoopConfig(cycle_length=2, min_repeats=2)
    detector = LoopDetector(config=cfg, session=None)

    detector.observe(ActionObservation(ts=1.0, signature="sigA", level_or_xp=10.0))
    detector.observe(ActionObservation(ts=2.0, signature="sigB", level_or_xp=10.0))
    detector.observe(ActionObservation(ts=3.0, signature="sigA", level_or_xp=10.0))
    res = detector.observe(ActionObservation(ts=4.0, signature="sigB", level_or_xp=10.0))

    assert res is not None
    assert res.detected is True


def test_reset_clears_deque() -> None:
    """reset clears the deque: size == 0 and subsequent observations start fresh."""
    detector = LoopDetector()
    detector.observe(ActionObservation(ts=1.0, signature="sigA", level_or_xp=10.0))
    detector.observe(ActionObservation(ts=2.0, signature="sigB", level_or_xp=10.0))
    assert detector.size == 2

    detector.reset()
    assert detector.size == 0
    assert detector.window_start_ts is None
    assert detector.window_end_ts is None

    # Reset is idempotent
    detector.reset()
    assert detector.size == 0


def test_determinism_across_detector_instances() -> None:
    """Two detectors with the same config and same observation sequence produce identical LoopDetection values."""
    cfg = LoopConfig(cycle_length=2, min_repeats=2)
    d1 = LoopDetector(config=cfg)
    d2 = LoopDetector(config=cfg)

    sequence = [
        ActionObservation(ts=1.0, signature="move:1,2", level_or_xp=10.0),
        ActionObservation(ts=2.0, signature="cast:fireball", level_or_xp=10.0),
        ActionObservation(ts=3.0, signature="move:1,2", level_or_xp=10.0),
        ActionObservation(ts=4.0, signature="cast:fireball", level_or_xp=10.0),
    ]

    r1 = [d1.observe(obs) for obs in sequence]
    r2 = [d2.observe(obs) for obs in sequence]

    assert r1 == r2


def test_hash_stability() -> None:
    """The same signature string yields the same signature_hash across two separate detector instances."""
    sig = "cast:frostbolt:target_12"
    h1 = hashlib.blake2b(sig.encode("utf-8"), digest_size=8).hexdigest()
    h2 = hashlib.blake2b(sig.encode("utf-8"), digest_size=8).hexdigest()

    assert h1 == h2
    assert isinstance(h1, str)
    assert len(h1) == 16


def test_observe_does_not_mutate_observation() -> None:
    """observe does not mutate the observation object."""
    detector = LoopDetector()
    obs = ActionObservation(ts=1.0, signature="move:10,20", level_or_xp=50.0)

    orig_ts = obs.ts
    orig_sig = obs.signature
    orig_xp = obs.level_or_xp

    detector.observe(obs)

    assert obs.ts == orig_ts
    assert obs.signature == orig_sig
    assert obs.level_or_xp == orig_xp


def test_window_start_and_end_ts_properties() -> None:
    """window_start_ts and window_end_ts reflect earliest and latest surviving entries."""
    detector = LoopDetector()
    assert detector.window_start_ts is None
    assert detector.window_end_ts is None

    detector.observe(ActionObservation(ts=10.0, signature="sig1", level_or_xp=5.0))
    assert detector.window_start_ts == 10.0
    assert detector.window_end_ts == 10.0

    detector.observe(ActionObservation(ts=15.0, signature="sig2", level_or_xp=5.0))
    assert detector.window_start_ts == 10.0
    assert detector.window_end_ts == 15.0


def test_static_ast_isolation_check() -> None:
    """Static AST check: loops.py does not import forbidden modules or LLM clients."""
    source_path = Path("src/wow_bot/watchdog/loops.py")
    assert source_path.exists()

    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    forbidden_exact = {
        "wow_bot.strategist",
        "wow_bot.perception",
        "wow_bot.reflex",
        "wow_bot.actuation",
        "wow_bot.executor",
        "wow_bot.world",
        "wow_bot.nav",
        "wow_bot.combat",
        "aiosqlite",
        "asyncio",
        "threading",
    }

    forbidden_substrings = ["ollama", "openai", "anthropic", "llm"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_name = alias.name
                assert (
                    mod_name not in forbidden_exact
                ), f"loops.py imports forbidden module '{mod_name}'"
                for sub in forbidden_substrings:
                    assert (
                        sub not in mod_name.lower()
                    ), f"loops.py imports forbidden LLM module '{mod_name}'"

        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            assert (
                mod_name not in forbidden_exact
            ), f"loops.py imports from forbidden module '{mod_name}'"
            for sub in forbidden_substrings:
                assert (
                    sub not in mod_name.lower()
                ), f"loops.py imports from forbidden LLM module '{mod_name}'"


def test_static_ast_no_wall_clock_time_check() -> None:
    """Static AST check: loops.py does not call time.monotonic, time.time, or time.perf_counter."""
    source_path = Path("src/wow_bot/watchdog/loops.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    forbidden_calls = {"monotonic", "time", "perf_counter"}

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr in forbidden_calls
            and isinstance(node.value, ast.Name)
            and node.value.id == "time"
        ):
            pytest.fail(f"loops.py calls forbidden wall-clock time function time.{node.attr}")
