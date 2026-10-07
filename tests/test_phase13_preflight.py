"""Unit tests for Phase 13 Real Perception pre-flight tooling and CLI integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.lab.live_soak import parse_args as parse_live_soak_args
from scripts.lab.run_farm_v2 import parse_args as parse_farm_args
from scripts.lab.verify_perception_calibration import (
    check_dependencies,
    check_rois_fit,
    check_template_assets,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_run_farm_v2_cli_parsing() -> None:
    """CLI parser in run_farm_v2 accepts Phase 13 real perception flags."""
    args = parse_farm_args([
        "--mode", "LAB",
        "--real-perception",
        "--perception-config", "config/perception.example.toml",
        "--capture-region", "100,200,800,600",
        "--monitor-idx", "2",
    ])
    assert args.mode == "LAB"
    assert args.real_perception is True
    assert args.perception_config == Path("config/perception.example.toml")
    assert args.capture_region == "100,200,800,600"
    assert args.monitor_idx == 2


def test_run_farm_v2_real_perception_rejected_in_mock_mode() -> None:
    """Real perception is strictly rejected in MOCK mode per Global Rule 1."""
    from scripts.lab.run_farm_v2 import main as farm_main

    with pytest.raises(ValueError, match="strictly prohibited in MOCK mode"):
        farm_main([
            "--mode", "MOCK",
            "--real-perception",
        ])


def test_live_soak_cli_parsing() -> None:
    """CLI parser in live_soak accepts all required and optional arguments."""
    args = parse_live_soak_args([
        "--config", "config/lab.example.toml",
        "--profile", "profiles/test_farm.toml",
        "--rotation", "config/rotations/default.toml",
        "--duration-s", "120.0",
        "--sample-interval-s", "0.5",
        "--driver", "pynput",
        "--window-title", "WoW-Test",
        "--capture-region", "0,0,1920,1080",
        "--max-cycles", "50",
    ])
    assert args.config == Path("config/lab.example.toml")
    assert args.profile == Path("profiles/test_farm.toml")
    assert args.rotation == Path("config/rotations/default.toml")
    assert args.duration_s == 120.0
    assert args.sample_interval_s == 0.5
    assert args.driver == "pynput"
    assert args.window_title == "WoW-Test"
    assert args.capture_region == "0,0,1920,1080"
    assert args.max_cycles == 50


def test_verify_perception_calibration_dependencies() -> None:
    """Dependency check detects installed perception packages."""
    deps = check_dependencies()
    assert "mss" in deps
    assert "opencv" in deps
    assert "pytesseract" in deps
    assert deps["mss"][0] is True
    assert deps["opencv"][0] is True
    assert deps["pytesseract"][0] is True


def test_verify_perception_calibration_template_assets() -> None:
    """Template check finds committed template images under models/."""
    from wow_bot.perception.perception_config import load_perception_config

    cfg_path = REPO_ROOT / "config" / "perception.example.toml"
    cfg = load_perception_config(cfg_path)
    templates = [
        cfg.target_name_template,
        cfg.target_frame_template,
        cfg.minimap_arrow_template,
    ]
    results = check_template_assets(REPO_ROOT, templates)
    for path_str in templates:
        assert path_str in results
        assert results[path_str][0] is True, f"Missing template: {path_str}"


def test_verify_perception_calibration_rois_fit() -> None:
    """Calibrated ROIs fit within the reference 1920x1080 frame geometry."""
    cfg_path = REPO_ROOT / "config" / "perception.example.toml"
    checks = check_rois_fit((1080, 1920), cfg_path)
    assert len(checks) >= 10
    for name, ok, detail in checks:
        assert ok is True, f"ROI {name} check failed: {detail}"


def test_phase13_governance_amendments() -> None:
    """Governance documents reflect Phase 13 Real Perception activation."""
    roadmap_text = (REPO_ROOT / "docs" / "lab_phase" / "LAB_PHASE_ROADMAP.md").read_text(encoding="utf-8")
    assert "RealPerceptionBackend (Phase 13) is permitted as an opt-in live producer" in roadmap_text
    assert "# Phase 13 — Real Perception Live Soak" in roadmap_text

    agents_text = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "`RealPerceptionBackend` / `PerceptionPort` permitted for live execution in Phase 13" in agents_text

    results_real = REPO_ROOT / "docs" / "lab_phase" / "RESULTS_REAL.md"
    assert results_real.exists()
    results_text = results_real.read_text(encoding="utf-8")
    assert "# Phase 13 Results — Real Perception Live Soak & Functional Outcomes" in results_text
    assert "Real Perception Performance Metrics" in results_text
    assert "In-Game Gameplay Outcome Findings" in results_text
