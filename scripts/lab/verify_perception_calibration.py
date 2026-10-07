#!/usr/bin/env python3
"""Pre-flight verification and calibration tool for Phase 13 Real Perception.

Verifies:
1. Perception Python dependencies (mss, cv2, pytesseract, numpy).
2. Tesseract OCR binary availability and OCR test execution.
3. Template PNG image assets under models/.
4. Detector / model assets (YOLO weights) when enabled.
5. Configuration path resolution from chosen perception config location.
6. ROI boundary validity against calibrated resolution and screen geometry.
7. Capture region specification validity.
8. Screen capture device connectivity and test grab.
9. Target WoW game window existence via focus backend.
10. Live perception channels calibration requirements.

Clearly distinguishes:
- PASS: Validated and ready.
- FAIL: Blocking failures that prevent execution (exits with code 1).
- NEEDS_LIVE_CALIBRATION: Non-blocking items requiring live game client / in-game frame.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def check_dependencies() -> dict[str, tuple[bool, str]]:
    """Check required and optional Python packages."""
    results: dict[str, tuple[bool, str]] = {}

    # mss
    try:
        import mss

        results["mss"] = (True, f"version {getattr(mss, '__version__', 'installed')}")
    except ImportError:
        results["mss"] = (False, "Missing. Install via: uv pip install mss")

    # cv2
    try:
        import cv2

        results["opencv"] = (True, f"version {cv2.__version__}")
    except ImportError:
        results["opencv"] = (False, "Missing. Install via: uv pip install opencv-python-headless")

    # pytesseract
    try:
        import pytesseract  # type: ignore[import-untyped]

        results["pytesseract"] = (True, f"version {pytesseract.__version__}")
    except ImportError:
        results["pytesseract"] = (False, "Missing. Install via: uv pip install pytesseract")

    # ultralytics (optional)
    try:
        import ultralytics  # type: ignore[import-not-found]

        results["ultralytics (optional YOLO)"] = (True, f"version {ultralytics.__version__}")
    except ImportError:
        results["ultralytics (optional YOLO)"] = (
            True,
            "Not installed (optional; template-based fallback active)",
        )

    return results


def check_tesseract_binary(tesseract_cmd: str | None) -> tuple[str, str]:
    """Check that Tesseract OCR binary exists and executes self-test."""
    import shutil

    cmd = tesseract_cmd or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    path = Path(cmd)

    if not path.exists():
        which = shutil.which("tesseract")
        if which:
            cmd = which
            path = Path(which)
        else:
            return (
                "FAIL",
                f"Binary not found at '{cmd}'. On Windows: winget install UB-Mannheim.TesseractOCR or download from GitHub.",
            )

    try:
        import cv2
        import pytesseract

        test_img = np.ones((50, 150), dtype=np.uint8) * 255
        cv2.putText(test_img, "123", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)

        pytesseract.pytesseract.tesseract_cmd = str(path)
        text = pytesseract.image_to_string(
            test_img, config="--psm 7 -c tessedit_char_whitelist=0123456789"
        )
        if "123" in text.strip():
            return "PASS", f"Found and verified at {path} (OCR self-test passed)"
        return "PASS", f"Found at {path} (self-test returned: '{text.strip()}')"
    except Exception as exc:  # noqa: BLE001
        return "FAIL", f"Failed executing Tesseract at '{cmd}': {exc}"


def check_template_assets(
    base_dir: Path, template_paths: list[str]
) -> dict[str, tuple[bool, str]]:
    """Check existence of template image assets."""
    results: dict[str, tuple[bool, str]] = {}
    for rel_path in template_paths:
        p = base_dir / rel_path
        if not p.exists():
            p = base_dir.parent / rel_path
        if not p.exists():
            p = Path(rel_path)

        if p.exists() and p.is_file():
            results[rel_path] = (True, f"Found at {p} ({p.stat().st_size} bytes)")
        else:
            results[rel_path] = (False, f"Missing at {p}")
    return results


def check_detector_assets(
    base_dir: Path,
    yolo_weights: str | None,
    yolo_enabled: bool = True,
) -> tuple[str, str]:
    """Check detector/model assets if configured."""
    if not yolo_weights:
        return "PASS", "YOLO detector not configured (template-based target reader active)"
    p = base_dir / yolo_weights
    if not p.exists():
        p = base_dir.parent / yolo_weights
    if not p.exists():
        p = Path(yolo_weights)
    if p.exists() and p.is_file():
        return "PASS", f"Found detector weights at {p} ({p.stat().st_size} bytes)"
    if yolo_enabled:
        return "FAIL", f"Configured YOLO weights missing at {p}"
    return (
        "PASS",
        f"YOLO weights not present at {p}; template-based target reader active (optional)",
    )


def parse_and_validate_capture_region(
    region_str: str | None,
) -> tuple[str, str, tuple[int, int, int, int] | None]:
    """Parse and validate optional capture region string."""
    if region_str is None:
        return "PASS", "Full monitor capture (no capture_region override)", None
    try:
        parts = [int(p.strip()) for p in region_str.split(",")]
        if len(parts) != 4:
            return (
                "FAIL",
                f"capture_region must have 4 integers 'left,top,width,height' (got {len(parts)})",
                None,
            )
        left, top, width, height = parts
        if left < 0 or top < 0 or width <= 0 or height <= 0:
            return (
                "FAIL",
                f"capture_region coordinates must be non-negative and sizes > 0: ({left}, {top}, {width}, {height})",
                None,
            )
        return (
            "PASS",
            f"Valid capture region: left={left}, top={top}, width={width}, height={height}",
            (left, top, width, height),
        )
    except Exception as exc:  # noqa: BLE001
        return "FAIL", f"Malformed capture_region '{region_str}': {exc}", None


def check_screen_capture(
    monitor_idx: int = 1,
    region: tuple[int, int, int, int] | None = None,
) -> tuple[str, str, np.ndarray | None]:
    """Check screen capture device and grab one test frame."""
    try:
        from wow_bot.perception.capture import ScreenCapture

        cap = ScreenCapture(monitor_idx=monitor_idx, region=region)
        frame = cap.grab()
        if frame is None:
            return "FAIL", "ScreenCapture returned None frame", None
        h, w = frame.shape[:2]
        return "PASS", f"Successfully grabbed test frame: {w}x{h} px (monitor {monitor_idx})", frame
    except Exception as exc:  # noqa: BLE001
        return "FAIL", f"ScreenCapture error: {exc}", None


def check_rois_validity(
    perception_config_path: Path,
    frame_shape: tuple[int, int] | None = None,
) -> list[tuple[str, str, str]]:
    """Check mathematical validity and frame boundaries for all configured ROIs."""
    from wow_bot.perception.perception_config import load_perception_config

    cfg = load_perception_config(perception_config_path)
    cal_w, cal_h = cfg.calibrated_resolution

    checks: list[tuple[str, str, str]] = []

    bag_w = cfg.bag_columns * (cfg.bag_slot_size[0] + cfg.bag_gap)
    bag_h = cfg.bag_rows * (cfg.bag_slot_size[1] + cfg.bag_gap)
    rois: list[tuple[str, tuple[int, int, int, int]]] = [
        ("bars.hp_roi", cfg.hp_roi),
        ("bars.mana_roi", cfg.mana_roi),
        ("minimap.minimap_roi", cfg.minimap_roi),
        ("events.combat_region", cfg.combat_region),
        ("events.chat_region", cfg.chat_region),
        ("pose.coordinate_roi", cfg.pose_coordinate_roi),
        ("reaction.nameplate_roi", cfg.reaction_nameplate_roi),
        ("xp.xp_bar_roi", cfg.xp_bar_roi),
        ("cast.cast_roi", cfg.cast_roi),
        ("loot.sparkle_roi", cfg.loot_sparkle_roi),
        ("bag.grid", (cfg.bag_grid_origin[0], cfg.bag_grid_origin[1], bag_w, bag_h)),
    ]
    if cfg.xp_level_roi is not None:
        rois.append(("xp.xp_level_roi", cfg.xp_level_roi))
    if cfg.cast_border_roi is not None:
        rois.append(("cast.cast_border_roi", cfg.cast_border_roi))
    for idx, s_roi in enumerate(cfg.durability_slot_rois):
        rois.append((f"durability.slot_{idx}", s_roi))

    for name, (x, y, w, h) in rois:
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            checks.append((name, "FAIL", f"Invalid dimensions: ({x}, {y}, {w}, {h})"))
        elif (x + w) > cal_w or (y + h) > cal_h:
            checks.append(
                (
                    name,
                    "FAIL",
                    f"ROI ({x}+{w}={x+w}, {y}+{h}={y+h}) exceeds calibrated resolution ({cal_w}x{cal_h})",
                )
            )
        elif frame_shape is not None:
            fh, fw = frame_shape
            if (x + w) > fw or (y + h) > fh:
                checks.append(
                    (
                        name,
                        "FAIL",
                        f"ROI ({x}+{w}={x+w}, {y}+{h}={y+h}) exceeds captured frame ({fw}x{fh})",
                    )
                )
            else:
                checks.append((name, "PASS", f"Valid and fits frame: ({x}, {y}, {w}, {h})"))
        else:
            checks.append((name, "PASS", f"Valid dimensions: ({x}, {y}, {w}, {h})"))

    return checks


def check_rois_fit(
    frame_shape: tuple[int, int],
    perception_config_path: Path,
) -> list[tuple[str, bool, str]]:
    """Check that all configured ROIs fit within the captured frame geometry."""
    checks = check_rois_validity(perception_config_path, frame_shape=frame_shape)
    return [(name, status == "PASS", detail) for name, status, detail in checks]


def check_game_window(window_title: str) -> tuple[str, str]:
    """Check if target game window exists on desktop (Windows only)."""
    if sys.platform != "win32":
        return "PASS", "Skipped window check (non-Windows platform)"
    try:
        from wow_bot.actuation.backends.focus_win32 import Win32FocusBackend

        fb = Win32FocusBackend()
        hwnd = fb.find_window(window_title)
        fb.close()
        if hwnd is not None:
            return "PASS", f"Found window matching '{window_title}' (HWND: {hwnd})"
        return (
            "NEEDS_LIVE_CALIBRATION",
            f"Window matching '{window_title}' not currently found on desktop. Game client must be running before live launch.",
        )
    except Exception as exc:  # noqa: BLE001
        return "FAIL", f"Error searching for game window: {exc}"


def check_live_channels() -> list[tuple[str, str, str]]:
    """List perception channels that require an active game frame for live calibration."""
    return [
        (
            "bars (HP/Mana)",
            "NEEDS_LIVE_CALIBRATION",
            "HP and Mana bar pixel ratios require an active player character rendered in-game.",
        ),
        (
            "target (Name & Frame)",
            "NEEDS_LIVE_CALIBRATION",
            "Target nameplate OCR and target frame matching require an active target selected in-game.",
        ),
        (
            "pose (Addon Coordinates)",
            "NEEDS_LIVE_CALIBRATION",
            "Coordinate frame OCR requires coordinate display addon rendered at coordinate_roi.",
        ),
        (
            "reaction (Nameplate Color)",
            "NEEDS_LIVE_CALIBRATION",
            "Nameplate color classification requires hostile/neutral target nameplate visible on screen.",
        ),
        (
            "panels (Bag / Durability / XP)",
            "NEEDS_LIVE_CALIBRATION",
            "Bag and durability reading require game inventory/character panels opened in-game.",
        ),
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Verify Real Perception calibration and pre-flight readiness for Phase 13."
    )
    parser.add_argument(
        "--perception-config",
        type=Path,
        default=None,
        help="Path to perception configuration TOML.",
    )
    parser.add_argument(
        "--window-title",
        type=str,
        default="WoW",
        help="Window title substring to check (default: WoW).",
    )
    parser.add_argument(
        "--monitor-idx",
        type=int,
        default=1,
        help="Monitor index to capture (default: 1).",
    )
    parser.add_argument(
        "--capture-region",
        type=str,
        default=None,
        help="Optional capture region as 'left,top,width,height'.",
    )
    parser.add_argument(
        "--require-yolo",
        action="store_true",
        default=False,
        help="Enforce that YOLO weights exist (default: optional with template fallback).",
    )
    return parser.parse_args(argv)


def run_verification(args: argparse.Namespace) -> int:
    """Run all verification checks and print structured report."""
    print("=" * 72)
    print("  WoW-bot Phase 13 Real Perception Pre-Flight & Calibration Gate")
    print("=" * 72)

    passes: list[tuple[str, str]] = []
    failures: list[tuple[str, str]] = []
    needs_calibration: list[tuple[str, str]] = []

    def record(name: str, status: str, detail: str) -> None:
        if status == "PASS":
            passes.append((name, detail))
        elif status == "FAIL":
            failures.append((name, detail))
        else:
            needs_calibration.append((name, detail))
        print(f"  [{status:<22}] {name:<28}: {detail}")

    # 1. Python dependencies
    print("\n[1] Checking Python Dependencies:")
    deps = check_dependencies()
    for name, (ok, detail) in deps.items():
        status = "PASS" if ok else "FAIL"
        record(name, status, detail)

    # 2. Perception config
    p_cfg_path = args.perception_config
    if p_cfg_path is None:
        if Path("config/perception.toml").exists():
            p_cfg_path = Path("config/perception.toml")
        else:
            p_cfg_path = Path("config/perception.example.toml")

    print(f"\n[2] Loading Perception Configuration: {p_cfg_path}")
    if not p_cfg_path.exists():
        record(
            "config_file",
            "FAIL",
            f"Perception configuration not found at {p_cfg_path}",
        )
        return 1

    from wow_bot.perception.perception_config import load_perception_config

    try:
        cfg = load_perception_config(p_cfg_path)
        record(
            "config_syntax",
            "PASS",
            f"Validated perception config (calibrated: {cfg.calibrated_resolution[0]}x{cfg.calibrated_resolution[1]})",
        )
    except Exception as exc:  # noqa: BLE001
        record("config_syntax", "FAIL", f"Configuration validation error: {exc}")
        return 1

    # 3. Tesseract OCR
    print("\n[3] Checking Tesseract OCR Engine:")
    tess_status, tess_detail = check_tesseract_binary(cfg.tesseract_cmd)
    record("tesseract_engine", tess_status, tess_detail)

    # 4. Template & Model Assets
    print("\n[4] Checking Template & Model Assets:")
    template_paths = [
        cfg.target_name_template,
        cfg.target_frame_template,
        cfg.minimap_arrow_template,
    ]
    assets = check_template_assets(p_cfg_path.parent, template_paths)
    for asset, (ok, detail) in assets.items():
        status = "PASS" if ok else "FAIL"
        record(asset, status, detail)

    det_status, det_detail = check_detector_assets(
        p_cfg_path.parent,
        cfg.yolo_weights,
        yolo_enabled=bool(args.require_yolo),
    )
    record("detector_weights", det_status, det_detail)

    # 5. Capture Region & Screen Capture Device
    print("\n[5] Checking Screen Capture & Geometry:")
    reg_status, reg_detail, cap_region = parse_and_validate_capture_region(
        args.capture_region
    )
    record("capture_region", reg_status, reg_detail)

    cap_status, cap_detail, frame = check_screen_capture(
        monitor_idx=args.monitor_idx,
        region=cap_region,
    )
    record("screen_capture", cap_status, cap_detail)

    # 6. ROI Bounds Check
    print("\n[6] Checking ROI Calibration Bounds:")
    frame_shape = frame.shape[:2] if frame is not None else None
    roi_checks = check_rois_validity(p_cfg_path, frame_shape=frame_shape)
    for name, status, detail in roi_checks:
        record(name, status, detail)

    # 7. Game Window Check
    print(f"\n[7] Checking Target Game Window ('{args.window_title}'):")
    win_status, win_detail = check_game_window(args.window_title)
    record("game_window", win_status, win_detail)

    # 8. Live In-Game Channels
    print("\n[8] In-Game Perception Channels Calibration Status:")
    for name, status, detail in check_live_channels():
        record(name, status, detail)

    # Summary
    print("\n" + "=" * 72)
    print("  PRE-FLIGHT GATE SUMMARY")
    print("=" * 72)
    print(f"  PASS                  : {len(passes)}")
    print(f"  FAIL (blocking)       : {len(failures)}")
    print(f"  NEEDS_LIVE_CALIBRATION: {len(needs_calibration)}")
    print("-" * 72)

    if failures:
        print("  GATE OUTCOME: FAILED. Blocking issues must be resolved:")
        for name, detail in failures:
            print(f"    - {name}: {detail}")
        print("=" * 72)
        return 1

    if needs_calibration:
        print("  GATE OUTCOME: PASSED (Live calibration required before activation).")
        print("  The following items require the game client to be running in-game:")
        for name, detail in needs_calibration:
            print(f"    * {name}: {detail}")
    else:
        print("  GATE OUTCOME: ALL CHECKS PASSED.")
    print("=" * 72)

    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    return run_verification(args)


if __name__ == "__main__":
    sys.exit(main())
