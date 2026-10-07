#!/usr/bin/env python3
"""Pre-flight verification and calibration tool for Phase 13 Real Perception.

Verifies:
1. Perception Python dependencies (mss, cv2, pytesseract, numpy).
2. Tesseract OCR binary availability and OCR test execution.
3. Template PNG image assets under models/.
4. Screen capture device connectivity, monitor resolution, and test grab.
5. ROI boundary validity against grabbed screen/window geometry.
6. Target game window existence via FocusBackend (optional).

Run before starting the game client and live bot execution:
    python scripts/lab/verify_perception_calibration.py
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
        results["ultralytics (optional YOLO)"] = (True, "Not installed (optional for template-based fallback)")

    return results


def check_tesseract_binary(tesseract_cmd: str | None) -> tuple[bool, str]:
    """Check that Tesseract OCR binary exists and runs."""
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
                False,
                f"Binary not found at '{cmd}'. On Windows: winget install UB-Mannheim.TesseractOCR or download from GitHub.",
            )

    try:
        import pytesseract

        # Create a small synthetic image with digits '123' to test OCR execution
        test_img = np.ones((50, 150), dtype=np.uint8) * 255
        import cv2

        cv2.putText(test_img, "123", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)

        pytesseract.pytesseract.tesseract_cmd = str(path)
        text = pytesseract.image_to_string(test_img, config="--psm 7 -c tessedit_char_whitelist=0123456789")
        if "123" in text.strip():
            return True, f"Found and verified at {path} (OCR self-test passed)"
        return True, f"Found at {path} (self-test returned: '{text.strip()}')"
    except Exception as exc:  # noqa: BLE001
        return False, f"Failed executing Tesseract at '{cmd}': {exc}"


def check_template_assets(base_dir: Path, template_paths: list[str]) -> dict[str, tuple[bool, str]]:
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


def check_screen_capture(
    monitor_idx: int = 1,
    region: tuple[int, int, int, int] | None = None,
) -> tuple[bool, str, np.ndarray | None]:
    """Check screen capture device and grab one test frame."""
    try:
        from wow_bot.perception.capture import ScreenCapture

        cap = ScreenCapture(monitor_idx=monitor_idx, region=region)
        frame = cap.grab()
        if frame is None:
            return False, "ScreenCapture returned None frame", None
        h, w = frame.shape[:2]
        return True, f"Successfully grabbed test frame: {w}x{h} px (monitor {monitor_idx})", frame
    except Exception as exc:  # noqa: BLE001
        return False, f"ScreenCapture error: {exc}", None


def check_rois_fit(frame_shape: tuple[int, int], perception_config_path: Path) -> list[tuple[str, bool, str]]:
    """Check that all configured ROIs fit within the captured frame geometry."""
    from wow_bot.perception.perception_config import load_perception_config

    cfg = load_perception_config(perception_config_path)
    fh, fw = frame_shape

    checks: list[tuple[str, bool, str]] = []

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
            checks.append((name, False, f"Invalid dimensions: ({x}, {y}, {w}, {h})"))
        elif (x + w) > fw or (y + h) > fh:
            checks.append(
                (name, False, f"ROI ({x}+{w}={x+w}, {y}+{h}={y+h}) exceeds frame bounds ({fw}x{fh})")
            )
        else:
            checks.append((name, True, f"Fits within frame: ({x}, {y}, {w}, {h})"))

    return checks


def check_game_window(window_title: str) -> tuple[bool, str]:
    """Check if the target game window is present (Windows only)."""
    if sys.platform != "win32":
        return True, "Skipped (non-Windows platform)"
    try:
        from wow_bot.actuation.backends.focus_win32 import Win32FocusBackend

        fb = Win32FocusBackend()
        hwnd = fb.find_window(window_title)
        fb.close()
        if hwnd is not None:
            return True, f"Found window matching '{window_title}' (HWND: {hwnd})"
        return False, f"Window matching '{window_title}' not currently found on desktop"
    except Exception as exc:  # noqa: BLE001
        return False, f"Error searching for window: {exc}"


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
    return parser.parse_args(argv)


def run_verification(args: argparse.Namespace) -> int:
    """Run all verification checks and print report."""
    print("=" * 70)
    print("  WoW-bot Phase 13 Real Perception Pre-Flight & Calibration Check")
    print("=" * 70)

    all_passed = True

    # 1. Python dependencies
    print("\n[1] Checking Python Dependencies:")
    deps = check_dependencies()
    for name, (ok, detail) in deps.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name:<26}: {detail}")
        if not ok:
            all_passed = False

    # 2. Perception config
    p_cfg_path = args.perception_config
    if p_cfg_path is None:
        if Path("config/perception.toml").exists():
            p_cfg_path = Path("config/perception.toml")
        else:
            p_cfg_path = Path("config/perception.example.toml")

    print(f"\n[2] Loading Perception Configuration: {p_cfg_path}")
    if not p_cfg_path.exists():
        print(f"  [FAIL] Perception configuration not found at {p_cfg_path}")
        return 1

    from wow_bot.perception.perception_config import load_perception_config

    try:
        cfg = load_perception_config(p_cfg_path)
        print("  [PASS] Configuration syntax and values validated successfully.")
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] Configuration validation error: {exc}")
        return 1

    # 3. Tesseract OCR
    print("\n[3] Checking Tesseract OCR:")
    tess_ok, tess_detail = check_tesseract_binary(cfg.tesseract_cmd)
    tess_status = "PASS" if tess_ok else "FAIL"
    print(f"  [{tess_status}] {tess_detail}")
    if not tess_ok:
        all_passed = False

    # 4. Template assets
    print("\n[4] Checking Template Assets:")
    template_paths = [
        cfg.target_name_template,
        cfg.target_frame_template,
        cfg.minimap_arrow_template,
    ]
    assets = check_template_assets(p_cfg_path.parent, template_paths)
    for asset, (ok, detail) in assets.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {asset:<30}: {detail}")
        if not ok:
            all_passed = False

    # 5. Screen Capture & Frame geometry
    print("\n[5] Checking Screen Capture Device:")
    cap_region = None
    if args.capture_region:
        parts = [int(p.strip()) for p in args.capture_region.split(",")]
        cap_region = (parts[0], parts[1], parts[2], parts[3])

    cap_ok, cap_detail, frame = check_screen_capture(
        monitor_idx=args.monitor_idx,
        region=cap_region,
    )
    cap_status = "PASS" if cap_ok else "FAIL"
    print(f"  [{cap_status}] {cap_detail}")
    if not cap_ok or frame is None:
        all_passed = False
    else:
        # 6. ROI Bounds Check
        print("\n[6] Checking ROI Calibration against Screen Geometry:")
        roi_checks = check_rois_fit(frame.shape[:2], p_cfg_path)
        for name, ok, detail in roi_checks:
            status = "PASS" if ok else "WARN"
            print(f"  [{status}] {name:<26}: {detail}")
            if not ok:
                all_passed = False

    # 7. Game Window Check
    print(f"\n[7] Checking Game Window ('{args.window_title}'):")
    win_ok, win_detail = check_game_window(args.window_title)
    win_status = "PASS" if win_ok else "INFO"
    print(f"  [{win_status}] {win_detail}")

    print("\n" + "=" * 70)
    if all_passed:
        print("  RESULT: PRE-FLIGHT CHECKS PASSED. Ready to run game client.")
    else:
        print("  RESULT: SOME CHECKS REQUIRE ATTENTION before live execution.")
    print("=" * 70)

    return 0 if all_passed else 1


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    return run_verification(args)


if __name__ == "__main__":
    sys.exit(main())
