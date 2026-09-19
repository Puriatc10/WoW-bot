#!/usr/bin/env python3
"""CLI script for Task 8.2 Timing Distribution Analysis.

Consumes Task 7.2 scenario report JSON, performs conditional log-normal model
validation using randomized PIT, calculates descriptive statistics and CV,
saves timing_analysis.json and timing_distribution.png to output directory.

Usage:
    python scripts/analyze_timing.py --input reports/combat_light_seed-42.json [--output-dir reports/analysis/combat_light]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure 'src' is in sys.path when script is executed directly
_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from wow_bot.analysis.timing import (
    PIT_RANDOM_SEED,
    build_timing_analysis,
    load_timing_report,
    plot_distribution,
    validate_timing_samples,
    write_analysis_output,
)


def main() -> None:
    """CLI execution entry point."""
    parser = argparse.ArgumentParser(
        description="Task 8.2 — Timing Distribution Analysis CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to input Task 7.2 scenario report JSON file.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where output timing_analysis.json and timing_distribution.png will be written.",
    )

    args = parser.parse_args()
    input_path = args.input.resolve()

    if not input_path.is_file():
        sys.stderr.write(f"Error: Input scenario report file not found: '{input_path}'\n")
        sys.exit(1)

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        output_dir = input_path.parent / f"{input_path.stem}_timing_analysis"

    try:
        report_dict = load_timing_report(input_path)
    except (ValueError, FileNotFoundError) as exc:
        sys.stderr.write(f"Error loading scenario report: {exc}\n")
        sys.exit(1)

    try:
        delays, detailed_samples = validate_timing_samples(report_dict.get("timing", {}))
        analysis_result = build_timing_analysis(
            report_dict,
            pit_seed=PIT_RANDOM_SEED,
        )

        analysis_json_path = output_dir / "timing_analysis.json"
        write_analysis_output(analysis_result, analysis_json_path)

        dist_png, pit_png = plot_distribution(
            analysis_result,
            delays,
            detailed_samples,
            output_dir,
        )

        sys.stdout.write("Timing distribution analysis succeeded:\n")
        sys.stdout.write(f"  Analysis JSON: {analysis_json_path}\n")
        sys.stdout.write(f"  Distribution Plot: {dist_png}\n")
        if pit_png is not None:
            sys.stdout.write(f"  PIT ECDF Plot: {pit_png}\n")

        cond_test = analysis_result.get("conditional_model_test", {})
        cond_status = cond_test.get("status")
        if cond_status == "available":
            p_val = cond_test.get("p_value")
            sys.stdout.write(f"  Conditional KS p-value: {p_val:.4f}\n")
        else:
            sys.stdout.write(f"  Conditional KS status: {cond_status}\n")

        cv_val = analysis_result.get("sample_summary", {}).get("cv", 0.0)
        sys.stdout.write(f"  Sample CV: {cv_val:.4f}\n")

    except Exception as exc:
        sys.stderr.write(f"Error executing timing analysis: {exc}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
