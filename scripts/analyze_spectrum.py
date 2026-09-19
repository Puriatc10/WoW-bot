#!/usr/bin/env python3
"""Scenario Report Spectrum Analysis CLI (Task 8.1).

Performs FFT and Welch Power Spectral Density (PSD) analysis on MetaState time series
data contained in a Task 7.2 scenario report JSON.

Usage:
    python scripts/analyze_spectrum.py --input reports/combat_light_seed-42_duration-600.json
    python scripts/analyze_spectrum.py --input reports/combat_light.json --output-dir reports/analysis/combat_light
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Ensure 'src' is in sys.path when script is executed directly
_src_dir = str(Path(__file__).resolve().parent.parent / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from wow_bot.analysis.spectrum import (
    SpectrumAnalysisError,
    analyze_spectrum,
    plot_psd,
    write_analysis_json,
)
from wow_bot.reporting.scenario import META_STATE_DIMENSIONS
from wow_bot.shared.logger import get_logger

logger = get_logger("SPECTRUM_ANALYZER")


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate command line arguments.

    Args:
        args: Command line arguments (defaults to sys.argv[1:]).

    Returns:
        Parsed Namespace object.
    """
    parser = argparse.ArgumentParser(
        description="Perform FFT / Spectrum Analysis on Task 7.2 scenario report data."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to Task 7.2 Scenario Report JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/analysis",
        help="Directory to store analysis artifacts (spectrum_analysis.json, spectrum_psd.png). Default: reports/analysis",
    )

    return parser.parse_args(args)


def print_summary_table(analysis_result: dict[str, Any]) -> None:
    """Print concise scientific analysis summary to console."""
    provenance = analysis_result.get("source_provenance", {})
    scenario = provenance.get("scenario") or "Unknown"
    seed = provenance.get("seed") or "Unknown"
    input_report = provenance.get("input_report")

    sampling = analysis_result.get("sampling", {})
    orig_n = sampling.get("original_sample_count")
    resamp_n = sampling.get("resampled_sample_count")
    rate_hz = sampling.get("sampling_rate_hz", 0.0)

    summary = analysis_result.get("summary", {})
    passed = summary.get("dimensions_within_target_range", 0)
    total = summary.get("total_dimensions", 5)

    print("\n" + "=" * 64)
    print("         SCENARIO REPORT SPECTRUM ANALYSIS SUMMARY")
    print("=" * 64)
    print(f"Scenario Report : {input_report}")
    print(f"Scenario / Seed : {scenario} / {seed}")
    print(f"Original Samples: {orig_n} | Resampled Grid: {resamp_n} ({rate_hz:.2f} Hz)")
    print("Target Range    : Slope in [-1.50, -0.50]")
    print(f"Dimensions Pass : {passed} / {total}")
    print("-" * 64)

    dims_info = analysis_result.get("dimensions", {})
    for dim_name in META_STATE_DIMENSIONS:
        dim_data = dims_info.get(dim_name, {})
        slope = dim_data.get("slope", 0.0)
        r2 = dim_data.get("r_squared", 0.0)
        within_target = dim_data.get("within_target_range", False)
        status_str = "PASS" if within_target else "OUTSIDE TARGET"
        print(f"  {dim_name:10s} : slope = {slope:6.2f}  (R² = {r2:.4f})  [{status_str}]")

    print("=" * 64 + "\n")


def main(args: Sequence[str] | None = None) -> None:
    """CLI entry point for spectrum analyzer script."""
    parsed = parse_args(args)
    input_path = Path(parsed.input)
    output_dir = Path(parsed.output_dir)

    logger.info(f"Starting spectrum analysis for report '{input_path}'...")

    try:
        analysis_result, per_dim_arrays = analyze_spectrum(input_path)
        json_path = write_analysis_json(analysis_result, output_dir)
        plot_path = plot_psd(analysis_result, per_dim_arrays, output_dir)

        logger.info(f"Analysis JSON successfully written to '{json_path}'.")
        logger.info(f"PSD plot successfully saved to '{plot_path}'.")

        print_summary_table(analysis_result)

    except SpectrumAnalysisError as exc:
        logger.error(f"Spectrum analysis error: {exc}")
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Unexpected analysis failure: {type(exc).__name__}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
