"""Analysis subpackage for WoW-Bot scenario metrics and spectral diagnostics."""

from wow_bot.analysis.spectrum import (
    ANALYSIS_SCHEMA_VERSION,
    MIN_FIT_POINTS,
    MIN_SPECTRUM_SAMPLES,
    TARGET_SLOPE_RANGE,
    SpectrumAnalysisError,
    analyze_spectrum,
    compute_dimension_spectrum,
    load_and_validate_scenario_report,
    plot_psd,
    resample_uniform,
    write_analysis_json,
)

__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "MIN_FIT_POINTS",
    "MIN_SPECTRUM_SAMPLES",
    "TARGET_SLOPE_RANGE",
    "SpectrumAnalysisError",
    "analyze_spectrum",
    "compute_dimension_spectrum",
    "load_and_validate_scenario_report",
    "plot_psd",
    "resample_uniform",
    "write_analysis_json",
]
