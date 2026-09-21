# Reporting & Analysis Reconciliation (T10.0)

This document is the normative reconciliation reference for the Phase 10 Reporting & Analysis overhaul (`wow_bot.reporting`, `wow_bot.analysis`, `wow_bot.internal_dynamics`). It audits all pre-lab reporting, analysis, and internal dynamics symbols, classifies their relationship to Phase 10 tasks (T10.1–T10.4), resolves structural conflicts C1–C7, establishes module naming conventions, defines the schema versioning plan, and sets frozen interfaces for Phase 10 development.

---

## Inventory

Below is the complete inventory of all public top-level symbols defined across `src/wow_bot/reporting/`, `src/wow_bot/analysis/`, and `src/wow_bot/internal_dynamics/`.

| Symbol Name | Kind | Module | One-Line Responsibility | Exported in `__init__.py` | Exercised in `tests/` | Depends on Schema v1? |
|---|---|---|---|---|---|---|
| `SCENARIO_REPORT_SCHEMA_VERSION` | constant | `src/wow_bot/reporting/scenario.py` | Schema version constant (int = 1) for pre-lab scenario reports. | No | Yes (`tests/unit/test_scenario_runner.py`, `tests/unit/test_timing_analysis.py`, `tests/unit/test_spectrum_analysis.py`) | Yes |
| `META_STATE_DIMENSIONS` | constant | `src/wow_bot/reporting/scenario.py` | Ordered tuple of 5 MetaState drive dimension names. | No | Yes (`tests/unit/test_spectrum_analysis.py`, `tests/integration/test_spectrum.py`) | Yes |
| `PipelineObserver` | class (Protocol) | `src/wow_bot/reporting/scenario.py` | Runtime protocol interface for passive scenario pipeline instrumentation callbacks. | No | Yes (`tests/unit/test_scenario_runner.py`, `tests/unit/test_soak_analysis.py`, `tests/unit/test_review_gate.py`) | Yes |
| `ScenarioReportCollector` | class | `src/wow_bot/reporting/scenario.py` | In-memory event aggregator building pre-lab scenario research report dicts. | No | Yes (`tests/unit/test_scenario_runner.py`, `tests/unit/test_timing_analysis.py`, `tests/unit/test_spectrum_analysis.py`) | Yes |
| `validate_report_dict` | function | `src/wow_bot/reporting/scenario.py` | Validates structural and numeric invariants for schema v1 scenario reports. | No | Yes (`tests/unit/test_scenario_runner.py`, `tests/unit/test_timing_analysis.py`, `tests/unit/test_spectrum_analysis.py`) | Yes |
| `write_report_atomically` | function | `src/wow_bot/reporting/scenario.py` | Atomically serializes and writes scenario report dicts to disk as formatted JSON. | No | Yes (`tests/unit/test_scenario_runner.py`) | Yes |
| `SpectrumAnalysisError` | exception | `src/wow_bot/analysis/spectrum.py` | Domain exception raised when spectral analysis cannot be performed on report input. | Yes | Yes (`tests/unit/test_spectrum_analysis.py`) | No |
| `ANALYSIS_SCHEMA_VERSION` | constant | `src/wow_bot/analysis/spectrum.py` | Schema version constant (int = 1) for spectrum analysis JSON output. | Yes | Yes (`tests/unit/test_spectrum_analysis.py`) | No |
| `MIN_SPECTRUM_SAMPLES` | constant | `src/wow_bot/analysis/spectrum.py` | Minimum required MetaState sample count threshold (128) for spectral analysis. | Yes | Yes (`tests/unit/test_spectrum_analysis.py`) | No |
| `MIN_FIT_POINTS` | constant | `src/wow_bot/analysis/spectrum.py` | Minimum required Welch PSD frequency fit points (8) for log-log linear regression. | Yes | Yes (`tests/unit/test_spectrum_analysis.py`) | No |
| `TARGET_SLOPE_RANGE` | constant | `src/wow_bot/analysis/spectrum.py` | Target spectral slope range tuple `(-1.5, -0.5)` for 1/f noise evaluation. | Yes | Yes (`tests/unit/test_spectrum_analysis.py`) | No |
| `load_and_validate_scenario_report` | function | `src/wow_bot/analysis/spectrum.py` | Loads and strictly validates a schema v1 scenario report JSON file for spectrum analysis. | Yes (`# DEPRECATED`) | Yes (`tests/unit/test_spectrum_analysis.py`) | Yes |
| `resample_uniform` | function | `src/wow_bot/analysis/spectrum.py` | Calculates sampling diagnostics and resamples drive time series onto a uniform grid. | Yes | Yes (`tests/unit/test_spectrum_analysis.py`) | No |
| `compute_dimension_spectrum` | function | `src/wow_bot/analysis/spectrum.py` | Computes FFT, Welch PSD, log-log fit, and slope target evaluation for a 1D drive signal. | Yes | Yes (`tests/unit/test_spectrum_analysis.py`) | No |
| `analyze_spectrum` | function | `src/wow_bot/analysis/spectrum.py` | High-level spectrum analysis pipeline operating on schema v1 scenario report JSON files. | Yes (`# DEPRECATED`) | Yes (`tests/unit/test_spectrum_analysis.py`) | Yes |
| `write_analysis_json` | function | `src/wow_bot/analysis/spectrum.py` | Writes structured spectrum analysis results to JSON atomically. | Yes | Yes (`tests/unit/test_spectrum_analysis.py`) | No |
| `plot_psd` | function | `src/wow_bot/analysis/spectrum.py` | Generates and saves per-dimension Welch PSD log-log plot PNGs using Matplotlib. | Yes | Yes (`tests/unit/test_spectrum_analysis.py`) | No |
| `PIT_RANDOM_SEED` | constant | `src/wow_bot/analysis/timing.py` | Deterministic RNG seed constant (8202) for randomized PIT transformation. | No | Yes (`tests/unit/test_timing_analysis.py`, `tests/unit/test_humanize.py`) | No |
| `MIN_TIMING_SAMPLES` | constant | `src/wow_bot/analysis/timing.py` | Minimum required sample count threshold (100) for timing distribution analysis. | No | Yes (`tests/unit/test_timing_analysis.py`) | No |
| `KS_ALPHA` | constant | `src/wow_bot/analysis/timing.py` | Significance level alpha (0.05) for one-sample Kolmogorov-Smirnov test. | No | Yes (`tests/unit/test_timing_analysis.py`) | No |
| `CV_THRESHOLD` | constant | `src/wow_bot/analysis/timing.py` | Minimum target threshold (0.3) for coefficient of variation (CV). | No | Yes (`tests/unit/test_timing_analysis.py`) | No |
| `LOWER_DELAY_MS` | constant | `src/wow_bot/analysis/timing.py` | Lower clipping bound (50.0 ms) for human delay sampling model. | No | Yes (`tests/unit/test_timing_analysis.py`) | No |
| `UPPER_DELAY_MS` | constant | `src/wow_bot/analysis/timing.py` | Upper clipping bound (2000.0 ms) for human delay sampling model. | No | Yes (`tests/unit/test_timing_analysis.py`) | No |
| `load_timing_report` | function | `src/wow_bot/analysis/timing.py` | Loads and performs basic validation on a schema v1 scenario report JSON file. | No | Yes (`tests/unit/test_timing_analysis.py`) | Yes |
| `validate_timing_samples` | function | `src/wow_bot/analysis/timing.py` | Validates raw delay milliseconds and detailed metadata from schema v1 timing sections. | No | Yes (`tests/unit/test_timing_analysis.py`, `tests/unit/test_review_gate.py`) | Yes |
| `compute_descriptive_statistics` | function | `src/wow_bot/analysis/timing.py` | Computes descriptive statistics (mean, std, CV, percentiles) on raw delay arrays. | No | Yes (`tests/unit/test_timing_analysis.py`) | No |
| `compute_randomized_pit` | function | `src/wow_bot/analysis/timing.py` | Calculates randomized Probability Integral Transform (PIT) values for conditional delays. | No | Yes (`tests/unit/test_timing_analysis.py`, `tests/unit/test_humanize.py`) | No |
| `run_conditional_ks` | function | `src/wow_bot/analysis/timing.py` | Runs one-sample Kolmogorov-Smirnov test comparing PIT values against Uniform(0,1). | No | Yes (`tests/unit/test_timing_analysis.py`) | No |
| `build_timing_analysis` | function | `src/wow_bot/analysis/timing.py` | High-level timing analysis builder operating on schema v1 scenario report dicts. | No | Yes (`tests/unit/test_timing_analysis.py`) | Yes |
| `plot_distribution` | function | `src/wow_bot/analysis/timing.py` | Generates delay distribution histogram and PIT ECDF plots using Matplotlib. | No | Yes (`tests/unit/test_timing_analysis.py`) | No |
| `write_analysis_output` | function | `src/wow_bot/analysis/timing.py` | Writes timing analysis dictionary atomically to JSON without NaN/Inf. | No | Yes (`tests/unit/test_timing_analysis.py`) | No |
| `SOAK_REPORT_SCHEMA_VERSION` | constant | `src/wow_bot/analysis/soak.py` | Schema version constant (int = 1) for soak test reports. | No | Yes (`tests/unit/test_soak_analysis.py`) | Yes |
| `ResourceSnapshot` | class (dataclass) | `src/wow_bot/analysis/soak.py` | Immutable dataclass capturing CPU percent, RSS memory MB, and process count. | No | Yes (`tests/unit/test_soak_analysis.py`, `tests/unit/test_review_gate.py`) | No |
| `SoakSample` | class (dataclass) | `src/wow_bot/analysis/soak.py` | Immutable operational sample recorded during soak test execution. | No | Yes (`tests/unit/test_soak_analysis.py`, `tests/unit/test_review_gate.py`) | No |
| `ResourceSampler` | class (Protocol) | `src/wow_bot/analysis/soak.py` | Protocol abstraction for operational resource sampling backends. | No | Yes (`tests/unit/test_soak_analysis.py`) | No |
| `ProcessResourceSampler` | class | `src/wow_bot/analysis/soak.py` | Cross-platform psutil sampler measuring process tree CPU and RSS memory. | No | Yes (`tests/unit/test_soak_analysis.py`) | No |
| `FakeResourceSampler` | class | `src/wow_bot/analysis/soak.py` | Deterministic test mock implementing ResourceSampler protocol for unit tests. | No | Yes (`tests/unit/test_soak_analysis.py`, `tests/unit/test_review_gate.py`) | No |
| `SoakObserver` | class | `src/wow_bot/analysis/soak.py` | Passive observer tracking progress tokens and FSM state for pre-lab soak runs. | No | Yes (`tests/unit/test_soak_analysis.py`) | Yes |
| `calculate_log_size` | function | `src/wow_bot/analysis/soak.py` | Recursively calculates byte size for log files or directories. | No | Yes (`tests/unit/test_soak_analysis.py`) | No |
| `resolve_log_path` | function | `src/wow_bot/analysis/soak.py` | Resolves target log file path or status string for soak measurements. | No | Yes (`tests/unit/test_soak_analysis.py`) | No |
| `calculate_memory_slope` | function | `src/wow_bot/analysis/soak.py` | Computes least-squares linear growth slope of RSS memory MB per hour. | No | Yes (`tests/unit/test_soak_analysis.py`) | No |
| `build_soak_summary` | function | `src/wow_bot/analysis/soak.py` | Calculates aggregate summary statistics across a sequence of soak operational samples. | No | Yes (`tests/unit/test_soak_analysis.py`) | No |
| `build_soak_report` | function | `src/wow_bot/analysis/soak.py` | Constructs a complete schema v1 soak test report dictionary. | No | Yes (`tests/unit/test_soak_analysis.py`, `tests/unit/test_review_gate.py`) | Yes |
| `write_soak_report_atomically` | function | `src/wow_bot/analysis/soak.py` | Atomically writes soak report dictionary to JSON with strict formatting. | No | Yes (`tests/unit/test_soak_analysis.py`) | No |
| `Drives` | class | `src/wow_bot/internal_dynamics/drives.py` | Core 5-dimensional drive integration model (hunger, fatigue, curiosity, aggression, social). | Yes | Yes (`tests/unit/test_drives.py`, `tests/unit/test_meta_state.py`, `tests/integration/test_spectrum.py`) | No |
| `LorenzAttractor` | class | `src/wow_bot/internal_dynamics/chaos.py` | RK4-integrated Lorenz 63 chaotic attractor model generating bounded chaotic modulation. | No | Yes (`tests/unit/test_chaos.py`, `tests/unit/test_meta_state.py`, `tests/integration/test_spectrum.py`) | No |
| `MemoryStore` | class | `src/wow_bot/internal_dynamics/memory.py` | Async SQLite persistent store for events and 5D drive vector similarity recall. | No | Yes (`tests/unit/test_memory.py`, `tests/unit/test_meta_state.py`, `tests/integration/test_spectrum.py`, `tests/unit/test_review_gate.py`) | No |
| `MetaStateGenerator` | class | `src/wow_bot/internal_dynamics/meta_state.py` | Subsystem coordinator generating MetaState snapshots and evaluating adaptive LLM trigger rules. | No | Yes (`tests/unit/test_meta_state.py`, `tests/integration/test_spectrum.py`) | No |
| `OscillatorBank` | class | `src/wow_bot/internal_dynamics/oscillators.py` | Bank of 5 coupled slow harmonic oscillators generating continuous rhythmic modulation. | No | Yes (`tests/unit/test_oscillators.py`, `tests/unit/test_meta_state.py`, `tests/integration/test_spectrum.py`) | No |

---

## Classification

Every public symbol is classified into exactly one of `{KEEP_AS_IS, KEEP_AND_EXTEND, RENAME, DEPRECATE, REPLACE, NEW}`:

### 1. `KEEP_AS_IS` (41 symbols)
- **Reporting Core Helpers:** `SCENARIO_REPORT_SCHEMA_VERSION`, `META_STATE_DIMENSIONS`, `validate_report_dict`, `write_report_atomically`. Reused by pre-lab scenario tests.
- **Spectrum Core Math & Output Helpers:** `SpectrumAnalysisError`, `ANALYSIS_SCHEMA_VERSION`, `MIN_SPECTRUM_SAMPLES`, `MIN_FIT_POINTS`, `TARGET_SLOPE_RANGE`, `resample_uniform`, `compute_dimension_spectrum`, `write_analysis_json`, `plot_psd`. Pure mathematical calculations and file output functions reused directly by Phase 10 versioned lab spectral analysis.
- **Timing Core Math & Output Helpers:** `PIT_RANDOM_SEED`, `MIN_TIMING_SAMPLES`, `KS_ALPHA`, `CV_THRESHOLD`, `LOWER_DELAY_MS`, `UPPER_DELAY_MS`, `compute_descriptive_statistics`, `compute_randomized_pit`, `run_conditional_ks`, `plot_distribution`, `write_analysis_output`. Pure statistical and plotting functions reused directly by Phase 10 versioned lab timing analysis.
- **Soak Measurement & Core Helpers:** `SOAK_REPORT_SCHEMA_VERSION`, `ResourceSnapshot`, `SoakSample`, `ResourceSampler`, `ProcessResourceSampler`, `FakeResourceSampler`, `calculate_log_size`, `resolve_log_path`, `calculate_memory_slope`, `build_soak_summary`, `build_soak_report`, `write_soak_report_atomically`. Reused by pre-lab soak tests and Phase 10 versioned soak analysis.
- **Internal Dynamics Subsystem:** `Drives`, `LorenzAttractor`, `MemoryStore`, `MetaStateGenerator`, `OscillatorBank`. Unmodified internal dynamics components providing synthetic agent internal state in MOCK_MODE.

### 2. `DEPRECATE` (8 symbols)
- `PipelineObserver` (`reporting/scenario.py`): Pre-lab observer protocol covering synthetic scenario events. Superseded by Phase 10 JSON Lines event stream processing (`events.jsonl`).
- `ScenarioReportCollector` (`reporting/scenario.py`): Pre-lab in-memory event collector. Superseded by `reporting/lab_pipeline_v2.py` in Phase 10.
- `load_and_validate_scenario_report` (`analysis/spectrum.py`): Pre-lab scenario report loader. Superseded by `analysis/lab_spectral_v2.py` reading schema v2 lab reports in Phase 10.
- `analyze_spectrum` (`analysis/spectrum.py`): Pre-lab spectrum pipeline operating on schema v1 scenario reports. Superseded by `analysis/lab_spectral_v2.py` in Phase 10.
- `load_timing_report` (`analysis/timing.py`): Pre-lab report loader for timing analysis. Superseded by `analysis/lab_timing_v2.py` reading schema v2 lab reports in Phase 10.
- `validate_timing_samples` (`analysis/timing.py`): Pre-lab timing sample validator expecting schema v1 report structure. Superseded by `analysis/lab_timing_v2.py` in Phase 10.
- `build_timing_analysis` (`analysis/timing.py`): Pre-lab timing analysis builder operating on schema v1 scenario report dicts. Superseded by `analysis/lab_timing_v2.py` in Phase 10.
- `SoakObserver` (`analysis/soak.py`): Pre-lab in-memory soak observer. Superseded by Phase 10 soak harness (`scripts/lab/soak_v2.py`) reading structured metric streams.

### 3. `KEEP_AND_EXTEND`, `RENAME`, `REPLACE` (0 symbols)
No pre-lab symbol is classified as `KEEP_AND_EXTEND`, `RENAME`, or `REPLACE`. Under Strategy A, all pre-lab symbols remain intact to guarantee 100% test compatibility.

---

## Conflict Resolution

This section state explicit resolution strategies for structural conflicts C1 through C7.

### Conflict C1: Two Report Formats Coexist
- **Problem:** Pre-lab uses nested JSON scenario reports (schema v1), whereas Phase 10 produces append-only JSON Lines (`events.jsonl`).
- **Affected Symbols:** `ScenarioReportCollector`, `PipelineObserver`, `validate_report_dict`, `write_report_atomically`.
- **Resolution:** Both report formats coexist. Schema v1 remains unmodified in `reporting/scenario.py`. T10.1 and T10.2 introduce `reporting/schema_v2.py` and `reporting/lab_pipeline_v2.py` to compile flat `events.jsonl` streams into schema v2 lab research reports (`report_v2.json`).
- **Migration Path:** Pre-lab mock tests continue using schema v1. Lab execution sessions write `events.jsonl`, which `lab_pipeline_v2.py` processes post-hoc into schema v2 reports.
- **Pre-lab Tests Updated:** No.

### Conflict C2: Spectral Analysis Import Coupling
- **Problem:** `analysis/spectrum.py` imports `SCENARIO_REPORT_SCHEMA_VERSION` and `META_STATE_DIMENSIONS` from `reporting/scenario.py`, binding it to schema v1.
- **Affected Symbols:** `analysis/spectrum.py` (`load_and_validate_scenario_report`, `analyze_spectrum`), `reporting/scenario.py`.
- **Resolution:** `analysis/spectrum.py` remains untouched for pre-lab compatibility. T10.3 introduces `analysis/lab_spectral_v2.py` which consumes schema v2 lab reports directly without importing `reporting/scenario.py`.
- **Migration Path:** Existing spectral tests continue using `analysis/spectrum.py`. Lab analysis scripts invoke `analysis/lab_spectral_v2.py`.
- **Pre-lab Tests Updated:** No.

### Conflict C3: Timing & Soak Analysis Import Coupling
- **Problem:** `analysis/timing.py` imports `SCENARIO_REPORT_SCHEMA_VERSION` from `reporting/scenario.py`. `soak.py` defines `SOAK_REPORT_SCHEMA_VERSION = 1`.
- **Affected Symbols:** `analysis/timing.py` (`load_timing_report`, `build_timing_analysis`), `analysis/soak.py`.
- **Resolution:** `analysis/timing.py` and `analysis/soak.py` remain untouched. T10.3 introduces `analysis/lab_timing_v2.py` and T10.4 introduces `analysis/lab_soak_v2.py` to process lab-grade schema v2 data.
- **Migration Path:** Pre-lab timing and soak unit tests run against existing modules. Phase 10 lab sessions use `lab_timing_v2.py` and `lab_soak_v2.py`.
- **Pre-lab Tests Updated:** No.

### Conflict C4: `PipelineObserver` Callback Protocol Scope Limits
- **Problem:** `PipelineObserver` protocol only covers synthetic pre-lab scenario events and cannot capture Phase 0–9 lab events (`safety_abort`, `fsm_transition`, `cooldown_allowed`, `vocab_accepted`, etc.).
- **Affected Symbols:** `PipelineObserver`, `ScenarioReportCollector`, `SoakObserver`.
- **Resolution:** `PipelineObserver` and `SoakObserver` are deprecated. Phase 10 relies on `Session` (T0.2) structured JSON Lines event logging (`events.jsonl`) rather than in-memory observer callbacks.
- **Migration Path:** Mock scenario runners retain `PipelineObserver`. Phase 10 lab reporting parses `events.jsonl` using event stream processors.
- **Pre-lab Tests Updated:** No.

### Conflict C5: Internal Dynamics Drive Vector Source Separation
- **Problem:** `internal_dynamics.Drives` produces `MetaState.vector` (5 dimensions), but LAB_MODE execution does not require internal dynamics for fast-loop actuation/reflex.
- **Affected Symbols:** `internal_dynamics/` (`Drives`, `MetaStateGenerator`, etc.).
- **Resolution:** `internal_dynamics` components are classified `KEEP_AS_IS`. In MOCK_MODE, `MetaState` events are written to `events.jsonl`. In LAB_MODE, spectral analysis runs on available continuous metrics logged in `events.jsonl` or is flagged as not applicable if `MetaState` is disabled.
- **Migration Path:** Unmodified internal dynamics modules remain available for MOCK_MODE research scenarios.
- **Pre-lab Tests Updated:** No.

### Conflict C6: Module Naming Discrepancies
- **Problem:** Pre-lab file names (`scenario.py`, `spectrum.py`, `timing.py`, `soak.py`) conflict with Phase 10 module naming requirements.
- **Affected Symbols:** All Phase 10 reporting and analysis modules.
- **Resolution:** Adopt **STRATEGY A (Versioned Modules)**. Pre-lab modules retain their file names. Phase 10 capabilities are implemented in dedicated versioned modules (`reporting/schema_v2.py`, `reporting/lab_pipeline_v2.py`, `analysis/lab_spectral_v2.py`, `analysis/lab_timing_v2.py`, `analysis/lab_soak_v2.py`).
- **Migration Path:** New versioned siblings created in T10.1–T10.4 without renaming pre-lab files.
- **Pre-lab Tests Updated:** No.

### Conflict C7: Schema Version Collisions
- **Problem:** Schema v1 (`SCENARIO_REPORT_SCHEMA_VERSION = 1`) and Schema v2 coexist.
- **Affected Symbols:** `SCENARIO_REPORT_SCHEMA_VERSION`, `SOAK_REPORT_SCHEMA_VERSION`, schema v2 (`SCHEMA_VERSION = 2`).
- **Resolution:** Schema v1 remains in `reporting/scenario.py` as `1`. Schema v2 is defined in `reporting/schema_v2.py` as `2`. Every report JSON file explicitly declares its `schema_version`. Analysis modules reject reports with mismatched schema versions. Callers must not mix v1 and v2 in a single analysis run.
- **Migration Path:** Pre-lab reports use schema v1. Lab Phase 10 reports use schema v2.
- **Pre-lab Tests Updated:** No.

---

## Naming Convention

**Chosen Strategy: STRATEGY A (versioned modules)**

Phase 10 implements versioned module names alongside existing pre-lab modules:
- Pre-lab modules retain their file names:
  - `src/wow_bot/reporting/scenario.py`
  - `src/wow_bot/analysis/spectrum.py`
  - `src/wow_bot/analysis/timing.py`
  - `src/wow_bot/analysis/soak.py`
- Phase 10 capabilities are added in dedicated, versioned modules:
  - `schemas/report_v2.json` (T10.1 JSON Schema specification)
  - `src/wow_bot/reporting/schema_v2.py` (T10.1 Report Schema v2 dataclasses & validator)
  - `src/wow_bot/reporting/lab_pipeline_v2.py` (T10.2 Lab event stream pipeline)
  - `src/wow_bot/analysis/lab_spectral_v2.py` (T10.3 Lab spectral analysis)
  - `src/wow_bot/analysis/lab_timing_v2.py` (T10.3 Lab timing analysis)
  - `src/wow_bot/analysis/lab_soak_v2.py` (T10.4 Lab soak analysis engine)
  - `scripts/lab/soak_v2.py` (T10.4 Long-running soak runner script)

**Rationale:** Strategy A eliminates all risk of breaking pre-lab unit and integration tests that import directly from pre-lab reporting and analysis modules. It provides a clean, side-by-side transition during Phase 10 development while ensuring pre-lab MOCK_MODE tests remain 100% passing and reproducible.

---

## Interface Freeze

The following contract specifies the strict interface boundaries for authors implementing Phase 10 tasks T10.1 through T10.4.

### 1. `src/wow_bot/reporting/scenario.py` (Coexists with T10.1/T10.2)
- **MUST NOT CHANGE:** `SCENARIO_REPORT_SCHEMA_VERSION`, `META_STATE_DIMENSIONS`, `PipelineObserver`, `ScenarioReportCollector`, `validate_report_dict`, `write_report_atomically`.
- **MAY ADD:** None. Pre-lab scenario reporting is frozen; Phase 10 schema and pipeline belong in `schema_v2.py` and `lab_pipeline_v2.py`.
- **MAY REFACTOR FREELY:** None.

### 2. `src/wow_bot/analysis/spectrum.py` (Coexists with T10.3 `lab_spectral_v2.py`)
- **MUST NOT CHANGE:** `SpectrumAnalysisError`, `ANALYSIS_SCHEMA_VERSION`, `MIN_SPECTRUM_SAMPLES`, `MIN_FIT_POINTS`, `TARGET_SLOPE_RANGE`, `load_and_validate_scenario_report`, `resample_uniform`, `compute_dimension_spectrum`, `analyze_spectrum`, `write_analysis_json`, `plot_psd`.
- **MAY ADD:** None. Pre-lab spectrum module is frozen; T10.3 lab spectral analysis belongs in `lab_spectral_v2.py`.
- **MAY REFACTOR FREELY:** Private math or plotting helpers if any.

### 3. `src/wow_bot/analysis/timing.py` (Coexists with T10.3 `lab_timing_v2.py`)
- **MUST NOT CHANGE:** `PIT_RANDOM_SEED`, `MIN_TIMING_SAMPLES`, `KS_ALPHA`, `CV_THRESHOLD`, `LOWER_DELAY_MS`, `UPPER_DELAY_MS`, `load_timing_report`, `validate_timing_samples`, `compute_descriptive_statistics`, `compute_randomized_pit`, `run_conditional_ks`, `build_timing_analysis`, `plot_distribution`, `write_analysis_output`.
- **MAY ADD:** None. Pre-lab timing module is frozen; T10.3 lab timing analysis belongs in `lab_timing_v2.py`.
- **MAY REFACTOR FREELY:** Private calculation helpers if any.

### 4. `src/wow_bot/analysis/soak.py` (Coexists with T10.4 `lab_soak_v2.py`)
- **MUST NOT CHANGE:** `SOAK_REPORT_SCHEMA_VERSION`, `ResourceSnapshot`, `SoakSample`, `ResourceSampler`, `ProcessResourceSampler`, `FakeResourceSampler`, `SoakObserver`, `calculate_log_size`, `resolve_log_path`, `calculate_memory_slope`, `build_soak_summary`, `build_soak_report`, `write_soak_report_atomically`.
- **MAY ADD:** None. Pre-lab soak module is frozen; T10.4 lab soak analysis belongs in `lab_soak_v2.py`.
- **MAY REFACTOR FREELY:** Private sampling helpers.

### 5. `src/wow_bot/internal_dynamics/*.py`
- **MUST NOT CHANGE:** `Drives`, `LorenzAttractor`, `MemoryStore`, `MetaStateGenerator`, `OscillatorBank`.
- **MAY ADD:** None. Internal dynamics modules are frozen.
- **MAY REFACTOR FREELY:** Private internal methods.

### 6. `src/wow_bot/reporting/__init__.py` and `src/wow_bot/analysis/__init__.py`
- **MUST NOT CHANGE:** Existing re-exported public symbols.
- **MAY ADD:** Re-exports for Phase 10 symbols as T10.1–T10.4 are completed.

---

## Schema Versioning Plan

Schema v1 and Schema v2 coexist without cross-contamination:

```
+------------------------------------+      +------------------------------------+
| Scenario Report v1 (schema_version=1)|      |     Lab Report v2 (schema_version=2) |
+------------------------------------+      +------------------------------------+
| - Produced by: ScenarioReportCollector|   | - Produced by: lab_pipeline_v2.py  |
| - Source: In-memory pipeline events|      | - Source: events.jsonl event stream|
| - Format: Nested JSON object       |      | - Format: Structured JSON sections |
| - Contains:                        |      | - Sections (optional by mode):     |
|     run (scenario, seed, duration) |      |     perception, action, reflex,    |
|     summary, strategist, fsm,      |      |     navigation, combat, humanizer, |
|     meta_state, timing, watchdog,  |      |     strategist, watchdog           |
|     deaths, idle_intents           |      | - Consumed by:                     |
| - Consumed by:                     |      |     analysis/lab_spectral_v2.py    |
|     analysis/spectrum.py           |      |     analysis/lab_timing_v2.py      |
|     analysis/timing.py             |      |     analysis/lab_soak_v2.py        |
+------------------------------------+      +------------------------------------+
```

### Invariants:
1. Every report JSON file MUST contain a top-level `"schema_version"` integer field (`1` or `2`).
2. Schema v1 reports are loaded and validated by `reporting/scenario.py`.
3. Schema v2 reports are loaded and validated by `reporting/schema_v2.py`.
4. Analysis modules MUST reject report files whose `schema_version` does not match their expected version.
5. An analysis run MUST process either schema v1 reports or schema v2 reports, never both in a single run.
