# Cross-Platform Resource Telemetry & Metric Semantics

This document defines the physical quantities, platform bindings, metric labelling,
and mathematical interpretation for operational resource monitoring in the WoW-bot
soak test harness (`wow_bot.analysis.lab_soak_v2`, `wow_bot.analysis.windows_sampler`,
and `scripts/lab/full_soak.py`).

---

## 1. Problem Statement

Prior to task T-FIX-14, the soak harness suffered from an undetected cross-platform
semantic divergence:

1. **POSIX (`ProcessResourceSampler`):** Sampled `resource.getrusage(RUSAGE_SELF).ru_maxrss`.
   This is **Peak Resident Set Size (Peak RSS)**, a monotonically non-decreasing high-water mark.
2. **Windows (`WindowsResourceSampler`):** Sampled `PROCESS_MEMORY_COUNTERS.WorkingSetSize`.
   This is **Current Working Set Size**, an instantaneous fluctuating measurement of physical
   memory pages currently assigned to the process.
3. **Log Size Telemetry:** `scripts/lab/full_soak.py` sampled `args.session_dir / "app.log"`,
   but never installed the logging handler to write to that file. Consequently, the monitored
   log size stayed at zero throughout soak runs despite active logging in `events.jsonl`.

Comparing a series of `ru_maxrss` against a series of `WorkingSetSize` produced invalid
cross-platform comparisons where high-water mark expansion was compared against fluctuating
memory usage.

---

## 2. Harmonized Metric Architecture

Under T-FIX-14, the metric semantics are harmonized, explicitly labelled, and documented.

### 2.1 Unified Default Metric: Peak Resident Memory (`peak_rss`)

By default, all platforms sample **Peak Resident Set Size** in bytes:

| Platform | Underlying OS Source API | Physical Quantity | Metric Mode | Metric Label | Monotonicity |
|---|---|---|---|---|---|
| **Linux** | `getrusage(RUSAGE_SELF).ru_maxrss * 1024` | Peak Resident Set Size (bytes) | `peak` | `peak_rss` | Monotonically non-decreasing |
| **macOS (Darwin)** | `getrusage(RUSAGE_SELF).ru_maxrss` | Peak Resident Set Size (bytes) | `peak` | `peak_rss` | Monotonically non-decreasing |
| **Windows** | `PROCESS_MEMORY_COUNTERS.PeakWorkingSetSize` | Peak Working Set Size (bytes) | `peak` | `peak_rss` | Monotonically non-decreasing |

Both POSIX `ru_maxrss` (scaled to bytes) and Windows `PeakWorkingSetSize` represent the
maximum resident physical memory allocated to the process since process launch.

### 2.2 Optional Windows Metric: Instantaneous Working Set (`current_working_set`)

On Windows, callers may optionally request instantaneous working set sampling via
`metric_mode="current"`:

| Platform | Underlying OS Source API | Physical Quantity | Metric Mode | Metric Label | Monotonicity |
|---|---|---|---|---|---|
| **Windows (opt-in)** | `PROCESS_MEMORY_COUNTERS.WorkingSetSize` | Current Working Set Size (bytes) | `current` | `current_working_set` | Fluctuating (non-monotonic) |

This metric is explicitly typed and labelled with `metric_label = "current_working_set"`,
ensuring it is never pooled or directly compared against a `peak_rss` series.

On POSIX platforms, `ProcessResourceSampler` only supports `metric_mode="peak"` because the
POSIX `getrusage(2)` API natively exposes only peak resident memory (`ru_maxrss`). Requesting
`metric_mode="current"` on POSIX raises a `ValueError`.

---

## 3. Explicit Typing and Labelling Contract

Every `ResourceSnapshot` emitted by a sampler carries an explicit `metric_label`:

```python
@dataclass(frozen=True)
class ResourceSnapshot:
    ts: float
    cpu_percent: float
    rss_bytes: int
    log_size_bytes: int
    metric_label: str = "peak_rss"  # 'peak_rss' or 'current_working_set'
```

Samplers expose:
- `sampler.metric_label`: The string label associated with the emitted snapshots.
- `sampler.metric_mode`: The operational mode (`"peak"` or `"current"`).

---

## 4. Slope Semantics: Peak vs Current

In `SoakSummary`, the metric `rss_slope_bytes_per_hour` is calculated via linear
regression over the memory window. The physical meaning of this slope depends on the metric mode:

- **When `metric_label == "peak_rss"` (Default):**
  Because peak RSS is monotonically non-decreasing, the slope measures the **rate of high-water mark growth**
  (the ratchet rate). A slope near zero indicates that the process memory ceiling has stabilized.
  A positive slope indicates that the process continues to establish higher memory ceilings over time.
- **When `metric_label == "current_working_set"` (Opt-in):**
  The slope measures the net linear trend of instantaneous working set allocations over time,
  which may be positive, negative, or zero as memory is allocated and reclaimed.

The peak-versus-current distinction is stated on `ResourceSnapshot`, `SoakSummary`, and
the respective sampler docstrings so slopes are never compared across different physical quantities.

---

## 5. Log Size Sampling Contract

The soak harness CLI (`scripts/lab/full_soak.py`) monitors `app.log` in the active session
directory:

1. **Handler Installation:** Before the execution loop begins and before resource sampling
   starts, the harness invokes `setup_logging(session, config.log_level)` to attach a file
   handler pointing to `session.path / "app.log"`.
2. **Initial Record Emission:** The harness emits an initialization record (`"Soak harness initialized"`),
   guaranteeing that `app.log` exists on disk and is non-empty before the first resource sample.
3. **Truthful Zero:** A sample of `log_size_bytes == 0` strictly represents an empty file,
   never an uninstalled or unwritten logger. Once the logger writes, `log_size_bytes > 0`.
4. **Deterministic Cleanup:** On soak completion or error, `run_soak_async` flushes and removes
   file handlers in its `finally` block to prevent Windows file-lock issues during session
   directory management.
