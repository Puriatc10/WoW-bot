# Cross-Session Aggregate Semantics & Percentile Proxy Documentation

**Authoritative Documentation for Task T-FIX-19**  
**Module:** `src/wow_bot/analysis/aggregate.py` (`AggregateReport`, `PerceptionAgnosticMetrics`)

---

## 1. Executive Summary

This document defines the mathematical semantics, approximation boundaries, and duplicate-session deduplication rules implemented in `wow_bot.analysis.aggregate.py`. 

Because cross-session aggregation operates over summarized reports (`ReportV2` and `SoakReport`) rather than raw, uncompressed sample streams, multi-session percentile and latency statistics are **conservative proxies**, not exact pooled percentiles. Consumers and automated evaluation harnesses must interpret these fields according to the precise definitions below.

---

## 2. Proxy Values vs. Pooled Percentiles

In single-session reporting (`schema_v2.py`), percentiles such as `p50`, `p95`, and `p99` are computed directly over raw in-session sample buffers. However, when aggregating across multiple sessions in `AggregateReport`, raw sample buffers are not preserved in session JSON reports to maintain storage bounds and strict module isolation.

Mathematically, percentiles cannot be reconstructed exactly from per-session summary percentiles without the full underlying sample multiset. The aggregate engine therefore computes deterministic **proxies**:
- **Central tendencies (p50 / median / mean):** Computed as weighted means across sessions, weighted by the corresponding per-session sample count.
- **Upper-tail bounds (p95 / p99):** Computed as the maximum of per-session percentiles across all input sessions.

### Concrete Field Definitions

The table below enumerates every latency and tick timing field in `PerceptionAgnosticMetrics`:

| Concrete Field Name | Single-Session Source | Aggregate Calculation Rule | Statistical Meaning & Proxy Interpretation |
|---|---|---|---|
| `reflex_tick_period_ms_mean` | `report["reflex"]["mean_tick_period_ms"]` | Weighted mean weighted by `report["reflex"]["tick_count"]` | **Exact pooled mean** of reflex tick periods across all reported sessions (assuming disjoint non-overlapping ticks). |
| `reflex_tick_period_ms_p95` | `report["reflex"]["p99_tick_period_ms"]` | $\max(\text{p99\_tick\_period\_ms}_i)$ across sessions | **Conservative upper bound proxy.** Upper tail is bounded by the worst observed p99 among all sessions. It is *not* an exact 95th percentile of the pooled sample distribution. |
| `action_latency_ms_p50` | `report["action"]["latency_ms_p50"]` | Weighted mean weighted by `report["action"]["actuator_result_count"]` | **Median proxy.** Weighted average of per-session medians. Accurately represents typical actuator round-trip latency when per-session latency distributions are unimodal and symmetric, but diverges from true median under skewed multi-modal distributions. |
| `action_latency_ms_p95` | `report["action"]["latency_ms_p95"]` | $\max(\text{latency\_ms\_p95}_i)$ across sessions | **Conservative upper bound proxy.** Maximum of 95th percentiles across sessions. Guarantees that the reported value is at least as high as the 95th percentile of the worst individual session. |
| `strategist_latency_ms_p50` | `report["strategist"]["latency_ms_p50"]` | Weighted mean weighted by `report["strategist"]["call_count"]` | **Median proxy.** Weighted average of per-session medians across all LLM strategist invocations. |
| `strategist_latency_ms_p95` | `report["strategist"]["latency_ms_p95"]` | $\max(\text{latency\_ms\_p95}_i)$ across sessions | **Conservative upper bound proxy.** Maximum observed 95th percentile LLM invocation latency across sessions. |

### Interpretation Guidance for Consumers

- **Do NOT assume pooled normality:** Consumers must not construct confidence intervals or assume these fields represent percentiles of a pooled empirical cumulative distribution function (ECDF).
- **Upper-tail conservatism:** Because $p95$ fields use $\max$, they err on the side of reporting worst-case latency boundaries. A single degraded session will raise the aggregate $p95$ metric, ensuring performance regressions are not masked by averaging.

---

## 3. Duplicate Session Deduplication Semantics

The `aggregate_sessions(session_dirs, ...)` function accepts a sequence of directory paths. Callers may inadvertently supply duplicate directory paths or distinct paths that share the same underlying `session_id`.

The system's implemented behavior is defined as follows:

1. **Session ID Deduplication:**
   - The returned `AggregateReport.session_ids` tuple is deduplicated and sorted:
     $$\text{session\_ids} = \text{tuple}(\text{sorted}(\{\text{src.session\_id} \mid \text{src} \in \text{session\_sources}\}))$$
   - The reported `AggregateReport.session_count` equals $\text{len}(\text{session\_ids})$.

2. **Session Sources Lineage:**
   - `AggregateReport.session_sources` records a `SessionSource` descriptor for **every input directory in the order provided**, including duplicates.

3. **Metric Accumulation (Double-Counting on Duplicates):**
   - Accumulation of event totals and weighted sums iterates over all loaded reports:
     $$\text{for report in loaded\_reports: accumulate(\dots)}$$
   - Consequently, if the same session directory is passed $K$ times:
     - `session_count` will remain $1$.
     - Event counts (e.g., `reflex_tick_count_total`, `action_result_count_total`) will be multiplied by $K$.
     - Weighted averages remain mathematically identical (since weights and values scale proportionally), but sample weights will reflect $K \times N$.

### Invariant Contract

Callers and automated test harnesses must pass disjoint session directories representing distinct operational runs. When duplicate session directories are supplied, metrics reflect the multiset sum of the input list while session IDs reflect the unique set.

---

## 4. Preservation of Protected Documents

In accordance with `AGENTS.md` and `PRE_REAL_PERCEPTION_FIX_ROADMAP.md`:
- `docs/SOAK_PROTOCOL.md` and `docs/non_claims.json` remain untouched.
- The `NON_CLAIMS` constant tuple in `src/wow_bot/analysis/aggregate.py` is byte-for-byte identical to `docs/non_claims.json`.
- All Phase 12 archived artifacts (specifically `runs/lab/aggregate-1h/aggregate_v1.json` and `runs/lab/soak-1h/soak_report.json`) remain frozen and continue to parse identically.
