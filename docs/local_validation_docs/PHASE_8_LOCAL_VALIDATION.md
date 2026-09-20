# Phase 8 — Local Validation Report

**Validation Date:** 2026-09-20  
**Scope:** Task 8.1, Task 8.2, Task 8.3  
**Environment:** Local Windows Environment  
**Perception Source:** Mock / Synthetic  
**LLM:** Local Ollama — `qwen2.5:7b`  
**Controller Mode:** Dry Run  
**Primary Scenario Data:** `combat_light`, seed `42`  
**Soak Scenario:** `peaceful_farm`, seed `42`

---

## 1. هدف فاز 8

هدف فاز 8 این بود که بعد از اثبات integration در فاز 7، سه دسته evidence مستقل جمع‌آوری شود:

```text
8.1
MetaState Spectrum Analysis

8.2
Timing Distribution Analysis

8.3
Long-Running Stability / Soak Test
```

این فاز قرار نیست فقط ثابت کند اسکریپت‌ها اجرا می‌شوند. هدف این است که بین سه موضوع تفاوت قائل شویم:

```text
Implementation correctness

Mock / synthetic model consistency

Real-world scientific validity
```

در Local Validation فعلی فقط دو مورد اول قابل بررسی کامل بودند.

Perception هنوز Mock است. بنابراین نتایج علمی مربوط به رفتار واقعی سیستم نباید نهایی تلقی شوند.

---

## 2. ورودی اصلی فاز 8

مهم‌ترین dataset مورد استفاده از Task 7.2 آمده است:

```text
reports/validation_7_2_combat_600s.json
```

این dataset حاصل اجرای واقعی ده‌دقیقه‌ای pipeline با مشخصات زیر بود:

```text
Scenario            combat_light
Seed                42
Requested Duration  600 s
Completed Duration  600.344 s
Completed Normally  true
```

این report شامل:

```text
GameState samples    5571
MetaState samples    5571
Timing samples       5571
FSM transitions      72
LLM queries          13
Strategies           13
Deaths               0
```

بود و به همین دلیل dataset اصلی Task 8.1 و Task 8.2 انتخاب شد.

---

# 3. Task 8.1 — Spectrum Analysis

## 3.1 هدف

هدف Task 8.1 بررسی رفتار طیفی پنج بعد اصلی MetaState بود:

```text
hunger
fatigue
curiosity
aggression
social
```

فرضیه‌ی اولیه‌ی roadmap این بود که Power Spectral Density این time-seriesها تقریباً رفتاری شبیه:

```text
1 / f
```

داشته باشد.

بازه‌ی هدف frozen شده:

```text
PSD slope target:

[-1.5, -0.5]
```

## 3.2 دستور اجرا

```powershell
uv run python scripts/analyze_spectrum.py `
  --input reports/validation_7_2_combat_600s.json `
  --output-dir reports/analysis/phase8_1_combat_600s
```

Artifacts تولیدشده:

```text
reports/analysis/phase8_1_combat_600s/spectrum_analysis.json
reports/analysis/phase8_1_combat_600s/spectrum_psd.png
```

---

# 4. Sampling Diagnostics

Analyzer ورودی اصلی را با مشخصات زیر دریافت کرد:

```text
Original Samples      5571
Resampled Samples     5460
Sampling Rate         9.0995 Hz
Mean dt               0.1077 s
Median dt             0.1099 s
Std dt                0.1229 s
dt CV                  1.1407
```

مقدار بالای:

```text
dt CV = 1.14
```

نشان می‌دهد sampling کاملاً uniform نبوده است.

این موضوع از قبل در Phase 7 مشاهده شده بود؛ در bootstrap اولیه یک gap نسبتاً بزرگ وجود دارد و سپس sampling تقریباً پایدار می‌شود.

بنابراین resampling قبل از تحلیل PSD تصمیم لازم و درستی بوده است.

---

# 5. Spectrum Results

نتایج Welch slope:

```text
hunger       -2.4351
fatigue      -2.0938
curiosity    -2.0778
aggression   -2.2480
social       -2.0885
```

هیچ‌یک از پنج dimension داخل target مورد انتظار قرار نگرفتند:

```text
Dimensions Passed:

0 / 5
```

Summary رسمی نیز:

```text
dimensions_within_target_range = 0

target_range = [-1.5, -0.5]
```

را ثبت کرده است.

---

# 6. کیفیت Fit

مقادیر ضریب تعیین بسیار بالا بودند:

```text
hunger       R² = 0.9470
fatigue      R² = 0.9897
curiosity    R² = 0.9885
aggression   R² = 0.9451
social       R² = 0.9900
```

بنابراین slopeهای تقریباً `-2` صرفاً حاصل fit کاملاً تصادفی یا بی‌کیفیت نیستند.

در frequency range مورد استفاده‌ی analyzer، رفتار داده واقعاً به‌صورت نسبتاً منظم به یک power-law با slope نزدیک به `-2` شبیه است.

---

# 7. Finding مهم درباره Methodology

بررسی implementation نشان داد Welch با تنظیم زیر اجرا می‌شود:

```python
actual_nperseg = min(256, len(centered))
```

و سپس:

```python
scipy.signal.welch(
    centered,
    fs=sampling_rate_hz,
    detrend="constant",
    nperseg=actual_nperseg,
)
```

سپس تمام binهای مثبت و finite وارد regression می‌شوند.

Policy گزارش‌شده:

```text
fit_policy = positive_welch_bins
```

است.

با نرخ نمونه‌برداری حدود:

```text
9.10 Hz
```

و:

```text
nperseg = 256
```

resolution تقریبی Welch:

```text
9.10 / 256
≈ 0.0355 Hz
```

است.

این یعنی پایین‌ترین فرکانس قابل استفاده تقریباً معادل دوره‌ای در حدود:

```text
28 seconds
```

است.

---

# 8. محدودیت مهم Spectrum Experiment

Oscillatorهای Internal Dynamics دارای periodهای تقریبی زیر هستند:

```text
1 minute
5 minutes
20 minutes
90 minutes
5 hours
```

اما اولین Welch bin تقریباً مربوط به period حدود 28 ثانیه است.

بنابراین حتی سریع‌ترین oscillator اصلی سیستم:

```text
1 minute
```

پایین‌تر از اولین frequency bin اصلی این تحلیل قرار می‌گیرد.

در نتیجه analyzer فعلی عمدتاً رفتار:

```text
short-timescale / high-frequency
```

را اندازه می‌گیرد و نه کل multi-timescale behavior طراحی‌شده را.

---

# 9. محدودیت طول Dataset

Dataset اصلی فقط ده دقیقه است.

در ده دقیقه تقریباً:

```text
1-minute oscillator     10 cycles
5-minute oscillator      2 cycles
20-minute oscillator     0.5 cycle
90-minute oscillator     0.11 cycle
5-hour oscillator        0.03 cycle
```

دیده می‌شود.

بنابراین dataset فعلی اصولاً نمی‌تواند ساختار زمانی مربوط به periodهای بلندتر را به‌صورت علمی validate کند.

---

# 10. تفسیر صحیح نتیجه 8.1

نتیجه‌ی درست این نیست که:

```text
Internal Dynamics is proven to be 1/f²
```

چیزی که فعلاً می‌توان گفت:

```text
Under the current mock combat_light dataset
and current Welch fitting methodology,
the observable short-timescale PSD
has slopes around -2.1 to -2.4.
```

است.

بنابراین وضعیت Task 8.1:

```text
Implementation Validation        PASS

Mock Spectrum Analysis           EXECUTED

Configured Target                FAIL
0 / 5 dimensions

Final Scientific Conclusion      INCONCLUSIVE

Real-Perception Validation       PENDING
```

---

# 11. مقایسه با Task 3.7

در Task 3.7 synthetic experiment قبلی slopeهایی تقریباً نزدیک:

```text
-6
```

تولید کرده بود.

در pipeline واقعی‌تر فاز 7 ولی هنوز با Mock Perception:

```text
~ -2.1 ... -2.4
```

مشاهده شد.

این تفاوت finding مهمی است، اما هنوز نباید به‌صورت مستقیم تفسیر شود، مگر اینکه هر دو experiment با estimator و frequency band یکسان تحلیل شوند.

---

# 12. تصمیم درباره Dynamics

بر اساس نتایج فعلی هیچ‌یک از موارد زیر تغییر نکرد:

```text
Lorenz parameters
Drive decay
Oscillator amplitudes
Event effects
Target slope
Adaptive trigger
```

دلیل:

```text
Scientific failure must be investigated,
not tuned away.
```

---

# 13. Task 8.2 — Timing Distribution Analysis

## 13.1 هدف

Task 8.2 برای بررسی consistency مدل timing مصنوعی طراحی شده است.

مدل تولید delay:

```text
Conditional Lognormal Distribution
```

است.

پارامتر distribution بر اساس state داخلی تغییر می‌کند.

بنابراین aggregate timing distribution یک Lognormal ثابت نیست، بلکه mixtureای از چند distribution شرطی است.

## 13.2 دستور اجرا

```powershell
uv run python scripts/analyze_timing.py `
  -i reports/validation_7_2_combat_600s.json `
  -o reports/analysis/phase8_2_combat_600s
```

Artifacts:

```text
timing_analysis.json
timing_distribution.png
timing_pit_ecdf.png
```

---

# 14. Timing Sample Summary

Dataset شامل:

```text
Sample Count    5571
Mean            217.60 ms
Median          198.66 ms
Std Dev          93.67 ms
Min              50.00 ms
Max             969.20 ms
P05             101.65 ms
P95             392.55 ms
```

بود.

---

# 15. Clipping

حد پایین و بالا:

```text
Lower Bound     50 ms
Upper Bound   2000 ms
```

نتیجه:

```text
Lower Clipped Samples     1
Upper Clipped Samples     0

Clipped Fraction:
0.0001795
≈ 0.018%
```

بنابراین clipping در dataset بسیار ناچیز بوده است.

---

# 16. Coefficient of Variation

معیار frozen شده:

```text
CV > 0.3
```

نتیجه:

```text
CV = 0.43047
```

بود.

بنابراین:

```text
PASS
```

---

# 17. Conditional PIT / KS Test

به‌جای fit کردن یک Lognormal ثابت روی تمام داده‌ها، Task 8.2 از مدل شرطی هر sample استفاده می‌کند.

برای handling صحیح clipping از:

```text
Randomized PIT
```

استفاده شده است.

Seed ثابت:

```text
8202
```

نتیجه:

```text
KS statistic = 0.007426

p-value = 0.91615
```

هدف:

```text
p-value > 0.05
```

بنابراین:

```text
PASS
```

---

# 18. Task 8.2 Overall Acceptance

Report رسمی:

```text
CV target      PASS
KS target      PASS

overall_within_target = true
```

را ثبت کرده است.

وضعیت:

```text
Implementation Validation       PASS

Mock Model Consistency          PASS

Roadmap Acceptance              PASS
```

---

# 19. محدودیت علمی Task 8.2

این PASS نباید با اثبات human realism اشتباه گرفته شود.

Source report صریحاً مدل timing را:

```text
lognormal_simulation
```

ثبت کرده است.

در واقع validation فعلی تقریباً این زنجیره را بررسی می‌کند:

```text
Conditional Lognormal Generator
            ↓
Scenario Instrumentation
            ↓
Stored Parameters
            ↓
Conditional PIT
            ↓
KS against Uniform(0,1)
```

پس نتیجه نشان می‌دهد generator و analyzer از نظر آماری با یکدیگر سازگارند.

اما ثابت نمی‌کند timing انسان واقعی همین distribution را دارد.

بنابراین:

```text
Real Human Timing Validation
NOT PERFORMED
```

حتی Real Perception به‌تنهایی برای این validation کافی نیست؛ برای بررسی external realism نیاز به dataset مستقل از timing رفتار واقعی وجود خواهد داشت.

---

# 20. Aggregate Timing Fit

Histogram یک aggregate Lognormal curve نیز رسم می‌کند:

```text
μ ≈ 5.299
σ ≈ 0.408
```

اما report صریحاً این curve را فقط diagnostic معرفی کرده:

```text
Fitted across heterogeneous samples
for plotting only;
not generative truth.
```

بنابراین این curve برای acceptance استفاده نشده است.

---

# 21. Task 8.3 — Soak / Stability Testing

Task 8.3 برای بررسی موارد زیر طراحی شده است:

```text
Long-running runtime stability
CPU behaviour
Memory behaviour
Progress continuity
Watchdog health
Log growth
Graceful shutdown
Crash detection
```

سناریوی مورد استفاده:

```text
peaceful_farm
```

بود تا stability baseline بدون failure-path عمدی بررسی شود.

---

# 22. 5-Minute Soak Smoke Test

دستور:

```powershell
uv run python scripts/run_soak_test.py `
  --scenario peaceful_farm `
  --duration 300 `
  --sample-interval 30 `
  --seed 42 `
  --output reports/analysis/phase8_3_soak_5m.json
```

نتیجه:

```text
Requested Duration   300.0 s
Completed Duration   300.55 s
Samples              12
Completed Normally   true
```

Termination:

```text
termination_reason = duration_completed

zero_crash_target_met = true
```

---

# 23. 5-Minute Progress

Progress Token:

```text
Initial    1
Final      2601
Delta      2600
```

بنابراین process فقط alive نبوده؛ pipeline در تمام run به‌صورت واقعی progress داشته است.

---

# 24. 5-Minute CPU

ابتدای startup:

```text
CPU ≈ 94.2%
```

ثبت شده است.

اما بعد از startup steady-state تقریباً:

```text
~0.6% ... 2.7%
```

بوده است.

بنابراین CPU runaway مشاهده نشد.

---

# 25. 5-Minute Memory

Memory در ابتدای process:

```text
149.39 MB
```

بود.

بعد از startup و بارگذاری child processها:

```text
~299 MB
```

رسید.

در طول runtime تقریباً روی همین plateau باقی ماند و بعد از shutdown به:

```text
153.75 MB
```

برگشت.

نتیجه:

```text
No obvious leak observed.

Formal memory stability:
not yet proven.
```

---

# 26. 1-Hour Soak Test

بعد از موفقیت smoke test، اجرای یک‌ساعته انجام شد:

```powershell
uv run python scripts/run_soak_test.py `
  --scenario peaceful_farm `
  --duration 3600 `
  --sample-interval 30 `
  --seed 42 `
  --output reports/analysis/phase8_3_soak_1h.json
```

نتیجه:

```text
Requested Duration   3600.0 s
Completed Duration   3600.58 s
Sample Count         121
```

---

# 27. 1-Hour Termination

Run به‌صورت کامل و normal تمام شد:

```text
completed_normally = true

termination_reason = duration_completed

failure_type = null
failure_message = null

zero_crash_target_met = true
```

در انتها نیز shutdown کاملاً clean بود:

```text
duration reached
→ controller stop
→ memory close
→ watchdog exit
→ watchdog join
→ report persist
```

---

# 28. 1-Hour CPU Behaviour

ابتدای run startup spike:

```text
94.7%
```

ثبت شد.

اما steady-state عمدتاً در بازه‌ی تقریبی:

```text
1% ... 3%
```

باقی ماند.

Summary:

```text
Average CPU    2.67%
Maximum CPU   94.7%
```

و peak دقیقاً در startup رخ داده است.

هیچ evidenceی از CPU runaway یا progressive CPU growth مشاهده نشد.

---

# 29. 1-Hour Memory Behaviour

مهم‌ترین نتیجه‌ی soak test مربوط به memory است.

Startup:

```text
149.66 MB
```

بعد از runtime initialization:

```text
~298–299 MB
```

و سپس طی بخش عمده‌ی ساعت:

```text
~294–296 MB
```

باقی ماند.

Summary:

```text
Maximum Memory               298.95 MB

Memory Growth Slope
0.2577 MB/hour

First Quarter Mean           291.02 MB
Last Quarter Mean            291.21 MB
```

---

# 30. تفسیر Memory

خروجی report عمداً:

```text
memory_stability_status =
manual_review_required
```

ثبت می‌کند.

هیچ threshold مصنوعی برای memory leak تعریف نشده است.

تفسیر فعلی:

```text
Runaway Memory Growth:
NOT OBSERVED

Stable Plateau:
OBSERVED

Definitive Long-Term Leak Absence:
NOT YET PROVEN
```

---

# 31. Progress در 1-Hour Run

Progress Token:

```text
Initial    1
Final      33915
```

ثبت شده است.

بنابراین حدود 34 هزار مرحله‌ی واقعی pipeline طی run انجام شده است.

هیچ نشانه‌ای از freeze، silent stall یا deadlock مشاهده نشد.

---

# 32. Log Growth

در run یک‌ساعته log file به‌صورت طبیعی رشد کرده است:

```text
Initial Size   305572 bytes
Final Size     613726 bytes

Growth         308154 bytes
```

این رشد linear-looking و قابل انتظار بوده و هیچ مشکل I/O یا resource pressure ایجاد نکرده است.

---

# 33. 24-Hour Soak Test

بر اساس roadmap اولیه، acceptance کامل Task 8.3 شامل run بیست‌وچهارساعته بود.

این run در Local Validation فعلی عمداً اجرا نشد.

تصمیم پروژه:

```text
24-hour soak test
DEFERRED
```

دلیل:

Perception فعلی هنوز Mock است.

Run بیست‌وچهارساعته هزینه زمانی بالایی دارد و ترجیح داده شد زمانی انجام شود که Real Perception وارد pipeline شده باشد تا evidence طولانی‌مدت روی architecture نزدیک‌تر به deployment نهایی تولید شود.

بنابراین وضعیت:

```text
5-minute Soak    PASS

1-hour Soak      PASS

24-hour Soak     PENDING
                 Deferred until real perception
```

است.

---

# 34. Task 8.3 Current Status

```text
Soak Harness Implementation     PASS

5-minute Smoke                  PASS

1-hour Intermediate Soak        PASS

Zero-Crash                      PASS

Progress Continuity             PASS

Watchdog Stability              PASS

CPU Stability                   PASS

Graceful Shutdown               PASS

Log Tracking                    PASS

No Runaway Memory Growth        OBSERVED

Formal Memory Review            MANUAL

24-hour Final Soak              PENDING
```

---

# 35. تفاوت مهم بین Taskهای فاز 8

نتایج سه task نباید با یک معنی تفسیر شوند.

## Task 8.1

این task یک hypothesis علمی را تست کرده است.

نتیجه Mock:

```text
Configured PSD Target:
FAIL
```

اما experiment فعلی محدودیت frequency resolution و dataset duration دارد.

بنابراین:

```text
Scientific conclusion:
INCONCLUSIVE
```

## Task 8.2

این task consistency مدل مصنوعی timing را بررسی کرده است.

نتیجه:

```text
Model Consistency:
PASS
```

اما این به معنی اثبات شباهت به انسان واقعی نیست.

## Task 8.3

این task عمدتاً architecture و runtime stability را بررسی می‌کند.

نتایج ۵ دقیقه و یک ساعت:

```text
PASS
```

هستند.

این نتیجه حتی با Mock Perception نیز ارزش عملی بالایی دارد.

---

# 36. Findings اصلی Phase 8

مهم‌ترین یافته‌های این فاز:

```text
1.
Spectrum analyzer technically works,
but current mock pipeline PSD is outside
the frozen target.

2.
Current Welch methodology mostly observes
short-timescale frequencies.

3.
A 10-minute dataset is insufficient
for validating the complete multi-hour
oscillator architecture.

4.
Timing generator and timing analyzer
are statistically consistent end-to-end.

5.
Timing PASS does not prove human realism.

6.
5-minute and 1-hour soak tests show
stable runtime behaviour.

7.
No crash, deadlock, Watchdog false-positive,
or progressive CPU growth was observed.

8.
Memory reaches a stable runtime plateau
with no evidence of runaway growth.

9.
24-hour validation remains pending.

10.
Final scientific validation should be
repeated after real perception integration.
```

---

# 37. تغییراتی که عمداً انجام نشد

هیچ‌یک از موارد زیر برای pass شدن validation تغییر داده نشدند:

```text
PSD target range
Lorenz parameters
Oscillator amplitudes
Drive decay
Event effects
Timing distribution
Clipping bounds
Watchdog thresholds
Memory thresholds
```

اصل حفظ‌شده:

```text
Validation discovers behaviour.

Validation must not tune the system
merely to make tests pass.
```

---

# 38. Artifacts تولیدشده

## Task 8.1

```text
reports/analysis/phase8_1_combat_600s/
    spectrum_analysis.json
    spectrum_psd.png
```

## Task 8.2

```text
reports/analysis/phase8_2_combat_600s/
    timing_analysis.json
    timing_distribution.png
    timing_pit_ecdf.png
```

## Task 8.3

```text
reports/analysis/phase8_3_soak_5m.json

reports/analysis/phase8_3_soak_1h.json
```

---

# 39. Final Phase 8 Status

وضعیت دقیق فاز 8 در پایان Local Validation:

```text
Task 8.1
Implementation                    PASS
Mock Analysis                     EXECUTED
Configured Scientific Target      FAIL
Final Real Scientific Validation  PENDING

Task 8.2
Implementation                    PASS
Mock Model Consistency            PASS
Roadmap Statistical Acceptance    PASS
Real Human-Timing Validation      PENDING

Task 8.3
Implementation                    PASS
5-Minute Soak                     PASS
1-Hour Soak                       PASS
24-Hour Soak                      PENDING
```

و وضعیت کلی:

```text
Phase 8 Implementation Validation:
PASS

Phase 8 Local Mock Validation:
PASS WITH SCIENTIFIC FINDINGS

Full Scientific Validation:
PENDING REAL DATA

24-Hour Stability Validation:
PENDING REAL PERCEPTION
```

---

# 40. Next Validation Stage

مرحله‌ی علمی بعدی بعد از اتصال Real Perception باید شامل این موارد باشد:

```text
Collect real-perception dataset

Re-run spectrum analysis

Review frequency-band methodology

Use longer-duration datasets

Compare mock vs real PSD

Re-evaluate timing assumptions

Run final 24-hour soak test

Document real-data acceptance
```

تا قبل از آن، نتایج فعلی باید به‌عنوان:

```text
Synthetic / Mock Baseline Evidence
```

نگهداری شوند، نه proof نهایی behavior واقعی.

---

# Final Conclusion

فاز 8 از نظر tooling، integration و runtime stability موفق بوده است.

Spectrum Analysis یک scientific finding واقعی تولید کرد: داده‌ی Mock فعلی در frequency range اندازه‌گیری‌شده رفتار مورد انتظار `1/f` را نشان نمی‌دهد. این failure پنهان یا tune نشد و به‌عنوان evidence حفظ شد.

Timing Analysis نشان داد مدل تولید delay و analyzer از نظر آماری بسیار consistent هستند و معیارهای frozen شده را پاس می‌کنند.

Soak Testing نیز نشان داد pipeline حداقل در بازه‌های پنج دقیقه و یک ساعت بدون crash، deadlock، Watchdog failure یا runaway resource growth اجرا می‌شود.

با این حال دو acceptance نهایی عمداً باز باقی مانده‌اند:

```text
Real-data scientific validation

24-hour soak validation
```

هر دو بعد از integration شدن Real Perception انجام خواهند شد.

```text
Phase 8 Local Validation
COMPLETE FOR CURRENT MOCK STAGE

Real-Data Validation
PENDING

24-Hour Final Soak
PENDING
```
