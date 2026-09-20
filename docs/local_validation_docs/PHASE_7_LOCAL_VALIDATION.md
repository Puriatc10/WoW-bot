# Phase 7 — Local Validation Report

**Validation Date:** 2026-09-20
**Scope:** Task 7.1 + Task 7.2
**Environment:** Windows / Local Machine / Local Ollama
**LLM:** `qwen2.5:7b`
**Execution Mode:** Mock/Synthetic Perception + Dry-Run Controller
**Overall Core Validation Status:** **PASS**

---

## 1. هدف این Validation

هدف Local Validation فاز 7 اثبات این بود که اجزایی که تا این مرحله به‌صورت جداگانه پیاده‌سازی و Unit Test شده‌اند، در اجرای واقعی و همزمان نیز به‌درستی با یکدیگر کار می‌کنند.

مسیر اصلی مورد بررسی:

```text
MockPerception
    ↓
GameState
    ↓
Internal Dynamics
    ↓
MetaState
    ↓
Strategist + Local Ollama
    ↓
Strategy
    ↓
ExecutorFSM
```

در کنار آن:

```text
Watchdog
```

به‌صورت یک process مستقل وضعیت runtime را مانیتور می‌کند.

مواردی که در این مرحله باید اثبات می‌شدند شامل موارد زیر بودند:

* راه‌اندازی صحیح تمام اجزای runtime
* اجرای همزمان loopها بدون deadlock
* تولید پیوسته GameState و MetaState
* اتصال واقعی به Ollama محلی
* تولید Strategy واقعی و معتبر
* ساخته‌شدن ExecutorFSM فقط پس از دریافت اولین Strategy
* به‌روزرسانی Strategy بدون recreate کردن FSM
* انتقال صحیح stateهای FSM
* عملکرد Watchdog
* graceful shutdown
* تولید report ساخت‌یافته توسط Scenario Runner
* حفظ consistency میان GameState، MetaState و progress
* جمع‌آوری داده‌های موردنیاز برای Phase 8

---

# 2. Task 7.1 — Async Application Pipeline

## 2.1 اولین اجرای exploratory

اولین اجرای مستقیم با دستور زیر انجام شد:

```powershell
uv run python -m wow_bot.main
```

در این اجرا تمامی اجزای اصلی runtime با موفقیت بالا آمدند:

```text
MemoryStore
Watchdog Process
Perception Loop
Dynamics Loop
Strategist Loop
Executor Loop
Watchdog Heartbeat Loop
Watchdog Shutdown Bridge
```

Watchdog نیز پس از startup وارد وضعیت زیر شد:

```text
INITIAL → HEALTHY
```

سپس Ollama محلی با موفقیت فراخوانی شد و Strategy اولیه تولید شد.

ExecutorFSM فقط پس از دریافت Strategy معتبر initialize شد که با contract طراحی‌شده برای Task 7.1 مطابقت داشت.

---

## 2.2 کشف رفتار نامناسب سناریوی پیش‌فرض برای Stability Validation

در اجرای مستقیم:

```text
python -m wow_bot.main
```

هیچ scenario مشخصی به runtime داده نمی‌شود.

در نتیجه:

```text
scenario=None
```

استفاده می‌شود.

بررسی MockPerception نشان داد profile پیش‌فرض شامل تمام event typeها، از جمله:

```text
DEATH
```

است.

در اجرای exploratory چند death event تصادفی تولید شد. پس از سومین death، Watchdog به‌درستی وارد وضعیت:

```text
HEALTHY → DEGRADED
reason = death_loop
```

شد.

پس از پنجمین death در همان sliding window، Watchdog وارد وضعیت:

```text
DEGRADED → CRITICAL
```

شد و shutdown اضطراری اما graceful را درخواست کرد. Main process درخواست shutdown را دریافت کرد، Controller متوقف شد، MemoryStore بسته شد و Watchdog به‌شکل clean خارج و join شد.

### نتیجه

این رفتار bug در Watchdog نبود.

برعکس، این اجرا به‌صورت عملی اثبات کرد که:

```text
Death event forwarding
Death-loop detection
DEGRADED transition
CRITICAL transition
Emergency shutdown bridge
Graceful cleanup
```

همگی درست کار می‌کنند.

اما profile پیش‌فرض برای baseline stability validation مناسب نیست، زیرا death تصادفی می‌تواند عمداً Watchdog را فعال کند.

بنابراین baseline رسمی Task 7.1 با scenario زیر انجام شد:

```text
peaceful_farm
```

---

# 3. Task 7.1 — Official 300-Second Acceptance

دستور اجرا:

```powershell
uv run python scripts/run_scenario.py \
  --scenario peaceful_farm \
  --duration 300 \
  --seed 42 \
  --output reports/validation_7_1_300s.json
```

اجرای واقعی:

```text
Requested Duration   300.000 s
Completed Duration   300.344 s
Completed Normally   true
Error                 none
Seed                  42
```

## 3.1 Pipeline Progress

در طول این اجرا:

```text
GameState Count       2624
MetaState Count       2624
Progress Steps        2624
Timing Samples        2624
```

این تطابق یک invariant مهم را تأیید می‌کند:

```text
GameState
    ↓
Dynamics processing
    ↓
MetaState
    ↓
Progress increment
```

هیچ evidenceای از توقف Dynamics یا از دست رفتن snapshotها مشاهده نشد.

---

## 3.2 Strategist

در پنج دقیقه:

```text
LLM Query Count       4
Generation Attempts   4
New Strategies        4
Fallbacks             0
```

هر چهار Strategy با موفقیت تولید شدند.

---

## 3.3 FSM

از آنجا که:

```text
peaceful_farm
```

عمداً combat ندارد، رفتار مورد انتظار FSM این بود:

```text
IDLE → SCANNING
```

و سپس باقی‌ماندن در وضعیت عادی scanning.

نتیجه نیز همین بود:

```text
FSM Transition Count  1
Deaths                0
```

---

## 3.4 Watchdog

در کل اجرای پنج‌دقیقه‌ای:

```text
Watchdog Health Transitions   0
Shutdown Requested            false
```

هیچ false-positive مربوط به heartbeat، progress، death-loop یا resource monitoring رخ نداد.

---

## 3.5 Graceful Shutdown

در انتهای دقیق bounded runtime:

```text
300s reached
    ↓
graceful shutdown requested
    ↓
Controller.stop_all()
    ↓
MemoryStore close
    ↓
Watchdog exits cleanly
    ↓
Watchdog join
    ↓
Scenario completed normally
```

انجام شد و هیچ exception غیرمنتظره‌ای مشاهده نشد.

### Task 7.1 Result

```text
Startup                     PASS
Local Ollama                PASS
Perception                  PASS
Dynamics                    PASS
Strategist                  PASS
ExecutorFSM                 PASS
Watchdog                    PASS
300-second Stability        PASS
Graceful Shutdown           PASS

Task 7.1 Local Acceptance   PASS
```

---

# 4. Task 7.2 — Scenario Runner

Task 7.2 برای اجرای کنترل‌شده scenarioها و تولید evidence ساخت‌یافته طراحی شده است.

Report مورد انتظار شامل بخش‌های زیر است:

```text
run
summary
strategist
fsm
meta_state
timing
watchdog
```

Validation در دو مرحله انجام شد:

```text
60-second smoke
600-second acceptance
```

scenario مورد استفاده:

```text
combat_light
```

seed:

```text
42
```

---

# 5. Task 7.2 — 60-Second Smoke Test

دستور:

```powershell
uv run python scripts/run_scenario.py \
  --scenario combat_light \
  --duration 60 \
  --seed 42 \
  --output reports/validation_7_2_combat_60s.json
```

نتیجه:

```text
Requested Duration   60.000 s
Completed Duration   60.406 s
Completed Normally   true
Error                 none
```

---

## 5.1 FSM Coverage

در همین اجرای کوتاه چند مسیر مهم FSM به‌صورت واقعی پوشش داده شدند:

```text
IDLE
→ SCANNING
→ COMBAT
→ LOOTING
→ SCANNING
→ FLEEING
→ SCANNING
```

در مجموع:

```text
FSM Transitions   6
```

ثبت شد.

یکی از مهم‌ترین validationها مربوط به FLEE threshold بود.

نمونه مشاهده‌شده:

```text
hp_pct = 0.30
risk_tolerance = 0.50
```

فرمول:

```text
threshold = 0.60 - 0.40 × risk_tolerance

threshold = 0.40
```

بنابراین:

```text
0.30 <= 0.40
```

و FSM به‌درستی وارد:

```text
FLEEING
```

شد.

---

## 5.2 Runtime Counters

نتیجه report:

```text
GameState         377
MetaState         377
Progress Steps    377
Timing Samples    377

FSM Transitions   6
LLM Queries       4
Strategies        3
Deaths            0
```

تفاوت:

```text
LLM Queries = 4
Strategies  = 3
```

نیز رفتار صحیح بود.

چهارمین LLM generation در لحظه پایان bounded run هنوز in-flight بود و هنگام shutdown cancel شد.

بنابراین metric مربوط به:

```text
logical query attempts
```

به‌درستی attempt را ثبت کرده است، حتی اگر completion قبل از پایان runtime دریافت نشده باشد.

---

## 5.3 Watchdog

در اجرای 60 ثانیه:

```text
health_transition_count = 0
shutdown_requested      = false
```

---

# 6. Task 7.2 — 600-Second Acceptance

دستور:

```powershell
uv run python scripts/run_scenario.py \
  --scenario combat_light \
  --duration 600 \
  --seed 42 \
  --output reports/validation_7_2_combat_600s.json
```

این run معیار اصلی Task 7.2 بود.

نتیجه:

```text
Requested Duration   600.000 s
Completed Duration   600.344 s
Completed Normally   true
Error                 none
Scenario              combat_light
Seed                  42
```

در کل run هیچ runtime error یا Watchdog shutdown مشاهده نشد.

---

# 7. Pipeline Consistency در Run ده‌دقیقه‌ای

مقادیر report:

```text
GameState Count       5571
MetaState Count       5571
Progress Steps        5571
Timing Samples        5571
```

این تطابق نشان می‌دهد که در کل runtime:

```text
Perception
Dynamics
Reporting
Progress Tracking
Timing Instrumentation
```

با یکدیگر consistent باقی مانده‌اند.

سایر metricها:

```text
FSM Transitions       72
Idle Intents          307
LLM Queries           13
Strategies            13
Deaths                0
```

---

# 8. FSM Behaviour در Run ده‌دقیقه‌ای

تعداد ورود به stateها:

```text
SCANNING   26
COMBAT     21
LOOTING    19
FLEEING     6
```

در مجموع:

```text
72 transitions
```

ثبت شد.

چندین چرخه کامل:

```text
SCANNING
→ COMBAT
→ LOOTING
→ SCANNING
```

مشاهده شد.

همچنین چندین بار مسیر:

```text
SCANNING / COMBAT
→ FLEEING
→ SCANNING
```

بر اساس HP و Strategy risk tolerance فعال شد.

بنابراین Task 7.2 عملاً مسیرهای اصلی FSM مربوط به combat scenario را پوشش داده است.

---

# 9. Strategist Behaviour

در اجرای 600 ثانیه:

```text
Generation Attempts   13
New Strategies        13
Fallbacks              0
```

توزیع goalها:

```text
explore       9
flee          3
farm_herbs    1
```

نرخ تقریبی replanning:

```text
13 calls / 600 seconds
≈ one query every 46 seconds
```

است.

این مقدار فعلاً failure محسوب نمی‌شود.

Triggerها عمدتاً adaptive بوده‌اند و evidence فعلی برای تغییر threshold یا Dynamics کافی نیست.

این موضوع به‌عنوان observation برای تحلیل‌های بعدی حفظ می‌شود.

---

# 10. Timing Data

برای run ده‌دقیقه‌ای:

```text
Sample Count   5571
Minimum        50.0 ms
Maximum        969.20 ms
Mean           217.60 ms
Std Dev        93.66 ms
```

ضریب تغییرات خام:

```text
CV = std / mean
   ≈ 0.430
```

است.

بنابراین به‌صورت preliminary:

```text
CV > 0.3
```

برقرار است.

با این حال این نتیجه هنوز acceptance رسمی Task 8.2 نیست.

Task 8.2 علاوه بر CV نیازمند:

```text
conditional randomized PIT
KS test
clipping-aware analysis
```

است.

---

# 11. MetaState Sampling

Dimension order در تمام reportها مطابق contract اصلی باقی ماند:

```text
hunger
fatigue
curiosity
aggression
social
```

timestampها نیز strictly increasing بودند.

در run ده‌دقیقه‌ای تنها یک gap بزرگ در ابتدای runtime مشاهده شد:

```text
1789894245.5362215
        ↓
1789894254.7959273
```

فاصله:

```text
≈ 9.26 seconds
```

بود.

بعد از آن sampling تقریباً در cadence معمول:

```text
~0.1 second
```

ادامه پیدا کرد.

در smoke شصت‌ثانیه‌ای نیز همین الگو وجود داشت؛ یک gap بزرگ در bootstrap و سپس sampling پایدار.

### Interpretation

محتمل‌ترین علت این gap انتظار برای Strategy اولیه و backpressure در startup است.

این موضوع فعلاً bug تأییدشده محسوب نمی‌شود، زیرا:

* فقط یک بار در bootstrap رخ می‌دهد؛
* در runtime پایدار تکرار نمی‌شود؛
* Dynamics پس از bootstrap به‌شکل مستمر پیشرفت می‌کند؛
* Task 8.1 صراحتاً irregular sampling را تشخیص می‌دهد و قبل از spectrum analysis resample انجام می‌دهد.

بنابراین فعلاً هیچ تغییر معماری برای این observation انجام نمی‌شود.

---

# 12. Watchdog Validation

در run ده‌دقیقه‌ای:

```text
health_transition_count = 0
shutdown_requested      = false
```

یعنی موارد زیر باعث false-positive نشدند:

```text
LLM latency
Combat transitions
FLEEING
Long-running async execution
Idle behaviour
Adaptive strategy refresh
```

در پایان 600 ثانیه:

```text
duration reached
→ graceful cleanup
→ Controller.stop_all()
→ MemoryStore close
→ Watchdog clean exit
→ Watchdog join
→ Scenario completed normally
→ report persisted
```

انجام شد.

---

# 13. Windows Ctrl+C Observation

در یکی از اجراهای دستی اولیه روی Windows، هنگام استفاده از:

```text
Ctrl+C
```

child process مربوط به Watchdog نیز console interrupt را دریافت کرد و یک:

```text
KeyboardInterrupt
```

در child process دیده شد.

Main process با این وجود cleanup خود را کامل انجام داد.

در bounded scenario runs این مشکل وجود نداشت و Watchdog در پایان runtime به‌صورت کاملاً clean shutdown شد.

بنابراین این مورد:

```text
Non-blocking Windows CLI polish issue
```

طبقه‌بندی می‌شود و blocker برای Task 7.1 یا Task 7.2 نیست.

در صورت نیاز در یک task کوچک مستقل می‌توان SIGINT ownership را برای child Watchdog اصلاح کرد.

---

# 14. Findings Summary

## Confirmed PASS

```text
Async pipeline startup
Concurrent loop execution
Local Ollama integration
Initial Strategy bootstrap
Adaptive Strategy refresh
FSM initialization ordering
FSM Strategy update without recreation
Combat transitions
Looting transitions
Flee transitions
Dry-run Controller boundary
Progress tracking
GameState / MetaState consistency
Timing instrumentation
Scenario report generation
Watchdog heartbeat monitoring
Watchdog death-loop detection
Watchdog emergency shutdown
Bounded graceful shutdown
Resource cleanup
Watchdog clean process exit
```

---

## Non-Blocking Observations

### A. Default scenario is unsuitable for baseline stability validation

```text
scenario=None
```

می‌تواند eventهای death تصادفی تولید کند.

این profile برای exploratory behaviour مناسب است ولی برای baseline 300-second acceptance نباید استفاده شود.

Baseline رسمی باید scenario مشخص داشته باشد.

---

### B. Single bootstrap sampling gap

یک gap بزرگ فقط هنگام startup مشاهده می‌شود.

در run اصلی 600 ثانیه‌ای:

```text
≈ 9.26 s
```

بود.

پس از bootstrap sampling پایدار است.

فعلاً fix لازم نیست.

---

### C. Adaptive replanning frequency

در run اصلی:

```text
13 queries / 600 s
```

یعنی تقریباً:

```text
1 query / 46 s
```

این عدد ثبت می‌شود ولی در این مرحله مبنایی برای تغییر adaptive trigger نیست.

---

### D. Interactive Ctrl+C on Windows

child Watchdog ممکن است مستقیماً console SIGINT دریافت کند.

Bounded shutdown سالم است و issue فعلاً blocker محسوب نمی‌شود.

---

# 15. Changes Explicitly Not Made

بر اساس validation فعلی هیچ دلیلی برای تغییر موارد زیر وجود ندارد:

```text
Watchdog death thresholds
Watchdog death window
FSM flee threshold
Strategy TTL
Adaptive trigger threshold
Internal Dynamics parameters
Mock scenario probabilities
```

Validation باید evidence تولید کند، نه اینکه مدل را صرفاً برای pass شدن تست‌ها tune کند.

---

# 16. Formal Validation Status

## Task 7.1

```text
Implementation Review        PASS
Local Runtime Boot           PASS
Local Ollama                 PASS
300s Stability               PASS
Watchdog                     PASS
Graceful Shutdown            PASS

STATUS:
LOCAL ACCEPTANCE PASS
```

## Task 7.2

```text
60s combat_light Smoke       PASS
600s combat_light Run        PASS
Structured Report            PASS
FSM Instrumentation          PASS
Strategist Instrumentation   PASS
MetaState Instrumentation    PASS
Timing Instrumentation       PASS
Watchdog Instrumentation     PASS

STATUS:
CORE LOCAL ACCEPTANCE PASS
```

---

# 17. Remaining Phase 7 Coverage

طبق Local Validation Roadmap اولیه، یک مورد coverage هنوز به‌صورت رسمی اجرا نشده است:

```text
Scenario Matrix
```

سناریوهای باقی‌مانده:

```text
death_loop
rare_loot_drought
stuck_repeatedly
```

در حال حاضر evidenceهای مهمی از death-loop به‌صورت incidental در اجرای generic scenario داریم، اما این جای یک scenario-matrix رسمی و reproducible را نمی‌گیرد.

بنابراین وضعیت دقیق فاز 7 بهتر است به شکل زیر ثبت شود:

```text
Phase 7 Core Local Acceptance: PASS

Extended Scenario Matrix:
PENDING
```

این مورد blocker برای ورود به Task 8.1 نیست، زیرا Task 8.1 به یک report واقعی و معتبر Task 7.2 نیاز دارد و اکنون report ده‌دقیقه‌ای معتبر زیر موجود است:

```text
reports/validation_7_2_combat_600s.json
```

---

# 18. Evidence Artifacts

Artifacts اصلی این validation:

```text
reports/validation_7_1_300s.json

reports/validation_7_2_combat_60s.json

reports/validation_7_2_combat_600s.json
```

مهم‌ترین artifact برای Phase 8:

```text
reports/validation_7_2_combat_600s.json
```

این report شامل:

```text
5571 GameState observations
5571 MetaState observations
5571 timing samples
72 FSM transitions
13 LLM queries
13 Strategies
307 idle intents
```

است و برای Spectrum Analysis و Timing Distribution Analysis قابل استفاده است.

---

# 19. Phase 8 Readiness

از نظر integration هیچ blocker شناخته‌شده‌ای برای شروع Task 8.1 وجود ندارد.

ورودی پیشنهادی:

```text
reports/validation_7_2_combat_600s.json
```

Task بعدی:

```text
8.1 — FFT / Spectrum Analysis
```

هدف اصلی Phase 8.1 این خواهد بود که مشخص شود آیا مشکل علمی مشاهده‌شده در Task 3.7:

```text
PSD slope ≈ -6
```

روی داده واقعی pipeline نیز تکرار می‌شود یا فقط مربوط به synthetic acceptance test قبلی بوده است.

تا قبل از گرفتن این evidence نباید Internal Dynamics برای نزدیک‌شدن مصنوعی به target تغییر داده شود.

---

# Final Conclusion

Local Validation فاز 7 نشان داد که معماری integrated سیستم در اجرای واقعی محلی پایدار است.

Task 7.1 پنج دقیقه و Task 7.2 ده دقیقه به‌صورت موفق اجرا شدند. Pipeline در طول اجرای اصلی هیچ crash، Watchdog false-positive، resource lifecycle failure یا state inconsistency نشان نداد.

مهم‌ترین نتیجه فاز 7 این است که اکنون یک evidence bundle واقعی داریم که می‌توان Phase 8 را بر اساس آن انجام داد، نه بر اساس synthetic assumptions.

```text
Task 7.1                     PASS
Task 7.2 Core Acceptance     PASS

Phase 7 Core Validation      PASS
Extended Scenario Matrix     PENDING

Ready for Task 8.1           YES
```
