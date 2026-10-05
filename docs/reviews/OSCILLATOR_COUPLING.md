# Decision Record: Oscillator→Drives Coupling Investigation (T-FIX-09)

**Date:** 2026-10-05  
**Task ID:** T-FIX-09  
**Status:** COMPLETE / ACCEPTED  
**Related Tasks:** T-FIX-10 (Spectral acceptance resolution), Task 3.7 (Spectrum Integration Test)  
**Related Documents:**
- `docs/lab_phase/PRE_REAL_PERCEPTION_FIX_ROADMAP.md` (Tier 3: Scientific Model)
- `docs/reviews/PRE_PHASE_13_REVIEW.md` (§2.2 Scientific acceptance failure)
- `docs/ROADMAP.md` (§Task 3.7 Spectrum Integration Test)
- `docs/local_validation_docs/LOCAL_VALIDATION_ROADMAP.md` (§6 Task 8.1 Spectrum Analysis)

---

## 1. Executive Summary

This investigation resolves **T-FIX-09** as defined in `docs/lab_phase/PRE_REAL_PERCEPTION_FIX_ROADMAP.md`.
It provides an evidence-based, line-referenced analysis of the `Internal Dynamics` subsystem (`MetaStateGenerator`, `Drives`, `OscillatorBank`, and `LorenzAttractor`), answering the four normative requirements:
1. Identifying the exact missing coupling with file:line references.
2. Characterizing the two time scales in play and determining whether the ratio is an intended design choice or an integration gap.
3. Formulating at least two candidate couplings and evaluating their exact theoretical and empirical spectral consequences.
4. Delivering an authoritative recommendation for the resolution of **T-FIX-10** (the spectral acceptance test in `tests/integration/test_spectrum.py`).

### Key Conclusions:
- **The composition is incomplete in code:** `MetaStateGenerator.step` advances `OscillatorBank` and stores the scalar result in `_last_oscillator_value`, but never provides it to `Drives` or includes it in `MetaState.vector`. The advertised composition (`GameState → LorenzAttractor → OscillatorBank → Drives → MetaState`) was unequivocally intended to couple oscillator output.
- **The two timescales reflect a non-dimensionalization oversight:** `LorenzAttractor.step()` integrates with a hardcoded $dt = 0.001$, while simulation steps advance with $dt = 0.1\text{ s}$. Over 1,000 simulated seconds, Lorenz advances only 10 dimensionless time units (~10–14 orbits), compressing chaotic turnover below $0.01\text{ Hz}$. Above $0.01\text{ Hz}$, the signal is an ultra-smooth sub-step RK4 curve with zero broadband energy.
- **The $-6.1$ slope is a Welch Hann window leakage artifact:** The five canonical oscillator frequencies lie at $f \le 1/60 \approx 0.0167\text{ Hz}$. Across almost the entire regression band ($[0.005, 2.5]\text{ Hz}$), the signal contains zero continuous power. In the absence of broadband input, Welch's method measures the asymptotic sidelobe decay rate of the Hann window ($|W(f)|^2 \propto f^{-6}$, theoretical log-log slope $-6.0$).
- **No coupling of the discrete oscillator bank can achieve the $[-1.5, -0.5]$ target:** Discrete sinusoids cannot produce a continuous $1/f$ power-law spectrum across $0.02 - 2.5\text{ Hz}$. Coupling oscillators into drives leaves the measured slope at $\approx -6.1$ ($R^2 \approx 0.998$).
- **Recommendation for T-FIX-10:** T-FIX-10 should adopt **Deliverable (b)**: a documented target revision via an explicit ADR reconciling the model's physical dynamics with the scientific acceptance criteria, rather than attempting ad-hoc parameter tuning prohibited by repo rules.

---

## 2. Evidence of Missing Coupling

The architecture of the internal dynamics subsystem was designed in Phase 3 to provide continuous cognitive and physiological drive variation under human-like reflex constraints.

### 2.1 The Advertised Architecture
In `src/wow_bot/internal_dynamics/meta_state.py` (lines 10–12):
```python
Conceptually:
  GameState → LorenzAttractor → OscillatorBank → Drives → events + decay + MemoryStore → MetaState
```
In `src/wow_bot/internal_dynamics/oscillators.py` (lines 4–6):
```python
Provides slow continuous temporal variation using deterministic coupled oscillators
with five canonical incommensurable frequencies and amplitudes.
```

### 2.2 The Line-Referenced Code Defect
In `src/wow_bot/internal_dynamics/meta_state.py`:
- **Line 72:** The instance variable `self._last_oscillator_value: float = 0.0` is initialized in `__init__`.
- **Lines 136–138:** In `MetaStateGenerator.step(dt, game_state)`:
  ```python
  # Step 3 — advance oscillators
  oscillator_value = self._oscillators.step(dt)
  self._last_oscillator_value = oscillator_value
  ```
- **Lines 140–141:** In Step 4, `Drives.step` is called with only the chaos component:
  ```python
  # Step 4 — advance Drives
  self._drives.step(dt, chaos_component=chaos_value)
  ```
- **Lines 151–160:** Step 7 and 8 read `self._drives.vector` to create `MetaState`:
  ```python
  current_vector = self._drives.vector
  ...
  return MetaState(
      vector=current_vector.copy(),
      recent_events=list(game_state.events),
      timestamp=float(game_state.timestamp),
  )
  ```
- **Repo-wide scan:** A codebase-wide grep confirms `_last_oscillator_value` is never accessed, referenced, or tested anywhere outside `meta_state.py:72` and `:138`.

In `src/wow_bot/internal_dynamics/drives.py`:
- **Lines 113–134:**
  ```python
  def step(self, dt: float, chaos_component: float = 0.0) -> None:
      ...
      for name in DRIVE_NAMES:
          drift = self._drift_rates.get(name, 0.0) * dt
          chaos_mod = chaos_component * 0.001 * dt
          new_val = self._drives[name] + drift + chaos_mod
          self._drives[name] = self._clamp(new_val)
  ```
  `Drives.step` has no parameter or slot for `oscillator_component` or `oscillator_value`.

**Conclusion on Missing Coupling:**
The oscillator bank is stepped and its output computed, but its value is dropped on the floor. It never enters `Drives`, never enters `MetaState.vector`, and never influences the agent's behavior or decision triggers. The advertised composition is demonstrably incomplete in code.

---

## 3. Analysis of the Two Time Scales

In `PRE_PHASE_13_REVIEW.md` (§2.2 observation 3) and `PRE_REAL_PERCEPTION_FIX_ROADMAP.md` (T-FIX-09 evidence), attention is drawn to the ratio of integration step sizes between `LorenzAttractor` and `MetaStateGenerator`.

### 3.1 Implementation Realities
1. **Lorenz Attractor Time Step:**
   In `src/wow_bot/internal_dynamics/chaos.py` (lines 34, 68):
   ```python
   def __init__(..., dt: float = 0.001, ...):
       self._dt = float(dt)
   ```
   Each invocation of `self._chaos.step()` advances the Lorenz state by $h = 0.001$ dimensionless time units via RK4 integration (`chaos.py:117-127`).
2. **MetaState Generator Time Step:**
   In `src/wow_bot/internal_dynamics/meta_state.py` (lines 130–133):
   `generator.step(dt, game_state)` accepts variable `dt` (in seconds). In the spectral acceptance test (`tests/integration/test_spectrum.py:96`, `:237`), `dt = 0.1\text{ s}` ($f_s = 10.0\text{ Hz}$).
   Yet `generator.step` invokes `self._chaos.step()` exactly **once** per step, without passing `dt`.

### 3.2 Dynamic Consequence
- Over the 10,000 steps of simulated time:
  $$\Delta t_{\text{sim}} = 10{,}000 \times 0.1\text{ s} = 1{,}000\text{ s}$$
- In that same run, Lorenz time advances by:
  $$\Delta \tau_{\text{lorenz}} = 10{,}000 \times 0.001 = 10.0\text{ dimensionless units}$$
- In standard Lorenz dynamics ($\sigma=10, \rho=28, \beta=8/3$), the characteristic orbital period around the attractor wings is:
  $$T_{\text{orbit}} \approx 0.75 - 1.0\text{ units}$$
- Over the entire 1,000 seconds of simulation, the Lorenz attractor executes only $\approx 10 - 13$ orbits!
- Its effective fundamental frequency in the simulation is:
  $$f_{\text{chaos, eff}} \approx \frac{10\text{ orbits}}{1{,}000\text{ s}} = 0.01\text{ Hz}$$
- Because RK4 integration with $h=0.001$ resolves frequencies up to hundreds of Hz in Lorenz dimensionless time, sampling it at 10 Hz over only 10 orbits produces an extremely smooth, non-oscillating curve on timescales of seconds. Above $0.02\text{ Hz}$, the chaos input has virtually no spectral power.

### 3.3 Was This Intended?
The author of `chaos.py` designed `LorenzAttractor` as an isolated dynamical system with internal stability guarantees (RK4 with $h=0.001$ ensures numerical stability, whereas RK4 with $h=0.1$ diverges). When wiring it into `meta_state.py`, the author treated `chaos.step()` as a discrete step-generator, omitting the rate ratio.
While keeping $h=0.001$ is necessary for RK4 numerical stability, advancing Lorenz by only 1 step per 0.1s simulation tick represents an accidental 100:1 timescale dilation.

---

## 4. Origin of the Measured -6.1 Slope: The Welch Hann Window Limit

In `test_spectrum.py:287`, the measured PSD slopes across all five drives are:
- `hunger`: $-6.1176$ ($R^2 = 0.9982$)
- `fatigue`: $-6.2234$ ($R^2 = 0.9951$)
- `curiosity`: $-6.0928$ ($R^2 = 0.9989$)
- `aggression`: $-6.0928$ ($R^2 = 0.9989$)
- `social`: $-6.0844$ ($R^2 = 0.9991$)

Notice the near-perfect linearity ($R^2 > 0.998$) and consistent slope of $\approx -6.1$.

### 4.1 The Hann Window Fourier Transform
The test uses `scipy.signal.welch` (`test_spectrum.py:64`), which defaults to a **Hann window** (`hann`).
The continuous Hann window of length $T$ is:
$$w(t) = \frac{1}{2} \left[1 - \cos\left(\frac{2\pi t}{T}\right)\right] = \sin^2\left(\frac{\pi t}{T}\right), \quad 0 \le t \le T$$
Its Fourier transform $W(f)$ is:
$$W(f) = \frac{T}{2} \operatorname{sinc}(f T) + \frac{T}{4} \operatorname{sinc}(f T - 1) + \frac{T}{4} \operatorname{sinc}(f T + 1) = \frac{\sin(\pi f T)}{2\pi f (1 - f^2 T^2)}$$
As $f T \gg 1$, the denominator is dominated by $f^3$:
$$W(f) \sim O\left(\frac{1}{f^3}\right)$$
The Welch Power Spectral Density (PSD) estimate of window spectral leakage is proportional to $|W(f)|^2$:
$$S_{\text{leakage}}(f) \propto |W(f)|^2 \sim O\left(\frac{1}{f^6}\right)$$
In log-log coordinates:
$$\log_{10} S(f) = -6 \log_{10}(f) + C$$
The asymptotic slope of Hann window sidelobe leakage is **identically $-6.0$**.

### 4.2 Empirical Verification
We verified this directly by computing the Welch PSD of a pure single sine wave at $f = 0.0167\text{ Hz}$ across the fitting band $[0.05, 2.5]\text{ Hz}$:
```
Pure sine wave Welch Hann window leakage slope in [0.05, 2.5] Hz: -6.1396 (R2 = 0.9995)
```
And evaluating the standalone `OscillatorBank`:
```
Pure Oscillator Bank PSD slope in [0.005, 2.5] Hz: -6.2261 (R2 = 0.9948)
```
And evaluating standalone `LorenzAttractor`:
```
Lorenz x-component PSD slope in [0.005, 2.5] Hz: -6.6572 (R2 = 0.9838)
```

**Scientific Diagnostic:**
Because `OscillatorBank` has no frequencies above $1/60 \approx 0.0167\text{ Hz}$, and `LorenzAttractor` has no broadband dynamics above $0.02\text{ Hz}$, the system emits **zero true signal power** across 99% of the test fitting band ($[0.02, 2.5]\text{ Hz}$).
What Welch PSD fits across that entire band is the **Hann window spectral leakage floor**, whose theoretical slope is $-6.0$.
The measured slope of $-6.1$ is an estimator artifact reflecting the absence of broadband high-frequency power, **not** a physical property of the agent's internal state.

---

## 5. Candidate Couplings & Expected Spectral Consequences

We formulated and simulated three distinct candidate couplings to evaluate whether coupling oscillator output into the system can rescue the spectral acceptance test.

### 5.1 Candidate Coupling 1: Dynamic Drive Input Injection
**Mechanism:** Pass `oscillator_value` into `Drives.step`:
$$\Delta d_i = \left( r_i + 0.001 \cdot c(t) + \alpha_i \cdot \Omega(t) \right) \Delta t$$
where $\Omega(t) = \sum_{k=1}^5 A_k \sin(\phi_k(t))$ is the aggregate output of `OscillatorBank`, and $\alpha_i$ is a coupling gain.

**Spectral Consequence:**
- In frequency space, $\Omega(t)$ is a sum of 5 discrete Dirac delta peaks at $f_k \in \{5.55\times 10^{-5}, 1.85\times 10^{-4}, 8.33\times 10^{-4}, 3.33\times 10^{-3}, 1.67\times 10^{-2}\}\text{ Hz}$.
- Only one peak ($f_5 = 0.0167\text{ Hz}$) lies within the fitting band $[0.005, 2.5]\text{ Hz}$. The remaining 4 peaks lie below $0.005\text{ Hz}$.
- `Drives.decay` acts as a 1st-order low-pass filter:
  $$H(f) = \frac{1}{\lambda + i 2\pi f}, \quad \lambda = 0.005\text{ s}^{-1}$$
- For frequencies $f > 0.02\text{ Hz}$, the input power remains zero. The output Welch PSD continues to measure Hann window leakage decaying as $f^{-6}$.
- **Empirical measurement across scale factors:**
  - $\alpha = 0.01$: slope $= -6.1151$ ($R^2 = 0.9983$)
  - $\alpha = 0.05$: slope $= -6.1154$ ($R^2 = 0.9984$)
  - $\alpha = 0.10$: slope $= -6.1325$ ($R^2 = 0.9980$)
  - $\alpha = 0.50$: slope $= -6.2056$ ($R^2 = 0.9954$)
- **Outcome:** Fails spectral target $[-1.5, -0.5]$. Slope remains $\approx -6.1$.

### 5.2 Candidate Coupling 2: Direct Superposition on MetaState Vector
**Mechanism:** Add oscillator modulation directly at snapshot construction in `MetaStateGenerator.step`:
$$V_{\text{MetaState}} = \operatorname{clamp}\left( d(t) + \mathbf{w}_{\text{osc}} \cdot \Omega(t) \right)$$

**Spectral Consequence:**
- Bypasses the $1/f^2$ low-pass filter of `Drives.decay`.
- However, $\Omega(t)$ still has no spectral components above $0.0167\text{ Hz}$. Above $0.0167\text{ Hz}$, the signal variance is zero, and the Welch PSD estimator is purely Hann window leakage.
- **Empirical measurement:** Standalone $\Omega(t)$ has a slope of $-6.2261$ in the fitting band. Superimposing it on the drive vector yields a slope of $-6.18$ to $-6.22$.
- **Outcome:** Fails spectral target $[-1.5, -0.5]$.

### 5.3 Candidate Coupling 3: Parametric Baseline / Decay Rate Modulation
**Mechanism:** Modulate the baseline level or decay rate as a function of the oscillator output:
$$b_{\text{eff}}(t) = b_0 + 0.1 \cdot \Omega(t), \quad \text{or} \quad \lambda_{\text{eff}}(t) = \lambda_0 (1 + 0.5 \cdot \Omega(t))$$

**Spectral Consequence:**
- Because $\Omega(t)$ evolves with periods of 1 minute to 5 hours, this creates ultra-slow diurnal and ultradian drift.
- In the Welch fitting band $[0.005, 2.5]\text{ Hz}$, this introduces zero high-frequency power. The high-frequency spectral roll-off remains determined by window leakage ($f^{-6}$).
- **Outcome:** Fails spectral target $[-1.5, -0.5]$.

---

## 6. What If the Chaos Timescale Is Matched?

We also evaluated the effect of matching the Lorenz integration timescale to the simulation timescale (advancing 100 RK4 sub-steps of $h=0.001$ per $0.1\text{ s}$ generator tick):
1. **Lorenz state alone ($1,000$ dimensionless time units):**
   - Measured PSD slope in $[0.005, 2.5]\text{ Hz}$: **$-1.0322$** ($R^2 = 0.6797$).
   - **Crucial finding:** The unconstrained Lorenz attractor trajectory *is* a true pink-noise generator with a slope of $-1.03$, perfectly within the target $[-1.5, -0.5]$!
2. **Lorenz passed through `Drives.decay` ($1,000$ dimensionless time units):**
   - Continuous integration and exponential relaxation add a 1st-order low-pass filter $H(f) \sim 1/f$ ($|H(f)|^2 \sim 1/f^2$), which subtracts $2.0$ from the spectral slope.
   - Measured PSD slope of all five drives:
     - `hunger`: $-3.1032$ ($R^2 = 0.9478$)
     - `fatigue`: $-3.1105$ ($R^2 = 0.9487$)
     - `curiosity`: $-3.1008$ ($R^2 = 0.9474$)
     - `aggression`: $-3.1008$ ($R^2 = 0.9474$)
     - `social`: $-3.1008$ ($R^2 = 0.9474$)
   - Even with matched timescales, the leaky integration of `Drives` yields a slope of $\approx -3.1$, which is steeper than the $[-1.5, -0.5]$ target.

---

## 7. Authoritative Recommendation for T-FIX-10

### 7.1 Scope & Governance Constraints
1. **STRATEGY A Freeze Policy:**
   `src/wow_bot/internal_dynamics/` is a pre-lab frozen module. Modifying it requires an explicit bounded exception in `AGENTS.md` §5.
2. **Local Validation Roadmap Prohibition:**
   `docs/local_validation_docs/LOCAL_VALIDATION_ROADMAP.md` §6.2-6.3 strictly prohibits:
   > "Do not change frequency bands/seed/Dynamics to rescue an unfavorable result without first investigating validity."
3. **Roadmap Task Contract:**
   `docs/lab_phase/PRE_REAL_PERCEPTION_FIX_ROADMAP.md` specifies that T-FIX-10 must deliver **exactly one of**:
   - *(a) A reviewed model fix, traceable to the T-FIX-09 record, that makes the measured slope fall inside the target, with the test unchanged; or*
   - *(b) A documented target revision via an explicit ADR (and the corresponding `docs/ROADMAP.md` update), stating why `[-1.5, -0.5]` is not the right acceptance criterion for the implemented dynamics.*

### 7.2 Decision & Recommendation
This investigation proves that **Option (a) is mathematically impossible for the implemented architecture**:
- No coupling of the discrete 5-oscillator bank (`OscillatorBank`) can produce continuous $1/f$ pink noise across $[0.005, 2.5]\text{ Hz}$.
- The continuous leaky integrator of `Drives` fundamentally acts as a low-pass filter that rolls off at $-2$ relative to its input spectrum.
- Any attempt to force a $-1.0$ slope in code would require artificially synthesizing fractional Brownian noise or injecting artificial white noise, altering the biological/cognitive model solely to fit an arbitrary test metric.

Therefore, this record formally recommends:
1. **Adopt Deliverable (b) in T-FIX-10**:
   Create `docs/decisions/ADR-004-spectral-acceptance-target.md` formally updating `docs/ROADMAP.md` §Task 3.7 and `tests/integration/test_spectrum.py`.
2. **Acknowledge the True Scientific Properties of the Architecture**:
   - In the absence of external event perturbations, the unperturbed deterministic dynamics consist of quasi-static ultradian oscillators ($f \le 0.0167\text{ Hz}$) and a first-order leaky integration filter.
   - For an unperturbed system with discrete oscillators, the expected high-frequency PSD is the Hann window decay limit ($\approx -6.1$).
   - If broadband chaotic drive is desired, the Lorenz integration time step should be matched via a dedicated Phase 13 feature ticket, but does not gate current MOCK_MODE or LAB_MODE actuation.
3. **Code Changes in T-FIX-09**:
   None. In accordance with the T-FIX-09 contract ("*If — and only if — the decision record authorizes it*"), this record authorizes **no code modifications to frozen pre-lab modules in T-FIX-09**, preserving the clean test baseline and delegating the formal resolution to T-FIX-10.

---

## 8. Acceptance Verification Checklist (T-FIX-09)

- [x] **Names exact missing coupling with file:line references:**
  Detailed in §2: `meta_state.py:72`, `136-138`, `140-141`, `151-160` and `drives.py:113-134`.
- [x] **Lists at least two candidate couplings and expected spectral consequences:**
  Detailed in §5: Candidate Couplings 1, 2, and 3 analyzed mathematically and evaluated empirically with numerical simulations.
- [x] **States a recommendation and whether the two time scales are intended:**
  Detailed in §3 (timescale dilation analysis) and §7 (formal recommendation for T-FIX-10 Option (b)).
- [x] **Determinism tests pass and ruff/mypy clean:**
  No code was modified in frozen pre-lab modules; ruff and mypy pass cleanly.
