# Clinical Specification — Python-Model-OT-Control

**Version**: 1.1.0
**Date**: 26 April 2026
**Status**: Specification of the engine's clinical use as currently
shipped. SWAT is the only implemented adapter; FSA-high-res is planned.

---

## 1. What the engine does, in clinical terms

Given a calibrated dynamical model of a patient (e.g. SWAT for
sleep-wake-testosterone) and a clinical target (e.g. "restore healthy
testosterone pulsatility"), the engine produces an **optimal daily
schedule of intervention controls** over a multi-week horizon.

The schedule is the set of daily values for the model's intervention
parameters — for SWAT, that is $(V_h, V_n, V_c)$ per day, i.e. how much
to push vitality, reduce chronic load, and shift circadian phase on
each day of the prescribed horizon.

The engine outputs:

1. **A schedule** — a $D \times n_c$ table of daily control values,
   exportable to CSV.
2. **A closed-loop verification** — Monte-Carlo simulation under the
   schedule confirming the patient's trajectory reaches the target
   distribution.
3. **Comparison with baselines** — the optimised schedule benchmarked
   against zero control, constant reference, and linear-interpolation
   schedules, on the same metrics.

This is decision-support output. Final treatment decisions remain with
the clinician.

---

## 2. The clinical question

> *Given this patient's current physiological state and a 14- to
> 21-day window, which sequence of interventions would most reliably
> restore their testosterone pulsatility (or fitness, or whatever the
> amplitude variable is) to a healthy distribution, with minimal
> perturbation from a clinically-conservative baseline?*

The engine answers this by treating the choice of daily intervention
as an optimisation problem: minimise the predicted distance between
the patient's terminal-day distribution of the amplitude variable and
a clinical target distribution, subject to a quadratic regulariser on
how aggressive the intervention is.

---

## 3. The clinical scenarios (SWAT)

The SWAT adapter ships three canonical scenarios. Each fixes the
patient's *initial state* and the *reference (uncontrolled) baseline
schedule*; the engine then optimises a 14-day intervention schedule.

### 3.1 Recovery — the easiest scenario

* **Patient phenotype**: previously healthy, recently insulted
  (e.g. acute stress event), low testosterone amplitude $T_0 = 0.05$.
* **Reference baseline**: $V_h = 1.0$ (normal vitality reserve),
  $V_n = 0.3$ (low chronic load), $V_c = 0$ (no phase shift). I.e.
  the patient's *underlying* parameters are healthy; the
  testosterone amplitude has merely been transiently knocked down and
  needs to recover.
* **Clinical question**: how aggressively (and along which axis) do
  we push the controls to bring $T$ back up?
* **Expected behaviour**: the optimised schedule modestly perturbs
  controls early in the horizon, then settles back toward the
  reference. Recovery driven by the model's own self-stabilising
  dynamics.

### 3.2 Insomnia — chronic amplitude collapse

* **Patient phenotype**: chronic primary insomnia, high chronic load,
  fragmented and shallow sleep, suppressed testosterone driven by
  amplitude failure (not phase shift). $T_0 = 0.05$ at the
  catastrophic-fold equilibrium.
* **Reference baseline**: $V_h = 0.2$ (collapsed vitality reserve),
  $V_n = 3.5$ (severe chronic load), $V_c = 0$ (phase OK). Sleep
  architecture is bad but the body clock is on time.
* **Clinical question**: must we treat the underlying load and
  vitality, or can phase tricks rescue the patient?
* **Expected behaviour** (predicted by SWAT theory): pushing $V_n$
  down and $V_h$ up is necessary; phase shifts are ineffective. The
  optimised schedule should reflect this physiology.

### 3.3 Shift work — phase misalignment

* **Patient phenotype**: night-shift worker with otherwise-normal
  sleep architecture (good total sleep time, normal deep/light/REM
  ratios), suppressed testosterone driven by phase misalignment, not
  amplitude. $T_0 = 0.05$.
* **Reference baseline**: $V_h = 1.0$, $V_n = 0.3$, $V_c = 6$
  (six-hour phase shift). Sleep is intact; the clock is wrong.
* **Clinical question**: is realigning the phase enough, or do we
  also need to attack vitality and load?
* **Expected behaviour** (predicted by SWAT theory): pushing $V_c \to 0$
  (rephasing toward the morning reference) is the dominant
  intervention. Other controls move only mildly.

These three scenarios are the SWAT "differential-diagnosis sandbox":
they let a clinician (or a clinical researcher) ask whether the model
prescribes the right physiological intervention for the right
phenotype.

---

## 4. The clinical target

The engine accepts a target distribution on the amplitude variable at
terminal time. For SWAT, the default healthy target is

$$
T(D) \sim \mathcal{N}(0.55, 0.05^2)
$$

— a tight Gaussian centred at the typical-healthy steady-state value
of testosterone pulsatility. Adapters can override; clinicians can
substitute a target derived from population reference distributions
or trial design constraints.

The basin indicator complements the MMD target. For SWAT it returns
true when *both* the entrainment quality $E$ at the terminal-day
controls exceeds the bifurcation threshold $E_\text{crit}$ AND
$T \geq 0.4$. The fraction of trajectories satisfying this in the
closed-loop verification is the **clinically-meaningful "success"
metric**: how reliably does the prescribed schedule lead the patient
into the healthy attractor?

---

## 5. The control-schedule output

A successful run produces a CSV like:

| day | V_h | V_n | V_c |
|:---:|:---:|:---:|:---:|
| 0 | 1.4 | 0.5 | 4.5 |
| 1 | 1.4 | 0.5 | 3.0 |
| 2 | 1.3 | 0.4 | 1.6 |
| ... | ... | ... | ... |
| 13 | 1.0 | 0.3 | 0.0 |

This is the engine's prescription. Mapping to clinical interventions
is the adapter author's responsibility:

* $V_h$ ↑ (vitality reserve) → e.g. resistance training, nutritional
  support, supervised exercise, sleep optimisation.
* $V_n$ ↓ (chronic load) → stress reduction, cognitive-behavioural
  therapy for insomnia, modifiable-load reduction.
* $V_c$ shift toward 0 → bright-light therapy in the morning,
  dim-light avoidance after sunset, scheduled-sleep restriction.

The engine does not prescribe specific doses or modalities. It
prescribes target physiological-parameter values per day; the
clinician chooses how to achieve them.

---

## 6. Limitations — read this before clinical use

* **Single-subject, single-trajectory**. The model assumes the patient's
  parameters are constant over the horizon. Acute illnesses, life
  events, schedule disruptions break this assumption.
* **Phase-1 constants**. SWAT currently treats $V_h, V_n$ as constants
  per fit; in reality they drift with season, age, training. The
  schedule should be re-optimised at most monthly to accommodate this.
* **Calibration is upstream**. The engine does not estimate the
  patient's parameters from data — that's the job of the SMC²
  estimation pipeline. Garbage in, garbage out.
* **Boundary-saturation pathology**. If a patient's optimised schedule
  pins a state variable (e.g. $W$) at the physical boundary on every
  step, the optimiser's gradient vanishes and the schedule may not be
  truly optimal. The closed-loop verification surfaces this as poor
  basin fraction.
* **Not a medical device**. The engine outputs a recommendation. Final
  decisions remain with the clinician. Regulatory-grade validation has
  not been performed.

---

## 7. What a clinician reads from the output

The standard run produces four figures (all optional, all generated by
`adapters/swat/plots.py`):

1. **`schedule.png`** — bar chart of daily $(V_h, V_n, V_c)$ values.
   *What does the prescription look like?*
2. **`latent_paths.png`** — sample trajectories of $W$, $\tilde Z$, $T$
   under the optimised schedule. *What does the patient's
   physiological trajectory look like under the schedule?*
3. **`terminal_amplitude.png`** — histogram of simulated $T(D)$ vs the
   target distribution. *Did we hit the target?*
4. **`loss_trace.png`** — total / terminal / transport / reference
   loss vs optimisation step. *Did the optimiser converge?*

The CSV `comparison.csv` (produced by `compare_schedules` /
`run_ot_pipeline`) lists the four scalar metrics for each schedule
(optimised vs. three baselines). The optimised schedule is the
clinical recommendation **iff** it strictly beats all three baselines
on the basin-fraction metric. If a baseline ties or wins, the engine
has not found a meaningful improvement and the schedule should not be
recommended.

---

## 8. Worked example — recovery scenario

Run:

```
python experiments/run_swat.py --scenario recovery --horizon 14 --output_dir runs/recovery_001
```

Produces (typical output for v1.1.0 on the default seed):

| Schedule | distance to T★ | basin frac | MMD² |
|:---|:---:|:---:|:---:|
| zero_control | 0.322 | 0.16 | 0.95 |
| constant_reference | 0.396 | 0.05 | 1.22 |
| linear_interpolation | 0.403 | 0.02 | 1.30 |
| **optimised** | **0.270** | **0.28** | **0.71** |

The optimised schedule reduces the terminal distance to the healthy T★
target by 16% and nearly doubles the fraction of trajectories landing
in the healthy basin (28% vs 16%) compared to the best baseline. This
is the signal that the engine has produced a clinically meaningful
recommendation for this patient phenotype.

---

## 9. Where the model and engine come from

* **Model spec**: see project-root `SWAT_Basic_Documentation.md`,
  `SWAT_Clinical_Specification.md`, `SWAT_Identifiability_Extension.md`.
* **Engine spec**: this repo's `docs/Mathematical_Specification.md`.
* **How to run experiments**: `docs/How_To_Run_Experiments.md`.
* **What's planned next**: `docs/Future_Features.md`.
