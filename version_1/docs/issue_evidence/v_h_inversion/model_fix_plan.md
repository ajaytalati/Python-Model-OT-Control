# Plan: make $V_h$ anabolic and $V_n$ catabolic in the SWAT model

**Companion to:** [`proof_v_h_optimum_at_zero.md`](proof_v_h_optimum_at_zero.md), which proves the current formulation has $V_h^\star = 0$ (anti-anabolic) under all clinically-relevant operating points.

**Scope:** This is an **upstream model-spec change** — it must land in `Python-Model-Development-Simulation/version_1/models/swat/` first, then be re-vendored into OT-Control. The OT-Control engine itself doesn't need code changes.

## Constraint summary

The fix must preserve:
- **33% sleep/wake ratio** under healthy controls (load-bearing for `beta_Z = 4.0`).
- **Healthy scenario equilibrium $T \approx 0.55$** for $V_h = 1, V_n = 0.3$.
- **Insomnia collapse** $T \to 0.12$ for $V_h = 0.2, V_n = 3.5$.
- **Recovery feasibility** from $T_0 = 0.05$ under healthy controls.
- **Phase-shift collapse** under $V_c = 6$.

The fix must produce:
- $T_\mathrm{end}$ **monotonically non-decreasing in $V_h$** at fixed $V_n$.
- $T_\mathrm{end}$ **monotonically non-increasing in $V_n$** at fixed $V_h$.
- The optimum $\mathrm{argmax}_{V_h} T_\mathrm{end}$ **strictly interior** in the bounded region (or at the upper bound, never at $V_h = 0$).

## Three candidate fixes

### Option A — direct anabolic coupling on $T$ *(recommended)*

**Change.** Remove $V_h$ from $u_W$ and from $\mu_W^\mathrm{slow}$. Add a direct anabolic term to the $T$ drift.

```
u_W           = lambda * C_eff + V_n - a - kappa * Z + alpha_T * T          # V_h removed
mu_W_slow     =                  V_n - a + alpha_T * T                       # V_h removed
u_Z           = -gamma_3 * W - V_n + beta_Z * a                              # unchanged
da/dt         = (W - a) / tau_a                                              # unchanged
dT/dt         = [(mu_0 + mu_E * E) * T - eta * T^3 + alpha_h * V_h] / tau_T  # alpha_h * V_h ADDED
```

with new parameter $\alpha_h > 0$. Suggested initial value: **$\alpha_h = 0.1$**, to be calibrated so that the healthy scenario gives $T \approx 0.55$.

**Why this works.**
- $V_h$ enters $T$'s drift directly. Increasing $V_h$ shifts the Stuart–Landau equilibrium upward by adding a forcing term. Anabolic ✓
- $V_n$ keeps its catabolic role — increasing it raises $u_W$, saturates $W$'s sigmoid, kills $\mathrm{amp}_W$ and hence $E$, drops $T$. Catabolic ✓
- Sleep/wake ratio is **unchanged** because $V_h$ no longer enters $u_W$. The user's `beta_Z = 4.0` choice continues to give the 33% sleep fraction.
- The dimensionality of the model is unchanged (4 stochastic states); only one new parameter.

**Why this is the smallest change.** It edits one term in $u_W$, one term in $\mu_W^\mathrm{slow}$, one term in $dT/dt$. Three lines in `_dynamics.py`, plus one parameter add in `simulation.py::PARAM_SET_A`.

**Concrete file changes** (upstream first):

1. [`models/swat/_dynamics.py`](https://github.com/Python-Model-Development-Simulation) — `compute_sigmoid_args`: drop `+ Vh` from `u_W`. `entrainment_quality`: drop `+ Vh` from `mu_W_slow`. `drift`: add `+ params[pi['alpha_h']] * Vh` to `dT`.
2. [`models/swat/simulation.py`](https://github.com/Python-Model-Development-Simulation) — same edits, plus add `'alpha_h': 0.1` to `PARAM_SET_A`. Update the spec docstring.
3. [`model_documentation/swat/SWAT_Basic_Documentation.md`](https://github.com/Python-Model-Development-Simulation) — update §4.1 (drift), §4.2 (Stuart–Landau drift), §5.1 ($\mu_W^\mathrm{slow}$), §2.2 Block T (parameter list).
4. [`models/swat/verify_swat_state.py`](https://github.com/Python-Model-Development-Simulation) — re-record expected $T_\mathrm{end}$ values with the new dynamics.

**OT-Control after re-vendoring:**
- Re-pull `_vendored_models/swat/dynamics_jax.py` and `parameters.py` from upstream.
- The engine itself doesn't change. `make_swat_problem` already passes `params` opaquely to the drift function.

### Option B — sign flip on $V_h$ in $u_W$

**Change.** Single sign flip:

```
u_W       = lambda * C_eff - V_h + V_n - a - kappa * Z + alpha_T * T
mu_W_slow =                 -V_h + V_n - a + alpha_T * T
```

Now increasing $V_h$ moves $\mu_W^\mathrm{slow}$ toward 0 *from above* (when $V_n + \alpha_T T \geq a$), boosting $\mathrm{amp}_W$.

**Why this works.** The bell-shape proof from `proof_v_h_optimum_at_zero.md` still applies, but the constrained optimum is now $V_h^\star = V_n + \alpha_T T - a$, which is *positive* whenever $V_n + \alpha_T T > a$ — exactly the clinically-relevant regime.

**Pros.** No new parameter. Smallest possible patch (one sign flip).

**Cons.**
- Conceptually awkward: "high vitality opposes wake drive." Physiologically you'd think exercise *promotes* wakefulness.
- Sleep/wake architecture changes: $V_h$ now enters $u_W$ with a negative sign, so high $V_h$ *suppresses* the wake state. This may interact with the user's calibrated `beta_Z = 4.0` and shift the sleep ratio.
- The constrained optimum is at $V_h^\star = V_n - a + \alpha_T T$, which means **higher $V_n$ requires higher $V_h^\star$** — interpretable as "the more chronic load, the more vitality intervention needed." Clinically defensible but also slightly counterintuitive (you might want $V_n$ and $V_h$ to be independent levers).

### Option C — $V_h$ as circadian-amplitude modulator

**Change.**

```
u_W = lambda * (1 + V_h) * C_eff + V_n - a - kappa * Z + alpha_T * T   # V_h scales lambda
```

$V_h$ scales the circadian forcing strength. Higher $V_h$ → larger oscillation amplitude → larger $\mathrm{amp}_W$ → larger $E$ → larger $T$.

**Pros.** Maps to a clean clinical interpretation: $V_h$ as "rhythm-strengthening" interventions (sleep optimisation, light therapy, exercise scheduling).

**Cons.**
- Most invasive: changes the structure of $u_W$.
- Still has potential saturation at very high $V_h$ (extremely strong forcing pushes $u_W$ to large values regardless of state, saturating the sigmoid).
- Needs more thought about the dimensional consistency of `(1 + V_h) * lambda`.

## Recommendation

**Go with Option A.** It's the smallest patch, the cleanest physiological story (direct anabolic coupling), and it preserves everything the user has calibrated (sleep ratio via `beta_Z = 4`).

## Calibration of $\alpha_h$ (Option A)

Pick $\alpha_h$ so that the healthy scenario ($V_h = 1, V_n = 0.3, V_c = 0$, $T_0 = 0.5$) yields a deterministic $T_\mathrm{end} \approx 0.55$.

**Rough analytical estimate.** At equilibrium with the new dynamics, $\dot T = 0$ gives:

$$
0 \;=\; (\mu_0 + \mu_E E) T - \eta T^3 + \alpha_h V_h
$$

Under healthy controls with $V_h$ removed from $u_W$ and $\mu_W^\mathrm{slow}$, $E$ becomes higher (because $V_h$ no longer contributes to saturation). Empirically (from the $V_h \times V_n$ scan with $V_h = 0$, $V_n = 0.3$): $E \approx 0.61$, so $\mu = 0.11$.

Solving the cubic $\eta T^3 - \mu T = \alpha_h V_h$ at $T = 0.55$, $\eta = 0.5$, $\mu = 0.11$, $V_h = 1$:

$$
\alpha_h \;=\; \eta T^3 - \mu T \;=\; 0.5 \cdot 0.166 - 0.11 \cdot 0.55 \;=\; 0.083 - 0.061 \;\approx\; 0.022
$$

So $\alpha_h \approx 0.02 - 0.05$ is the right ballpark. Refine empirically.

**Stability check.** The added term $\alpha_h V_h / \tau_T$ must not destabilise the bifurcation structure. Since it's a constant forcing (positive, bounded), it shifts the equilibrium without removing the cubic $T^3$ saturation — the SDE remains well-posed.

## Implementation steps (recommended path)

**On upstream `Python-Model-Development-Simulation`:**

1. Edit `_dynamics.py`, `simulation.py` per Option A.
2. Add `alpha_h` to `PARAM_SET_A` (start with $0.05$).
3. Run `verify_swat_state.py` — adjust $\alpha_h$ until scenario A passes with $T_\mathrm{end} \approx 0.55$ (mean across seeds).
4. Verify scenarios B, C, D still pass their thresholds.
5. Run the new gating tests from [`upstream_gating_tests.md`](upstream_gating_tests.md) — V_h must be monotonically anabolic, V_n must be monotonically catabolic.
6. Update `SWAT_Basic_Documentation.md` and `Clinical_Specification.md` with the new equations.
7. Bump upstream model version (e.g. v2.0).

**Downstream (OT-Control):**

1. Re-pull `_vendored_models/swat/dynamics_jax.py` and `parameters.py`. Update vendored README's "Source" line.
2. Re-run the existing test suite (`pytest tests/`).
3. Re-run the recovery scenario through `experiments/run_swat.py` — the optimised schedule should now reach high $V_h$ for severe scenarios (visible on the schedule plot). Document in CHANGELOG.
4. Bump OT-Control engine version (e.g. v1.3.0).
5. Close issues #3 and #4.

## What this fix does NOT address

- **The "spec target $T = 0.55$" mythology.** That was always a single-seed value; even after this fix, expect ~$0.55 \pm 0.07$ across seeds. Spec language should be updated to "deterministic equilibrium ≈ 0.55."
- **The minimum-time formulation (issue #3).** That issue's reframing — using the model-derived target — still applies after this fix. The new principled target will be the empirical pool under (sensibly) maximal $V_h$, minimal $V_n$.
- **Clinical bound on $V_h$.** Currently $[0, 4]$. Probably needs re-thinking once $V_h$ is anabolic — does $V_h = 4$ make clinical sense, or should the upper bound also be tightened?
