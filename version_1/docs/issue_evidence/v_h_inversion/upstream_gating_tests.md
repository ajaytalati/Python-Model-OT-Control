# Upstream gating tests for vendoring into OT-Control

**Purpose.** A model that ships into the OT-Control engine has its $V_h, V_n, V_c$ promoted from per-subject *constants* to time-varying *controls*. The optimiser will explore the entire $(V_h, V_n, V_c)$ box and exploit any clinically-inverted gradient. **Issue #4** showed this happens silently when the upstream model has the wrong sign or wrong structure on a control: V_h was anti-anabolic, the optimiser found this, and produced clinically-invalid schedules.

These tests sit in the upstream repo (`Python-Model-Development-Simulation/version_1/models/swat/`) and **must pass before the vendored copy is updated in OT-Control**. The bar is "the model's response to each control is monotone in the clinically-claimed direction across the entire bounded box."

**Where these tests live.** Suggest `models/swat/test_control_responses.py`, run as part of the upstream's CI.

**How they should be wired.** OT-Control's `_vendored_models/swat/README_vendored.md` should add a line: *"Upstream version vendored: <hash>. Gating tests passed: <date>."* Refuse to bump the vendored hash if the gating tests have not been re-run.

---

## Common test fixture

All tests use deterministic simulation (noise temperatures forced to zero) to remove single-seed variance, and run for $D = 14$ days from initial state $(W_0, \tilde Z_0, a_0, T_0) = (0.5, 3.5, 0.5, T_0^\mathrm{scenario})$. Use scipy's `solve_deterministic` (BDF) or fine-dt explicit Euler (dt ≤ 0.001 days). Take $T_\mathrm{end}$ as the last-day mean of $T(t)$.

```python
def t_end_under_constant_controls(V_h, V_n, V_c, T_0=0.5, D=14):
    """Return last-day mean T under constant (V_h, V_n, V_c), noise off."""
    p = dict(PARAM_SET_A)
    p['T_W'] = p['T_Z'] = p['T_a'] = p['T_T'] = 0.0
    init = dict(INIT_STATE_A); init['Vh'] = V_h; init['Vn'] = V_n; init['T_0'] = T_0
    p['V_c'] = V_c
    t_grid = np.arange(0.0, D * 24.0, 5.0/60.0)
    traj = solve_deterministic(SWAT_MODEL, p, init, t_grid)
    one_day = int(round(24.0 / (5.0/60.0)))
    return float(traj[-one_day:, 3].mean())
```

---

## Test 1 — V_h anabolicity (monotonicity in V_h at fixed V_n)

**Purpose.** Verify that increasing the vitality reserve increases the testosterone amplitude.

**Procedure.** Sweep $V_h \in \{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0\}$ at $V_n = 0.3$, $V_c = 0$, $T_0 = 0.5$.

**Pass criterion.**

```python
def test_V_h_anabolic():
    Ts = [t_end_under_constant_controls(V_h, V_n=0.3, V_c=0)
          for V_h in [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]]
    # Strictly non-decreasing
    for i in range(1, len(Ts)):
        assert Ts[i] >= Ts[i-1] - 0.01, (
            f"V_h is not anabolic: T_end at V_h-step {i} dropped: {Ts}")
    # Stronger: T(V_h=2) > T(V_h=0.5) by a clinically meaningful amount
    assert Ts[4] > Ts[1] + 0.05, (
        f"V_h response is too weak: T(V_h=2)={Ts[4]:.3f} vs "
        f"T(V_h=0.5)={Ts[1]:.3f}")
```

The current upstream model **fails this test** — the heatmap in `coarse_vh_vn_scan.png` shows $T_\mathrm{end}$ falling from 0.37 at $V_h = 0$ to 0.08 at $V_h = 4$.

---

## Test 2 — V_n catabolicity (monotonicity in V_n at fixed V_h)

**Purpose.** Verify that increasing chronic load decreases testosterone amplitude.

**Procedure.** Sweep $V_n \in \{0, 0.3, 0.5, 1.0, 2.0, 3.5, 5.0\}$ at $V_h = 1.0$, $V_c = 0$, $T_0 = 0.5$.

**Pass criterion.**

```python
def test_V_n_catabolic():
    Ts = [t_end_under_constant_controls(V_h=1.0, V_n=V_n, V_c=0)
          for V_n in [0, 0.3, 0.5, 1.0, 2.0, 3.5, 5.0]]
    # Strictly non-increasing
    for i in range(1, len(Ts)):
        assert Ts[i] <= Ts[i-1] + 0.01, (
            f"V_n is not catabolic: T_end at V_n-step {i} rose: {Ts}")
    # Stronger: T(V_n=0.3) > T(V_n=3.5) by a clinically meaningful amount
    assert Ts[1] > Ts[5] + 0.10, (
        f"V_n response is too weak: T(V_n=0.3)={Ts[1]:.3f} vs "
        f"T(V_n=3.5)={Ts[5]:.3f}")
```

---

## Test 3 — Antagonistic dose-response

**Purpose.** Confirm that the canonical clinical scenarios sit on a monotone trajectory: more vitality + less load → higher $T$.

**Procedure.** Compare four operating points spanning healthy → severe pathological:

| label | $V_h$ | $V_n$ | scenario |
|:---|:-:|:-:|:---|
| robust | 2.0 | 0.1 | high vitality, low load |
| healthy | 1.0 | 0.3 | canonical Set A |
| stressed | 0.5 | 1.0 | low vitality, high load |
| insomnia | 0.2 | 3.5 | canonical Set B |

**Pass criterion.**

```python
def test_dose_response_ordering():
    T_robust   = t_end_under_constant_controls(2.0, 0.1, 0)
    T_healthy  = t_end_under_constant_controls(1.0, 0.3, 0)
    T_stressed = t_end_under_constant_controls(0.5, 1.0, 0)
    T_insomnia = t_end_under_constant_controls(0.2, 3.5, 0)
    assert T_robust > T_healthy > T_stressed > T_insomnia, (
        f"Dose-response wrong: robust={T_robust:.3f}  healthy={T_healthy:.3f}  "
        f"stressed={T_stressed:.3f}  insomnia={T_insomnia:.3f}")
```

The current upstream **fails this test** — `T_robust` (0.086 from the diagnostic) is *lower* than `T_healthy` (~0.46), the wrong way round.

---

## Test 4 — No backwards optimum on the clinical box

**Purpose.** Confirm the maximum of $T_\mathrm{end}$ over the bounded $(V_h, V_n)$ box is **not** at the lower edge $V_h = 0$.

**Procedure.** Grid over $V_h \in \{0, 0.5, 1, 2, 4\}$ and $V_n \in \{0, 0.3, 1, 2, 4\}$.

**Pass criterion.**

```python
def test_no_backwards_optimum():
    grid = np.zeros((5, 5))
    V_h_vals = [0, 0.5, 1, 2, 4]
    V_n_vals = [0, 0.3, 1, 2, 4]
    for i, V_h in enumerate(V_h_vals):
        for j, V_n in enumerate(V_n_vals):
            grid[i, j] = t_end_under_constant_controls(V_h, V_n, 0)
    argmax_i, argmax_j = np.unravel_index(grid.argmax(), grid.shape)
    assert argmax_i > 0, (
        f"Backwards optimum: argmax T_end is at V_h={V_h_vals[argmax_i]} "
        f"(lower bound). Grid:\n{grid}")
    assert argmax_j < len(V_n_vals) - 1, (
        f"Suspicious optimum at V_n upper bound: argmax at "
        f"V_n={V_n_vals[argmax_j]}. Should be at moderate V_n.")
```

This is the **smoke-detector test** — if it fails, V_h is structurally wrong (issue #4).

---

## Test 5 — Healthy equilibrium reaches its claimed value

**Purpose.** Verify the spec's "T* ≈ 0.55 under healthy controls" claim is actually the deterministic equilibrium, not a single-seed artefact.

**Procedure.** Run scenario A deterministically for $D = 30$ days (long enough to fully equilibrate; $\tau_T = 2$ days, so 15 relaxation times).

**Pass criterion.**

```python
def test_healthy_equilibrium_is_055():
    T_end = t_end_under_constant_controls(V_h=1.0, V_n=0.3, V_c=0,
                                            T_0=0.5, D=30)
    # Spec claims T* ≈ 0.55. Allow ±0.05 for finite-time / discretisation slack.
    assert 0.50 <= T_end <= 0.60, (
        f"Healthy equilibrium {T_end:.3f} outside [0.50, 0.60]. "
        "Either the spec value is wrong or the model is.")
```

The current upstream **fails this test** — deterministic equilibrium is 0.46, not 0.55. The spec needs to be updated either to match the actual equilibrium or to fix the formulation so 0.55 is reached.

---

## Test 6 — Pathological scenarios collapse appropriately

**Purpose.** Verify the four canonical scenarios produce the expected outcomes.

```python
def test_canonical_scenarios():
    # A: healthy
    T_A = t_end_under_constant_controls(1.0, 0.3, 0, T_0=0.5)
    assert T_A >= 0.45, f"Healthy scenario A produces low T: {T_A:.3f}"
    # B: insomnia (severe load)
    T_B = t_end_under_constant_controls(0.2, 3.5, 0, T_0=0.5)
    assert T_B <= 0.20, f"Insomnia scenario B doesn't collapse: {T_B:.3f}"
    # C: recovery (start from low T)
    T_C = t_end_under_constant_controls(1.0, 0.3, 0, T_0=0.05)
    assert T_C >= 0.30, f"Recovery scenario C doesn't recover: {T_C:.3f}"
    # D: phase-shift collapse
    T_D = t_end_under_constant_controls(1.0, 0.3, 6.0, T_0=0.5)
    assert T_D <= 0.20, f"Phase-shift D doesn't collapse: {T_D:.3f}"
```

---

## Test 7 — Sleep fraction within physiological window

**Purpose.** The user's `beta_Z = 4.0` choice was specifically to give a 33% sleep fraction. Verify it stays in window after any model change.

```python
def test_sleep_fraction_healthy():
    """Healthy scenario A produces 25-40% sleep time."""
    p = dict(PARAM_SET_A)
    init = dict(INIT_STATE_A)
    t_grid = np.arange(0.0, 14 * 24.0, 5.0/60.0)
    traj = solve_sde(SWAT_MODEL, p, init, t_grid, seed=42, n_substeps=10)
    # sleep = Zt > c_tilde
    sleep_frac = float((traj[:, 1] >= p['c_tilde']).mean())
    assert 0.25 <= sleep_frac <= 0.40, (
        f"Sleep fraction {sleep_frac*100:.1f}% outside 25-40% window")
```

---

## Test 8 — Phase-quality term enters E correctly

**Purpose.** Verify $V_c \neq 0$ collapses $E$ symmetrically and that $V_c = \pm 6$h zeros it.

```python
def test_V_c_phase_collapse():
    T_aligned = t_end_under_constant_controls(1.0, 0.3, V_c=0)
    T_shift_pos = t_end_under_constant_controls(1.0, 0.3, V_c=6.0)
    T_shift_neg = t_end_under_constant_controls(1.0, 0.3, V_c=-6.0)
    assert T_shift_pos <= 0.20 and T_shift_neg <= 0.20, (
        "V_c = ±6 should collapse T to near zero")
    # Symmetry
    assert abs(T_shift_pos - T_shift_neg) < 0.05, (
        f"V_c collapse not symmetric: +6 -> {T_shift_pos:.3f}, "
        f"-6 -> {T_shift_neg:.3f}")
```

---

## Test 9 — Bifurcation threshold

**Purpose.** Verify Stuart-Landau collapses below $E_\mathrm{crit}$ and grows above.

```python
def test_bifurcation():
    """Find a (V_h, V_n) that produces E < E_crit and one that produces E > E_crit.
    The first should drive T to ~0; the second should sustain T > 0.3."""
    # E < 0.5: take any control with very high V_h that saturates amp_W
    T_subcrit = t_end_under_constant_controls(V_h=4.0, V_n=0, V_c=0)
    assert T_subcrit < 0.20, (
        f"Sub-critical E should collapse T: got {T_subcrit:.3f}")
    # E > 0.5: take controls that maximize amp_W and amp_Z
    T_supercrit = t_end_under_constant_controls(V_h=1.0, V_n=0.3, V_c=0)
    assert T_supercrit > 0.30, (
        f"Super-critical E should sustain T: got {T_supercrit:.3f}")
```

After the fix from Option A in `model_fix_plan.md`, `T_subcrit` will be at its $V_h$-dependent floor (driven by the new $\alpha_h V_h$ term), so this test may need adjusting — keep it as a smoke check but don't expect $T \to 0$ exactly.

---

## How these tests would have caught issue #4

| Test | Caught? | Why |
|:---|:-:|:---|
| 1 — V_h anabolicity | **YES** | Heatmap shows monotone *decrease* in V_h |
| 2 — V_n catabolicity | partial | V_n is non-monotone in current model |
| 3 — dose-response ordering | **YES** | T_robust < T_healthy in current model |
| 4 — no backwards optimum | **YES** | argmax is at V_h=0 |
| 5 — equilibrium = 0.55 | **YES** | actual is 0.46 |
| 6 — canonical scenarios | partial | scenarios A/C marginal |
| 7 — sleep fraction | no | unrelated to V_h |
| 8 — V_c phase collapse | no | V_c works correctly |
| 9 — bifurcation | no | bifurcation works correctly |

So **5 of 9 tests would have failed** on the upstream model in its current form, blocking the vendored update. Issue #4 would have been raised against upstream months earlier, and the OT-Control engine would never have been built on a clinically-inverted substrate.

## Suggested CI integration

```yaml
# .github/workflows/swat_gating.yml in upstream
- name: Run SWAT control-response gating tests
  run: |
    pytest models/swat/test_control_responses.py -v
- name: If passing, tag the commit as 'vendor-ready'
  if: success()
  run: git tag "swat-vendor-ready-$(date +%Y%m%d)"
```

The OT-Control side would only re-pull from upstream commits tagged `swat-vendor-ready-*`, codified in the vendor sync script.
