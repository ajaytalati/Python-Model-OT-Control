# Vendored FSA-High-Res Model

**Source:** `Python-Model-Development-Simulation/version_1/models/fsa_high_res/`
**Vendored on:** 26 April 2026
**Purpose:** Self-contained JAX-native re-implementation of the
FSA-high-res SDE for use by the OT-Control engine. Avoids a hard
dependency on the upstream model-dev repo at runtime.

## What's here

- `dynamics_jax.py` — JAX drift, diffusion, state-clip, amplitude
  projector, and healthy-attractor predicate for the three-state
  $(B, F, A)$ SDE.
- `parameters.py` — the 13-parameter dynamics dictionary tuned for the
  14-day proof-of-principle (per upstream `simulation.py` line 463).

## What's NOT here (and why)

- The **observation model** (HR, sleep, stress, steps channels) and
  its 19 observation parameters. The OT-Control engine works on the
  latent state directly; observation channels are an estimation
  concern.
- The **15-min bin lookup machinery** for time-varying $T_B(t),
  \Phi(t)$. The OT engine receives controls from a daily-resolution
  policy (`PiecewiseConstant`), so the vendored `fsa_drift` takes
  scalar `u = (T_B, Phi)` directly. The "high_res" aspect of the
  upstream simulator is for the observation channel, not the control.
- The **scipy-NumPy** simulator for cross-validation.
- The **estimation** routines (Gaussian-kernel particle filter,
  guided proposals).

## Consistency with upstream

Time is in **days** throughout, matching upstream. No unit conversion
is needed (unlike SWAT, where upstream is in hours).

The only deviation from upstream is the control signature: upstream
takes `(T_B_arr, Phi_arr)` with shape `(96 * n_days,)` indexed by bin
position; this vendored copy takes scalar `(T_B, Phi)` from the OT
policy. Drift formulae are otherwise identical.

The `epsilon_A` regulariser is hardcoded at `1e-4` inside
`dynamics_jax.fsa_diffusion` (matching upstream's `EPS_A_FROZEN`)
rather than appearing in the parameter dict — it is a non-estimable
implementation detail.

## Update procedure

If the upstream model spec changes:

1. Edit `parameters.py` to match the new defaults from upstream's
   `DEFAULT_PARAMS`.
2. Edit `dynamics_jax.py` if drift or diffusion formulae change.
3. Update the version block at the top of each file.
4. Re-run the engine tests; specifically `tests/adapters/test_fsa_adapter.py`.
