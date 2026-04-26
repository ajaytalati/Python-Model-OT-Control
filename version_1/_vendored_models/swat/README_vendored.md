# Vendored SWAT Model

**Source:** `Python-Model-Development-Simulation/version_1/models/swat/`
**Vendored on:** 26 April 2026
**Purpose:** Self-contained JAX-native re-implementation of the SWAT
SDE for use by the OT-Control engine. Avoids a hard dependency on the
upstream model-dev repo during development.

## What's here

- `dynamics_jax.py` — JAX drift, diffusion, and entrainment-quality
  helpers for the four-state $(W, \tilde Z, a, T)$ SDE.
- `parameters.py` — the default 28-parameter dictionary from the
  spec (see `SWAT_Basic_Documentation.md` §2.2 in the upstream repo).

## What's NOT here (and why)

- The **observation model** (HR, sleep stages, steps, stress channels).
  The OT-Control engine works on the latent state directly; observation
  channels are an estimation concern, not a control concern.
- The **identification machinery** (Fisher information, Lyapunov
  proofs). These are properties of the model, not used in control.
- The **scipy-NumPy** simulator for cross-validation. JAX is the only
  path the OT-Control engine uses.

## Consistency with upstream

Time is converted from **hours** (upstream convention) to **days** (the
OT engine's convention). Specifically:

- All timescales $\tau_i$ are divided by 24 to become days.
- The diffusion temperatures $T_i$ are multiplied by 24 (variance
  scales linearly with time, so $T'_i\,d t' = T_i\,(d t' \cdot 24)$
  when $d t'$ is in days and the original $T_i$ was per-hour).
- The circadian formula $\sin(2\pi t / 24 + \phi)$ becomes
  $\sin(2\pi t + \phi)$ when $t$ is in days.

## Update procedure

If the upstream model spec changes:

1. Edit `parameters.py` to match the new defaults.
2. Edit `dynamics_jax.py` if drift or diffusion changes.
3. Update the version block at the top of each file.
4. Re-run the engine tests; specifically `tests/adapters/test_swat_adapter.py`.
