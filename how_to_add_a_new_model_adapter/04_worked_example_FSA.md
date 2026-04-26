# 04 — Worked example: FSA-high-res adapter

A complete walkthrough using the FSA-high-res model
(Fitness-Strain-Amplitude, 3 states, 2 controls), specified in the
project root as `Gem_-_Mathematical_Specification_-_3-State_FSA_Model_v4_1.md`.

This example is designed to show the pattern when the controls are
**different** from SWAT (continuous training-load `T_B` and strain
`Phi` instead of `V_h, V_n, V_c`) and the amplitude variable is
**fitness-driven endocrine amplitude** rather than testosterone.

The code below is illustrative — verified against the spec but not
shipped in v1.2.0 (FSA-high-res adapter is Phase 6, deferred).

## The FSA model in 60 seconds

State $x = (B, F, A)$, all in days:

* $B \in [0, 1]$: **fitness** (Jacobi diffusion, target = training load $T_B$).
* $F \geq 0$: **strain** (CIR diffusion, fed by training intensity $\Phi$, drained by fitness and amplitude).
* $A \geq 0$: **endocrine amplitude** (regularised Landau, controlled by bifurcation parameter $\mu(B, F)$).

Controls $u = (T_B, \Phi)$:

* $T_B \in [0, 1]$: training-load target. Pushes $B$ up.
* $\Phi \geq 0$: training intensity (strain production). Pushes $F$ up.

The clinical question: *"Given this athlete's current fitness, strain,
and endocrine state, what 14-day training schedule restores healthy
endocrine pulsatility?"* — i.e. drive $A$ from low into the healthy
attractor.

## Step 1 — Vendor `_vendored_models/fsa_high_res/`

### `dynamics_jax.py`

```python
"""
_vendored_models/fsa_high_res/dynamics_jax.py
==============================================
JAX-native dynamics for the 3-State FSA model v4.1.

Source:  Python-Model-Development-Simulation @ <commit>,
         version_1/models/fsa_high_res/_dynamics.py
Spec:    Gem_-_Mathematical_Specification_-_3-State_FSA_Model_v4_1.md
Vendored: <date>
"""

from __future__ import annotations
from typing import Dict
import jax.numpy as jnp


def _bifurcation_parameter(B: jnp.ndarray, F: jnp.ndarray,
                            params: Dict[str, float]) -> jnp.ndarray:
    """mu(B, F) — drives the Hopf bifurcation in A."""
    return (params['mu_0']
            + params['mu_B'] * B
            - params['mu_F'] * F
            - params['mu_FF'] * (F ** 2))


def fsa_drift(t: jnp.ndarray,
              x: jnp.ndarray,
              u: jnp.ndarray,
              params: Dict[str, float]) -> jnp.ndarray:
    """Drift for (B, F, A) under controls (T_B, Phi)."""
    del t                                    # autonomous in t
    B, F, A = x[0], x[1], x[2]
    T_B, Phi = u[0], u[1]

    # Effective fitness adaptation rate boosted by amplitude
    inv_tau_B_eff = (1.0 + params['alpha_A'] * A) / params['tau_B']
    dB = inv_tau_B_eff * (T_B - B)

    # Effective strain recovery rate boosted by fitness and amplitude
    inv_tau_F_eff = ((1.0 + params['lambda_B'] * B
                      + params['lambda_A'] * A) / params['tau_F'])
    dF = Phi - inv_tau_F_eff * F

    # Stuart-Landau amplitude
    mu = _bifurcation_parameter(B, F, params)
    dA = mu * A - params['eta'] * A ** 3

    return jnp.array([dB, dF, dA])


def fsa_diffusion(x: jnp.ndarray,
                  params: Dict[str, float]) -> jnp.ndarray:
    """State-dependent diffusion (Jacobi / CIR / Landau)."""
    B, F, A = x[0], x[1], x[2]
    sigma_B = params['sigma_B'] * jnp.sqrt(B * (1.0 - B))
    sigma_F = params['sigma_F'] * jnp.sqrt(F)
    sigma_A = params['sigma_A'] * jnp.sqrt(A + params['epsilon_A'])
    return jnp.array([sigma_B, sigma_F, sigma_A])


def fsa_state_clip(x: jnp.ndarray,
                   params: Dict[str, float] | None = None) -> jnp.ndarray:
    """B in [0, 1], F >= 0, A >= 0."""
    del params
    return jnp.array([
        jnp.clip(x[0], 0.0, 1.0),    # B
        jnp.maximum(x[1], 0.0),      # F
        jnp.maximum(x[2], 0.0),      # A
    ])


def amplitude_of_fsa(x: jnp.ndarray) -> jnp.ndarray:
    """A is index 2 in the (B, F, A) state vector."""
    return x[2]


def healthy_attractor_check(B: jnp.ndarray, F: jnp.ndarray,
                             params: Dict[str, float]) -> jnp.ndarray:
    """True if mu(B, F) > 0 — the Hopf bifurcation is super-critical."""
    return _bifurcation_parameter(B, F, params) > 0.0
```

### `parameters.py`

```python
"""
_vendored_models/fsa_high_res/parameters.py
============================================
Default parameters for FSA v4.1, all timescales in DAYS.
"""
from typing import Dict


def default_fsa_parameters() -> Dict[str, float]:
    return {
        # Fitness block
        'tau_B': 14.0,           # days
        'alpha_A': 1.0,          # 1/amplitude

        # Strain block
        'tau_F': 7.0,            # days
        'lambda_B': 1.0,
        'lambda_A': 0.5,

        # Amplitude block (Landau)
        'mu_0': -0.3,            # 1/day, baseline negative (no pulsatility at zero fitness)
        'mu_B': 1.0,             # 1/day
        'mu_F': 0.5,             # 1/(day*strain)
        'mu_FF': 1.5,            # 1/(day*strain^2) — overtraining cliff
        'eta': 1.0,              # 1/(day*amp^2)

        # Frozen process noises (per spec §2)
        'sigma_B': 0.01,
        'sigma_F': 0.005,
        'sigma_A': 0.02,
        'epsilon_A': 1e-4,
    }
```

### `__init__.py`

```python
from _vendored_models.fsa_high_res.dynamics_jax import (
    fsa_drift, fsa_diffusion, fsa_state_clip,
    amplitude_of_fsa, healthy_attractor_check,
)
from _vendored_models.fsa_high_res.parameters import (
    default_fsa_parameters,
)
__all__ = [
    "fsa_drift", "fsa_diffusion", "fsa_state_clip",
    "amplitude_of_fsa", "healthy_attractor_check",
    "default_fsa_parameters",
]
```

## Step 2 — Define scenarios

Three clinically-meaningful scenarios for FSA:

| Scenario | Initial $(B_0, F_0, A_0)$ | Reference $(T_B, \Phi)$ | Clinical question |
|:---|:---:|:---:|:---|
| `unfit_recovery` | (0.05, 0.10, 0.01) | (0.0, 0.0) | Sedentary, low fitness, no pulsatility. Build fitness from rest. |
| `over_trained` | (0.40, 0.50, 0.10) | (0.6, 0.8) | Athlete pinned past the overtraining cliff. Reduce strain, restore amplitude. |
| `detrained_athlete` | (0.20, 0.05, 0.05) | (0.3, 0.2) | Returning athlete after layoff. Build back. |

```python
_SCENARIOS = {
    'unfit_recovery':    {'T_B': 0.0, 'Phi': 0.0,
                           'B_0': 0.05, 'F_0': 0.10, 'A_0': 0.01},
    'over_trained':      {'T_B': 0.6, 'Phi': 0.8,
                           'B_0': 0.40, 'F_0': 0.50, 'A_0': 0.10},
    'detrained_athlete': {'T_B': 0.3, 'Phi': 0.2,
                           'B_0': 0.20, 'F_0': 0.05, 'A_0': 0.05},
}
```

## Step 3 — Adapter constants

```python
FSA_CONTROL_NAMES = ('T_B', 'Phi')

# T_B in [0, 1] is dimensionless training-load target.
# Phi >= 0 is strain production (training intensity).
# Both are non-negative — no negative training. Cap chosen
# to be physiologically extreme but bounded.
_FSA_CONTROL_BOUNDS = (
    (0.0, 1.0),          # T_B: training-load target
    (0.0, 2.0),          # Phi: training intensity
)
```

## Step 4 — Initial-state sampler

```python
def _make_initial_sampler(B_0: float, F_0: float, A_0: float):
    def sample_init(rng, n):
        rng_b, rng_f, rng_a = jax.random.split(rng, 3)
        B = jnp.clip(B_0 + 0.02 * jax.random.normal(rng_b, (n,)), 0.0, 1.0)
        F = jnp.maximum(F_0 + 0.02 * jax.random.normal(rng_f, (n,)), 0.0)
        A = jnp.maximum(A_0 + 0.005 * jax.random.normal(rng_a, (n,)), 0.0)
        return jnp.stack([B, F, A], axis=1)
    return sample_init
```

## Step 5 — Model-derived target sampler

The clinical target is "healthy endocrine pulsatility" — the
deterministic Landau equilibrium when $\mu > 0$. Construct from a
one-off simulation under "idealised healthy" controls (moderate $T_B$,
low $\Phi$):

```python
_HEALTHY_REFERENCE_CONTROLS = {'T_B': 0.7, 'Phi': 0.3}
_HEALTHY_REFERENCE_INIT = {'B_0': 0.7, 'F_0': 0.3, 'A_0': 0.5}


def _build_healthy_target_sampler(horizon_days, dt_days, params,
                                    n_pool=1024, seed=0xBEEF):
    from ot_engine.simulator import simulate_latent
    from ot_engine.types import BridgeProblem
    from ot_engine.policies.piecewise_constant import PiecewiseConstant

    h = _HEALTHY_REFERENCE_CONTROLS
    healthy_init = _make_initial_sampler(**_HEALTHY_REFERENCE_INIT)
    healthy_ref = jnp.tile(jnp.array([h['T_B'], h['Phi']]),
                             (horizon_days, 1))
    healthy_sigma = jnp.ones((horizon_days, 2))

    def _state_clip(x): return fsa_state_clip(x, params)

    healthy_problem = BridgeProblem(
        name='fsa_healthy_pool',
        drift_fn_jax=fsa_drift,
        diffusion_fn_jax=fsa_diffusion,
        model_params=params,
        sample_initial_state=healthy_init,
        sample_target_amplitude=lambda r, n: jnp.zeros(n),
        amplitude_of=amplitude_of_fsa,
        state_clip_fn=_state_clip,
        n_controls=2,
        control_bounds=_FSA_CONTROL_BOUNDS,
        control_names=FSA_CONTROL_NAMES,
        horizon_days=horizon_days,
        reference_schedule=healthy_ref,
        reference_sigma=healthy_sigma,
        n_particles=n_pool,
        dt_days=dt_days,
    )
    pol = PiecewiseConstant(horizon_days, 2)
    rng = jax.random.PRNGKey(seed)
    _, A_D_pool, _ = simulate_latent(rng, healthy_problem, pol, healthy_ref)

    def sample(rng_key, n):
        idx = jax.random.randint(rng_key, (n,), 0, A_D_pool.shape[0])
        return A_D_pool[idx]
    return sample, A_D_pool
```

## Step 6 — Basin indicator

The FSA basin condition combines the empirical pool central interval
on $A$ with a check that $\mu(B, F) > 0$ at the terminal day's
controls (i.e. the patient's parameters genuinely support a healthy
attractor):

```python
def _make_basin_indicator(target_pool, params):
    A_lo = float(jnp.percentile(target_pool, 10.0))
    A_hi = float(jnp.percentile(target_pool, 90.0))

    def basin(x, u_terminal, params_):
        B, F, A = x[0], x[1], x[2]
        A_in_range = jnp.logical_and(A >= A_lo, A <= A_hi)
        mu_positive = healthy_attractor_check(B, F, params_)
        return jnp.logical_and(A_in_range, mu_positive)
    return basin
```

## Step 7 — Top-level constructor

```python
def make_fsa_problem(
    scenario: str,
    horizon_days: int = 14,
    n_particles: int = 256,
    dt_days: float = 0.1,
    optim_steps: int = 2000,
    learning_rate: float = 1e-2,
    alpha_terminal: float = 1.0,
    alpha_transport: float = 0.05,
    alpha_reference: float = 0.01,
    reference_sigma: float = 0.3,
    model_params=None,
) -> BridgeProblem:
    if scenario not in _SCENARIOS:
        raise ValueError(f"Unknown FSA scenario {scenario!r}.")
    sc = _SCENARIOS[scenario]
    params = (model_params if model_params is not None
              else default_fsa_parameters())

    ref_schedule = jnp.tile(jnp.array([sc['T_B'], sc['Phi']]),
                              (horizon_days, 1))
    ref_sigma_array = jnp.ones((horizon_days, 2)) * reference_sigma

    target_sampler, target_pool = _build_healthy_target_sampler(
        horizon_days=horizon_days, dt_days=dt_days, params=params,
    )
    basin_indicator = _make_basin_indicator(target_pool, params)

    def _state_clip_with_params(x):
        return fsa_state_clip(x, params)

    return BridgeProblem(
        name=f'fsa_{scenario}',
        drift_fn_jax=fsa_drift,
        diffusion_fn_jax=fsa_diffusion,
        model_params=params,
        sample_initial_state=_make_initial_sampler(
            sc['B_0'], sc['F_0'], sc['A_0']),
        sample_target_amplitude=target_sampler,
        amplitude_of=amplitude_of_fsa,
        state_clip_fn=_state_clip_with_params,
        basin_indicator_fn=basin_indicator,
        n_controls=2,
        control_bounds=_FSA_CONTROL_BOUNDS,
        control_names=FSA_CONTROL_NAMES,
        horizon_days=horizon_days,
        reference_schedule=ref_schedule,
        reference_sigma=ref_sigma_array,
        alpha_terminal=alpha_terminal,
        alpha_transport=alpha_transport,
        alpha_reference=alpha_reference,
        n_particles=n_particles,
        dt_days=dt_days,
        optim_steps=optim_steps,
        learning_rate=learning_rate,
    )
```

## Step 8 — Use it

```python
import jax
from ot_engine import (PiecewiseConstant, optimise_schedule,
                        simulate_closed_loop, compare_schedules,
                        zero_control_schedule, constant_reference_schedule)
from adapters.fsa_high_res import make_fsa_problem

problem = make_fsa_problem(scenario='over_trained', horizon_days=14)
pol = PiecewiseConstant.from_problem(problem)

# Optimise
schedule, trace = optimise_schedule(problem, pol, jax.random.PRNGKey(0))

# Verify
result = simulate_closed_loop(problem, pol, schedule,
                                jax.random.PRNGKey(1), n_realisations=512)
print(f"basin fraction: {result.fraction_in_healthy_basin:.2f}")
print(f"MMD to target: {result.mmd_target:.4f}")
```

## What to expect

For the **over_trained** scenario, the optimised schedule should:

* Push $\Phi$ **down** sharply early in the horizon — the patient is
  past the overtraining cliff and needs strain to drain.
* Hold $T_B$ at a moderate level — too low and fitness drains, losing
  the only thing pushing $\mu$ positive.
* Allow $\Phi$ to gradually rise toward the reference as fitness
  rebuilds.

For the **unfit_recovery** scenario, the optimised schedule should:

* Push $T_B$ **up** monotonically — the patient is starting from
  $B = 0.05$ and needs to build fitness.
* Push $\Phi$ moderately, but not so high that $F$ overwhelms $\mu$.

If your output schedules don't show this pattern, run through the
debugging checklist in §8 of `03_step_by_step_guide.md`.

## Differences from SWAT (highlights)

| | SWAT | FSA-high-res |
|:---|:---|:---|
| state dim | 4 (W, Z, a, T) | 3 (B, F, A) |
| controls | 3 (V_h, V_n, V_c) | 2 (T_B, Phi) |
| signed control? | yes (V_c is phase shift) | no (both non-negative) |
| amplitude variable | T (testosterone) | A (endocrine amplitude) |
| bifurcation parameter | $E_\text{dyn}$ (entrainment quality) | $\mu(B, F)$ |
| diffusion | additive (state-independent) | state-dependent (Jacobi / CIR / Landau) |
| timescales | hours $\to$ days at vendor time | days throughout (no conversion) |
| state clip | all four components | $B \in [0,1]$, $F, A \geq 0$ |

The pattern is the same. The contents are different. That's the whole
point of the engine being model-agnostic.

## Done

If this all works, you have:

* `version_1/_vendored_models/fsa_high_res/` — vendored model
* `version_1/adapters/fsa_high_res/adapter.py` — adapter + scenarios
* `version_1/adapters/fsa_high_res/plots.py` — adapter-specific plots
* `version_1/experiments/run_fsa.py` — CLI runner
* `version_1/tests/adapters/test_fsa_adapter.py` — tests

…all on top of the same engine. No engine changes.

That is the proof of the abstraction.
