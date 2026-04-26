# 03 — Step-by-step guide

This is the procedure. Assumes you've read `01_overview.md` and
skimmed `02_bridge_problem_contract.md`.

The example throughout uses `your_model` as a placeholder. Substitute
your actual model name (lowercase, no spaces, e.g. `fsa_high_res`).

## Step 1 — Vendor the model dynamics

Goal: a self-contained JAX implementation under
`version_1/_vendored_models/your_model/` so the adapter has no hard
dependency on the upstream model-dev repo at runtime.

Create:

```
version_1/_vendored_models/your_model/
├── __init__.py
├── README_vendored.md        # provenance
├── dynamics_jax.py            # drift, diffusion, optional state_clip
└── parameters.py              # default parameter dictionary
```

### `dynamics_jax.py` skeleton

```python
"""
_vendored_models/your_model/dynamics_jax.py
============================================
JAX-native dynamics for the Your-Model SDE.

Source:  Python-Model-Development-Simulation @ <commit-sha>,
         version_1/models/your_model/_dynamics.py
Vendored: <date>
"""

from __future__ import annotations
from typing import Dict, Optional

import jax.numpy as jnp


def your_drift(t: jnp.ndarray,
               x: jnp.ndarray,
               u: jnp.ndarray,
               params: Dict[str, float],
               ) -> jnp.ndarray:
    """Drift f(t, x, u) for the Your-Model SDE.

    Args:
        t: scalar time in DAYS.
        x: latent state, shape (dim_state,).
        u: control vector, shape (n_controls,).
        params: model parameter dictionary.

    Returns:
        Drift vector, shape (dim_state,).
    """
    # Unpack state and controls
    # state_1, state_2, ... = x[0], x[1], ...
    # ctrl_1, ctrl_2, ... = u[0], u[1], ...
    # Compute drift component by component
    # Return jnp.array([d_state_1, d_state_2, ...])
    ...


def your_diffusion(x: jnp.ndarray,
                   params: Dict[str, float],
                   ) -> jnp.ndarray:
    """Per-component noise amplitudes sigma_i(x).

    Returns:
        Shape (dim_state,) of per-component sigmas. The simulator
        applies sigma * sqrt(dt) * noise_i where noise_i ~ N(0, 1).
    """
    # If state-independent, ignore x:
    del x
    return jnp.sqrt(2.0 * jnp.array([
        params['T_state_1'],
        params['T_state_2'],
        # ...
    ]))


def your_state_clip(x: jnp.ndarray,
                    params: Optional[Dict[str, float]] = None,
                    ) -> jnp.ndarray:
    """Optional: clip x to physical bounds after each EM step.

    Some state components are physically bounded (sleep depth in
    [0, A_scale], testosterone >= 0, etc). Euler-Maruyama can push
    them past these bounds; clipping after each step restores them.
    """
    return jnp.array([
        jnp.clip(x[0], 0.0, 1.0),       # e.g. wakefulness
        jnp.maximum(x[1], 0.0),         # e.g. amplitude variable
        # ...
    ])


def amplitude_of_your_model(x: jnp.ndarray) -> jnp.ndarray:
    """Project latent state onto the scalar amplitude variable."""
    return x[<index_of_amplitude_component>]
```

### `parameters.py` skeleton

```python
"""
_vendored_models/your_model/parameters.py
==========================================
Default parameter dictionary for Your-Model.

Source:  Python-Model-Development-Simulation @ <commit-sha>
"""

from typing import Dict


_HOURS_PER_DAY = 24.0


def default_your_model_parameters() -> Dict[str, float]:
    """Healthy-baseline parameter dict, with all timescales in DAYS."""
    p = {
        # Drift / coupling parameters (dimensionless or in 1/days)
        'kappa':  6.67,
        # ...
    }

    # If upstream is in hours, convert here ONCE
    p['tau_state_1'] = 2.0 / _HOURS_PER_DAY    # hours -> days
    p['T_state_1']   = 0.05 * _HOURS_PER_DAY    # per-hour -> per-day variance

    return p
```

### `__init__.py`

```python
"""Your-Model vendored dynamics."""
from _vendored_models.your_model.dynamics_jax import (
    your_drift,
    your_diffusion,
    your_state_clip,
    amplitude_of_your_model,
)
from _vendored_models.your_model.parameters import (
    default_your_model_parameters,
)
__all__ = [
    "your_drift",
    "your_diffusion",
    "your_state_clip",
    "amplitude_of_your_model",
    "default_your_model_parameters",
]
```

### `README_vendored.md`

Use SWAT's `version_1/_vendored_models/swat/README_vendored.md` as a
template. Document:

* Source repo, commit, and date of vendoring.
* What's NOT vendored (observation model, identifiability machinery,
  scipy-NumPy simulator) and why.
* Time-unit conversion conventions.
* Update procedure for when the upstream model changes.

## Step 2 — Define your scenarios

A "scenario" is a clinically-meaningful combination of an initial-state
distribution + a reference-control schedule + (optionally) loss-weight
defaults. SWAT has three: `'insomnia'`, `'recovery'`, `'shift_work'`.

Plan your scenarios on paper first. For each:

* What's the **patient phenotype** at $t = 0$? (Determines
  `sample_initial_state`.)
* What are the **pre-intervention controls** the patient has been
  living with? (Determines `reference_schedule`.)
* What's the **clinical question**? (Helps you sanity-check whether
  the scenario is well-posed: is the patient meant to recover, stay
  stable, or worsen?)

Then write them as a dictionary:

```python
_SCENARIOS = {
    'baseline_unfit':   {'T_B': 0.5, 'Phi': 0.0, 'A_0': 0.1},
    'over_trained':     {'T_B': 1.0, 'Phi': 1.5, 'A_0': 0.3},
    # ...
}
```

## Step 3 — Write the adapter module

Create `version_1/adapters/your_model/adapter.py` with the following
structure. Five subsections, in order.

### 3.1 Imports + scenario table

```python
"""
adapters/your_model/adapter.py — Your-Model adapter for OT-Control.
"""
from __future__ import annotations
from typing import Dict, Optional

import jax
import jax.numpy as jnp

from ot_engine.types import BridgeProblem
from _vendored_models.your_model import (
    your_drift, your_diffusion, your_state_clip,
    amplitude_of_your_model, default_your_model_parameters,
)

# Optional helpers used in basin indicator
from _vendored_models.your_model.dynamics_jax import (
    healthy_attractor_check,    # adapter-specific
)


_SCENARIOS = {
    'scenario_a': {...},
    'scenario_b': {...},
}

YOUR_CONTROL_NAMES = ('T_B', 'Phi')  # human-readable

# Clinical control bounds — non-negative for unsigned quantities,
# signed for genuinely-signed quantities like phase shifts.
_YOUR_CONTROL_BOUNDS = (
    (0.0, 2.0),     # T_B: target intensity, non-negative
    (-3.0, 3.0),    # Phi: training phase, signed
)


def list_scenarios() -> tuple:
    return tuple(_SCENARIOS.keys())
```

### 3.2 Initial-state sampler

For each scenario the *patient phenotype* determines the initial
state distribution. Slow components (sleep architecture, baseline
hormones) are common across scenarios; the scenario-specific component
(testosterone amplitude in SWAT, fitness in FSA) is what changes.

```python
def _make_initial_sampler(scenario_amplitude_0: float):
    """Build an initial-state sampler for a given amplitude_0.

    Returns a callable rng, n -> jnp.ndarray of shape (n, dim_state).
    """
    def sample_init(rng: jax.Array, n: int) -> jnp.ndarray:
        rng_a, rng_b, ... = jax.random.split(rng, dim_state)
        state_1 = mid_value_1 + 0.05 * jax.random.normal(rng_a, (n,))
        state_1 = jnp.clip(state_1, lo_1, hi_1)
        # ...
        amp = scenario_amplitude_0 + 0.02 * jax.random.normal(rng_amp, (n,))
        amp = jnp.maximum(amp, 0.0)
        return jnp.stack([state_1, ..., amp], axis=1)
    return sample_init
```

### 3.3 Target sampler — DO THIS RIGHT

This is the step where SWAT had a bug in v1.0–v1.1: the hardcoded
target was unreachable. The recommended pattern is **model-derived
empirical target**: simulate the model once at adapter-build time
under "idealised healthy" controls, take the empirical distribution of
the amplitude variable at $t = D$ as the target.

```python
_HEALTHY_REFERENCE_CONTROLS = {'T_B': 1.5, 'Phi': 0.0}    # adjust per model
_HEALTHY_REFERENCE_AMP_0 = 1.0    # near deterministic equilibrium


def _build_healthy_target_sampler(horizon_days, dt_days, params,
                                    n_pool=1024, seed=0xBEEF):
    """One-off simulation under idealised-healthy controls; return sampler."""
    # Lazy local import to avoid circular import at module load
    from ot_engine.simulator import simulate_latent
    from ot_engine.types import BridgeProblem
    from ot_engine.policies.piecewise_constant import PiecewiseConstant

    h = _HEALTHY_REFERENCE_CONTROLS
    healthy_init = _make_initial_sampler(_HEALTHY_REFERENCE_AMP_0)
    healthy_ref = jnp.tile(
        jnp.array(list(h.values())), (horizon_days, 1)
    )
    healthy_sigma = jnp.ones((horizon_days, len(h)))

    # Stub problem — only the SDE bits and shape bits are used
    healthy_problem = BridgeProblem(
        name='your_model_healthy_pool',
        drift_fn_jax=your_drift,
        diffusion_fn_jax=your_diffusion,
        model_params=params,
        sample_initial_state=healthy_init,
        sample_target_amplitude=lambda r, n: jnp.zeros(n),  # stub, unused
        amplitude_of=amplitude_of_your_model,
        state_clip_fn=lambda x: your_state_clip(x, params),
        n_controls=len(h),
        control_bounds=_YOUR_CONTROL_BOUNDS,
        control_names=YOUR_CONTROL_NAMES,
        horizon_days=horizon_days,
        reference_schedule=healthy_ref,
        reference_sigma=healthy_sigma,
        n_particles=n_pool,
        dt_days=dt_days,
    )
    pol = PiecewiseConstant(horizon_days, len(h))
    rng = jax.random.PRNGKey(seed)
    _, A_D_pool, _ = simulate_latent(rng, healthy_problem, pol, healthy_ref)

    def sample_from_pool(rng_key, n):
        idx = jax.random.randint(rng_key, (n,), 0, A_D_pool.shape[0])
        return A_D_pool[idx]

    return sample_from_pool, A_D_pool
```

The pool is computed once at problem-construction time, not on every
loss evaluation.

### 3.4 Basin indicator

The basin indicator answers "is the patient in the healthy attractor
at terminal time?" — used by the closed-loop verification.

Build it from the empirical target pool's central interval:

```python
def _make_basin_indicator(target_pool: jnp.ndarray):
    A_lo = float(jnp.percentile(target_pool, 10.0))
    A_hi = float(jnp.percentile(target_pool, 90.0))

    def basin(x: jnp.ndarray, u_terminal: jnp.ndarray,
                params: Dict[str, float]) -> jnp.ndarray:
        amp = amplitude_of_your_model(x)
        amp_in_range = jnp.logical_and(amp >= A_lo, amp <= A_hi)
        # Optionally also test the model's deterministic stability at
        # u_terminal — see SWAT's entrainment_quality check.
        return amp_in_range
    return basin
```

### 3.5 Top-level constructor

This is the public API users call.

```python
def make_your_model_problem(
    scenario: str,
    horizon_days: int = 14,
    n_particles: int = 256,
    dt_days: float = 0.05,
    optim_steps: int = 2000,
    learning_rate: float = 1e-2,
    alpha_terminal: float = 1.0,
    alpha_transport: float = 0.01,
    alpha_reference: float = 0.001,
    reference_sigma: float = 1.0,
    model_params: Optional[Dict[str, float]] = None,
) -> BridgeProblem:
    if scenario not in _SCENARIOS:
        raise ValueError(
            f"Unknown scenario {scenario!r}. "
            f"Available: {list(_SCENARIOS.keys())}"
        )
    sc = _SCENARIOS[scenario]
    params = (model_params if model_params is not None
              else default_your_model_parameters())

    ref_schedule = jnp.tile(
        jnp.array([sc['T_B'], sc['Phi']]), (horizon_days, 1)
    )
    ref_sigma_array = jnp.ones(
        (horizon_days, len(YOUR_CONTROL_NAMES))
    ) * reference_sigma

    target_sampler, target_pool = _build_healthy_target_sampler(
        horizon_days=horizon_days, dt_days=dt_days, params=params
    )
    basin_indicator = _make_basin_indicator(target_pool)

    def _state_clip_with_params(x):
        return your_state_clip(x, params)

    return BridgeProblem(
        name=f'your_model_{scenario}',
        drift_fn_jax=your_drift,
        diffusion_fn_jax=your_diffusion,
        model_params=params,
        sample_initial_state=_make_initial_sampler(sc['A_0']),
        sample_target_amplitude=target_sampler,
        amplitude_of=amplitude_of_your_model,
        state_clip_fn=_state_clip_with_params,
        basin_indicator_fn=basin_indicator,
        n_controls=len(YOUR_CONTROL_NAMES),
        control_bounds=_YOUR_CONTROL_BOUNDS,
        control_names=YOUR_CONTROL_NAMES,
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

### 3.6 Adapter `__init__.py`

```python
"""Your-Model adapter for the OT-Control engine."""
from adapters.your_model.adapter import (
    make_your_model_problem,
    list_scenarios,
    YOUR_CONTROL_NAMES,
)
__all__ = [
    "make_your_model_problem",
    "list_scenarios",
    "YOUR_CONTROL_NAMES",
]
```

## Step 4 — Write a CLI runner

Clone `experiments/run_swat.py` to `experiments/run_your_model.py`.
Replace SWAT-specific bits (scenario list, control names, plot
calls). The boilerplate is mostly the same.

Key calls:

```python
problem = make_your_model_problem(scenario=args.scenario, ...)
pol = PiecewiseConstant.from_problem(problem)
schedule, trace = optimise_schedule(problem, pol, jax.random.PRNGKey(args.seed))
result = simulate_closed_loop(problem, pol, schedule, jax.random.PRNGKey(args.seed + 1),
                                n_realisations=args.n_realisations)

# Comparison with baselines
results = compare_schedules(problem, [
    schedule,
    zero_control_schedule(problem, pol),
    constant_reference_schedule(problem, pol),
    linear_interpolation_schedule(problem, pol, end_value=jnp.array([...])),
], jax.random.PRNGKey(args.seed + 2), n_realisations=args.n_realisations)
```

## Step 5 — Adapter-specific plots (optional but recommended)

`adapters/your_model/plots.py` — analogous to `swat/plots.py`. Four
standard figures:

1. `plot_schedule(schedule, reference_schedule)` — daily controls.
2. `plot_latent_paths(trajectories, t_grid, n_show=10)` — sample
   trajectories of key state components over time.
3. `plot_terminal_amplitude(amplitude_at_D, target_samples)` —
   histogram of simulated vs target amplitude marginal at $t = D$.
4. `plot_loss_trace(trace)` — total + per-component loss vs
   optimisation step.

## Step 6 — Tests

`tests/adapters/test_your_model_adapter.py` should at minimum cover:

* `make_your_model_problem(scenario)` returns a valid `BridgeProblem`
  for every scenario.
* Unknown scenario raises `ValueError`.
* `your_drift` is JIT-compatible and produces shape-correct output.
* End-to-end: optimise a problem with small `optim_steps` and
  `n_particles`, confirm the loss decreases.

`tests/adapters/test_your_model_phase5.py` should run the comparison-
with-baselines pipeline and assert:

1. The schedule respects `control_bounds` (no out-of-bounds values).
2. MMD within 5× best baseline.
3. Beats `zero_control` on at least one of (MMD, basin_fraction) on
   a scenario where intervention should genuinely help.

Use SWAT's `tests/adapters/test_swat_adapter.py` and
`tests/adapters/test_swat_phase5.py` as templates.

## Step 7 — Run it

```bash
cd version_1
pytest tests/adapters/test_your_model_adapter.py -v
pytest tests/adapters/test_your_model_phase5.py -v
python -m experiments.run_your_model --scenario scenario_a
```

Inspect the output schedule plot. If the controls are in clinically
meaningful directions for the scenario, you're done. If not, see the
debugging checklist in §8 below.

## Step 8 — Debugging checklist

If the optimised schedule looks wrong, work through these in order:

* [ ] Are all values inside `control_bounds`? If yes but they look
      saturated at a bound, the optimiser found that bound is binding.
      Either tighten/loosen the bound or accept it as the answer.
* [ ] Is the reference schedule inside the bounds? If not, `init_params`
      starts the optimiser at a clipped point and the reference penalty
      pulls toward an unreachable target.
* [ ] Is the target distribution actually reachable? Run a healthy-
      reference simulation manually and compare its terminal amplitude
      distribution to your target. If they're very different, the
      optimiser will hunt counterintuitively. Use the model-derived
      empirical target pattern.
* [ ] Are signs correct in `drift_fn_jax`? Specifically: does pushing a
      control "up" produce the expected effect on the relevant state
      component?
* [ ] Are time units consistent? Check that timescales are in days and
      noise temperatures are scaled correspondingly.
* [ ] Is `dt_days` small enough? Try halving it; if the schedule
      changes substantially, your EM step is too coarse.
* [ ] Is the loss actually decreasing? If `trace.losses_total[-1]` is
      higher than `losses_total[0]`, the optimiser is diverging — try a
      smaller `learning_rate`.

## Next

`04_worked_example_FSA.md` — a complete walkthrough using a different
model and different controls.
