"""
adapters/fsa_high_res/adapter.py — FSA adapter for the OT-Control engine.
==========================================================================
Date:    26 April 2026
Version: 1.0.0

Constructs BridgeProblem instances for the FSA-high-res
(Fitness-Strain-Amplitude) model. Three canonical clinical scenarios:

    * 'unfit_recovery'    B_0=0.05, F_0=0.05, A_0=0.30,  ref (T_B=0.0, Phi=0.00)
                          --- sedentary patient with low fitness, low strain,
                              and A near a weak fixed point. Goal: build B
                              up so mu(B,F) rises and A grows toward the
                              healthy target ~ 0.78.

    * 'over_trained'      B_0=0.60, F_0=0.60, A_0=0.05,  ref (T_B=0.7, Phi=0.50)
                          --- athlete past the overtraining cliff: F so
                              high that mu(B,F) is barely positive, A
                              drained. Goal: drop Phi to let F drain and
                              mu recover; keep T_B moderate.

    * 'detrained_athlete' B_0=0.20, F_0=0.05, A_0=0.30,  ref (T_B=0.3, Phi=0.15)
                          --- returning athlete after a layoff. Mid-range
                              B, low F, A at local steady state. Goal:
                              raise T_B and add modest Phi to rebuild.

For each scenario the *patient's current state* determines rho_0; the
*reference schedule* holds the patient's pre-intervention controls
constant; the OT optimisation finds a deviation that drives endocrine
amplitude A toward the model-derived healthy target distribution.

This module never imports the OT engine's internals -- only the public
contract (BridgeProblem). It also never imports the FSA model's
internals beyond the JAX-native `fsa_drift`, `fsa_diffusion`,
`fsa_state_clip`, `amplitude_of_fsa`, `healthy_attractor_check`, and
`default_fsa_parameters`.
"""

from __future__ import annotations

from typing import Dict, Optional

import jax
import jax.numpy as jnp

from ot_engine.types import BridgeProblem
from _vendored_models.fsa_high_res import (
    fsa_drift,
    fsa_diffusion,
    fsa_state_clip,
    amplitude_of_fsa,
    healthy_attractor_check,
    default_fsa_parameters,
)


# =========================================================================
# Scenario specifications
# =========================================================================

# Each entry maps a scenario name to:
#   - the patient's pre-intervention controls (T_B, Phi) used as the
#     reference schedule (the "do nothing / current habit" baseline);
#   - the patient's initial state (B_0, F_0, A_0) at t = 0.

_SCENARIOS: Dict[str, Dict[str, float]] = {
    # Sedentary patient: low fitness, low strain. mu(B,F) is barely
    # positive at this state, so A has a viable but weak fixed point
    # near sqrt(0.029/0.20) ~= 0.38. Goal: build B (T_B up) to push mu
    # higher and grow A toward the healthy target ~ 0.85.
    'unfit_recovery':    {'T_B': 0.0, 'Phi': 0.0,
                           'B_0': 0.05, 'F_0': 0.05, 'A_0': 0.30},

    # Overtrained athlete: high B, but F so high that mu(B,F) ~ 0,
    # tanking A. Goal: drop Phi sharply to let F drain and mu recover;
    # T_B should be moderate (don't drop fitness while recovering).
    'over_trained':      {'T_B': 0.7, 'Phi': 0.5,
                           'B_0': 0.60, 'F_0': 0.60, 'A_0': 0.05},

    # Returning athlete: mid-range B, low F, A at local steady state.
    # Goal: increase T_B to rebuild fitness toward the healthy target,
    # add modest Phi without crashing F.
    'detrained_athlete': {'T_B': 0.3, 'Phi': 0.15,
                           'B_0': 0.20, 'F_0': 0.05, 'A_0': 0.30},
}


# Idealised "moderately healthy" reference controls used to derive the
# clinical target distribution mu_D^A by simulation. Same approach as
# the SWAT adapter: tie the target to *what the model predicts is
# achievable* under realistic intervention, ensuring the target is
# always reachable.
#
# Under these settings the model produces a target distribution
# tightly concentrated around A ~ 0.78 (sd ~ 0.03) at terminal time
# (verified empirically; stable across PRNG seeds). This is below the
# deterministic A* = sqrt(mu/eta) ~ 0.88 fixed point because the
# horizon is finite and noise broadens the marginal.
_HEALTHY_REFERENCE_CONTROLS = {'T_B': 0.5, 'Phi': 0.05}
_HEALTHY_REFERENCE_INIT = {'B_0': 0.3, 'F_0': 0.05, 'A_0': 0.40}


# Clinical control bounds. T_B is a dimensionless target in [0, 1]
# (cannot prescribe negative training-load nor more than 100% target).
# Phi is non-negative training intensity; upper bound 2.0 covers
# physiologically extreme volume blocks while ruling out runaway
# values that don't reflect any real prescription.
_FSA_CONTROL_BOUNDS = (
    (0.0, 1.0),     # T_B: training-load target
    (0.0, 2.0),     # Phi: training intensity
)


FSA_CONTROL_NAMES = ('T_B', 'Phi')

# Display constant: median of the model-derived empirical target
# distribution under _HEALTHY_REFERENCE_CONTROLS / _HEALTHY_REFERENCE_INIT.
# Verified empirically at 0.78 with sd ~ 0.001 across PRNG seeds (see
# tests/adapters/test_fsa_adapter.py::test_a_star_healthy_matches_pool).
# Used as a reference line on plots and in distance-to-target metrics.
# Decoupled from the actual loss target (which is the empirical pool;
# see _build_healthy_target_sampler) — this is the headline scalar.
A_STAR_HEALTHY = 0.78


def list_scenarios() -> tuple:
    """Return the names of available FSA scenarios."""
    return tuple(_SCENARIOS.keys())


# =========================================================================
# Initial-state sampler
# =========================================================================

def _make_initial_sampler(B_0: float, F_0: float, A_0: float):
    """Build an initial-state sampler for given (B_0, F_0, A_0).

    Slow components (B, F, A) are sampled from small Gaussians around
    their nominal values, with physical-bound clipping. The same
    spread is used across scenarios; only the centres differ.

    Args:
        B_0, F_0, A_0: scenario-specific nominal initial values.

    Returns:
        A callable rng, n -> jnp.ndarray of shape (n, 3).
    """
    def sample_init(rng: jax.Array, n: int) -> jnp.ndarray:
        rng_b, rng_f, rng_a = jax.random.split(rng, 3)
        B = jnp.clip(B_0 + 0.02 * jax.random.normal(rng_b, (n,)), 0.0, 1.0)
        F = jnp.maximum(F_0 + 0.02 * jax.random.normal(rng_f, (n,)), 0.0)
        A = jnp.maximum(A_0 + 0.005 * jax.random.normal(rng_a, (n,)), 0.0)
        return jnp.stack([B, F, A], axis=1)
    return sample_init


# =========================================================================
# Model-derived target sampler
# =========================================================================

def _build_healthy_target_sampler(horizon_days: int, dt_days: float,
                                    params: Dict[str, float],
                                    n_pool: int = 1024,
                                    seed: int = 0xBEEF):
    """Build a target sampler from a one-off "healthy reference" simulation.

    See SWAT's `_build_healthy_target_sampler` for the rationale. In
    one paragraph: hardcoding a target for A can be unreachable when
    the model's noise level or parameters change. Here we run the FSA
    SDE for `horizon_days` from a healthy initial state (A_0 = 0.40,
    below the Stuart-Landau fixed point so the trajectory has room to
    grow) under moderate-volume controls (T_B=0.5, Phi=0.05). The
    empirical distribution of A at terminal time concentrates around
    0.78 ± 0.03 (verified across seeds); this is the loss target,
    so it is always reachable by something close to the healthy
    trajectory.

    Args:
        horizon_days: schedule horizon D in days.
        dt_days: EM step size for the reference simulation.
        params: FSA parameter dictionary.
        n_pool: number of particles in the empirical pool.
        seed: fixed seed for reproducibility.

    Returns:
        Pair (sample_fn, target_pool):
            sample_fn(rng, n) -> (n,) bootstrap draws from the pool.
            target_pool: the (n_pool,) empirical sample, exposed so
                callers (basin indicator, plot helpers) can use the
                full pool rather than just bootstrap draws.
    """
    # Lazy local imports to avoid a circular import at module load.
    from ot_engine.simulator import simulate_latent
    from ot_engine.types import BridgeProblem
    from ot_engine.policies.piecewise_constant import PiecewiseConstant

    h = _HEALTHY_REFERENCE_CONTROLS
    healthy_init = _make_initial_sampler(**_HEALTHY_REFERENCE_INIT)
    healthy_ref = jnp.tile(jnp.array([h['T_B'], h['Phi']]),
                             (horizon_days, 1))
    healthy_sigma = jnp.ones((horizon_days, 2))

    def _stub_target(rng, n):
        return jnp.zeros(n)

    def _state_clip(x):
        return fsa_state_clip(x, params)

    healthy_problem = BridgeProblem(
        name='fsa_healthy_target_pool',
        drift_fn_jax=fsa_drift,
        diffusion_fn_jax=fsa_diffusion,
        model_params=params,
        sample_initial_state=healthy_init,
        sample_target_amplitude=_stub_target,
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

    def sample_from_pool(rng_key: jax.Array, n: int) -> jnp.ndarray:
        idx = jax.random.randint(rng_key, (n,), 0, A_D_pool.shape[0])
        return A_D_pool[idx]

    return sample_from_pool, A_D_pool


# =========================================================================
# Basin indicator
# =========================================================================

def _make_basin_indicator(target_pool: jnp.ndarray):
    """Build a basin-indicator closure over the empirical target pool.

    Healthy basin condition (closed-loop verification metric):
      - Terminal A in the empirical target pool's central 80% interval;
      - mu(B, F) > 0 at the terminal state -- the patient's
        physiology genuinely supports a healthy attractor, not just a
        transiently-elevated A.

    This pairs an *empirical* test on A (matches what the schedule
    achieved) with a *structural* test on (B, F) (matches whether the
    achievement is sustainable under the model's own dynamics).

    Args:
        target_pool: 1-D array of A(D) samples from the healthy
            reference simulation.

    Returns:
        A function basin(x, u_terminal, params) -> jnp scalar bool.
    """
    A_lo = float(jnp.percentile(target_pool, 10.0))
    A_hi = float(jnp.percentile(target_pool, 90.0))

    def basin(x: jnp.ndarray, u_terminal: jnp.ndarray,
                params: Dict[str, float]) -> jnp.ndarray:
        del u_terminal     # FSA basin is a property of the state, not the schedule's terminal control
        B, F, A = x[0], x[1], x[2]
        A_in_range = jnp.logical_and(A >= A_lo, A <= A_hi)
        mu_positive = healthy_attractor_check(B, F, params)
        return jnp.logical_and(A_in_range, mu_positive)

    return basin


# =========================================================================
# Reference schedule
# =========================================================================

def _build_reference_schedule(T_B: float, Phi: float,
                                horizon_days: int) -> jnp.ndarray:
    """Reference = pre-intervention controls held constant ('do nothing')."""
    return jnp.tile(jnp.array([T_B, Phi]), (horizon_days, 1))


# =========================================================================
# Top-level adapter constructor
# =========================================================================

def make_fsa_problem(
    scenario: str,
    horizon_days: int = 14,
    n_particles: int = 256,
    dt_days: float = 0.05,
    optim_steps: int = 2000,
    learning_rate: float = 5e-3,
    alpha_terminal: float = 1.0,
    alpha_transport: float = 0.05,
    alpha_reference: float = 0.001,
    reference_sigma: float = 1.0,
    model_params: Optional[Dict[str, float]] = None,
) -> BridgeProblem:
    """Build a BridgeProblem for the named FSA scenario.

    Notes on optimisation quality
    -----------------------------
    The FSA optimiser produces bounds-respecting schedules but can
    pin to clinically-counterintuitive points for two related reasons:

    1. **Boundary saturation**. For scenarios whose reference is at a
       control bound (notably `unfit_recovery` with reference at
       T_B=0, Phi=0, both at the lower bound), the reference penalty
       pulls toward the bound and Adam's clipped sub-gradient cannot
       push past it. Result: the schedule pins at the reference.

    2. **Single-bandwidth MMD geometry**. For scenarios where the
       source A-distribution at theta_0 is far from the target pool
       (mean 0.78), the Gaussian-kernel MMD's curvature is low along
       the clinically-meaningful descent direction. Adam still
       receives finite, non-zero gradient, but the descent can favour
       directions that don't reduce the *clinically-relevant* gap.

    Multi-bandwidth MMD (F2 in docs/Future_Features.md) addresses (2)
    and is the principled fix. In the interim, users can:
      - Tighten the target by overriding `_HEALTHY_REFERENCE_INIT`
        and `_HEALTHY_REFERENCE_CONTROLS` (closer target = stronger
        gradient at theta_0).
      - Warm-start the optimiser at a hand-crafted theta closer to a
        sensible solution and let Adam refine.
      - For scenarios with reference at a bound, move the reference
        slightly inside the bound so Adam can move in either direction.

    Args:
        scenario: One of 'unfit_recovery', 'over_trained',
            'detrained_athlete'. See list_scenarios().
        horizon_days: Schedule length in days. Default 14 matches the
            upstream proof-of-principle horizon.
        n_particles: Monte-Carlo particle count for the optimiser's
            inner simulator.
        dt_days: Euler-Maruyama timestep in days. Default 0.05 (=
            1.2 hours) is fine-grained enough that state-dependent
            diffusion sqrt(F) and sqrt(A+eps) don't blow up near the
            boundary.
        optim_steps: Maximum Adam steps in the optimisation. The
            convergence detector typically stops earlier.
        learning_rate: Adam learning rate. Default 5e-3 is
            conservative -- larger values can blow up the schedule
            on the unfit_recovery scenario where the patient is at
            the boundary of viable mu(B,F) > 0.
        alpha_terminal, alpha_transport, alpha_reference: Loss-term
            weights. Defaults: terminal heavily weighted (we want the
            optimiser to chase the target distribution); transport
            mild (don't penalise reasonable-magnitude prescriptions);
            reference very mild (the recovery scenario's reference is
            pathological for two of the three cases, so the optimiser
            should be free to deviate).
        reference_sigma: Per-day, per-control width of the Gaussian
            reference. Default 1.0 is loose -- the optimiser can
            deviate substantially from baseline.
        model_params: Override the FSA parameter dictionary. Defaults
            to default_fsa_parameters().

    Returns:
        A BridgeProblem ready to pass to optimise_schedule(...).

    Raises:
        ValueError: If the scenario name is not recognised.
    """
    if scenario not in _SCENARIOS:
        raise ValueError(
            f"Unknown FSA scenario {scenario!r}. "
            f"Available: {list(_SCENARIOS.keys())}"
        )
    sc = _SCENARIOS[scenario]
    params = (model_params if model_params is not None
              else default_fsa_parameters())

    ref_schedule = _build_reference_schedule(
        T_B=sc['T_B'], Phi=sc['Phi'], horizon_days=horizon_days,
    )
    ref_sigma_array = jnp.ones((horizon_days, 2)) * reference_sigma

    # Bind params into the state-clip closure for signature consistency
    # with SWAT (and to be future-proof if FSA's clip ever needs params).
    def _state_clip_with_params(x):
        return fsa_state_clip(x, params)

    # Build the model-derived target sampler and basin indicator.
    target_sampler, target_pool = _build_healthy_target_sampler(
        horizon_days=horizon_days, dt_days=dt_days, params=params
    )
    basin_indicator = _make_basin_indicator(target_pool)

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
