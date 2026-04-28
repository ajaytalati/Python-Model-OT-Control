"""
adapters/swat/adapter.py — SWAT adapter for the OT-Control engine.
====================================================================
Date:    26 April 2026
Version: 1.0.0

Constructs BridgeProblem instances for the SWAT (Sleep-Wake-Adenosine-
Testosterone) model. Three canonical clinical scenarios are exposed:

    * 'insomnia'     — V_h=0.2, V_n=3.5, V_c=0:    chronic-stress amplitude collapse
    * 'recovery'     — V_h=1.0, V_n=0.3, V_c=0:    rising T from flatline (T_0=0.05)
    * 'shift_work'   — V_h=1.0, V_n=0.3, V_c=6.0:  phase-shift collapse

In each scenario the *patient's current state* is what determines the
initial-state distribution rho_0; the *reference schedule* holds the
patient's pre-intervention controls constant (the "do nothing" baseline)
so the OT optimisation finds a deviation that drives testosterone to the
healthy target T_star ≈ 0.55.

This module never imports the OT engine's internals — only the public
contract (BridgeProblem). It also never imports the SWAT model's
internals beyond the JAX-native `swat_drift`, `swat_diffusion`,
`swat_state_clip`, `amplitude_of_swat`, and `default_swat_parameters`.
"""

from __future__ import annotations

from typing import Dict, Optional

import jax
import jax.numpy as jnp

from ot_engine.types import BridgeProblem
from _vendored_models.swat import (
    swat_drift,
    swat_diffusion,
    swat_state_clip,
    amplitude_of_swat,
    entrainment_quality,
    default_swat_parameters,
)


# =========================================================================
# Scenario specifications
# =========================================================================

# Each entry maps a scenario name to the patient's pre-intervention
# (V_h, V_n, V_c) and initial testosterone amplitude T_0. The other
# states (W, Z, a) are sampled from physically reasonable priors common
# to all scenarios.

_SCENARIOS: Dict[str, Dict[str, float]] = {
    'insomnia':   {'V_h': 0.2, 'V_n': 3.5, 'V_c': 0.0, 'T_0': 0.5},
    'recovery':   {'V_h': 1.0, 'V_n': 0.3, 'V_c': 0.0, 'T_0': 0.05},
    'shift_work': {'V_h': 1.0, 'V_n': 0.3, 'V_c': 6.0, 'T_0': 0.5},
}

# "Idealised healthy" controls used to derive the clinical target
# distribution mu_D^A by simulation. Higher V_h, lower V_n, no phase
# shift -- maximally favourable controls within the clinical box. The
# target is then the empirical distribution of T(D) under these controls
# starting from a healthy initial state. This ties the target to *what
# the model predicts is achievable for a healthy patient*, which is a
# moving function of the model's parameters and noise levels.
_HEALTHY_REFERENCE_CONTROLS = {'V_h': 2.0, 'V_n': 0.1, 'V_c': 0.0}
_HEALTHY_REFERENCE_T_0 = 0.55     # near deterministic Stuart-Landau equilibrium

# Display constant — the deterministic Stuart-Landau equilibrium value
# of T at perfect entrainment (E ~ 1) and the historical "spec target"
# from §7 of the SWAT documentation. Kept as a public constant for use
# as a reference line on plots and for distance-to-target reporting in
# `run_swat.py`. This is NOT the actual loss target — the loss target
# is the model-derived empirical distribution constructed at problem-
# build time by `_build_healthy_target_sampler`. The two are decoupled
# because under the post-2026-04-26 noise levels the deterministic
# equilibrium is not reachable by any clinically-sensible schedule.
#
# Bumped 0.55 -> 0.9 on 2026-04-28 alongside the V_h-anabolic
# structural fix: under the corrected dynamics the healthy reference
# (V_h=2, V_n=0.1, V_c=0) produces T(D) ~ 0.85-0.95, so the display
# constant is set to the rounded mid-band.
T_STAR_HEALTHY = 0.9


# Clinical control bounds (review fix: V_h lower bound corrected from
# -2.0 to 0.0 -- vitality reserve cannot be negative; the previous bound
# allowed clinically-meaningless schedules. V_c remains signed because
# phase shifts are genuinely signed -- positive = morningward, negative
# = eveningward).
_SWAT_CONTROL_BOUNDS = ((0.0, 4.0),    # V_h: vitality reserve, non-negative
                         (0.0, 5.0),    # V_n: chronic load, non-negative
                         (-12.0, 12.0))  # V_c: phase shift in hours, signed


def list_scenarios() -> tuple:
    """Return the names of available SWAT scenarios."""
    return tuple(_SCENARIOS.keys())


# =========================================================================
# Initial-state and target samplers
# =========================================================================

def _make_initial_sampler(scenario_T_0: float):
    """Initial state x_0 = (W, Z, a, T) for the patient.

    Slow states W, Z, a are sampled from a small Gaussian around their
    typical mid-range values. T is sampled near the scenario's pathological
    initial value with a small spread.
    """
    def sample_init(rng: jax.Array, n: int) -> jnp.ndarray:
        rng_w, rng_z, rng_a, rng_t = jax.random.split(rng, 4)
        W = 0.5 + 0.05 * jax.random.normal(rng_w, (n,))
        Z = 3.5 + 0.3 * jax.random.normal(rng_z, (n,))
        a = 0.5 + 0.05 * jax.random.normal(rng_a, (n,))
        T = scenario_T_0 + 0.02 * jax.random.normal(rng_t, (n,))
        # Apply physical clipping to the initial states.
        W = jnp.clip(W, 0.0, 1.0)
        Z = jnp.clip(Z, 0.0, 6.0)
        a = jnp.maximum(a, 0.0)
        T = jnp.maximum(T, 0.0)
        return jnp.stack([W, Z, a, T], axis=1)
    return sample_init


def _build_healthy_target_sampler(horizon_days: int, dt_days: float,
                                    params: Dict[str, float],
                                    n_pool: int = 1024,
                                    seed: int = 0xBEEF):
    """Build a target sampler from a one-off "healthy reference" simulation.

    Rationale
    ---------
    The clinical target distribution mu_D^A should reflect what the
    *model* predicts T(D) looks like for an idealised healthy patient
    over the same horizon. Hardcoding a target like N(0.55, 0.05^2)
    can be unreachable when the model's noise level (or a parameter
    re-tune) shifts the achievable amplitude downward; the optimiser
    then hunts in clinically-counterintuitive directions chasing a
    target the model literally cannot reach.

    Construction
    ------------
    Run the SWAT SDE for `horizon_days` from a healthy initial state
    (T_0 ~ 0.55) under maximally-favourable controls (V_h=2, V_n=0.1,
    V_c=0) with `n_pool` particles. Take the empirical distribution of
    T at terminal time as the target. Each call to the returned
    sampler draws bootstrap samples from this fixed empirical pool, so
    the target is deterministic given the seed. The pool is computed
    once at adapter-build time, not on every loss evaluation.

    Args:
        horizon_days: Schedule horizon D in days.
        dt_days: EM step size for the reference simulation.
        params: SWAT parameter dictionary.
        n_pool: Number of particles in the empirical pool. 1024 gives
            <1% sampling error at typical kernel bandwidths.
        seed: Fixed seed for reproducibility of the reference pool.

    Returns:
        A function `sample(rng, n) -> (n,) array` drawing from the
        empirical pool.
    """
    # Lazy local import to avoid a circular import at module load time.
    from ot_engine.simulator import simulate_latent
    from ot_engine.types import BridgeProblem
    from ot_engine.policies.piecewise_constant import PiecewiseConstant

    # Healthy initial state: T near steady state, W/Z/a at mid-range.
    h = _HEALTHY_REFERENCE_CONTROLS
    init_T_0 = _HEALTHY_REFERENCE_T_0

    healthy_init_sampler = _make_initial_sampler(init_T_0)
    healthy_ref = jnp.tile(
        jnp.array([h['V_h'], h['V_n'], h['V_c']]), (horizon_days, 1)
    )
    healthy_sigma = jnp.ones((horizon_days, 3))

    # Build a stub problem with healthy controls. The simulator only
    # reads drift/diffusion/initial-state/clip, plus n_particles and
    # control_bounds — nothing else here matters for the forward
    # simulation. We need to satisfy the BridgeProblem validator, so
    # we supply minimal-but-valid stubs for the unused fields.
    def _stub_target(rng, n):
        return jnp.zeros(n)
    def _state_clip_with_params(x):
        return swat_state_clip(x, params)
    healthy_problem = BridgeProblem(
        name='swat_healthy_target_pool',
        drift_fn_jax=swat_drift,
        diffusion_fn_jax=swat_diffusion,
        model_params=params,
        sample_initial_state=healthy_init_sampler,
        sample_target_amplitude=_stub_target,
        amplitude_of=amplitude_of_swat,
        state_clip_fn=_state_clip_with_params,
        n_controls=3,
        control_bounds=_SWAT_CONTROL_BOUNDS,
        control_names=('V_h', 'V_n', 'V_c'),
        horizon_days=horizon_days,
        reference_schedule=healthy_ref,
        reference_sigma=healthy_sigma,
        n_particles=n_pool,
        dt_days=dt_days,
    )
    pol = PiecewiseConstant(horizon_days, 3)
    rng = jax.random.PRNGKey(seed)
    _, A_D_pool, _ = simulate_latent(rng, healthy_problem, pol, healthy_ref)
    # A_D_pool is the empirical "what does the model say healthy T(D)
    # looks like under maximally-favourable controls" distribution.

    def sample_from_pool(rng_key: jax.Array, n: int) -> jnp.ndarray:
        idx = jax.random.randint(rng_key, (n,), 0, A_D_pool.shape[0])
        return A_D_pool[idx]

    return sample_from_pool, A_D_pool


# =========================================================================
# Basin indicator (for closed-loop verification, Phase 5)
# =========================================================================

def _make_basin_indicator(target_pool: jnp.ndarray):
    """Build a basin-indicator closure over the empirical target pool.

    The healthy basin is now defined as: terminal entrainment is
    super-critical AND terminal T is within the empirical target pool's
    central 80% interval. This replaces the previous "T within 30% of
    0.55" check, which was hardcoded to the (no-longer-reachable) value
    0.55 and missed the post-noise-correction shift in the model's
    achievable T.

    Args:
        target_pool: 1-D array of T(D) samples from the healthy
            reference simulation (output of _build_healthy_target_sampler).

    Returns:
        A function `basin(x, u_terminal, params) -> jnp scalar bool`.
    """
    T_lo = float(jnp.percentile(target_pool, 10.0))
    T_hi = float(jnp.percentile(target_pool, 90.0))

    def basin(x: jnp.ndarray, u_terminal: jnp.ndarray,
                params: Dict[str, float]) -> jnp.ndarray:
        V_h, V_n, V_c = u_terminal[0], u_terminal[1], u_terminal[2]
        E = entrainment_quality(x[0], x[1], x[2], x[3], V_h, V_n, V_c, params)
        E_crit = -params['mu_0'] / params['mu_E']
        T_in_range = jnp.logical_and(x[3] >= T_lo, x[3] <= T_hi)
        return jnp.logical_and(E > E_crit, T_in_range)

    return basin


# =========================================================================
# Reference schedule
# =========================================================================

def _build_reference_schedule(V_h: float, V_n: float, V_c: float,
                                horizon_days: int) -> jnp.ndarray:
    """Reference = pathological values held constant ("do nothing")."""
    return jnp.tile(jnp.array([V_h, V_n, V_c]), (horizon_days, 1))


# =========================================================================
# Top-level adapter constructor
# =========================================================================

def make_swat_problem(
    scenario: str,
    horizon_days: int = 21,
    n_particles: int = 128,
    dt_days: float = 0.05,
    optim_steps: int = 800,
    learning_rate: float = 5e-2,
    alpha_terminal: float = 1.0,
    alpha_transport: float = 0.01,
    alpha_reference: float = 0.001,
    reference_sigma: float = 1.0,
    model_params: Optional[Dict[str, float]] = None,
) -> BridgeProblem:
    """Build a BridgeProblem for the named SWAT scenario.

    Args:
        scenario: One of 'insomnia', 'recovery', 'shift_work'. See
            list_scenarios() for the canonical list.
        horizon_days: Schedule length in days (default 21 = 3 weeks).
        n_particles: Monte-Carlo particle count.
        dt_days: Euler-Maruyama timestep in days. Default 0.05 = 1.2 hours
            which resolves tau_W = 2 hours with ~1.7 steps per
            characteristic time. Set lower for higher precision.
        optim_steps: Maximum Adam steps in optimisation.
        learning_rate: Adam learning rate.
        alpha_terminal, alpha_transport, alpha_reference: Loss-term weights.
            Defaults are tuned for SWAT — terminal heavily weighted, the
            other two light because the schedule has plenty of room to
            move from the pathological reference.
        reference_sigma: Per-day, per-control width of the Gaussian
            reference. Larger values make the reference looser and let
            the schedule deviate further. Default 1.0 covers ±3 std-devs
            of intervention magnitude.
        model_params: Override the SWAT parameter dictionary. Defaults to
            `default_swat_parameters()`.

    Returns:
        A BridgeProblem ready to pass to `optimise_schedule(...)`.

    Raises:
        ValueError: If the scenario name is not recognised.
    """
    if scenario not in _SCENARIOS:
        raise ValueError(
            f"Unknown SWAT scenario '{scenario}'. "
            f"Available: {list(_SCENARIOS.keys())}"
        )
    sc = _SCENARIOS[scenario]
    params = model_params if model_params is not None else default_swat_parameters()

    ref_schedule = _build_reference_schedule(
        V_h=sc['V_h'], V_n=sc['V_n'], V_c=sc['V_c'],
        horizon_days=horizon_days,
    )
    ref_sigma_array = jnp.ones((horizon_days, 3)) * reference_sigma

    # Bind params into the state-clip closure so A_scale stays consistent
    # with the configured model_params (review fix L-3).
    def _state_clip_with_params(x):
        return swat_state_clip(x, params)

    # Build the model-derived target sampler and basin indicator.
    target_sampler, target_pool = _build_healthy_target_sampler(
        horizon_days=horizon_days, dt_days=dt_days, params=params
    )
    basin_indicator = _make_basin_indicator(target_pool)

    return BridgeProblem(
        name=f'swat_{scenario}',
        drift_fn_jax=swat_drift,
        diffusion_fn_jax=swat_diffusion,
        model_params=params,
        sample_initial_state=_make_initial_sampler(sc['T_0']),
        sample_target_amplitude=target_sampler,
        amplitude_of=amplitude_of_swat,
        state_clip_fn=_state_clip_with_params,
        basin_indicator_fn=basin_indicator,
        n_controls=3,
        control_bounds=_SWAT_CONTROL_BOUNDS,
        control_names=SWAT_CONTROL_NAMES,
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


# Convenience tuple of control names — same as the BridgeProblem.control_names
# field; exported for callers that need it without going through a problem.
SWAT_CONTROL_NAMES = ('V_h', 'V_n', 'V_c')
