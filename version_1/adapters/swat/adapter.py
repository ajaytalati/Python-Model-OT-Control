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

# Healthy steady-state target for T. From the spec §7: T_star ≈ 0.55 when
# E_dyn ≈ 0.64 (V_h=1, V_n=0.3, V_c=0).
T_STAR_HEALTHY = 0.55
T_STAR_TARGET_STD = 0.05


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


def _sample_target_T_healthy(rng: jax.Array, n: int) -> jnp.ndarray:
    """Target distribution: T near the healthy steady state."""
    return T_STAR_HEALTHY + T_STAR_TARGET_STD * jax.random.normal(rng, (n,))


# =========================================================================
# Basin indicator (for closed-loop verification, Phase 5)
# =========================================================================

def _basin_indicator(x: jnp.ndarray, u_terminal: jnp.ndarray,
                      params: Dict[str, float]) -> jnp.ndarray:
    """Healthy basin indicator: physiological state in the recovered region.

    A patient is in the healthy basin at terminal time iff:
      - the entrainment quality (computed with their *terminal* schedule
        controls) exceeds the bifurcation threshold (so that the
        Stuart-Landau dynamics are super-critical and T is attracted to
        a non-zero stable point);
      - the testosterone amplitude is within 30% of T_star.

    We deliberately do NOT impose magnitude constraints on the control
    vector itself: any V_h, V_n, V_c are allowed if they produce the
    healthy physiological state. The clinician's preference for
    physiological-magnitude controls is captured by the reference KL
    term in the loss, not by the basin indicator.

    Args:
        x: Latent state at terminal time, shape (4,).
        u_terminal: Terminal-day control vector (V_h, V_n, V_c).
        params: SWAT model parameter dictionary.

    Returns:
        Boolean (jnp scalar) — True iff in healthy basin.
    """
    V_h, V_n, V_c = u_terminal[0], u_terminal[1], u_terminal[2]
    E = entrainment_quality(x[0], x[1], x[2], x[3], V_h, V_n, V_c, params)
    E_crit = -params['mu_0'] / params['mu_E']
    T_in_range = jnp.abs(x[3] - T_STAR_HEALTHY) < 0.3 * T_STAR_HEALTHY
    return jnp.logical_and(E > E_crit, T_in_range)


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

    return BridgeProblem(
        name=f'swat_{scenario}',
        drift_fn_jax=swat_drift,
        diffusion_fn_jax=swat_diffusion,
        model_params=params,
        sample_initial_state=_make_initial_sampler(sc['T_0']),
        sample_target_amplitude=_sample_target_T_healthy,
        amplitude_of=amplitude_of_swat,
        state_clip_fn=_state_clip_with_params,
        basin_indicator_fn=_basin_indicator,
        n_controls=3,
        control_bounds=((-2.0, 4.0), (0.0, 5.0), (-12.0, 12.0)),
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
