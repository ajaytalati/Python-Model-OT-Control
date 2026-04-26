"""
ot_engine/simulator.py — JAX Euler-Maruyama latent forward simulator.
=======================================================================
Date:    25 April 2026
Version: 1.0.0
Status:  Phase 2 deliverable.

AD-traced Euler-Maruyama driver for the latent SDE under a parameterised
control schedule. The simulator is the central computational primitive of
the engine — every gradient ultimately flows through it.

KEY DESIGN POINTS
-----------------
* Reproducible noise via JAX key splits — every call with the same theta
  and the same root rng gives bit-identical trajectories, so jax.grad is
  well-defined and the optimisation is deterministic.
* The Brownian increments are *frozen as inputs*, not part of theta. The
  gradient flows through the dependence of the trajectory on theta via
  the drift and the controls (the standard "reparameterisation trick").
* vmap over particles + lax.scan over time. Both are jit-friendly.
* The policy object is closed over (Python-level), not a pytree leaf.
  The integrator only sees jnp arrays inside the scan body.
* Returns the *full* trajectory (for plotting and verification) and a
  separate amplitude_at_D vector (for the loss).

NUMERICS
--------
The Euler-Maruyama update for a state component i is
    x_{t+dt}[i] = x_t[i] + f_i(t, x_t, u_t) * dt + sigma_i(x_t) * sqrt(dt) * xi
where xi ~ N(0, 1) is independent per (particle, time, dim).
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from ot_engine.policies._abstract import ControlPolicy
from ot_engine.types import BridgeProblem


# =========================================================================
# CORE SIMULATOR
# =========================================================================

def simulate_latent(
    rng: jax.Array,
    problem: BridgeProblem,
    policy: ControlPolicy,
    theta: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Forward-simulate the latent SDE under the parameterised schedule.

    Args:
        rng: Root JAX PRNG key. Two splits are taken: one for initial
            states, one for the per-step Brownian increments.
        problem: BridgeProblem with the SDE specification.
        policy: ControlPolicy mapping (t, theta) -> u(t).
        theta: Schedule parameters, shape depends on policy.

    Returns:
        trajectory: shape (n_particles, n_steps + 1, dim_state). Includes
            the initial state at index 0.
        amplitude_at_D: shape (n_particles,). The amplitude component at
            the terminal time, extracted via problem.amplitude_of.
        t_grid: shape (n_steps + 1,). Time in days, in [0, horizon_days].
    """
    rng_init, rng_noise = jax.random.split(rng, 2)

    # --- Setup: time grid, particle initial states ---
    # horizon_days and dt_days are static Python scalars on BridgeProblem,
    # so we compute n_steps with Python arithmetic. Using jnp.round here
    # would create a traced value that breaks under jit.
    horizon = float(problem.horizon_days)
    dt = float(problem.dt_days)
    n_steps = int(round(horizon / dt))
    t_grid = jnp.linspace(0.0, horizon, n_steps + 1)

    x0_batch = problem.sample_initial_state(rng_init, problem.n_particles)
    if x0_batch.ndim != 2:
        raise ValueError(
            f"sample_initial_state must return a 2-D array of shape "
            f"(n_particles, dim_state); got shape {x0_batch.shape}."
        )
    n_particles = x0_batch.shape[0]
    dim_state = x0_batch.shape[1]

    # --- Pre-sample all Brownian increments: shape (n_steps, n_particles, dim) ---
    # We pre-sample so that each call with the same rng gives identical noise,
    # which is what makes jax.grad well-defined.
    xi = jax.random.normal(
        rng_noise, shape=(n_steps, n_particles, dim_state)
    )

    # --- Per-particle scan body ---
    sqrt_dt = jnp.sqrt(dt)
    model_params = problem.model_params
    drift_fn = problem.drift_fn_jax
    diffusion_fn = problem.diffusion_fn_jax
    clip_fn = problem.state_clip_fn  # may be None

    def step(x, scan_input):
        """Single Euler-Maruyama step on one particle.

        x: shape (dim_state,)
        scan_input: (t, xi_step) where xi_step has shape (dim_state,)
        """
        t, xi_step = scan_input
        u = policy.evaluate(t, theta)
        f = drift_fn(t, x, u, model_params)
        sigma = diffusion_fn(x, model_params)
        x_next = x + f * dt + sigma * sqrt_dt * xi_step
        # Optional adapter-supplied state clipping. Applied unconditionally
        # at trace time when present; the closure captures None when absent.
        if clip_fn is not None:
            x_next = clip_fn(x_next)
        return x_next, x_next

    def integrate_one_particle(x0, xi_particle):
        """Integrate one particle over the full time grid.

        x0:           shape (dim_state,)
        xi_particle:  shape (n_steps, dim_state)
        Returns:
            trajectory shape (n_steps + 1, dim_state) including x0 at index 0
        """
        # scan over timesteps; t at step k is t_grid[k] (start of step).
        t_scan = t_grid[:-1]                            # (n_steps,)
        scan_inputs = (t_scan, xi_particle)
        _, traj_after = jax.lax.scan(step, x0, scan_inputs)
        return jnp.concatenate([x0[None, :], traj_after], axis=0)

    # --- vmap over particles ---
    # xi has shape (n_steps, n_particles, dim) -> (n_particles, n_steps, dim).
    xi_per_particle = jnp.transpose(xi, (1, 0, 2))
    trajectories = jax.vmap(integrate_one_particle)(x0_batch, xi_per_particle)
    # shape: (n_particles, n_steps + 1, dim_state)

    # --- Extract terminal amplitudes ---
    amplitude_at_D = jax.vmap(problem.amplitude_of)(trajectories[:, -1, :])

    return trajectories, amplitude_at_D, t_grid
