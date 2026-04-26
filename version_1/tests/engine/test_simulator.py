"""
tests/engine/test_simulator.py — JAX Euler-Maruyama correctness.
==================================================================
Phase 2 unit tests for ot_engine.simulator.simulate_latent.

Strategy:
    Use a 1-D Ornstein-Uhlenbeck SDE with no controls, for which the
    terminal distribution has a closed form. The simulator's empirical
    mean and variance at t = D must match the analytical values.

    Then check determinism: same rng + same theta = bit-identical traj.

The OU SDE is
    dx = -theta_0 * x * dt + sigma_0 * dW
where theta_0 > 0 and sigma_0 > 0 are constants (set as
problem.model_params here).

Closed form for x_0 deterministic at value x_init:
    x(T) ~ Normal(
        mean = x_init * exp(-theta_0 * T),
        var  = sigma_0^2 / (2 * theta_0) * (1 - exp(-2 * theta_0 * T))
    )

We compare empirical to analytical to within a Monte-Carlo standard
error.
"""

from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ot_engine import (
    BridgeProblem,
    PiecewiseConstant,
    simulate_latent,
)


# =========================================================================
# OU model — drift, diffusion, fixed initial state
# =========================================================================

def _ou_drift(t, x, u, p):
    """OU drift: -theta_0 * x. Controls u are accepted but ignored."""
    del t, u
    return -p['theta_0'] * x


def _ou_diffusion(x, p):
    """OU diffusion: constant sigma_0 along the single dim."""
    return jnp.full_like(x, p['sigma_0'])


def _ou_init_sampler(x_init: float):
    """Returns a sample_initial_state callable for fixed x_init."""
    def _sampler(rng, n):
        del rng
        return jnp.full((n, 1), x_init)
    return _sampler


def _ou_dummy_target(rng, n):
    del rng
    return jnp.zeros(n)


def _ou_amplitude(x):
    """Single state component is the 'amplitude'."""
    return x[0]


def _make_ou_problem(
    x_init: float,
    theta_0: float,
    sigma_0: float,
    horizon: int = 5,
    dt: float = 0.01,
    n_particles: int = 4096,
) -> BridgeProblem:
    """Build a minimal BridgeProblem wrapping the OU SDE."""
    n_c = 1
    return BridgeProblem(
        name='ou_test',
        drift_fn_jax=_ou_drift,
        diffusion_fn_jax=_ou_diffusion,
        model_params={'theta_0': theta_0, 'sigma_0': sigma_0},
        sample_initial_state=_ou_init_sampler(x_init),
        sample_target_amplitude=_ou_dummy_target,
        amplitude_of=_ou_amplitude,
        n_controls=n_c,
        control_bounds=((0.0, 1.0),),
        horizon_days=horizon,
        reference_schedule=jnp.zeros((horizon, n_c)),
        reference_sigma=jnp.ones((horizon, n_c)),
        n_particles=n_particles,
        dt_days=dt,
    )


# =========================================================================
# Tests
# =========================================================================

def test_simulator_output_shapes():
    """Trajectory and amplitude_at_D must have the documented shapes."""
    problem = _make_ou_problem(x_init=1.0, theta_0=0.5, sigma_0=0.5,
                               horizon=4, dt=0.1, n_particles=64)
    pol = PiecewiseConstant(horizon_days=4, n_controls=1)
    theta = jnp.zeros((4, 1))
    rng = jax.random.PRNGKey(0)
    traj, A_D, t_grid = simulate_latent(rng, problem, pol, theta)
    n_steps = int(round(4.0 / 0.1))
    assert traj.shape == (64, n_steps + 1, 1)
    assert A_D.shape == (64,)
    assert t_grid.shape == (n_steps + 1,)
    # Initial state stored at index 0.
    assert jnp.allclose(traj[:, 0, 0], 1.0)


def test_simulator_determinism():
    """Same rng + theta -> bit-identical trajectories (gradient correctness)."""
    problem = _make_ou_problem(x_init=0.7, theta_0=0.4, sigma_0=0.3,
                               horizon=3, dt=0.05, n_particles=32)
    pol = PiecewiseConstant(horizon_days=3, n_controls=1)
    theta = jnp.zeros((3, 1))
    rng = jax.random.PRNGKey(42)
    traj_a, A_a, _ = simulate_latent(rng, problem, pol, theta)
    traj_b, A_b, _ = simulate_latent(rng, problem, pol, theta)
    assert jnp.array_equal(traj_a, traj_b)
    assert jnp.array_equal(A_a, A_b)


def test_simulator_ou_terminal_mean_matches_analytical():
    """Empirical mean of x(T) must match x_init * exp(-theta_0 * T)."""
    x_init, theta_0, sigma_0, T = 1.5, 0.4, 0.5, 5.0
    problem = _make_ou_problem(
        x_init=x_init, theta_0=theta_0, sigma_0=sigma_0,
        horizon=int(T), dt=0.01, n_particles=4096,
    )
    pol = PiecewiseConstant(horizon_days=int(T), n_controls=1)
    theta = jnp.zeros((int(T), 1))
    rng = jax.random.PRNGKey(7)
    _, A_D, _ = simulate_latent(rng, problem, pol, theta)

    expected_mean = x_init * np.exp(-theta_0 * T)
    empirical_mean = float(jnp.mean(A_D))
    # Monte-Carlo SE: sigma_terminal / sqrt(N). With var below ~0.3, N=4096,
    # SE ~ 0.009. Tolerance 0.03 = ~3 SE.
    assert abs(empirical_mean - expected_mean) < 0.03, \
        f"mean mismatch: empirical {empirical_mean}, expected {expected_mean}"


def test_simulator_ou_terminal_variance_matches_analytical():
    """Empirical var of x(T) must match the OU steady-state-like formula."""
    x_init, theta_0, sigma_0, T = 1.5, 0.4, 0.5, 5.0
    problem = _make_ou_problem(
        x_init=x_init, theta_0=theta_0, sigma_0=sigma_0,
        horizon=int(T), dt=0.01, n_particles=4096,
    )
    pol = PiecewiseConstant(horizon_days=int(T), n_controls=1)
    theta = jnp.zeros((int(T), 1))
    rng = jax.random.PRNGKey(8)
    _, A_D, _ = simulate_latent(rng, problem, pol, theta)

    expected_var = (sigma_0 ** 2) / (2.0 * theta_0) * \
                   (1.0 - np.exp(-2.0 * theta_0 * T))
    empirical_var = float(jnp.var(A_D))
    # Variance has higher MC error than mean; tolerance 10% relative.
    rel_err = abs(empirical_var - expected_var) / expected_var
    assert rel_err < 0.10, \
        f"var mismatch: empirical {empirical_var}, expected {expected_var}, "\
        f"rel_err {rel_err}"


def test_simulator_ou_no_noise_matches_deterministic_decay():
    """With sigma_0 = 0 the SDE collapses to the deterministic ODE."""
    x_init, theta_0, T = 2.0, 0.5, 4.0
    problem = _make_ou_problem(
        x_init=x_init, theta_0=theta_0, sigma_0=0.0,
        horizon=int(T), dt=0.01, n_particles=8,
    )
    pol = PiecewiseConstant(horizon_days=int(T), n_controls=1)
    theta = jnp.zeros((int(T), 1))
    rng = jax.random.PRNGKey(0)
    _, A_D, _ = simulate_latent(rng, problem, pol, theta)
    expected = x_init * np.exp(-theta_0 * T)
    # All particles should be identical (no noise) and equal to the
    # deterministic decay. Euler-Maruyama with step 0.01 -> error <1%.
    assert jnp.allclose(A_D, A_D[0])  # all particles equal
    assert abs(float(A_D[0]) - expected) / expected < 0.01, \
        f"got {A_D[0]}, expected {expected}"


def test_simulator_runs_under_jit():
    """The simulator should be jit-compilable end-to-end."""
    problem = _make_ou_problem(x_init=1.0, theta_0=0.5, sigma_0=0.3,
                               horizon=3, dt=0.05, n_particles=64)
    pol = PiecewiseConstant(horizon_days=3, n_controls=1)
    theta = jnp.zeros((3, 1))
    rng = jax.random.PRNGKey(1)

    @jax.jit
    def _sim(rng, theta):
        traj, A_D, t = simulate_latent(rng, problem, pol, theta)
        return A_D

    A_D = _sim(rng, theta)
    assert A_D.shape == (64,)
    assert jnp.all(jnp.isfinite(A_D))
