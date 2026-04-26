"""
tests/engine/test_loss.py — Loss composition and end-to-end gradients.
=======================================================================
Phase 2 unit tests for ot_engine.loss.

Strategy:
    Build a small OU-with-control problem (control adds an additive
    forcing term to the drift). Verify:

    1. loss_fn returns finite scalar + the right components dict
    2. Component weights are applied correctly
    3. jax.grad of the total loss w.r.t. theta is finite, non-zero, and
       correctly shaped
    4. Determinism: same (theta, rng) -> same loss
    5. Transport-cost-only mode produces an analytical gradient that
       matches the closed form alpha * theta * dt
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ot_engine import (
    BridgeProblem,
    PiecewiseConstant,
    make_loss_fn,
    transport_cost_piecewise_constant,
)


# =========================================================================
# Toy problem: OU with linear control
#
#     dx = (-theta_0 * x + u(t)) dt + sigma_0 * dW
#
# At terminal time the pulled-toward target is achieved by setting u in
# the right direction.
# =========================================================================

def _ou_drift_with_control(t, x, u, p):
    del t
    return -p['theta_0'] * x + u  # u is a 1-D control vector here


def _ou_diffusion(x, p):
    return jnp.full_like(x, p['sigma_0'])


def _init_at(x_init: float):
    def _sampler(rng, n):
        del rng
        return jnp.full((n, 1), x_init)
    return _sampler


def _target_at(x_target: float, sigma_target: float):
    def _sampler(rng, n):
        return x_target + sigma_target * jax.random.normal(rng, shape=(n,))
    return _sampler


def _amp(x):
    return x[0]


def _make_problem(
    horizon: int = 4,
    n_particles: int = 64,
    alpha_terminal: float = 1.0,
    alpha_transport: float = 0.1,
    alpha_reference: float = 0.1,
) -> BridgeProblem:
    return BridgeProblem(
        name='ou_loss_test',
        drift_fn_jax=_ou_drift_with_control,
        diffusion_fn_jax=_ou_diffusion,
        model_params={'theta_0': 0.5, 'sigma_0': 0.3},
        sample_initial_state=_init_at(0.2),
        sample_target_amplitude=_target_at(1.5, 0.1),
        amplitude_of=_amp,
        n_controls=1,
        control_bounds=((0.0, 2.0),),
        horizon_days=horizon,
        reference_schedule=jnp.zeros((horizon, 1)),
        reference_sigma=jnp.ones((horizon, 1)) * 0.5,
        alpha_terminal=alpha_terminal,
        alpha_transport=alpha_transport,
        alpha_reference=alpha_reference,
        n_particles=n_particles,
        dt_days=0.1,
    )


# =========================================================================
# Tests
# =========================================================================

def test_loss_returns_scalar_and_dict():
    problem = _make_problem()
    pol = PiecewiseConstant(horizon_days=4, n_controls=1)
    loss_fn = make_loss_fn(problem, pol)
    theta = jnp.zeros((4, 1))
    rng = jax.random.PRNGKey(0)
    total, components = loss_fn(theta, rng)
    assert total.shape == ()
    assert jnp.isfinite(total)
    assert set(components.keys()) == {'terminal', 'transport', 'reference', 'total'}
    for name, val in components.items():
        assert jnp.isfinite(val), f"{name} is not finite: {val}"


def test_loss_components_weighted_correctly():
    """Components dict should hold *weighted* values (after alpha)."""
    problem = _make_problem(
        alpha_terminal=2.0, alpha_transport=3.0, alpha_reference=4.0,
    )
    pol = PiecewiseConstant(horizon_days=4, n_controls=1)
    loss_fn = make_loss_fn(problem, pol)
    theta = jnp.ones((4, 1)) * 0.3
    rng = jax.random.PRNGKey(1)
    total, components = loss_fn(theta, rng)
    # total = sum of three weighted components
    summed = (components['terminal']
              + components['transport']
              + components['reference'])
    assert jnp.allclose(total, summed, atol=1e-6)


def test_loss_determinism():
    """Same theta + rng -> identical loss (gradient correctness)."""
    problem = _make_problem()
    pol = PiecewiseConstant(horizon_days=4, n_controls=1)
    loss_fn = make_loss_fn(problem, pol)
    theta = jnp.array([[0.1], [0.2], [0.3], [0.4]])
    rng = jax.random.PRNGKey(99)
    total_a, _ = loss_fn(theta, rng)
    total_b, _ = loss_fn(theta, rng)
    assert total_a == total_b


def test_loss_grad_is_finite_and_nonzero():
    """jax.grad must produce a finite gradient of the right shape."""
    problem = _make_problem()
    pol = PiecewiseConstant(horizon_days=4, n_controls=1)
    loss_fn = make_loss_fn(problem, pol)
    theta_init = jnp.zeros((4, 1))
    rng = jax.random.PRNGKey(2)

    def total_loss(theta):
        total, _ = loss_fn(theta, rng)
        return total

    grad = jax.grad(total_loss)(theta_init)
    assert grad.shape == theta_init.shape
    assert jnp.all(jnp.isfinite(grad))
    # With theta=0 the reference KL has zero gradient, but the terminal
    # MMD pushes mass toward the target so the gradient must be non-zero.
    assert jnp.linalg.norm(grad) > 1e-3


def test_transport_cost_closed_form():
    """transport_cost_piecewise_constant must equal 0.5 * sum(u^2) * dt."""
    daily = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    expected = 0.5 * (1 + 4 + 9 + 16) * 1.0
    val = transport_cost_piecewise_constant(daily, dt_per_day=1.0)
    assert val == pytest.approx(expected)


def test_transport_grad_matches_analytical():
    """grad of transport cost w.r.t. daily values is u * dt."""
    daily = jnp.array([[0.5, 1.0], [-0.5, 2.0]])

    def cost(d):
        return transport_cost_piecewise_constant(d, dt_per_day=1.0)

    grad = jax.grad(cost)(daily)
    # d/du [0.5 * u^2 * dt] = u * dt; here dt = 1.
    assert jnp.allclose(grad, daily, atol=1e-6)


def test_loss_finite_difference_check():
    """Compare jax.grad against a 2-sided finite-difference estimate
    on a single component of theta. This is the gold-standard correctness
    check that all the JAX/AD tracing is honest end-to-end.

    Uses a *fixed* MMD bandwidth via a custom loss function, removing the
    median-bandwidth's order-statistic non-smoothness which makes FD
    unreliable at any but the smallest eps. Also uses eps = 1e-5: at this
    scale on float64 the FD truncation error is well below AD precision
    and the agreement should be ~5 digits.
    """
    import jax
    from ot_engine import simulate_latent, mmd_squared, gaussian_iid_kl
    from ot_engine.loss import transport_cost_piecewise_constant

    problem = _make_problem(n_particles=128)
    pol = PiecewiseConstant(horizon_days=4, n_controls=1)
    rng = jax.random.PRNGKey(3)

    # Fixed bandwidth removes the median order-statistic non-smoothness.
    fixed_bandwidth = jnp.float64(1.0)

    def total_loss(theta):
        rng_sim, rng_target = jax.random.split(rng, 2)
        _, A_sim, _ = simulate_latent(rng_sim, problem, pol, theta)
        A_tgt = problem.sample_target_amplitude(rng_target,
                                                 problem.n_particles)
        L_term = mmd_squared(A_sim, A_tgt, bandwidth=fixed_bandwidth)
        L_tran = transport_cost_piecewise_constant(pol.evaluate_daily(theta))
        L_ref = gaussian_iid_kl(theta,
                                problem.reference_schedule,
                                problem.reference_sigma)
        return (problem.alpha_terminal * L_term
                + problem.alpha_transport * L_tran
                + problem.alpha_reference * L_ref)

    theta_0 = jnp.zeros((4, 1)) + 0.2
    grad_ad = jax.grad(total_loss)(theta_0)

    # Tiny eps; 2-sided FD on float64.
    eps = 1e-5
    i, j = 1, 0
    bump = jnp.zeros_like(theta_0).at[i, j].set(eps)
    fd = (total_loss(theta_0 + bump) - total_loss(theta_0 - bump)) / (2.0 * eps)
    err = abs(float(grad_ad[i, j]) - float(fd))
    rel = err / max(abs(float(fd)), 1e-3)
    assert rel < 0.01, f"FD={fd}, AD={grad_ad[i,j]}, rel_err={rel}"
