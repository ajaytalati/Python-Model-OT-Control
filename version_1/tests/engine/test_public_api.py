"""
tests/engine/test_public_api.py — Phase 1 sanity check.
=========================================================
Exercises that:
    - Public API surface from ot_engine.__init__ is importable
    - Components compose as the plan describes
    - No import cycles, no missing symbols

This is the "all the pieces line up" test. Solver-level integration
tests come in Phase 2 onwards.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


def test_public_api_imports():
    """Every name advertised in __all__ must be importable at the top level."""
    import ot_engine
    expected = {
        "BridgeProblem", "Schedule", "OptimisationTrace", "ClosedLoopResult",
        "PolicyKind", "TerminalCostKind", "ReferenceKind",
        "ControlPolicy", "PiecewiseConstant",
        "mmd_squared", "median_bandwidth",
        "gaussian_iid_kl",
    }
    assert expected.issubset(set(ot_engine.__all__))
    for name in expected:
        assert hasattr(ot_engine, name), f"ot_engine missing {name}"


def test_phase_1_pieces_compose():
    """End-to-end Phase 1: build a policy, evaluate it, compute MMD on a
    fake terminal sample, compute reference KL on theta. No solver yet —
    just that the loss-component primitives line up shapewise.
    """
    from ot_engine import (
        PiecewiseConstant, mmd_squared, gaussian_iid_kl
    )
    D, n_c = 14, 2
    rng = jax.random.PRNGKey(0)
    rng_x, rng_y = jax.random.split(rng)

    # 1) Build a policy and initialise theta from a reference baseline.
    pol = PiecewiseConstant(horizon_days=D, n_controls=n_c)
    mu_ref = jnp.ones((D, n_c)) * 0.5
    sigma_ref = jnp.ones((D, n_c)) * 0.2
    theta = pol.init_params(mu_ref)

    # 2) Evaluate the daily schedule (would feed the simulator in Phase 2).
    daily = pol.evaluate_daily(theta)
    assert daily.shape == (D, n_c)

    # 3) Stand in for the simulator: pretend we've simulated and got
    #    terminal amplitude samples. They're a function of theta in the
    #    real engine; here, just a fixed array so we can call mmd.
    fake_simulated_amplitudes = jax.random.normal(rng_x, shape=(50,)) + 0.3
    target_amplitudes = jax.random.normal(rng_y, shape=(50,))

    # 4) Three loss-component scalars.
    loss_terminal = mmd_squared(fake_simulated_amplitudes, target_amplitudes)
    # Transport cost on a piecewise-constant schedule, dt=1 day:
    transport = 0.5 * jnp.sum(daily * daily) * 1.0
    loss_reference = gaussian_iid_kl(theta, mu_ref, sigma_ref)

    # 5) Each piece is a finite scalar.
    assert jnp.isfinite(loss_terminal)
    assert jnp.isfinite(transport)
    assert jnp.isfinite(loss_reference)

    # 6) Reference loss should be 0 because theta == mu_ref.
    assert loss_reference == pytest.approx(0.0)


def test_loss_components_differentiable_in_theta():
    """jax.grad through the reference KL part must produce finite output —
    this is the key correctness property for the optimisation loop."""
    from ot_engine import PiecewiseConstant, gaussian_iid_kl
    D, n_c = 7, 2
    pol = PiecewiseConstant(horizon_days=D, n_controls=n_c)
    mu_ref = jnp.zeros((D, n_c))
    sigma_ref = jnp.ones((D, n_c)) * 0.5
    theta_init = pol.init_params(mu_ref + 0.3)  # offset from reference

    def total_kl(theta):
        return gaussian_iid_kl(theta, mu_ref, sigma_ref)

    grad = jax.grad(total_kl)(theta_init)
    assert grad.shape == theta_init.shape
    assert jnp.all(jnp.isfinite(grad))
    assert jnp.linalg.norm(grad) > 0
