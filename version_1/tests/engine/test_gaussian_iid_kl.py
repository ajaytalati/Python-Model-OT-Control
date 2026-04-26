"""
tests/engine/test_gaussian_iid_kl.py — Reference KL term.
===========================================================
Phase 1 unit tests for ot_engine.reference.gaussian_iid.gaussian_iid_kl.

Covers:
    - KL is zero when theta == mu_ref
    - KL is positive when theta != mu_ref
    - Hand-computed value matches the analytical formula
    - Shape mismatch raises ValueError
    - jax.grad produces the expected analytical gradient
      (theta - mu_ref) / sigma_ref^2
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ot_engine.reference import gaussian_iid_kl


def test_kl_zero_at_reference():
    D, n_c = 14, 2
    mu_ref = jnp.ones((D, n_c)) * 3.0
    sigma_ref = jnp.ones((D, n_c)) * 0.5
    theta = mu_ref.copy()
    val = gaussian_iid_kl(theta, mu_ref, sigma_ref)
    assert val == pytest.approx(0.0)


def test_kl_positive_when_offset():
    D, n_c = 14, 2
    mu_ref = jnp.zeros((D, n_c))
    sigma_ref = jnp.ones((D, n_c))
    theta = jnp.ones((D, n_c))  # offset of 1 everywhere
    val = gaussian_iid_kl(theta, mu_ref, sigma_ref)
    # 0.5 * sum( (1 - 0)^2 / 1^2 ) = 0.5 * D * n_c = 0.5 * 28 = 14
    assert val == pytest.approx(0.5 * D * n_c)


def test_kl_explicit_known_value():
    """Hand-computed reference."""
    theta = jnp.array([[2.0, 1.0], [3.0, 4.0]])
    mu_ref = jnp.array([[0.0, 1.0], [1.0, 2.0]])
    sigma_ref = jnp.array([[1.0, 1.0], [2.0, 2.0]])
    # Quadratic terms:
    #   (2-0)^2 / 1^2 = 4
    #   (1-1)^2 / 1^2 = 0
    #   (3-1)^2 / 2^2 = 4/4 = 1
    #   (4-2)^2 / 2^2 = 4/4 = 1
    # Sum = 6, divided by 2 = 3.0
    val = gaussian_iid_kl(theta, mu_ref, sigma_ref)
    assert val == pytest.approx(3.0)


def test_shape_mismatch_raises():
    theta = jnp.zeros((14, 2))
    mu_ref = jnp.zeros((14, 3))   # wrong n_controls
    sigma_ref = jnp.ones((14, 2))
    with pytest.raises(ValueError, match="Shape mismatch"):
        gaussian_iid_kl(theta, mu_ref, sigma_ref)


def test_grad_matches_analytical():
    """Gradient should be (theta - mu_ref) / sigma_ref^2."""
    theta = jnp.array([[2.0, 1.0], [3.0, 4.0]])
    mu_ref = jnp.array([[0.0, 1.0], [1.0, 2.0]])
    sigma_ref = jnp.array([[1.0, 1.0], [2.0, 2.0]])

    def loss(t):
        return gaussian_iid_kl(t, mu_ref, sigma_ref)

    g = jax.grad(loss)(theta)
    expected = (theta - mu_ref) / (sigma_ref * sigma_ref)
    assert jnp.allclose(g, expected, atol=1e-6)


def test_kl_scales_with_inverse_variance():
    """Tighter sigma_ref => larger KL for same offset."""
    theta = jnp.ones((5, 1))
    mu_ref = jnp.zeros((5, 1))
    val_loose = gaussian_iid_kl(theta, mu_ref, jnp.ones((5, 1)))
    val_tight = gaussian_iid_kl(theta, mu_ref, jnp.ones((5, 1)) * 0.1)
    # KL with sigma=0.1 should be 100x KL with sigma=1.0 (since variance ratio is 1/100).
    assert val_tight == pytest.approx(100 * val_loose, rel=1e-5)
