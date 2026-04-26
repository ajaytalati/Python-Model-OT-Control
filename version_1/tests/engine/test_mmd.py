"""
tests/engine/test_mmd.py — MMD with Gaussian kernel.
=====================================================
Phase 1 unit tests for ot_engine.terminal_cost.mmd.

Covers:
    - MMD between identical samples is approximately 0
    - MMD between samples with same mean (different from 0) is small
    - MMD between samples with different means is positive and ordered
      monotonically with the separation
    - Median-bandwidth heuristic returns a positive scalar
    - Vector samples (2-D) work as well as scalar samples (1-D)
    - jax.grad of MMD w.r.t. X produces finite gradients
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ot_engine.terminal_cost import mmd_squared, median_bandwidth


def test_mmd_zero_for_identical_samples():
    rng = jax.random.PRNGKey(0)
    X = jax.random.normal(rng, shape=(200,))
    # MMD between X and itself uses the same N for both — the formula
    # gives K_XX/N^2 + K_YY/M^2 - 2*K_XY/(NM) which collapses to 0 when
    # X == Y exactly.
    val = mmd_squared(X, X)
    assert jnp.abs(val) < 1e-6, f"Got {val}"


def test_mmd_small_for_same_distribution():
    """X and Y drawn iid from same Gaussian -> MMD^2 ~ O(1/N) bias."""
    rng = jax.random.PRNGKey(1)
    rng_x, rng_y = jax.random.split(rng)
    X = jax.random.normal(rng_x, shape=(500,))
    Y = jax.random.normal(rng_y, shape=(500,))
    val = mmd_squared(X, Y)
    # The biased MMD^2 has O(1/N) bias for samples from the same
    # distribution. With N = 500 we expect val < ~0.05.
    assert 0.0 <= val < 0.05, f"Got {val}"


def test_mmd_increases_with_mean_separation():
    rng = jax.random.PRNGKey(2)
    rng_x, rng_y = jax.random.split(rng)
    X = jax.random.normal(rng_x, shape=(300,))
    # Three Y's at increasing distance.
    deltas = [0.0, 1.0, 3.0]
    vals = []
    for delta in deltas:
        Y = jax.random.normal(rng_y, shape=(300,)) + delta
        vals.append(float(mmd_squared(X, Y)))
    # Values should be increasing.
    assert vals[0] < vals[1] < vals[2], f"Got {vals}"
    # Big separation should be much larger than small.
    assert vals[2] > vals[1] * 2


def test_median_bandwidth_positive():
    rng = jax.random.PRNGKey(3)
    X = jax.random.normal(rng, shape=(50,))
    Y = jax.random.normal(rng, shape=(50,)) + 2.0
    h = median_bandwidth(X, Y)
    assert h > 0


def test_median_bandwidth_floor_for_identical_samples():
    """When all samples are identical, median bandwidth must not be zero
    or the kernel collapses and the gradient vanishes. The floor of 1e-8
    inside median_bandwidth keeps things stable."""
    X = jnp.zeros(20)
    Y = jnp.zeros(20)
    h = median_bandwidth(X, Y)
    assert h > 0
    # Specifically, sqrt of the floor 1e-8.
    assert h == pytest.approx(jnp.sqrt(1e-8), rel=1e-3)


def test_mmd_vector_samples():
    """MMD also works on vector samples (shape (N, d))."""
    rng = jax.random.PRNGKey(4)
    rng_x, rng_y = jax.random.split(rng)
    X = jax.random.normal(rng_x, shape=(100, 3))
    Y = jax.random.normal(rng_y, shape=(100, 3))
    val = mmd_squared(X, Y)
    assert 0.0 <= val < 0.1, f"Got {val}"


def test_mmd_grad_is_finite_and_nonzero():
    """jax.grad of MMD w.r.t. X must produce finite, non-zero gradients
    when X and Y differ — this is the core property the optimiser relies on."""
    rng = jax.random.PRNGKey(5)
    rng_x, rng_y = jax.random.split(rng)
    X = jax.random.normal(rng_x, shape=(80,)) + 1.0
    Y = jax.random.normal(rng_y, shape=(80,))

    def loss(X_):
        return mmd_squared(X_, Y)

    g = jax.grad(loss)(X)
    assert g.shape == X.shape
    assert jnp.all(jnp.isfinite(g))
    # Gradient must not be the zero vector.
    assert jnp.linalg.norm(g) > 1e-3


def test_mmd_explicit_known_value():
    """Hand-computed MMD^2 with bandwidth = 1, scalar samples.

    X = [0.0], Y = [2.0]:
        K_XX = exp(0)            = 1
        K_YY = exp(0)            = 1
        K_XY = exp(-(0-2)^2/2)  = exp(-2)
        MMD^2 = 1/1 + 1/1 - 2 * exp(-2)
              = 2 - 2 * exp(-2)
              ≈ 2 - 0.270671  =  1.729329
    """
    X = jnp.array([0.0])
    Y = jnp.array([2.0])
    val = mmd_squared(X, Y, bandwidth=jnp.float32(1.0))
    expected = 2.0 - 2.0 * float(jnp.exp(-2.0))
    assert val == pytest.approx(expected, rel=1e-5)
