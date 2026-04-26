"""
tests/engine/test_piecewise_constant.py
========================================
Phase 1 unit tests for ot_engine.policies.piecewise_constant.PiecewiseConstant.

Covers:
    - init_params returns the reference schedule unchanged
    - evaluate at t = 0.5 returns day-0 value
    - evaluate at t = 1.5 returns day-1 value
    - evaluate at t = D - 0.001 returns day-(D-1) value
    - evaluate_daily round-trips
    - n_params == D * n_controls
    - input validation
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from ot_engine.policies import PiecewiseConstant


def test_n_params():
    pol = PiecewiseConstant(horizon_days=21, n_controls=3)
    assert pol.n_params == 63


def test_init_params_matches_reference():
    D, n_c = 14, 2
    pol = PiecewiseConstant(horizon_days=D, n_controls=n_c)
    ref = jnp.arange(D * n_c, dtype=jnp.float32).reshape(D, n_c)
    theta = pol.init_params(ref)
    assert theta.shape == (D, n_c)
    # init_params should equal the reference exactly
    assert jnp.allclose(theta, ref)


def test_init_params_rejects_wrong_shape():
    pol = PiecewiseConstant(horizon_days=14, n_controls=2)
    bad_ref = jnp.zeros((10, 2))  # wrong D
    with pytest.raises(ValueError, match="reference_schedule shape mismatch"):
        pol.init_params(bad_ref)


def test_evaluate_at_day_boundaries():
    D, n_c = 5, 2
    pol = PiecewiseConstant(horizon_days=D, n_controls=n_c)
    # Distinct value per day per control, so we can identify which day was returned.
    theta = jnp.array([
        [10.0, 100.0],
        [11.0, 110.0],
        [12.0, 120.0],
        [13.0, 130.0],
        [14.0, 140.0],
    ])
    # Day 0
    u = pol.evaluate(jnp.float32(0.5), theta)
    assert jnp.allclose(u, jnp.array([10.0, 100.0]))
    # Day 1
    u = pol.evaluate(jnp.float32(1.5), theta)
    assert jnp.allclose(u, jnp.array([11.0, 110.0]))
    # Day 4 (last day)
    u = pol.evaluate(jnp.float32(4.999), theta)
    assert jnp.allclose(u, jnp.array([14.0, 140.0]))


def test_evaluate_clips_at_horizon():
    D, n_c = 3, 1
    pol = PiecewiseConstant(horizon_days=D, n_controls=n_c)
    theta = jnp.array([[1.0], [2.0], [3.0]])
    # t = D should clip to last day
    u = pol.evaluate(jnp.float32(D), theta)
    assert jnp.allclose(u, jnp.array([3.0]))
    # t > D also clips
    u = pol.evaluate(jnp.float32(D + 5.0), theta)
    assert jnp.allclose(u, jnp.array([3.0]))


def test_evaluate_at_negative_t_clips_to_day_zero():
    pol = PiecewiseConstant(horizon_days=3, n_controls=1)
    theta = jnp.array([[1.0], [2.0], [3.0]])
    u = pol.evaluate(jnp.float32(-1.5), theta)
    assert jnp.allclose(u, jnp.array([1.0]))


def test_evaluate_daily_returns_theta():
    pol = PiecewiseConstant(horizon_days=7, n_controls=3)
    theta = jnp.arange(21, dtype=jnp.float32).reshape(7, 3)
    daily = pol.evaluate_daily(theta)
    assert jnp.allclose(daily, theta)


def test_init_rejects_bad_dimensions():
    with pytest.raises(ValueError, match="horizon_days"):
        PiecewiseConstant(horizon_days=0, n_controls=2)
    with pytest.raises(ValueError, match="horizon_days"):
        PiecewiseConstant(horizon_days=-1, n_controls=2)
    with pytest.raises(ValueError, match="n_controls"):
        PiecewiseConstant(horizon_days=14, n_controls=0)


def test_init_params_preserves_float64():
    """Review fix C-1: init_params must NOT downcast float64 to float32."""
    import jax.numpy as jnp
    from ot_engine.policies.piecewise_constant import PiecewiseConstant
    pol = PiecewiseConstant(4, 2)
    ref64 = jnp.zeros((4, 2), dtype=jnp.float64)
    theta = pol.init_params(ref64)
    assert theta.dtype == jnp.float64, (
        f"init_params downcast float64 to {theta.dtype}"
    )
