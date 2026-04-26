"""
tests/engine/test_closed_loop_compare_pipeline.py — Phase 5 unit tests.
========================================================================
Phase 5 unit tests for the closed-loop verification, the naive
baselines, and the top-level pipeline composition.

Strategy:
    Use a small OU-with-control problem (the same fixture used in the
    Phase 2/3 tests) plus a hand-written basin indicator. Verify:

    1. simulate_closed_loop returns a ClosedLoopResult with the
       documented shapes; basin_fraction is NaN if the adapter did not
       supply an indicator.
    2. zero_control_schedule, constant_reference_schedule,
       linear_interpolation_schedule produce the right shapes and
       values; bad inputs raise.
    3. run_ot_pipeline composes optimise + closed_loop end-to-end.
    4. compare_schedules accepts a list of schedules and returns a
       label-keyed dict.

End-to-end SWAT smoke is in tests/adapters/test_swat_phase5.py.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ot_engine import (
    BridgeProblem,
    PiecewiseConstant,
    Schedule,
    simulate_closed_loop,
    zero_control_schedule,
    constant_reference_schedule,
    linear_interpolation_schedule,
    run_ot_pipeline,
    compare_schedules,
)


# =========================================================================
# OU-with-control fixture (the same one used in Phase 2/3 tests)
# =========================================================================

def _drift(t, x, u, p):
    del t
    return -p['theta_0'] * x + u


def _diffusion(x, p):
    return jnp.full_like(x, p['sigma_0'])


def _init_zero(rng, n):
    del rng
    return jnp.zeros((n, 1))


def _target_at_one(rng, n):
    return 1.0 + 0.05 * jax.random.normal(rng, shape=(n,))


def _amp(x):
    return x[0]


def _basin(x, u_terminal, params):
    """Simple toy basin indicator: x[0] > 0.5 AND |u_terminal[0]| < 1.0."""
    del params
    return jnp.logical_and(x[0] > 0.5, jnp.abs(u_terminal[0]) < 1.0)


def _make_problem(horizon=4, n_particles=64, with_basin=True):
    return BridgeProblem(
        name='ou_phase5_test',
        drift_fn_jax=_drift,
        diffusion_fn_jax=_diffusion,
        model_params={'theta_0': 0.5, 'sigma_0': 0.15},
        sample_initial_state=_init_zero,
        sample_target_amplitude=_target_at_one,
        amplitude_of=_amp,
        n_controls=1,
        control_bounds=((-2.0, 2.0),),
        horizon_days=horizon,
        reference_schedule=jnp.zeros((horizon, 1)),
        reference_sigma=jnp.ones((horizon, 1)) * 1.0,
        alpha_terminal=1.0,
        alpha_transport=0.01,
        alpha_reference=0.001,
        n_particles=n_particles,
        dt_days=0.1,
        optim_steps=200,
        learning_rate=5e-2,
        basin_indicator_fn=_basin if with_basin else None,
    )


# =========================================================================
# simulate_closed_loop
# =========================================================================

def test_closed_loop_returns_correct_shapes():
    problem = _make_problem()
    pol = PiecewiseConstant(horizon_days=4, n_controls=1)
    schedule = Schedule(
        theta=jnp.ones((4, 1)) * 0.5,
        daily_values=jnp.ones((4, 1)) * 0.5,
        horizon_days=4, n_controls=1, control_names=('u_0',),
    )
    rng = jax.random.PRNGKey(0)
    result = simulate_closed_loop(problem, pol, schedule, rng,
                                    n_realisations=32)
    n_steps = int(round(4.0 / 0.1))
    assert result.trajectories.shape == (32, n_steps + 1, 1)
    assert result.amplitude_at_D.shape == (32,)
    assert result.target_samples.shape == (32,)
    assert isinstance(result.mmd_target, float)
    assert jnp.isfinite(jnp.float64(result.mmd_target))


def test_closed_loop_basin_fraction_is_nan_when_no_indicator():
    problem = _make_problem(with_basin=False)
    pol = PiecewiseConstant(horizon_days=4, n_controls=1)
    schedule = Schedule(
        theta=jnp.zeros((4, 1)),
        daily_values=jnp.zeros((4, 1)),
        horizon_days=4, n_controls=1, control_names=('u_0',),
    )
    rng = jax.random.PRNGKey(0)
    result = simulate_closed_loop(problem, pol, schedule, rng,
                                    n_realisations=16)
    # NaN check via != self.
    assert result.fraction_in_healthy_basin != result.fraction_in_healthy_basin


def test_closed_loop_basin_fraction_in_unit_interval_with_indicator():
    problem = _make_problem(with_basin=True)
    pol = PiecewiseConstant(horizon_days=4, n_controls=1)
    # A schedule with u=0.5 should leave x oscillating around 1.0 at
    # terminal time, so quite a few will satisfy the basin condition
    # (x > 0.5 AND u < 1.0).
    schedule = Schedule(
        theta=jnp.ones((4, 1)) * 0.5,
        daily_values=jnp.ones((4, 1)) * 0.5,
        horizon_days=4, n_controls=1, control_names=('u_0',),
    )
    rng = jax.random.PRNGKey(0)
    result = simulate_closed_loop(problem, pol, schedule, rng,
                                    n_realisations=64)
    assert 0.0 <= result.fraction_in_healthy_basin <= 1.0


# =========================================================================
# Naive baselines
# =========================================================================

def test_zero_control_schedule_is_zero():
    problem = _make_problem(horizon=5)
    pol = PiecewiseConstant(horizon_days=5, n_controls=1)
    sch = zero_control_schedule(problem, pol)
    assert sch.theta.shape == (5, 1)
    assert jnp.all(sch.theta == 0.0)
    assert sch.metadata['label'] == 'zero_control'


def test_constant_reference_equals_problem_reference():
    """Adapter ref schedule = (V_h, V_n, V_c) per day; baseline mirrors it."""
    horizon = 5
    ref = jnp.tile(jnp.array([[0.7]]), (horizon, 1))   # custom reference
    problem = BridgeProblem(
        name='_ref_test',
        drift_fn_jax=_drift, diffusion_fn_jax=_diffusion,
        model_params={'theta_0': 0.5, 'sigma_0': 0.1},
        sample_initial_state=_init_zero,
        sample_target_amplitude=_target_at_one,
        amplitude_of=_amp,
        n_controls=1, control_bounds=((-2.0, 2.0),),
        horizon_days=horizon,
        reference_schedule=ref,
        reference_sigma=jnp.ones((horizon, 1)),
        n_particles=16, dt_days=0.1,
    )
    pol = PiecewiseConstant(horizon_days=horizon, n_controls=1)
    sch = constant_reference_schedule(problem, pol)
    assert jnp.allclose(sch.theta, ref)
    assert sch.metadata['label'] == 'constant_reference'


def test_linear_interpolation_endpoints():
    """First day = reference[0]; last day = theta_target."""
    problem = _make_problem(horizon=5)  # reference all zeros, n_controls=1
    pol = PiecewiseConstant(horizon_days=5, n_controls=1)
    target = jnp.array([2.0])
    sch = linear_interpolation_schedule(problem, pol, target)
    assert sch.theta.shape == (5, 1)
    # Day 0 should equal reference day 0 (= 0).
    assert jnp.allclose(sch.theta[0], jnp.zeros(1))
    # Day D-1 should equal target.
    assert jnp.allclose(sch.theta[-1], target)
    # Linearly increasing (each day strictly greater than the previous).
    diffs = jnp.diff(sch.theta[:, 0])
    assert jnp.all(diffs > 0)


def test_linear_interpolation_rejects_wrong_target_shape():
    problem = _make_problem(horizon=5)
    pol = PiecewiseConstant(horizon_days=5, n_controls=1)
    with pytest.raises(ValueError, match='theta_target must have shape'):
        linear_interpolation_schedule(problem, pol,
                                        jnp.array([1.0, 2.0]))  # wrong dim


# =========================================================================
# Top-level pipeline
# =========================================================================

def test_run_ot_pipeline_returns_three_artefacts():
    """run_ot_pipeline should produce a Schedule, OptimisationTrace,
    and ClosedLoopResult; each with the documented field shapes."""
    problem = _make_problem(horizon=4, n_particles=64)
    rng = jax.random.PRNGKey(0)
    schedule, trace, closed_loop = run_ot_pipeline(
        problem, rng, n_realisations=32,
        optimise_kwargs={'convergence_window': 20, 'convergence_tol': 1e-2},
    )
    assert schedule.theta.shape == (4, 1)
    assert trace.losses_total.shape[0] == trace.n_steps_run
    assert closed_loop.amplitude_at_D.shape == (32,)
    # Basin indicator was supplied; basin fraction should be a real number.
    bf = float(closed_loop.fraction_in_healthy_basin)
    assert 0.0 <= bf <= 1.0


def test_compare_schedules_keyed_by_label():
    """compare_schedules should accept a list and key results by label."""
    problem = _make_problem(horizon=4, n_particles=32)
    pol = PiecewiseConstant(horizon_days=4, n_controls=1)
    s1 = zero_control_schedule(problem, pol)
    s2 = constant_reference_schedule(problem, pol)
    s3 = linear_interpolation_schedule(problem, pol, jnp.array([1.0]))
    rng = jax.random.PRNGKey(0)
    results = compare_schedules(problem, [s1, s2, s3], rng,
                                  n_realisations=32)
    assert set(results.keys()) == {
        'zero_control', 'constant_reference', 'linear_interpolation',
    }
    for label, res in results.items():
        assert res.amplitude_at_D.shape == (32,)


def test_simulate_closed_loop_rejects_horizon_mismatch():
    """Review fix H-7: a schedule built for a different horizon must error."""
    from ot_engine import simulate_closed_loop, Schedule
    problem = _make_problem(horizon=4, n_particles=8)
    pol = PiecewiseConstant(horizon_days=4, n_controls=1)
    bad = Schedule(
        theta=jnp.zeros((5, 1)),
        daily_values=jnp.zeros((5, 1)),
        horizon_days=5, n_controls=1, control_names=('u_0',),
    )
    import pytest
    with pytest.raises(ValueError, match='horizon_days'):
        simulate_closed_loop(problem, pol, bad, jax.random.PRNGKey(0))


def test_simulate_closed_loop_rejects_zero_n_realisations():
    """Review fix H-6: n_realisations=0 must raise, not silently fall back."""
    from ot_engine import simulate_closed_loop
    problem = _make_problem(horizon=4, n_particles=16)
    pol = PiecewiseConstant(horizon_days=4, n_controls=1)
    sch = zero_control_schedule(problem, pol)
    import pytest
    with pytest.raises(ValueError, match='n_realisations'):
        simulate_closed_loop(problem, pol, sch, jax.random.PRNGKey(0),
                             n_realisations=0)
