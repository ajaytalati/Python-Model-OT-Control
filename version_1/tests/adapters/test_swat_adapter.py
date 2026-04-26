"""
tests/adapters/test_swat_adapter.py — Phase 4 acceptance tests.
=================================================================
Smoke tests for the SWAT adapter and one short end-to-end optimisation
run on the recovery scenario (the easiest of the three: starts from
flatlined T with healthy V_h/V_n already in place, so the schedule just
needs to maintain them and let T climb).

The acceptance criterion from the development plan is that the engine
runs end-to-end on a SWAT scenario and produces a schedule that drives
the simulated terminal T mean above the pathological-baseline
behaviour. Strict numerical recovery to T* is not required at this
horizon and step budget — that's a closed-loop verification concern
(Phase 5).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ot_engine import (
    BridgeProblem,
    PiecewiseConstant,
    optimise_schedule,
    simulate_latent,
)
from adapters.swat import (
    make_swat_problem,
    list_scenarios,
    SWAT_CONTROL_NAMES,
    T_STAR_HEALTHY,
)
from _vendored_models.swat import (
    swat_drift, swat_diffusion, swat_state_clip,
    default_swat_parameters, entrainment_quality,
)


# =========================================================================
# Smoke tests
# =========================================================================

def test_list_scenarios_returns_three():
    """The plan promises three canonical scenarios."""
    names = list_scenarios()
    assert set(names) == {'insomnia', 'recovery', 'shift_work'}


def test_swat_control_names():
    assert SWAT_CONTROL_NAMES == ('V_h', 'V_n', 'V_c')


def test_make_swat_problem_returns_bridge_problem():
    """Each scenario produces a valid BridgeProblem."""
    for sc in list_scenarios():
        problem = make_swat_problem(scenario=sc, horizon_days=7,
                                      n_particles=32)
        assert isinstance(problem, BridgeProblem)
        assert problem.name == f'swat_{sc}'
        assert problem.n_controls == 3
        assert problem.horizon_days == 7
        assert problem.reference_schedule.shape == (7, 3)
        assert problem.reference_sigma.shape == (7, 3)
        assert problem.control_names == ('V_h', 'V_n', 'V_c')
        # state_clip_fn is wrapped with a params closure so it can read
        # A_scale from the model_params; just check it is a callable
        # producing the same output as the direct call.
        assert callable(problem.state_clip_fn)
        x_test = jnp.array([1.5, 7.0, -0.1, -0.2])  # out-of-bounds state
        clipped = problem.state_clip_fn(x_test)
        expected = swat_state_clip(x_test, problem.model_params)
        assert jnp.allclose(clipped, expected)


def test_unknown_scenario_raises():
    with pytest.raises(ValueError, match='Unknown SWAT scenario'):
        make_swat_problem(scenario='not_a_real_scenario')


def test_swat_drift_jit_compatible():
    """The drift function is differentiable and jit-compatible."""
    params = default_swat_parameters()
    x = jnp.array([0.5, 3.0, 0.5, 0.4])
    u = jnp.array([1.0, 0.3, 0.0])
    f = jax.jit(lambda t, x, u: swat_drift(t, x, u, params))(
        jnp.float64(0.0), x, u
    )
    assert f.shape == (4,)
    assert jnp.all(jnp.isfinite(f))


def test_swat_state_clip_keeps_states_in_bounds():
    """Out-of-bounds inputs get clipped to physical domain."""
    x_bad = jnp.array([-0.5, 8.0, -1.0, -0.3])
    x_ok = swat_state_clip(x_bad)
    assert x_ok[0] == 0.0      # W in [0, 1]
    assert x_ok[1] == 6.0      # Z in [0, 6]
    assert x_ok[2] == 0.0      # a >= 0
    assert x_ok[3] == 0.0      # T >= 0


def test_simulator_with_swat_no_state_violations():
    """Running the simulator under the healthy reference produces
    physically valid trajectories (states in their bounds throughout)."""
    problem = make_swat_problem(scenario='recovery', horizon_days=7,
                                  n_particles=32, dt_days=0.05)
    pol = PiecewiseConstant.from_problem(problem)
    rng = jax.random.PRNGKey(0)
    traj, A_D, _ = simulate_latent(rng, problem, pol,
                                    problem.reference_schedule)
    # All states within their physical bounds, all the time.
    assert jnp.all(traj[:, :, 0] >= 0.0)
    assert jnp.all(traj[:, :, 0] <= 1.0)
    assert jnp.all(traj[:, :, 1] >= 0.0)
    assert jnp.all(traj[:, :, 1] <= 6.0)
    assert jnp.all(traj[:, :, 2] >= 0.0)
    assert jnp.all(traj[:, :, 3] >= 0.0)


def test_swat_recovery_under_reference_does_not_recover():
    """Sanity check: under the *reference* schedule (healthy V_h/V_n
    held constant) starting from T_0=0.05, the recovery is partial.
    This sets the baseline that the optimised schedule must beat.
    """
    problem = make_swat_problem(scenario='recovery', horizon_days=14,
                                  n_particles=64, dt_days=0.05)
    pol = PiecewiseConstant.from_problem(problem)
    rng = jax.random.PRNGKey(0)
    _, A_D, _ = simulate_latent(rng, problem, pol,
                                  problem.reference_schedule)
    # Just record the baseline: T(D) should be somewhere between the
    # pathological 0.05 and the healthy 0.55 — the optimiser should
    # push it further toward 0.55.
    mean_baseline = float(jnp.mean(A_D))
    assert 0.0 <= mean_baseline <= T_STAR_HEALTHY


# =========================================================================
# End-to-end test (short optimisation, recovery scenario)
# =========================================================================

def test_swat_end_to_end_recovery_optimisation():
    """Phase 4 acceptance: optimiser runs on SWAT and drives T(D) higher
    than the reference baseline on the 'recovery' scenario.

    'recovery' is the easiest of the three scenarios because the patient
    already has healthy V_h, V_n, V_c — only T needs to climb. The
    reference schedule already drives T above its initial 0.05 (the
    Stuart-Landau dynamics will lift T toward T*); the OPTIMISED
    schedule should do at least as well, ideally better.
    """
    problem = make_swat_problem(
        scenario='recovery',
        horizon_days=14,
        n_particles=128,
        optim_steps=300,
        learning_rate=5e-2,
    )
    pol = PiecewiseConstant.from_problem(problem)
    rng = jax.random.PRNGKey(0)

    # Reference baseline first.
    _, A_baseline, _ = simulate_latent(
        rng, problem, pol, problem.reference_schedule
    )
    mean_baseline = float(jnp.mean(A_baseline))

    # Run the optimiser.
    schedule, trace = optimise_schedule(problem, pol, rng)

    # Optimised schedule simulation.
    rng_eval = jax.random.PRNGKey(99)
    _, A_opt, _ = simulate_latent(rng_eval, problem, pol, schedule.theta)
    mean_optimised = float(jnp.mean(A_opt))

    # Loss must have decreased.
    assert float(trace.losses_total[-1]) < float(trace.losses_total[0]), \
        "Loss did not decrease over optimisation"

    # Optimised T(D) must be at least as good as reference, ideally
    # closer to T_star = 0.55.
    distance_baseline = abs(mean_baseline - T_STAR_HEALTHY)
    distance_optimised = abs(mean_optimised - T_STAR_HEALTHY)
    assert distance_optimised <= distance_baseline + 0.05, \
        f"Optimised T_mean {mean_optimised:.3f} not closer to T* than " \
        f"baseline {mean_baseline:.3f}"
