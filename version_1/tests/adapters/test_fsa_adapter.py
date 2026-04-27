"""
tests/adapters/test_fsa_adapter.py — Structural tests for the FSA adapter.
==========================================================================

Scope: structural correctness only.

  - make_fsa_problem returns a valid BridgeProblem for each scenario.
  - Unknown scenarios raise ValueError.
  - Drift / diffusion are JIT-compatible and produce finite output.
  - State-clip keeps states in physical bounds.
  - Forward simulation under any schedule produces finite trajectories.
  - End-to-end optimisation runs without crashing for one scenario;
    we assert the loss is finite and the schedule respects bounds, but
    NOT that the schedule is clinically optimal.

Why no convergence-quality tests
--------------------------------
The FSA dynamics expose the single-bandwidth-MMD gradient-vanishing
pathology (see F2 in docs/Future_Features.md and the docstring of
adapters/fsa_high_res/adapter.py::make_fsa_problem). The optimiser
produces valid bounds-respecting schedules but may land in
clinically-counterintuitive local minima for some scenarios. A true
"phase 5" acceptance test (beats baselines, MMD within k * best) will
land alongside the multi-bandwidth-MMD fix, in a follow-up release.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from ot_engine.types import BridgeProblem
from adapters.fsa_high_res import (
    make_fsa_problem, list_scenarios,
    FSA_CONTROL_NAMES, A_STAR_HEALTHY,
)
from _vendored_models.fsa_high_res import (
    fsa_drift, fsa_diffusion, fsa_state_clip, amplitude_of_fsa,
    healthy_attractor_check, default_fsa_parameters,
)


# =========================================================================
# Catalogue tests
# =========================================================================

def test_list_scenarios_returns_three():
    scs = list_scenarios()
    assert isinstance(scs, tuple)
    assert len(scs) == 3
    assert set(scs) == {'unfit_recovery', 'over_trained', 'detrained_athlete'}


def test_fsa_control_names():
    assert FSA_CONTROL_NAMES == ('T_B', 'Phi')
    assert isinstance(A_STAR_HEALTHY, float)
    assert 0.0 <= A_STAR_HEALTHY <= 2.0


# =========================================================================
# make_fsa_problem tests
# =========================================================================

@pytest.mark.parametrize('scenario', list_scenarios())
def test_make_fsa_problem_returns_bridge_problem(scenario: str):
    p = make_fsa_problem(scenario, horizon_days=14, n_particles=64)
    assert isinstance(p, BridgeProblem)
    assert p.name == f'fsa_{scenario}'
    assert p.n_controls == 2
    assert p.horizon_days == 14
    assert p.control_names == FSA_CONTROL_NAMES
    # Reference schedule shape
    assert p.reference_schedule.shape == (14, 2)
    # Bounds are non-empty 2-tuples of (lo, hi) with lo < hi
    assert len(p.control_bounds) == 2
    for lo, hi in p.control_bounds:
        assert lo < hi
    # All required callables are present
    assert callable(p.drift_fn_jax)
    assert callable(p.diffusion_fn_jax)
    assert callable(p.sample_initial_state)
    assert callable(p.sample_target_amplitude)
    assert callable(p.amplitude_of)
    # Optional state-clip and basin-indicator are present
    assert callable(p.state_clip_fn)
    assert callable(p.basin_indicator_fn)


def test_unknown_scenario_raises():
    with pytest.raises(ValueError, match='Unknown FSA scenario'):
        make_fsa_problem('not_a_scenario')


# =========================================================================
# Initial-state sampler tests
# =========================================================================

@pytest.mark.parametrize('scenario', list_scenarios())
def test_initial_state_sampler_shapes_and_bounds(scenario: str):
    p = make_fsa_problem(scenario, horizon_days=14, n_particles=64)
    rng = jax.random.PRNGKey(0)
    samples = p.sample_initial_state(rng, 100)
    assert samples.shape == (100, 3)
    # B in [0, 1]
    assert jnp.all(samples[:, 0] >= 0.0)
    assert jnp.all(samples[:, 0] <= 1.0)
    # F >= 0
    assert jnp.all(samples[:, 1] >= 0.0)
    # A >= 0
    assert jnp.all(samples[:, 2] >= 0.0)
    # All finite
    assert jnp.all(jnp.isfinite(samples))


# =========================================================================
# Target sampler tests
# =========================================================================

@pytest.mark.parametrize('scenario', list_scenarios())
def test_target_sampler_returns_finite_nonnegative_amplitudes(scenario: str):
    p = make_fsa_problem(scenario, horizon_days=14, n_particles=64)
    rng = jax.random.PRNGKey(99)
    samples = p.sample_target_amplitude(rng, 200)
    assert samples.shape == (200,)
    assert jnp.all(jnp.isfinite(samples))
    assert jnp.all(samples >= 0.0)
    # The target should be meaningfully positive (clinical relevance)
    assert float(jnp.mean(samples)) > 0.1


# =========================================================================
# Drift / diffusion tests
# =========================================================================

def test_fsa_drift_jit_compatible():
    drift_jit = jax.jit(fsa_drift)
    p = default_fsa_parameters()
    x = jnp.array([0.3, 0.1, 0.4])
    u = jnp.array([0.5, 0.1])
    dx = drift_jit(0.0, x, u, p)
    assert dx.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(dx)))


def test_fsa_drift_responds_to_T_B():
    """If T_B > B, B should drift up (positive dB)."""
    p = default_fsa_parameters()
    x = jnp.array([0.1, 0.05, 0.4])
    # T_B = 0.5 > B = 0.1, so dB should be positive
    u_hi = jnp.array([0.5, 0.05])
    dx_hi = fsa_drift(0.0, x, u_hi, p)
    assert float(dx_hi[0]) > 0.0
    # T_B = 0.0 < B = 0.1, so dB should be negative
    u_lo = jnp.array([0.0, 0.05])
    dx_lo = fsa_drift(0.0, x, u_lo, p)
    assert float(dx_lo[0]) < 0.0


def test_fsa_drift_responds_to_Phi():
    """Higher Phi increases dF (more strain produced)."""
    p = default_fsa_parameters()
    x = jnp.array([0.3, 0.1, 0.4])
    u_lo = jnp.array([0.5, 0.0])
    u_hi = jnp.array([0.5, 0.5])
    dx_lo = fsa_drift(0.0, x, u_lo, p)
    dx_hi = fsa_drift(0.0, x, u_hi, p)
    # dF is higher under higher Phi
    assert float(dx_hi[1]) > float(dx_lo[1])


def test_fsa_diffusion_jit_compatible_and_nonnegative():
    diff_jit = jax.jit(fsa_diffusion)
    p = default_fsa_parameters()
    for x in [jnp.array([0.5, 0.1, 0.5]),
                jnp.array([0.0, 0.0, 0.0]),    # all-boundary edge case
                jnp.array([1.0, 0.0, 0.0]),    # B at upper bound
                jnp.array([0.5, 2.0, 1.5])]:
        sig = diff_jit(x, p)
        assert sig.shape == (3,)
        assert bool(jnp.all(jnp.isfinite(sig)))
        assert bool(jnp.all(sig >= 0.0))


def test_fsa_diffusion_gradient_finite_at_boundary():
    """Critical: gradient of diffusion at the (B, F, A) = (0, 0, 0)
    boundary must be finite. Without epsilon-regularisation the
    sqrt(B*(1-B)), sqrt(F), sqrt(A) gradients explode and crash AD."""
    p = default_fsa_parameters()

    def total_diffusion(x):
        return jnp.sum(fsa_diffusion(x, p) ** 2)

    grad_fn = jax.grad(total_diffusion)
    g0 = grad_fn(jnp.array([0.0, 0.0, 0.0]))
    assert bool(jnp.all(jnp.isfinite(g0)))
    g1 = grad_fn(jnp.array([1.0, 0.0, 0.0]))
    assert bool(jnp.all(jnp.isfinite(g1)))


# =========================================================================
# State-clip tests
# =========================================================================

def test_fsa_state_clip_keeps_states_in_bounds():
    p = default_fsa_parameters()
    # Out-of-bounds state
    x = jnp.array([1.5, -0.3, -0.1])
    clipped = fsa_state_clip(x, p)
    assert 0.0 <= float(clipped[0]) <= 1.0
    assert float(clipped[1]) >= 0.0
    assert float(clipped[2]) >= 0.0
    # In-bounds state should be unchanged
    x_ok = jnp.array([0.3, 0.1, 0.5])
    clipped_ok = fsa_state_clip(x_ok, p)
    assert bool(jnp.allclose(clipped_ok, x_ok))


# =========================================================================
# Healthy-attractor predicate tests
# =========================================================================

def test_healthy_attractor_check_returns_bool():
    p = default_fsa_parameters()
    # B=0.5, F=0.1: mu = 0.02 + 0.15 - 0.01 - 0.004 = 0.156 > 0
    assert bool(healthy_attractor_check(jnp.array(0.5), jnp.array(0.1), p))
    # B=0.0, F=1.0: mu = 0.02 + 0 - 0.1 - 0.4 = -0.48 < 0
    assert not bool(healthy_attractor_check(jnp.array(0.0), jnp.array(1.0), p))


# =========================================================================
# Amplitude projector
# =========================================================================

def test_amplitude_of_fsa_picks_index_2():
    x = jnp.array([0.3, 0.1, 0.7])
    assert float(amplitude_of_fsa(x)) == pytest.approx(0.7)


# =========================================================================
# End-to-end smoke tests
# =========================================================================

def test_fsa_simulator_no_state_violations():
    """Forward simulation under reference schedule keeps states bounded."""
    from ot_engine import PiecewiseConstant, simulate_latent
    p = make_fsa_problem('detrained_athlete', horizon_days=14, n_particles=64,
                          dt_days=0.05)
    pol = PiecewiseConstant.from_problem(p)
    rng = jax.random.PRNGKey(0)
    traj, A_D, t_grid = simulate_latent(rng, p, pol, p.reference_schedule)
    assert bool(jnp.all(jnp.isfinite(traj)))
    assert bool(jnp.all(traj[:, :, 0] >= -1e-6))     # B >= 0
    assert bool(jnp.all(traj[:, :, 0] <= 1.0 + 1e-6))  # B <= 1
    assert bool(jnp.all(traj[:, :, 1] >= -1e-6))     # F >= 0
    assert bool(jnp.all(traj[:, :, 2] >= -1e-6))     # A >= 0
    assert bool(jnp.all(jnp.isfinite(A_D)))


def test_fsa_loss_finite_at_reference():
    """Loss and gradient at theta = reference must be finite (no NaN)."""
    from ot_engine import PiecewiseConstant
    from ot_engine.loss import make_loss_fn

    p = make_fsa_problem('unfit_recovery', horizon_days=14, n_particles=64,
                          dt_days=0.05)
    pol = PiecewiseConstant.from_problem(p)
    loss_fn = make_loss_fn(p, pol)

    loss, _ = loss_fn(p.reference_schedule, jax.random.PRNGKey(0))
    assert bool(jnp.isfinite(loss))

    grad_fn = jax.grad(lambda t, r: loss_fn(t, r)[0])
    g = grad_fn(p.reference_schedule, jax.random.PRNGKey(0))
    assert bool(jnp.all(jnp.isfinite(g))), "Gradient at theta_0 has non-finite entries"


def test_fsa_optimisation_runs_and_respects_bounds():
    """End-to-end optimisation completes and produces a bounds-respecting
    schedule. We do NOT assert convergence quality (see file docstring)."""
    from ot_engine import PiecewiseConstant, optimise_schedule

    p = make_fsa_problem('detrained_athlete', horizon_days=7, n_particles=64,
                          dt_days=0.1, optim_steps=50)
    pol = PiecewiseConstant.from_problem(p)
    sch, trace = optimise_schedule(p, pol, jax.random.PRNGKey(0))

    # Schedule respects bounds elementwise
    daily = jnp.asarray(sch.daily_values)
    for c, (lo, hi) in enumerate(p.control_bounds):
        assert bool(jnp.all(daily[:, c] >= lo - 1e-6)), \
            f"Control {c} below lower bound"
        assert bool(jnp.all(daily[:, c] <= hi + 1e-6)), \
            f"Control {c} above upper bound"

    # Final loss is finite
    assert bool(jnp.isfinite(trace.losses_total[-1])), \
        "Final loss is NaN or Inf"
