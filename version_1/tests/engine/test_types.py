"""
tests/engine/test_types.py — Dataclass instantiation and contract checks.
==========================================================================
Phase 1 unit tests for ot_engine.types.

Covers:
    - All four dataclasses instantiate with valid fields
    - All three enums have the v1 members
    - Frozen-ness: cannot mutate fields after construction
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from ot_engine.types import (
    BridgeProblem,
    Schedule,
    OptimisationTrace,
    ClosedLoopResult,
    PolicyKind,
    TerminalCostKind,
    ReferenceKind,
)


# =========================================================================
# Enum sanity checks
# =========================================================================

def test_policy_kind_has_piecewise_constant():
    assert PolicyKind.PIECEWISE_CONSTANT.value == "piecewise_constant"


def test_terminal_cost_kind_has_mmd():
    assert TerminalCostKind.MMD.value == "mmd"


def test_reference_kind_has_gaussian_iid():
    assert ReferenceKind.GAUSSIAN_IID.value == "gaussian_iid"


# =========================================================================
# Schedule
# =========================================================================

def test_schedule_instantiates():
    sch = Schedule(
        theta=jnp.zeros((14, 2)),
        daily_values=jnp.zeros((14, 2)),
        horizon_days=14,
        n_controls=2,
        control_names=("u_0", "u_1"),
    )
    assert sch.horizon_days == 14
    assert sch.n_controls == 2
    assert sch.theta.shape == (14, 2)
    assert sch.metadata == {}


def test_schedule_is_frozen():
    sch = Schedule(
        theta=jnp.zeros((14, 2)),
        daily_values=jnp.zeros((14, 2)),
        horizon_days=14,
        n_controls=2,
        control_names=("u_0", "u_1"),
    )
    with pytest.raises((AttributeError, Exception)):
        sch.horizon_days = 21  # type: ignore[misc]


# =========================================================================
# OptimisationTrace
# =========================================================================

def test_optimisation_trace_instantiates():
    n_steps = 100
    tr = OptimisationTrace(
        losses_total=jnp.zeros(n_steps),
        losses_terminal=jnp.zeros(n_steps),
        losses_transport=jnp.zeros(n_steps),
        losses_reference=jnp.zeros(n_steps),
        grad_norms=jnp.zeros(n_steps),
        converged=True,
        n_steps_run=n_steps,
    )
    assert tr.n_steps_run == n_steps
    assert tr.converged is True
    assert tr.losses_total.shape == (n_steps,)


# =========================================================================
# ClosedLoopResult
# =========================================================================

def test_closed_loop_result_instantiates():
    n_real = 50
    n_t = 100
    dim = 4
    res = ClosedLoopResult(
        t=jnp.linspace(0.0, 14.0, n_t),
        trajectories=jnp.zeros((n_real, n_t, dim)),
        amplitude_at_D=jnp.zeros(n_real),
        target_samples=jnp.zeros(200),
        mmd_target=0.05,
        fraction_in_healthy_basin=0.92,
    )
    assert res.trajectories.shape == (n_real, n_t, dim)
    assert res.mmd_target == pytest.approx(0.05)
    assert res.fraction_in_healthy_basin == pytest.approx(0.92)


# =========================================================================
# BridgeProblem — uses minimal stub callables
# =========================================================================

def _stub_drift(t, x, u, p):
    return jnp.zeros_like(x)


def _stub_diffusion(x, p):
    return jnp.ones_like(x) * 0.1


def _stub_sample_init(rng, n):
    return jnp.zeros((n, 4))


def _stub_sample_target(rng, n):
    return jnp.zeros(n)


def _stub_amplitude(x):
    return x[-1]


def test_bridge_problem_instantiates():
    D = 14
    n_c = 2
    problem = BridgeProblem(
        name="test_problem",
        drift_fn_jax=_stub_drift,
        diffusion_fn_jax=_stub_diffusion,
        model_params={"sigma": 0.1},
        sample_initial_state=_stub_sample_init,
        sample_target_amplitude=_stub_sample_target,
        amplitude_of=_stub_amplitude,
        n_controls=n_c,
        control_bounds=((0.0, 1.0), (0.0, 0.5)),
        horizon_days=D,
        reference_schedule=jnp.zeros((D, n_c)),
        reference_sigma=jnp.ones((D, n_c)) * 0.1,
    )
    assert problem.horizon_days == D
    assert problem.n_controls == n_c
    assert problem.policy_kind == PolicyKind.PIECEWISE_CONSTANT
    assert problem.terminal_cost_kind == TerminalCostKind.MMD
    assert problem.reference_kind == ReferenceKind.GAUSSIAN_IID
    assert problem.alpha_terminal == pytest.approx(1.0)
    assert problem.alpha_transport == pytest.approx(0.1)
    assert problem.alpha_reference == pytest.approx(0.1)
    # Stub callables work
    assert problem.amplitude_of(jnp.array([1.0, 2.0, 3.0, 4.0])) == 4.0
    # control_names defaults to None
    assert problem.control_names is None


# =========================================================================
# BridgeProblem validation (post-review hardening)
# =========================================================================

def _valid_kwargs(D=4, n_c=1):
    """Build a kwargs dict that produces a valid BridgeProblem."""
    return dict(
        name='valid',
        drift_fn_jax=_stub_drift,
        diffusion_fn_jax=_stub_diffusion,
        model_params={},
        sample_initial_state=_stub_sample_init,
        sample_target_amplitude=_stub_sample_target,
        amplitude_of=_stub_amplitude,
        n_controls=n_c,
        control_bounds=tuple((0.0, 1.0) for _ in range(n_c)),
        horizon_days=D,
        reference_schedule=jnp.zeros((D, n_c)),
        reference_sigma=jnp.ones((D, n_c)),
    )


def test_bridge_problem_rejects_none_callables():
    kw = _valid_kwargs()
    kw['drift_fn_jax'] = None
    with pytest.raises(ValueError, match='drift_fn_jax'):
        BridgeProblem(**kw)


def test_bridge_problem_rejects_zero_horizon():
    kw = _valid_kwargs()
    kw['horizon_days'] = 0
    with pytest.raises(ValueError, match='horizon_days'):
        BridgeProblem(**kw)


def test_bridge_problem_rejects_zero_n_controls():
    kw = _valid_kwargs()
    kw['n_controls'] = 0
    with pytest.raises(ValueError, match='n_controls'):
        BridgeProblem(**kw)


def test_bridge_problem_rejects_zero_sigma_ref():
    kw = _valid_kwargs()
    kw['reference_sigma'] = jnp.zeros((kw['horizon_days'], kw['n_controls']))
    with pytest.raises(ValueError, match='reference_sigma'):
        BridgeProblem(**kw)


def test_bridge_problem_rejects_shape_mismatch():
    kw = _valid_kwargs(D=4, n_c=2)
    kw['reference_schedule'] = jnp.zeros((4, 1))   # wrong shape
    with pytest.raises(ValueError, match='reference_schedule shape'):
        BridgeProblem(**kw)


def test_bridge_problem_rejects_negative_alpha():
    kw = _valid_kwargs()
    kw['alpha_transport'] = -0.1
    with pytest.raises(ValueError, match='alpha_transport'):
        BridgeProblem(**kw)


def test_bridge_problem_rejects_bad_control_names_length():
    kw = _valid_kwargs(n_c=3)
    kw['control_bounds'] = ((-1, 1), (-1, 1), (-1, 1))
    kw['control_names'] = ('V_h', 'V_n')   # only 2, expected 3
    with pytest.raises(ValueError, match='control_names'):
        BridgeProblem(**kw)


def test_bridge_problem_accepts_explicit_control_names():
    kw = _valid_kwargs(n_c=3)
    kw['control_bounds'] = ((-1, 1), (-1, 1), (-1, 1))
    kw['control_names'] = ('V_h', 'V_n', 'V_c')
    p = BridgeProblem(**kw)
    assert p.control_names == ('V_h', 'V_n', 'V_c')


def test_default_control_names_helper():
    from ot_engine.types import default_control_names
    assert default_control_names(3) == ('u_0', 'u_1', 'u_2')
    assert default_control_names(1) == ('u_0',)
