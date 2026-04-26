"""
ot_engine/compare.py — Naive-baseline schedules for closed-loop comparison.
============================================================================
Date:    26 April 2026
Version: 1.0.0
Status:  Phase 5 deliverable.

Three baseline schedules an optimised schedule should beat:

    zero_control_schedule(problem, policy)
        Schedule with theta = 0 everywhere. The "do absolutely nothing"
        baseline. For most adapters this is *not* the same as the
        reference baseline (which holds the patient's pre-intervention
        controls); it represents a hypothetical world where every
        control is forced to zero.

    constant_reference_schedule(problem, policy)
        Schedule with theta equal to the adapter's reference baseline.
        This is the patient's "do nothing different" trajectory — leave
        them with whatever V_h, V_n, V_c they came in with. The
        clinically realistic null-hypothesis baseline.

    linear_interpolation_schedule(problem, policy, theta_target)
        Schedule with theta linearly interpolated between the reference
        baseline at day 0 and theta_target at day D-1. The cheapest-
        possible "smooth ramp" baseline. theta_target is supplied by
        the caller (typically the adapter's clinically-ideal control
        vector, e.g. (V_h=1.0, V_n=0.3, V_c=0) for SWAT).

The optimised schedule should beat all three on the closed-loop metrics
(terminal MMD, basin fraction, terminal-amplitude mean).
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp

from ot_engine.policies._abstract import ControlPolicy
from ot_engine.types import BridgeProblem, Schedule, default_control_names


# =========================================================================
# Helpers
# =========================================================================

def _wrap_as_schedule(theta: jnp.ndarray, policy: ControlPolicy,
                       problem: BridgeProblem,
                       label: str) -> Schedule:
    """Wrap a raw theta into a Schedule dataclass for use by closed_loop.

    Honours adapter-supplied control names if `problem.control_names` is
    set; otherwise falls back to ('u_0', 'u_1', ...).

    Args:
        theta: Schedule parameters, shape (D, n_controls).
        policy: ControlPolicy instance.
        problem: BridgeProblem (used for control-name + shape defaults).
        label: Short descriptor stored in metadata.

    Returns:
        A Schedule with the daily values and a metadata['label'].
    """
    daily = policy.evaluate_daily(theta)
    control_names = (problem.control_names
                     if problem.control_names is not None
                     else default_control_names(problem.n_controls))
    return Schedule(
        theta=theta,
        daily_values=daily,
        horizon_days=problem.horizon_days,
        n_controls=problem.n_controls,
        control_names=control_names,
        metadata={'label': label},
    )


# =========================================================================
# Baseline schedules
# =========================================================================

def zero_control_schedule(problem: BridgeProblem,
                            policy: ControlPolicy) -> Schedule:
    """Schedule with all controls set to zero.

    Args:
        problem: BridgeProblem from the adapter.
        policy: ControlPolicy instance.

    Returns:
        A Schedule with theta = 0.
    """
    theta = jnp.zeros((problem.horizon_days, problem.n_controls))
    return _wrap_as_schedule(theta, policy, problem, label='zero_control')


def constant_reference_schedule(problem: BridgeProblem,
                                  policy: ControlPolicy) -> Schedule:
    """Schedule equal to the adapter's reference baseline.

    Args:
        problem: BridgeProblem from the adapter.
        policy: ControlPolicy instance.

    Returns:
        A Schedule equal to problem.reference_schedule.
    """
    theta = jnp.asarray(problem.reference_schedule)
    return _wrap_as_schedule(theta, policy, problem, label='constant_reference')


def linear_interpolation_schedule(
    problem: BridgeProblem,
    policy: ControlPolicy,
    theta_target: Sequence[float],
) -> Schedule:
    """Schedule linearly interpolated from reference (day 0) to theta_target (day D-1).

    Args:
        problem: BridgeProblem from the adapter.
        policy: ControlPolicy instance.
        theta_target: Target control vector at terminal day, shape
            (n_controls,). Typically the adapter's clinically-ideal
            controls.

    Returns:
        A Schedule whose daily values ramp linearly between the
        reference's first day and theta_target.
    """
    theta_target_arr = jnp.asarray(theta_target)
    if theta_target_arr.shape != (problem.n_controls,):
        raise ValueError(
            f"theta_target must have shape ({problem.n_controls},); "
            f"got {theta_target_arr.shape}"
        )
    ref = jnp.asarray(problem.reference_schedule)
    start = ref[0]                   # control values on day 0 of reference
    end = theta_target_arr
    D = problem.horizon_days
    # Linear interpolation: alpha goes from 0 (day 0) to 1 (day D-1).
    alpha = jnp.linspace(0.0, 1.0, D).reshape(D, 1)
    theta = start[None, :] * (1.0 - alpha) + end[None, :] * alpha
    return _wrap_as_schedule(theta, policy, problem, label='linear_interpolation')
