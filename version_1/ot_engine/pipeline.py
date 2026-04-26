"""
ot_engine/pipeline.py — End-to-end pipeline composition.
==========================================================
Date:    26 April 2026
Version: 1.0.0
Status:  Phase 5 deliverable.

Top-level entry points that compose the engine's individual stages:

    run_ot_pipeline(problem, rng, ...) -> (Schedule, OptimisationTrace, ClosedLoopResult)
        The whole flow: optimise, then verify in closed loop.

    compare_schedules(problem, policy, schedules, rng, ...) -> dict
        Run closed-loop verification on a set of schedules and return a
        dict keyed by schedule label, each holding the closed-loop
        metrics. Used by experiment scripts to compare an optimised
        schedule against the naive baselines.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import jax

from ot_engine.closed_loop import simulate_closed_loop
from ot_engine.optimise import optimise_schedule
from ot_engine.policies._abstract import ControlPolicy
from ot_engine.policies.piecewise_constant import PiecewiseConstant
from ot_engine.types import (
    BridgeProblem, ClosedLoopResult, OptimisationTrace, PolicyKind, Schedule,
)


# =========================================================================
# Top-level pipeline
# =========================================================================

def _build_policy(problem: BridgeProblem) -> ControlPolicy:
    """Construct the appropriate ControlPolicy from problem.policy_kind."""
    if problem.policy_kind == PolicyKind.PIECEWISE_CONSTANT:
        return PiecewiseConstant(
            horizon_days=problem.horizon_days,
            n_controls=problem.n_controls,
            control_bounds=problem.control_bounds,
        )
    raise NotImplementedError(
        f"policy_kind {problem.policy_kind} not supported in v1"
    )


def run_ot_pipeline(
    problem: BridgeProblem,
    rng: jax.Array,
    n_realisations: Optional[int] = None,
    optimise_kwargs: Optional[Dict] = None,
) -> tuple[Schedule, OptimisationTrace, ClosedLoopResult]:
    """Run the whole engine: optimise the schedule, then verify in closed loop.

    Args:
        problem: BridgeProblem from an adapter.
        rng: Root JAX PRNG key. The function splits this internally for
            optimisation and verification so the two steps are
            independent.
        n_realisations: Particle count for the closed-loop verification.
            None defaults to problem.n_particles. Larger values tighten
            the verification metrics.
        optimise_kwargs: Optional dict of extra kwargs passed through to
            `optimise_schedule` (e.g. `convergence_tol`, `verbose`).

    Returns:
        Tuple of:
            schedule: Optimised Schedule.
            trace: OptimisationTrace from the Adam loop.
            closed_loop: ClosedLoopResult with verification metrics.
    """
    rng_opt, rng_eval = jax.random.split(rng, 2)
    policy = _build_policy(problem)

    schedule, trace = optimise_schedule(
        problem, policy, rng_opt,
        **(optimise_kwargs or {}),
    )
    closed_loop = simulate_closed_loop(
        problem, policy, schedule, rng_eval,
        n_realisations=n_realisations,
    )
    return schedule, trace, closed_loop


# =========================================================================
# Multi-schedule comparison
# =========================================================================

def compare_schedules(
    problem: BridgeProblem,
    schedules: Sequence[Schedule],
    rng: jax.Array,
    n_realisations: Optional[int] = None,
) -> Dict[str, ClosedLoopResult]:
    """Run closed-loop verification on each schedule and tag by label.

    Args:
        problem: BridgeProblem from the adapter.
        schedules: Iterable of Schedule objects to compare. Each is
            expected to have metadata['label']; if absent, a positional
            label 'schedule_<i>' is used.
        rng: Fresh JAX PRNG key. A separate split is used per schedule
            so the closed-loop comparisons are mutually independent.
        n_realisations: Closed-loop particle count.

    Returns:
        Dict mapping label -> ClosedLoopResult.
    """
    policy = _build_policy(problem)
    results: Dict[str, ClosedLoopResult] = {}
    rng_iter = rng
    for i, sch in enumerate(schedules):
        rng_iter, rng_step = jax.random.split(rng_iter)
        label = sch.metadata.get('label', f'schedule_{i}')
        results[label] = simulate_closed_loop(
            problem, policy, sch, rng_step,
            n_realisations=n_realisations,
        )
    return results
