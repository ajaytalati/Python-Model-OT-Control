"""
ot_engine/closed_loop.py — Post-optimisation Monte-Carlo verification.
=======================================================================
Date:    26 April 2026
Version: 1.1.0
Status:  Phase 5 deliverable + post-review hardening.

Once the optimiser has produced a schedule, we do not just trust the
loss value. We re-simulate the latent SDE under the optimised schedule
with a fresh, independent Monte-Carlo cohort — typically larger than the
training cohort — and pack two metrics on a ClosedLoopResult:

    1. mmd_target — terminal MMD^2 between simulated and target
       amplitude marginals.
    2. fraction_in_healthy_basin — fraction of trajectories in the
       adapter-supplied "healthy basin" at terminal time. This is the
       metric that matters clinically.

The full trajectories and the terminal-amplitude vector are also
stored on the result so the caller can compute additional summary
statistics (mean, std, percentiles, etc.) without re-running the
simulation.

The basin indicator is supplied by each adapter via
BridgeProblem.basin_indicator_fn and has signature
(x, u_terminal, model_params) -> bool. If it is None the engine
returns NaN for the basin metric without failing the run.

This module re-uses simulate_latent for the forward pass — the only
difference from the optimiser's inner simulation is (a) the rng is
fresh and independent and (b) the particle count can be raised
arbitrarily without affecting gradient quality.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import jax
import jax.numpy as jnp

from ot_engine.policies._abstract import ControlPolicy
from ot_engine.simulator import simulate_latent
from ot_engine.terminal_cost import mmd_squared
from ot_engine.types import BridgeProblem, ClosedLoopResult, Schedule


# =========================================================================
# CLOSED-LOOP SIMULATOR
# =========================================================================

def simulate_closed_loop(
    problem: BridgeProblem,
    policy: ControlPolicy,
    schedule: Schedule,
    rng: jax.Array,
    n_realisations: Optional[int] = None,
) -> ClosedLoopResult:
    """Run the latent SDE under the schedule with a fresh MC cohort.

    Args:
        problem: BridgeProblem from the adapter.
        policy: ControlPolicy used to construct the schedule (typically
            PiecewiseConstant in v1).
        schedule: Optimised schedule from the engine. Must have
            schedule.horizon_days == problem.horizon_days and matching
            n_controls.
        rng: Fresh JAX PRNG key, independent of any used in optimisation.
        n_realisations: Particle count for verification. If None, falls
            back to problem.n_particles. Larger M tightens the MMD
            estimate; M = 256 is usually plenty. Must be >= 1.

    Returns:
        A ClosedLoopResult with the trajectory, the amplitude marginal at
        terminal time, the MMD against the target distribution, and (if
        the adapter supplied it) the fraction of trajectories in the
        healthy basin at t = D.

    Raises:
        ValueError: If schedule shape does not match problem, or if
            n_realisations is not None and < 1.
    """
    # --- Schedule / problem consistency (review fix H-7) ---
    if int(schedule.horizon_days) != int(problem.horizon_days):
        raise ValueError(
            f"schedule.horizon_days ({schedule.horizon_days}) != "
            f"problem.horizon_days ({problem.horizon_days})."
        )
    if int(schedule.n_controls) != int(problem.n_controls):
        raise ValueError(
            f"schedule.n_controls ({schedule.n_controls}) != "
            f"problem.n_controls ({problem.n_controls})."
        )

    # --- Particle count (review fix H-6: explicit None check) ---
    if n_realisations is None:
        M = int(problem.n_particles)
    else:
        M = int(n_realisations)
        if M < 1:
            raise ValueError(
                f"n_realisations must be >= 1 if specified; got {M}."
            )

    # Build a problem with the verification particle count, otherwise
    # identical to the original.
    problem_M = replace(problem, n_particles=M)

    # --- Forward simulation
    rng_sim, rng_target = jax.random.split(rng, 2)
    trajectories, amplitude_at_D, t_grid = simulate_latent(
        rng_sim, problem_M, policy, schedule.theta
    )

    # --- Target distribution samples (same M for matched MC noise)
    target_samples = problem.sample_target_amplitude(rng_target, M)

    # --- Terminal MMD
    mmd_value = float(mmd_squared(amplitude_at_D, target_samples))

    # --- Basin fraction (adapter-optional)
    if problem.basin_indicator_fn is None:
        basin_fraction = float('nan')
    else:
        # Terminal-day controls: last row of daily_values.
        u_terminal = jnp.asarray(schedule.daily_values)[-1]
        terminal_states = trajectories[:, -1, :]      # (M, dim_state)
        # vmap basin_indicator across particles.
        flags = jax.vmap(
            lambda x: problem.basin_indicator_fn(x, u_terminal,
                                                   problem.model_params)
        )(terminal_states)
        basin_fraction = float(jnp.mean(flags.astype(jnp.float64)))

    return ClosedLoopResult(
        t=t_grid,
        trajectories=trajectories,
        amplitude_at_D=amplitude_at_D,
        target_samples=target_samples,
        mmd_target=mmd_value,
        fraction_in_healthy_basin=basin_fraction,
    )
