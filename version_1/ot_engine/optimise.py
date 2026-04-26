"""
ot_engine/optimise.py — Optax-based gradient-descent loop.
============================================================
Date:    25 April 2026
Version: 1.0.0
Status:  Phase 3 deliverable.

The optimisation loop driving the schedule parameters theta to minimise
the three-term loss. Standard recipe for differentiable transport:

    1. Build a JAX-jitted single-step function:
         (theta, opt_state, rng) -> (theta_new, opt_state_new, loss_components, grad_norm)
    2. Loop in Python, calling step() each iteration and recording the
       per-step diagnostics.
    3. Sliding-window convergence check halts early when relative-mean
       change drops below tolerance.

DESIGN CHOICES
--------------
* Optax Adam with gradient clipping by global norm (default 1.0). The
  loss has terms with quite different scales (MMD is O(0.1), reference
  KL can be O(10)) so clipping protects against early-step blow-ups.
* By default the same root rng is used for every Adam step, making the
  loss a deterministic function of theta. This is standard practice in
  the OT-Flow / Schrödinger-Föllmer family — sample the Monte-Carlo
  cohort once, then optimise against that cohort. A `resample_every`
  parameter is provided for users who prefer mini-batch-style refresh.
* The single-step function is jit'd once. In benchmarks (Phase 2) the
  forward simulator runs at ~10 ms per call jitted; the full step is
  forward + backward + optax update, expected ~30-50 ms per step. A
  2000-step run is therefore ~60-100 s on CPU JAX — comfortably under
  the plan's 30-s target for the 1-D OU toy problem (which uses far
  fewer steps and a smaller particle count).
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import optax

from ot_engine.diagnostics import convergence_check
from ot_engine.loss import make_loss_fn
from ot_engine.policies._abstract import ControlPolicy
from ot_engine.types import (
    BridgeProblem, OptimisationTrace, Schedule, default_control_names,
)


# =========================================================================
# OPTIMISER FACTORY
# =========================================================================

def _build_optimiser(
    learning_rate: float,
    grad_clip_norm: float = 1.0,
) -> optax.GradientTransformation:
    """Adam with global-norm gradient clipping.

    Args:
        learning_rate: Adam learning rate.
        grad_clip_norm: Maximum gradient L2 norm; entries above are
            rescaled. Set to a large value (e.g. 1e6) to disable.

    Returns:
        An Optax GradientTransformation chain.
    """
    return optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adam(learning_rate),
    )


# =========================================================================
# OPTIMISE_SCHEDULE — the public entry point
# =========================================================================

def optimise_schedule(
    problem: BridgeProblem,
    policy: ControlPolicy,
    rng: jax.Array,
    grad_clip_norm: float = 1.0,
    convergence_window: int = 50,
    convergence_tol: float = 1e-4,
    resample_every: int = 0,
    verbose: bool = False,
) -> Tuple[Schedule, OptimisationTrace]:
    """Run Adam on the loss surface to optimise the schedule.

    Args:
        problem: The BridgeProblem from an adapter. n_particles, dt_days,
            optim_steps, learning_rate are all read from here.
        policy: The ControlPolicy instance.
        rng: Root JAX PRNG key. Used either as the single fixed seed for
            all Monte-Carlo evaluations (default) or refreshed every
            resample_every steps (see below).
        grad_clip_norm: Global-norm clip threshold for gradients.
        convergence_window: Steps in each half of the sliding-window test.
        convergence_tol: Relative-mean-change tolerance.
        resample_every: If > 0, refresh the Monte-Carlo rng every this
            many steps. Default 0 means use the same rng for all steps —
            the loss is then a deterministic function of theta. Standard
            practice in the OT-Flow / Schrödinger-Föllmer family. Set to
            a positive integer (e.g. 100) for stochastic mini-batch-style
            training; the loss becomes noisier but the optimiser is less
            prone to overfitting the particular MC sample.
        verbose: If True, print loss every 100 steps. Off by default.

    Returns:
        schedule: Schedule dataclass with the final theta and daily values.
        trace: OptimisationTrace with per-step diagnostics.
    """
    # --- Setup ---
    loss_fn = make_loss_fn(problem, policy)
    optimiser = _build_optimiser(problem.learning_rate, grad_clip_norm)

    theta = policy.init_params(jnp.asarray(problem.reference_schedule))
    opt_state = optimiser.init(theta)

    # --- JIT the inner step (compiles once) ---
    def step(theta, opt_state, rng_step):
        """Single Adam step: compute grad, apply update, return diagnostics."""
        def loss_only(t):
            total, components = loss_fn(t, rng_step)
            return total, components
        (total, components), grad = jax.value_and_grad(
            loss_only, has_aux=True
        )(theta)
        updates, opt_state_new = optimiser.update(grad, opt_state, theta)
        theta_new = optax.apply_updates(theta, updates)
        grad_norm = optax.global_norm(grad)
        return theta_new, opt_state_new, components, grad_norm

    step = jax.jit(step)

    # --- Python loop with sliding-window convergence check ---
    n_steps_max = problem.optim_steps

    # Strategy for the per-step rng:
    #   resample_every == 0  ->  always reuse `rng` (deterministic loss)
    #   resample_every > 0   ->  split off a new rng every that many steps
    rng_current = rng
    rng_master = rng

    losses_total = []
    losses_terminal = []
    losses_transport = []
    losses_reference = []
    grad_norms = []

    converged = False
    n_steps_run = 0

    for k in range(n_steps_max):
        if resample_every > 0 and k % resample_every == 0:
            rng_master, rng_current = jax.random.split(rng_master)
        # else: rng_current stays fixed at its initial value.

        theta, opt_state, components, grad_norm = step(
            theta, opt_state, rng_current
        )

        losses_total.append(float(components['total']))
        losses_terminal.append(float(components['terminal']))
        losses_transport.append(float(components['transport']))
        losses_reference.append(float(components['reference']))
        grad_norms.append(float(grad_norm))
        n_steps_run = k + 1

        if verbose and (k + 1) % 100 == 0:
            print(f"  step {k+1:5d}  L={losses_total[-1]:.6f}  "
                  f"|g|={grad_norms[-1]:.4f}")

        # Convergence check — only fires after at least 2*window steps.
        if (k + 1) >= 2 * convergence_window and (k + 1) % convergence_window == 0:
            if convergence_check(
                jnp.asarray(losses_total),
                window=convergence_window,
                tol=convergence_tol,
            ):
                converged = True
                break

    # --- Pack the trace and the schedule ---
    trace = OptimisationTrace(
        losses_total=jnp.asarray(losses_total),
        losses_terminal=jnp.asarray(losses_terminal),
        losses_transport=jnp.asarray(losses_transport),
        losses_reference=jnp.asarray(losses_reference),
        grad_norms=jnp.asarray(grad_norms),
        converged=converged,
        n_steps_run=n_steps_run,
    )

    daily = policy.evaluate_daily(theta)
    # Honour adapter-supplied control names if present; fall back to
    # generic ('u_0', 'u_1', ...) otherwise. (Review fix H-3.)
    control_names = (problem.control_names
                     if problem.control_names is not None
                     else default_control_names(problem.n_controls))
    # Guard against optim_steps == 0 / convergence-on-step-0 (review fix C-2).
    final_loss = losses_total[-1] if losses_total else float('nan')
    schedule = Schedule(
        theta=theta,
        daily_values=daily,
        horizon_days=problem.horizon_days,
        n_controls=problem.n_controls,
        control_names=control_names,
        metadata={
            'final_loss': final_loss,
            'converged': converged,
            'n_steps_run': n_steps_run,
        },
    )

    return schedule, trace
