"""
ot_engine/loss.py — Three-term loss composition.
==================================================
Date:    26 April 2026
Version: 1.1.0
Status:  Phase 2 deliverable + post-review documentation hardening.

The loss the optimiser minimises is

    L(theta) = alpha_terminal   * MMD^2(simulated A_D, target A)
             + alpha_transport  * sum_d  0.5 * ||u_d||^2 * dt_d
             + alpha_reference  * P(theta || N(mu_ref, sigma_ref^2))

where the third term P is the iid Gaussian negative-log-density penalty
(see ot_engine/reference/gaussian_iid.py). Each term plays a distinct
role:

    * Terminal MMD pulls the simulated terminal-amplitude marginal
      toward the clinical target distribution.
    * Transport regulariser favours small control magnitudes (the
      Benamou-Brenier kinetic-energy interpretation holds when controls
      have homogeneous units; for multi-control problems with mixed
      units it acts as a quadratic regulariser).
    * Reference penalty pulls theta toward the adapter's pre-intervention
      baseline, expressing a clinical preference for small deviations.

The factory `make_loss_fn` closes over the BridgeProblem and the
ControlPolicy, returning a function loss_fn(theta, rng) suitable for
jax.value_and_grad. It returns a (scalar, dict) pair so the caller can
log the three components separately.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp

from ot_engine.policies._abstract import ControlPolicy
from ot_engine.reference import gaussian_iid_kl
from ot_engine.simulator import simulate_latent
from ot_engine.terminal_cost import mmd_squared
from ot_engine.types import BridgeProblem


# =========================================================================
# TRANSPORT COST  (closed form for piecewise-constant policy)
# =========================================================================

def transport_cost_piecewise_constant(
    daily_values: jnp.ndarray,
    dt_per_day: float = 1.0,
) -> jnp.ndarray:
    """Quadratic-in-control transport regulariser for piecewise-constant u.

    For piecewise-constant u(t) = u_d on day d, the integral

        \\int_0^D 0.5 ||u(t)||^2 dt = sum_d 0.5 ||u_d||^2 * dt_per_day.

    When all controls have the same units (e.g. forces, currents) this
    is exactly the Benamou-Brenier L^2 kinetic-energy cost. When
    controls have heterogeneous units (the SWAT case: V_h dimensionless,
    V_c in hours) the strict Benamou-Brenier interpretation breaks down;
    the term then acts as a quadratic regulariser pulling each control
    component toward zero.

    Args:
        daily_values: Schedule values per day, shape (D, n_controls).
        dt_per_day: Time width of each constant segment (1 day by default
            for piecewise-constant). Kept as an argument so the same
            function works for finer step structures in future.

    Returns:
        Scalar transport cost.
    """
    return 0.5 * jnp.sum(daily_values * daily_values) * dt_per_day


# =========================================================================
# LOSS FACTORY
# =========================================================================

def make_loss_fn(
    problem: BridgeProblem,
    policy: ControlPolicy,
) -> Callable:
    """Build the loss function closed over the problem and policy.

    The returned loss_fn signature is
        loss_fn(theta, rng) -> (total: scalar, components: dict)

    where components contains:
        'terminal':  alpha_terminal  * MMD^2(simulated, target)
        'transport': alpha_transport * 0.5 sum ||u_d||^2 dt
        'reference': alpha_reference * gaussian_iid_kl(theta, mu_ref, sigma_ref)
        'total':     sum of the three weighted terms

    Both the rng and theta are explicit arguments so the same loss_fn can
    be used (i) inside `jax.value_and_grad(loss_fn)` for the optimiser, and
    (ii) standalone for diagnostics.

    Args:
        problem: BridgeProblem from the adapter.
        policy: ControlPolicy instance.

    Returns:
        loss_fn: callable with signature (theta, rng) -> (scalar, dict).
    """
    alpha_terminal = problem.alpha_terminal
    alpha_transport = problem.alpha_transport
    alpha_reference = problem.alpha_reference

    mu_ref = jnp.asarray(problem.reference_schedule)
    sigma_ref = jnp.asarray(problem.reference_sigma)

    def loss_fn(
        theta: jnp.ndarray,
        rng: jax.Array,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        # 1) Forward-simulate under the candidate schedule.
        rng_sim, rng_target = jax.random.split(rng, 2)
        _trajectories, A_simulated, _t_grid = simulate_latent(
            rng_sim, problem, policy, theta
        )

        # 2) Sample target distribution. n_target_samples = n_particles
        #    keeps the MMD estimator's two empirical sums on equal footing.
        A_target = problem.sample_target_amplitude(
            rng_target, problem.n_particles
        )

        # 3) Three loss components.
        L_terminal = mmd_squared(A_simulated, A_target)
        daily_values = policy.evaluate_daily(theta)
        L_transport = transport_cost_piecewise_constant(daily_values)
        L_reference = gaussian_iid_kl(theta, mu_ref, sigma_ref)

        # 4) Weighted total.
        total = (alpha_terminal * L_terminal
                 + alpha_transport * L_transport
                 + alpha_reference * L_reference)

        components = {
            'terminal':  alpha_terminal  * L_terminal,
            'transport': alpha_transport * L_transport,
            'reference': alpha_reference * L_reference,
            'total':     total,
        }
        return total, components

    return loss_fn
