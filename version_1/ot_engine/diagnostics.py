"""
ot_engine/diagnostics.py — Convergence-check + trace-summary helpers.
======================================================================
Date:    25 April 2026
Version: 1.0.0
Status:  Phase 3 deliverable.

Pure functions used by the optimisation loop and by post-hoc analysis.
No JAX state, no global mutation, no I/O. The convergence_check
function is the *only* termination criterion in optimise.py beyond the
hard step cap.

Convergence rule
----------------
Sliding-window relative-mean check: with window W and tolerance tol,
declare converged at step k (k >= 2W) if

    |mean(L[k-W : k])  -  mean(L[k-2W : k-W])|
    -------------------------------------------   <  tol
    max(|mean(L[k-2W : k-W])|,  abs_floor)

Defaults are W = 50 and tol = 1e-4. The denominator floor guards against
divide-by-near-zero when the loss decays toward zero.
"""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp


# =========================================================================
# CONVERGENCE CHECK
# =========================================================================

def convergence_check(
    loss_history: jnp.ndarray,
    window: int = 50,
    tol: float = 1e-4,
    abs_floor: float = 1e-8,
) -> bool:
    """Sliding-window convergence test.

    Args:
        loss_history: Loss values per Adam step, shape (n_steps,).
        window: Window size W; need >= 2W steps before this can fire.
        tol: Relative-mean-change tolerance.
        abs_floor: Floor on the denominator to avoid divide-by-zero.

    Returns:
        True if the relative change between the most-recent window and
        the previous window is below tol; False otherwise (including when
        there are not yet 2W steps).
    """
    n = int(loss_history.shape[0])
    if n < 2 * window:
        return False
    recent = jnp.mean(loss_history[n - window: n])
    prior = jnp.mean(loss_history[n - 2 * window: n - window])
    rel = float(jnp.abs(recent - prior) / jnp.maximum(jnp.abs(prior), abs_floor))
    return rel < tol


# =========================================================================
# TRACE SUMMARY
# =========================================================================

def summarise_trace(trace) -> Dict[str, float]:
    """Reduce an OptimisationTrace into a small scalar summary dict.

    Convenient for logging and for storing alongside outputs in a
    metadata JSON sibling file.

    Args:
        trace: An OptimisationTrace dataclass instance.

    Returns:
        Dictionary with keys:
            'final_total':     last loss value (NaN if trace is empty)
            'final_terminal':  last terminal-cost component
            'final_transport': last transport-cost component
            'final_reference': last reference-KL component
            'min_total':       best loss value seen (NaN if trace is empty)
            'final_grad_norm': last grad norm
            'n_steps_run':     steps actually executed
            'converged':       whether convergence fired
    """
    n = int(trace.n_steps_run)
    if n == 0:
        # Empty trace — return NaN scalars so callers don't crash.
        return {
            'final_total':     float('nan'),
            'final_terminal':  float('nan'),
            'final_transport': float('nan'),
            'final_reference': float('nan'),
            'min_total':       float('nan'),
            'final_grad_norm': float('nan'),
            'n_steps_run':     0,
            'converged':       bool(trace.converged),
        }
    return {
        'final_total':     float(trace.losses_total[-1]),
        'final_terminal':  float(trace.losses_terminal[-1]),
        'final_transport': float(trace.losses_transport[-1]),
        'final_reference': float(trace.losses_reference[-1]),
        'min_total':       float(jnp.min(trace.losses_total)),
        'final_grad_norm': float(trace.grad_norms[-1]),
        'n_steps_run':     n,
        'converged':       bool(trace.converged),
    }
