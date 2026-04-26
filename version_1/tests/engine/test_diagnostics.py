"""
tests/engine/test_diagnostics.py — Convergence + trace summary helpers.
========================================================================
Phase 3 unit tests for ot_engine.diagnostics.

Covers convergence_check on synthetic loss histories and summarise_trace
on a hand-built OptimisationTrace.
"""

from __future__ import annotations

import jax.numpy as jnp

from ot_engine import (
    OptimisationTrace,
    convergence_check,
    summarise_trace,
)


def test_convergence_false_when_history_short():
    """Need at least 2*window steps before any check fires."""
    hist = jnp.zeros(50)  # window default 50, so need 100
    assert convergence_check(hist, window=50, tol=1e-4) is False


def test_convergence_true_for_constant_history():
    """Constant loss history -> 0 relative change -> converged."""
    hist = jnp.ones(200) * 0.5
    assert convergence_check(hist, window=50, tol=1e-4) is True


def test_convergence_false_for_strictly_decreasing_history():
    """A history still actively decreasing should not be flagged converged."""
    # Linear decrease from 1.0 to 0.1 over 200 steps.
    hist = jnp.linspace(1.0, 0.1, 200)
    # Means of last 50 vs prior 50 differ by ~0.225 — well above tol.
    assert convergence_check(hist, window=50, tol=1e-3) is False


def test_convergence_true_after_loss_flattens():
    """Decreasing then plateau -> converges once the plateau dominates."""
    decreasing = jnp.linspace(1.0, 0.1, 100)
    plateau = jnp.full(100, 0.1) + 1e-6 * jnp.arange(100)
    hist = jnp.concatenate([decreasing, plateau])
    assert convergence_check(hist, window=50, tol=1e-3) is True


def test_summarise_trace_returns_expected_keys():
    n = 10
    trace = OptimisationTrace(
        losses_total=jnp.linspace(1.0, 0.1, n),
        losses_terminal=jnp.linspace(0.5, 0.05, n),
        losses_transport=jnp.linspace(0.3, 0.03, n),
        losses_reference=jnp.linspace(0.2, 0.02, n),
        grad_norms=jnp.linspace(2.0, 0.1, n),
        converged=True,
        n_steps_run=n,
    )
    summary = summarise_trace(trace)
    expected = {
        'final_total', 'final_terminal', 'final_transport',
        'final_reference', 'min_total', 'final_grad_norm',
        'n_steps_run', 'converged',
    }
    assert set(summary.keys()) == expected
    assert summary['final_total'] == 0.1
    assert summary['min_total'] == 0.1
    assert summary['n_steps_run'] == n
    assert summary['converged'] is True


def test_summarise_trace_handles_empty_trace():
    """Review fix C-2: empty trace must not crash summarise_trace."""
    import math
    trace = OptimisationTrace(
        losses_total=jnp.zeros(0),
        losses_terminal=jnp.zeros(0),
        losses_transport=jnp.zeros(0),
        losses_reference=jnp.zeros(0),
        grad_norms=jnp.zeros(0),
        converged=False,
        n_steps_run=0,
    )
    summary = summarise_trace(trace)
    assert summary['n_steps_run'] == 0
    assert summary['converged'] is False
    assert math.isnan(summary['final_total'])
    assert math.isnan(summary['min_total'])
    assert math.isnan(summary['final_grad_norm'])
