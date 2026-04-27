"""
adapters/fsa_high_res/plots.py — FSA-specific plot helpers.
============================================================
Date:    26 April 2026
Version: 1.0.0

Four standard figures, mirroring SWAT's plot module:

  1. plot_schedule       — daily (T_B, Phi) bars + reference line.
  2. plot_latent_paths   — sample (B, F, A) trajectories vs t.
  3. plot_terminal_amplitude — A(D) histogram + target overlay.
  4. plot_loss_trace     — total + per-component loss vs step.

All functions return matplotlib Figures the caller can save or display.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ot_engine.types import Schedule, OptimisationTrace
from adapters.fsa_high_res.adapter import FSA_CONTROL_NAMES, A_STAR_HEALTHY


# Visual conventions: colour per control / state component.
_T_B_COLOUR = '#1f77b4'         # blue
_PHI_COLOUR = '#ff7f0e'         # orange

_B_COLOUR = '#2ca02c'           # green   (fitness)
_F_COLOUR = '#d62728'           # red     (strain)
_A_COLOUR = '#9467bd'           # purple  (amplitude)


def plot_schedule(schedule: Schedule,
                   reference_schedule: Optional[np.ndarray] = None,
                   ) -> plt.Figure:
    """Bar chart of daily (T_B, Phi) values, with optional reference line.

    Args:
        schedule: Optimised Schedule from `optimise_schedule`.
        reference_schedule: Optional (D, 2) array of reference values
            shown as a dashed horizontal line per control.

    Returns:
        matplotlib Figure with two subplots stacked vertically.
    """
    daily = np.asarray(schedule.daily_values)
    D = daily.shape[0]
    days = np.arange(D)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].bar(days, daily[:, 0], color=_T_B_COLOUR, alpha=0.8, label='T_B')
    if reference_schedule is not None:
        axes[0].axhline(reference_schedule[0, 0], color=_T_B_COLOUR,
                          linestyle='--', alpha=0.5, label='reference')
    axes[0].set_ylabel('T_B (training-load target)')
    axes[0].legend(loc='upper right')
    axes[0].grid(alpha=0.3)

    axes[1].bar(days, daily[:, 1], color=_PHI_COLOUR, alpha=0.8, label='Phi')
    if reference_schedule is not None:
        axes[1].axhline(reference_schedule[0, 1], color=_PHI_COLOUR,
                          linestyle='--', alpha=0.5, label='reference')
    axes[1].set_ylabel('Phi (training intensity)')
    axes[1].set_xlabel('Day')
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.3)

    fig.suptitle("FSA optimised control schedule")
    fig.tight_layout()
    return fig


def plot_latent_paths(trajectories: jnp.ndarray, t_grid: jnp.ndarray,
                       n_show: int = 10) -> plt.Figure:
    """Sample latent trajectories of (B, F, A) over the horizon.

    Args:
        trajectories: shape (n_particles, n_steps + 1, 3).
        t_grid: shape (n_steps + 1,) in days.
        n_show: number of sample paths to draw (selected from the front
            of the array; particles are exchangeable).

    Returns:
        matplotlib Figure with three subplots.
    """
    traj = np.asarray(trajectories)
    t = np.asarray(t_grid)
    n = min(n_show, traj.shape[0])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    for i in range(n):
        axes[0].plot(t, traj[i, :, 0], color=_B_COLOUR, alpha=0.4, lw=0.8)
        axes[1].plot(t, traj[i, :, 1], color=_F_COLOUR, alpha=0.4, lw=0.8)
        axes[2].plot(t, traj[i, :, 2], color=_A_COLOUR, alpha=0.4, lw=0.8)

    # Mean overlay
    axes[0].plot(t, traj[:, :, 0].mean(axis=0), color=_B_COLOUR, lw=2.0, label='mean')
    axes[1].plot(t, traj[:, :, 1].mean(axis=0), color=_F_COLOUR, lw=2.0, label='mean')
    axes[2].plot(t, traj[:, :, 2].mean(axis=0), color=_A_COLOUR, lw=2.0, label='mean')

    axes[0].set_ylabel('B (fitness)')
    axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel('F (strain)')
    axes[2].set_ylabel('A (amplitude)')
    axes[2].axhline(A_STAR_HEALTHY, color='grey', linestyle='--',
                     alpha=0.6, label=f'A* (healthy)')
    axes[2].set_xlabel('t (days)')

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)

    fig.suptitle(f"Latent trajectories under optimised schedule  (n={n} particles shown)")
    fig.tight_layout()
    return fig


def plot_terminal_amplitude(amplitude_at_D: jnp.ndarray,
                              target_samples: Optional[jnp.ndarray] = None,
                              ) -> plt.Figure:
    """Histogram of simulated terminal A vs target distribution.

    Args:
        amplitude_at_D: (n_realisations,) simulated A at terminal time.
        target_samples: (n_target,) samples from the clinical target.

    Returns:
        matplotlib Figure.
    """
    A_D = np.asarray(amplitude_at_D)
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(A_D, bins=40, density=True, alpha=0.55, color=_A_COLOUR,
              edgecolor='black', linewidth=0.5, label='simulated A(D)')

    if target_samples is not None:
        target = np.asarray(target_samples)
        ax.hist(target, bins=40, density=True, alpha=0.4, color='grey',
                  edgecolor='black', linewidth=0.5, label='target distribution')

    ax.axvline(A_STAR_HEALTHY, color='black', linestyle='--', alpha=0.7,
                 label=f'A* = {A_STAR_HEALTHY}')
    ax.set_xlabel('A (endocrine amplitude at t = D)')
    ax.set_ylabel('density')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.suptitle("Terminal-amplitude marginal vs clinical target")
    fig.tight_layout()
    return fig


def plot_loss_trace(trace: OptimisationTrace) -> plt.Figure:
    """Total + per-component loss vs Adam step.

    Args:
        trace: OptimisationTrace from `optimise_schedule`.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    losses_total = np.asarray(trace.losses_total)
    losses_terminal = np.asarray(trace.losses_terminal)
    losses_transport = np.asarray(trace.losses_transport)
    losses_reference = np.asarray(trace.losses_reference)
    n = len(losses_total)
    steps = np.arange(n)

    ax.plot(steps, losses_total, color='black', lw=1.8, label='total')
    ax.plot(steps, losses_terminal, color='#1f77b4', lw=1.0, alpha=0.8,
              label='terminal MMD')
    ax.plot(steps, losses_transport, color='#2ca02c', lw=1.0, alpha=0.8,
              label='transport (1/2 |u|^2)')
    ax.plot(steps, losses_reference, color='#d62728', lw=1.0, alpha=0.8,
              label='reference (Gaussian iid)')

    ax.set_xlabel('Adam step')
    ax.set_ylabel('loss')
    ax.set_yscale('symlog', linthresh=0.01)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.suptitle(f"Optimisation loss trace  ({n} steps)")
    fig.tight_layout()
    return fig
