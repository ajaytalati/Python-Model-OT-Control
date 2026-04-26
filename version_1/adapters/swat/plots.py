"""
adapters/swat/plots.py — SWAT-specific plotting.
==================================================
Date:    26 April 2026
Version: 1.1.0

Four standard figures for SWAT optimisation runs:

    plot_schedule(schedule, reference_schedule=None, ax=None)
        Daily V_h, V_n, V_c bars over the schedule horizon.

    plot_latent_paths(trajectories, t_grid, n_show=10)
        Sample latent trajectories — W (top), Z (mid), T (bottom) overlaid
        across three stacked subplots. Always creates its own figure.

    plot_terminal_amplitude(amplitude_at_D, target_samples, ax=None)
        Histogram of simulated T(D) versus the target distribution.

    plot_loss_trace(trace, ax=None)
        Total / terminal / transport / reference loss vs. step.

Each single-axes plotting function accepts an optional matplotlib Axes;
if None, a new figure is created. plot_latent_paths is the exception —
it always builds its own 3-panel figure. All four functions return the
matplotlib `Figure`.

These functions use only matplotlib + numpy. They convert JAX arrays
to numpy at the boundary.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ot_engine.types import OptimisationTrace, Schedule


def _to_numpy(x):
    """Convert a JAX array (or any array-like) to a NumPy array."""
    return np.asarray(x)


# =========================================================================
# Schedule
# =========================================================================

def plot_schedule(schedule: Schedule,
                   reference_schedule: Optional[np.ndarray] = None,
                   ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Plot the daily V_h, V_n, V_c values from an optimised SWAT schedule.

    Args:
        schedule: Schedule dataclass from the engine.
        reference_schedule: Optional baseline to overlay as dashed lines.
        ax: Existing Axes to draw into; if None, a new Figure is created.

    Returns:
        The Figure containing the schedule plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
    daily = _to_numpy(schedule.daily_values)
    days = np.arange(daily.shape[0])
    width = 0.27
    labels = ('V_h', 'V_n', 'V_c (h)')
    colours = ('#1b9e77', '#d95f02', '#7570b3')
    for i, (label, c) in enumerate(zip(labels, colours)):
        ax.bar(days + (i - 1) * width, daily[:, i], width=width,
               label=label, color=c, alpha=0.85)
        if reference_schedule is not None:
            ref = _to_numpy(reference_schedule)
            ax.plot(days + (i - 1) * width, ref[:, i], color=c,
                    linestyle='--', alpha=0.6, marker='_', markersize=12)
    ax.set_xlabel('Day')
    ax.set_ylabel('Control value')
    ax.set_title('SWAT optimised control schedule')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# =========================================================================
# Sample latent paths
# =========================================================================

def plot_latent_paths(trajectories: np.ndarray, t_grid: np.ndarray,
                       n_show: int = 10,
                       ) -> plt.Figure:
    """Overlay W, Z, T sample paths from the simulator output.

    This function always builds its own 3-panel figure (one row each
    for W, Z, T) — unlike the other plotters it does not accept an
    external Axes.

    Args:
        trajectories: Shape (n_particles, n_steps + 1, 4) — full state.
        t_grid: Shape (n_steps + 1,) in days.
        n_show: Number of sample paths to plot per panel.

    Returns:
        Figure with three stacked subplots (W, Z, T).
    """
    traj = _to_numpy(trajectories)
    t = _to_numpy(t_grid)
    n_show = min(n_show, traj.shape[0])
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)

    # W
    for i in range(n_show):
        axes[0].plot(t, traj[i, :, 0], alpha=0.4, lw=0.8)
    axes[0].plot(t, np.mean(traj[:, :, 0], axis=0), color='black', lw=2,
                  label='mean')
    axes[0].set_ylabel('W (wakefulness)')
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc='upper right', fontsize=9)

    # Z
    for i in range(n_show):
        axes[1].plot(t, traj[i, :, 1], alpha=0.4, lw=0.8)
    axes[1].plot(t, np.mean(traj[:, :, 1], axis=0), color='black', lw=2)
    axes[1].set_ylabel(r'$\tilde Z$ (sleep depth)')
    axes[1].set_ylim(-0.5, 6.5)
    axes[1].grid(alpha=0.3)

    # T
    for i in range(n_show):
        axes[2].plot(t, traj[i, :, 3], alpha=0.4, lw=0.8)
    axes[2].plot(t, np.mean(traj[:, :, 3], axis=0), color='black', lw=2)
    axes[2].axhline(0.55, color='green', linestyle=':', lw=1.5,
                     label='T*  (healthy)')
    axes[2].set_ylabel('T (testosterone amplitude)')
    axes[2].set_xlabel('Time (days)')
    axes[2].set_ylim(-0.05, 1.0)
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc='lower right', fontsize=9)

    fig.suptitle('SWAT latent state trajectories under optimised schedule')
    fig.tight_layout()
    return fig


# =========================================================================
# Terminal amplitude vs target
# =========================================================================

def plot_terminal_amplitude(amplitude_at_D: np.ndarray,
                              target_samples: np.ndarray,
                              ax: Optional[plt.Axes] = None
                              ) -> plt.Figure:
    """Histogram of simulated T(D) overlaid with target distribution samples.

    Args:
        amplitude_at_D: Shape (n_particles,).
        target_samples: Shape (n_target,).
        ax: Existing Axes to draw into.

    Returns:
        The Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.figure
    A = _to_numpy(amplitude_at_D)
    Tgt = _to_numpy(target_samples)
    bins = np.linspace(0.0, 1.0, 30)
    ax.hist(A, bins=bins, density=True, alpha=0.6, color='#7570b3',
            label='Simulated T(D)', edgecolor='white')
    ax.hist(Tgt, bins=bins, density=True, alpha=0.4, color='#1b9e77',
            label='Target T*', edgecolor='white')
    ax.set_xlabel('Terminal testosterone amplitude T(D)')
    ax.set_ylabel('Density')
    ax.set_title('Terminal amplitude — simulated vs. target')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# =========================================================================
# Loss trace
# =========================================================================

def plot_loss_trace(trace: OptimisationTrace,
                     ax: Optional[plt.Axes] = None) -> plt.Figure:
    """Plot the four-loss trace from an OptimisationTrace.

    Args:
        trace: OptimisationTrace from the engine.
        ax: Existing Axes; if None, makes a new Figure.

    Returns:
        The Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure
    steps = np.arange(int(trace.n_steps_run))
    ax.plot(steps, _to_numpy(trace.losses_total), label='total', lw=2)
    ax.plot(steps, _to_numpy(trace.losses_terminal),
            label=r'$\alpha_1\,$MMD$^2$', alpha=0.7)
    ax.plot(steps, _to_numpy(trace.losses_transport),
            label=r'$\alpha_2\,\int\,||u||^2$', alpha=0.7)
    ax.plot(steps, _to_numpy(trace.losses_reference),
            label=r'$\alpha_3\,$KL ref', alpha=0.7)
    ax.set_xlabel('Adam step')
    ax.set_ylabel('Loss')
    ax.set_title('Optimisation trace')
    ax.set_yscale('symlog', linthresh=1e-3)
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig
