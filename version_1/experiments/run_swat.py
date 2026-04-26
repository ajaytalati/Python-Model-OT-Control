"""
experiments/run_swat.py — End-to-end SWAT optimisation script.
================================================================
Date:    26 April 2026
Version: 1.1.0
Status:  Phase 5 — now compares against naive baselines.

Runs the OT-Control engine on a named SWAT scenario, saves the
optimised schedule, simulates under the schedule and three naive
baselines (zero-control, constant-reference, linear-interpolation),
and produces the standard SWAT figures plus a baseline-comparison CSV.

Usage
-----
    python -m experiments.run_swat --scenario insomnia --horizon 21
    python -m experiments.run_swat --scenario recovery --horizon 14 --steps 800
    python -m experiments.run_swat --scenario shift_work

Outputs land in `outputs/swat_<scenario>_h<horizon>_<timestamp>/`.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import csv
import datetime as _dt
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for headless runs
import matplotlib.pyplot as plt

from ot_engine import (
    compare_schedules,
    constant_reference_schedule,
    linear_interpolation_schedule,
    optimise_schedule,
    PiecewiseConstant,
    summarise_trace,
    zero_control_schedule,
)
from adapters.swat import (
    make_swat_problem, list_scenarios, SWAT_CONTROL_NAMES, T_STAR_HEALTHY,
)
from adapters.swat.plots import (
    plot_schedule, plot_latent_paths, plot_terminal_amplitude, plot_loss_trace,
)


def _parse_args():
    p = argparse.ArgumentParser(description="SWAT OT-Control end-to-end")
    p.add_argument('--scenario', default='insomnia',
                   choices=list_scenarios(),
                   help='SWAT clinical scenario')
    p.add_argument('--horizon', type=int, default=21,
                   help='Schedule length in days')
    p.add_argument('--steps', type=int, default=800,
                   help='Maximum Adam optimisation steps')
    p.add_argument('--n-particles', type=int, default=128,
                   help='Monte-Carlo particle count for optimisation')
    p.add_argument('--n-verify', type=int, default=256,
                   help='Particle count for closed-loop verification')
    p.add_argument('--lr', type=float, default=5e-2,
                   help='Adam learning rate')
    p.add_argument('--seed', type=int, default=0,
                   help='JAX PRNG seed')
    p.add_argument('--output-dir', default='outputs',
                   help='Directory under which a per-run folder is created')
    return p.parse_args()


def _write_comparison_csv(results: dict, csv_path: Path):
    """Write the baseline-comparison table to CSV."""
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'schedule_label', 'mean_T_at_D', 'std_T_at_D',
            'mmd_to_target', 'basin_fraction',
            'distance_to_T_star',
        ])
        writer.writeheader()
        for label, res in results.items():
            mean_T = float(jnp.mean(res.amplitude_at_D))
            writer.writerow({
                'schedule_label':       label,
                'mean_T_at_D':          mean_T,
                'std_T_at_D':           float(jnp.std(res.amplitude_at_D)),
                'mmd_to_target':        float(res.mmd_target),
                'basin_fraction':       float(res.fraction_in_healthy_basin),
                'distance_to_T_star':   abs(mean_T - T_STAR_HEALTHY),
            })


def main():
    args = _parse_args()

    timestamp = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = (Path(args.output_dir)
               / f'swat_{args.scenario}_h{args.horizon}_{timestamp}')
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== SWAT OT-Control run ===")
    print(f"  Scenario:   {args.scenario}")
    print(f"  Horizon:    {args.horizon} days")
    print(f"  Steps:      {args.steps}")
    print(f"  Output:     {run_dir}")
    print()

    # 1) Build the problem.
    problem = make_swat_problem(
        scenario=args.scenario,
        horizon_days=args.horizon,
        n_particles=args.n_particles,
        optim_steps=args.steps,
        learning_rate=args.lr,
    )
    pol = PiecewiseConstant(horizon_days=args.horizon, n_controls=3)

    # 2) Run the optimiser.
    rng = jax.random.PRNGKey(args.seed)
    rng_opt, rng_eval = jax.random.split(rng, 2)
    print("Optimising schedule ...")
    optimised, trace = optimise_schedule(problem, pol, rng_opt, verbose=True)
    summary = summarise_trace(trace)
    print()
    print(f"Final loss:       {summary['final_total']:.4f}")
    print(f"  terminal:       {summary['final_terminal']:.4f}")
    print(f"  transport:      {summary['final_transport']:.4f}")
    print(f"  reference:      {summary['final_reference']:.4f}")
    print(f"Steps run:        {summary['n_steps_run']}")
    print(f"Converged:        {summary['converged']}")

    # 3) Build naive baselines.
    print("\nBuilding naive baselines ...")
    sch_zero = zero_control_schedule(problem, pol)
    sch_const = constant_reference_schedule(problem, pol)
    sch_lin = linear_interpolation_schedule(
        problem, pol, jnp.array([1.0, 0.3, 0.0])  # clinically-ideal terminal
    )
    # Tag the optimised schedule for compare_schedules' label dict.
    from dataclasses import replace
    optimised_labeled = replace(
        optimised,
        metadata={**optimised.metadata, 'label': 'optimised'},
    )
    schedules = [sch_zero, sch_const, sch_lin, optimised_labeled]

    # 4) Closed-loop comparison.
    print("\nRunning closed-loop comparison ...")
    results = compare_schedules(
        problem, schedules, rng_eval,
        n_realisations=args.n_verify,
    )
    print(f"\n{'Schedule':25s}  {'mean T(D)':10s}  {'distance':10s}  "
          f"{'basin frac':12s}  {'MMD':10s}")
    for label, res in results.items():
        m = float(jnp.mean(res.amplitude_at_D))
        print(f"  {label:23s}  {m:10.3f}  "
              f"{abs(m - T_STAR_HEALTHY):10.3f}  "
              f"{res.fraction_in_healthy_basin:12.3f}  "
              f"{res.mmd_target:10.4f}")

    # 5) Save artifacts.
    print("\nSaving outputs ...")
    daily = jnp.asarray(optimised.daily_values)
    metadata = {
        'scenario': args.scenario,
        'horizon_days': args.horizon,
        'control_names': list(SWAT_CONTROL_NAMES),
        'reference_schedule': problem.reference_schedule.tolist(),
        'optimised_daily_values': daily.tolist(),
        'target_T_star': T_STAR_HEALTHY,
        'optim_summary': summary,
        'seed': args.seed,
    }
    (run_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    jnp.save(run_dir / 'theta.npy', optimised.theta)
    _write_comparison_csv(results, run_dir / 'baseline_comparison.csv')

    # Trajectory + amplitudes from the OPTIMISED closed-loop run, for
    # plotting.
    opt_result = results['optimised']
    jnp.save(run_dir / 'trajectories.npy', opt_result.trajectories)
    jnp.save(run_dir / 'amplitude_at_D.npy', opt_result.amplitude_at_D)

    # 6) Figures.
    fig1 = plot_schedule(optimised,
                          reference_schedule=problem.reference_schedule)
    fig1.savefig(run_dir / 'schedule.png', dpi=120)
    fig2 = plot_latent_paths(opt_result.trajectories, opt_result.t)
    fig2.savefig(run_dir / 'latent_paths.png', dpi=120)
    fig3 = plot_terminal_amplitude(
        opt_result.amplitude_at_D, opt_result.target_samples
    )
    fig3.savefig(run_dir / 'terminal_amplitude.png', dpi=120)
    fig4 = plot_loss_trace(trace)
    fig4.savefig(run_dir / 'loss_trace.png', dpi=120)
    plt.close('all')

    print(f"Done. Outputs in: {run_dir}")


if __name__ == '__main__':
    main()
