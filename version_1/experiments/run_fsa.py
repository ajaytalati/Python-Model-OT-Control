"""
experiments/run_fsa.py — End-to-end FSA-high-res optimisation script.
======================================================================
Date:    26 April 2026
Version: 1.0.0
Status:  Phase 6 — FSA-high-res adapter + same Phase-5 comparison flow.

Runs the OT-Control engine on a named FSA scenario, saves the
optimised schedule, simulates under the schedule and three naive
baselines (zero-control, constant-reference, linear-interpolation),
and produces the standard FSA figures plus a baseline-comparison CSV.

Known limitation
----------------
The FSA optimisation landscape exhibits the single-bandwidth-MMD
gradient-vanishing pathology (F2 in docs/Future_Features.md) for some
scenarios. The optimiser produces a valid schedule (bounds-respected,
finite, completes), but may end up in a clinically-counterintuitive
local minimum. Multi-bandwidth MMD is the principled fix and is
deferred to a follow-up release. For now, treat optimised FSA
schedules as candidate solutions to be sanity-checked manually rather
than as gold-standard recommendations.

Usage
-----
    python -m experiments.run_fsa --scenario unfit_recovery --horizon 14
    python -m experiments.run_fsa --scenario over_trained --steps 1500
    python -m experiments.run_fsa --scenario detrained_athlete

Outputs land in `outputs/fsa_<scenario>_h<horizon>_<timestamp>/`.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')      # non-interactive backend for headless runs
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
from adapters.fsa_high_res import (
    make_fsa_problem, list_scenarios, FSA_CONTROL_NAMES, A_STAR_HEALTHY,
)
from adapters.fsa_high_res.plots import (
    plot_schedule, plot_latent_paths,
    plot_terminal_amplitude, plot_loss_trace,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run FSA optimisation + verification.")
    p.add_argument('--scenario', type=str, default='unfit_recovery',
                    choices=list_scenarios(),
                    help="Clinical scenario to run.")
    p.add_argument('--horizon', type=int, default=14, help="Horizon D in days.")
    p.add_argument('--n-particles', type=int, default=128,
                    help="Optimiser inner-simulator particle count.")
    p.add_argument('--n-realisations', type=int, default=256,
                    help="Closed-loop verification particle count.")
    p.add_argument('--steps', type=int, default=2000,
                    help="Maximum Adam optimisation steps.")
    p.add_argument('--lr', type=float, default=5e-3, help="Adam learning rate.")
    p.add_argument('--dt', type=float, default=0.05, help="EM step in days.")
    p.add_argument('--alpha-terminal', type=float, default=1.0)
    p.add_argument('--alpha-transport', type=float, default=0.05)
    p.add_argument('--alpha-reference', type=float, default=0.001)
    p.add_argument('--reference-sigma', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output-dir', type=Path, default=None,
                    help="Output dir (auto-named if not provided).")
    p.add_argument('--no-plots', action='store_true',
                    help="Skip plot generation.")
    return p.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    ts = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    return Path('outputs') / f'fsa_{args.scenario}_h{args.horizon}_{ts}'


def _summarise_results(results, output_dir: Path) -> Path:
    """Write the comparison CSV and return its path."""
    csv_path = output_dir / 'comparison.csv'
    with csv_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['schedule_label', 'mean_amplitude_at_D',
                    'distance_to_A_star', 'fraction_in_healthy_basin',
                    'mmd_target'])
        for label, res in results.items():
            mean_A = float(jnp.mean(res.amplitude_at_D))
            d = abs(mean_A - A_STAR_HEALTHY)
            basin = float(res.fraction_in_healthy_basin)    # NaN if adapter has no basin_indicator_fn
            w.writerow([label, f"{mean_A:.4f}", f"{d:.4f}",
                          f"{basin:.4f}", f"{float(res.mmd_target):.4f}"])
    return csv_path


def main():
    args = parse_args()
    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"FSA scenario: {args.scenario}")
    print(f"Horizon:      {args.horizon} days")
    print(f"Output dir:   {output_dir}")

    problem = make_fsa_problem(
        scenario=args.scenario,
        horizon_days=args.horizon,
        n_particles=args.n_particles,
        dt_days=args.dt,
        optim_steps=args.steps,
        learning_rate=args.lr,
        alpha_terminal=args.alpha_terminal,
        alpha_transport=args.alpha_transport,
        alpha_reference=args.alpha_reference,
        reference_sigma=args.reference_sigma,
    )
    pol = PiecewiseConstant.from_problem(problem)

    print("\nOptimising schedule ...")
    rng_opt = jax.random.PRNGKey(args.seed)
    schedule, trace = optimise_schedule(problem, pol, rng_opt)
    summary = summarise_trace(trace)
    print(f"\nFinal loss:   {summary['final_total']:.4f}")
    print(f"  terminal:   {summary['final_terminal']:.4f}")
    print(f"  transport:  {summary['final_transport']:.4f}")
    print(f"  reference:  {summary['final_reference']:.4f}")
    print(f"  steps run:  {summary['n_steps_run']}")
    print(f"  converged:  {summary['converged']}")

    # Write schedule CSV
    sch_csv = output_dir / 'schedule.csv'
    with sch_csv.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['day'] + list(FSA_CONTROL_NAMES))
        daily = jnp.asarray(schedule.daily_values)
        for d in range(args.horizon):
            w.writerow([d] + [f"{float(daily[d, c]):.4f}"
                                for c in range(2)])

    # Compare against three naive baselines
    print("\nRunning closed-loop comparison ...")
    baselines = [
        schedule,
        zero_control_schedule(problem, pol),
        constant_reference_schedule(problem, pol),
        linear_interpolation_schedule(
            problem, pol,
            theta_target=jnp.array([0.5, 0.1])    # linearly to a moderately-healthy endpoint
        ),
    ]
    results = compare_schedules(
        problem, baselines, jax.random.PRNGKey(args.seed + 1),
        n_realisations=args.n_realisations,
    )
    print(f"\n  {'Schedule':28s}  {'mean A(D)':>10s}  {'distance':>10s}  "
          f"{'basin frac':>11s}  {'MMD':>8s}")
    for label, res in results.items():
        m = float(jnp.mean(res.amplitude_at_D))
        b = float(res.fraction_in_healthy_basin)    # NaN if no basin_indicator_fn
        print(f"  {label:28s}  {m:>10.3f}  {abs(m - A_STAR_HEALTHY):>10.3f}  "
              f"{b:>11.3f}  {float(res.mmd_target):>8.4f}")

    csv_path = _summarise_results(results, output_dir)
    print(f"\nWrote {csv_path}")

    # Optional plots
    if not args.no_plots:
        print("\nGenerating plots ...")
        ref = jnp.asarray(problem.reference_schedule)
        plot_schedule(schedule, reference_schedule=ref) \
            .savefig(output_dir / 'schedule.png', dpi=110, bbox_inches='tight')
        plt.close('all')

        opt_result = results['schedule_0']
        plot_latent_paths(opt_result.trajectories, opt_result.t,
                            n_show=12) \
            .savefig(output_dir / 'latent_paths.png', dpi=110, bbox_inches='tight')
        plot_terminal_amplitude(
            opt_result.amplitude_at_D,
            target_samples=problem.sample_target_amplitude(
                jax.random.PRNGKey(args.seed + 99), 500
            ),
        ).savefig(output_dir / 'terminal_amplitude.png',
                    dpi=110, bbox_inches='tight')
        plot_loss_trace(trace) \
            .savefig(output_dir / 'loss_trace.png', dpi=110, bbox_inches='tight')
        plt.close('all')

    summary_path = output_dir / 'optim_summary.json'
    with summary_path.open('w') as f:
        json.dump({
            'scenario': args.scenario,
            'horizon': args.horizon,
            'A_star_healthy': A_STAR_HEALTHY,
            **summary,
        }, f, indent=2)
    print(f"\nDone. Outputs in {output_dir}")


if __name__ == '__main__':
    main()
