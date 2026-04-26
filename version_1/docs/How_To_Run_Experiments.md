# How to Run Experiments — Python-Model-OT-Control

**Version**: 1.1.0
**Date**: 26 April 2026

A pragmatic guide: install, run a SWAT optimisation, generate the
plots, run the test suite, and write a new experiment.

---

## 1. Install

Tested on Python 3.12 with JAX (CPU build). Float64 mode is enabled by
the engine on import.

```bash
pip install --upgrade jax jaxlib optax matplotlib numpy pytest
```

The engine has no native build step; clone the repo and add
`Python-Model-OT-Control/version_1` to `PYTHONPATH`:

```bash
git clone <this repo>
cd Python-Model-OT-Control/version_1
export PYTHONPATH=$(pwd)
```

Verify the installation:

```bash
python -c "import ot_engine; print('OK', ot_engine.__version__)"
# OK 1.1.0
```

---

## 2. Run the test suite

```bash
cd Python-Model-OT-Control/version_1
pytest tests/ -q
# 87 passed in ~200s
```

The suite covers:

* Engine units (types, policies, MMD, KL, simulator, loss, optimise,
  diagnostics, closed_loop, compare, pipeline).
* SWAT adapter (drift/diffusion/clipping smoke tests, end-to-end
  pipeline test).
* Phase-5 acceptance test (optimised schedule beats all three baselines
  on all metrics).

If anything fails, re-read the error before editing — almost all
failures are stale environments or path issues, not code regressions.

---

## 3. Run the SWAT runner — the standard experiment

The CLI lives at `experiments/run_swat.py`. Three scenarios are built
in: `recovery`, `insomnia`, `shift_work`.

```bash
python experiments/run_swat.py \
  --scenario recovery \
  --horizon 14 \
  --output-dir runs/recovery_001
```

Default options (override on the CLI):

| Flag | Default | Meaning |
|:---|:---:|:---|
| `--scenario` | insomnia | one of recovery / insomnia / shift_work |
| `--horizon` | 21 | schedule length in days |
| `--steps` | 800 | maximum Adam optimisation steps |
| `--n-particles` | 128 | Monte-Carlo particle count for optimisation |
| `--n-verify` | 256 | particle count for closed-loop verification |
| `--lr` | 5e-2 | Adam learning rate |
| `--seed` | 0 | JAX PRNG seed |
| `--output-dir` | outputs | parent directory; a per-run subdir is created beneath it |

Artifacts land in `<output-dir>/swat_<scenario>_h<horizon>_<timestamp>/`,
not directly in `--output-dir`. Each per-run directory contains:

```
metadata.json              # scenario, horizon, control names, reference schedule,
                           # optimised daily values, target T*, optim summary, seed
theta.npy                  # raw policy parameters
trajectories.npy           # closed-loop W/Z/T trajectories under the optimised schedule
amplitude_at_D.npy         # terminal amplitude samples
baseline_comparison.csv    # metrics for optimised + 3 baselines
schedule.png               # bar chart of the optimised schedule
latent_paths.png           # W, Z, T sample trajectories
terminal_amplitude.png     # histogram of T(D) vs target
loss_trace.png             # total / per-component loss vs step
```

---

## 4. Run from Python (programmatic)

For more flexibility — custom targets, custom reference schedules,
non-default scenarios — call the API directly. Minimal example:

```python
import jax
from ot_engine import (
    PiecewiseConstant, optimise_schedule,
    simulate_closed_loop, compare_schedules,
    zero_control_schedule, constant_reference_schedule,
    linear_interpolation_schedule,
)
from adapters.swat import make_swat_problem

# 1) Build the problem.
problem = make_swat_problem(
    scenario='recovery',
    horizon_days=14,
    n_particles=256,
    dt_days=0.05,
    optim_steps=2000,
    learning_rate=1e-2,
)

# 2) Build the policy.
pol = PiecewiseConstant(horizon_days=14, n_controls=3)

# 3) Optimise.
rng = jax.random.PRNGKey(0)
schedule, trace = optimise_schedule(problem, pol, rng,
                                       convergence_window=50,
                                       convergence_tol=1e-4)
print(f"final loss: {trace.losses_total[-1]:.4f}")
print(f"converged after {trace.n_steps_run} steps")

# 4) Closed-loop verification.
result = simulate_closed_loop(problem, pol, schedule,
                                jax.random.PRNGKey(1),
                                n_realisations=512)
print(f"MMD to target:    {result.mmd_target:.4f}")
print(f"basin fraction:   {result.fraction_in_healthy_basin:.3f}")

# 5) Baseline comparison.
import jax.numpy as jnp
baselines = [
    zero_control_schedule(problem, pol),
    constant_reference_schedule(problem, pol),
    linear_interpolation_schedule(problem, pol,
                                    end_value=jnp.array([1.0, 0.3, 0.0])),
]
results = compare_schedules(problem, [schedule] + baselines,
                              jax.random.PRNGKey(2),
                              n_realisations=512)
for label, res in results.items():
    print(f"{label:30s}  basin={res.fraction_in_healthy_basin:.3f}")
```

---

## 5. Use the built-in pipeline helper

If the only thing you want is "optimise + verify + compare + dump
artefacts", use `run_ot_pipeline`:

```python
from ot_engine import run_ot_pipeline

schedule, trace, closed_loop = run_ot_pipeline(
    problem, pol,
    rng=jax.random.PRNGKey(0),
    n_realisations=512,
    optimise_kwargs={'convergence_window': 50, 'convergence_tol': 1e-4},
)
```

This runs the same steps as §4 in a single call. Adapter-specific
plotting is up to the caller — the SWAT plot helpers live in
`adapters/swat/plots.py` and accept `Schedule` and `OptimisationTrace`
directly.

---

## 6. Generate plots manually

If you want to render the four standard figures from a saved schedule:

```python
import matplotlib.pyplot as plt
from adapters.swat.plots import (
    plot_schedule, plot_latent_paths,
    plot_terminal_amplitude, plot_loss_trace,
)

# schedule, trace, closed_loop are the outputs from §4 or §5

plot_schedule(schedule,
                reference_schedule=problem.reference_schedule).savefig('schedule.png')
plot_latent_paths(closed_loop.trajectories, closed_loop.t,
                    n_show=10).savefig('latent_paths.png')
plot_terminal_amplitude(closed_loop.amplitude_at_D,
                          closed_loop.target_samples).savefig('terminal.png')
plot_loss_trace(trace).savefig('loss_trace.png')
plt.close('all')
```

---

## 7. Sweeping hyperparameters

To run a grid over $\alpha_2$ (transport weight) and look at how the
schedule's aggressiveness varies:

```python
import numpy as np
import csv

results = []
for alpha2 in [0.01, 0.05, 0.1, 0.5, 1.0]:
    p = make_swat_problem('recovery', horizon_days=14,
                            n_particles=256, dt_days=0.05,
                            optim_steps=1000, alpha_transport=alpha2)
    sch, tr = optimise_schedule(p, pol, jax.random.PRNGKey(0))
    cl = simulate_closed_loop(p, pol, sch, jax.random.PRNGKey(1),
                                n_realisations=256)
    results.append({
        'alpha_transport': alpha2,
        'final_loss': float(tr.losses_total[-1]),
        'mmd_target': cl.mmd_target,
        'basin_frac': cl.fraction_in_healthy_basin,
        'theta_norm': float(np.linalg.norm(sch.theta - p.reference_schedule)),
    })

with open('alpha2_sweep.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)
```

Higher $\alpha_2$ → schedule stays closer to baseline. Lower $\alpha_2$
→ more aggressive intervention. The right value is application- and
clinician-dependent.

---

## 8. Reproducibility

Same `--seed` produces bit-identical results (JAX float64 + PRNGKey
chain is deterministic). To verify:

```bash
python experiments/run_swat.py --scenario recovery --seed 42 --output_dir /tmp/run1
python experiments/run_swat.py --scenario recovery --seed 42 --output_dir /tmp/run2
diff /tmp/run1/schedule.csv /tmp/run2/schedule.csv
# (no output — identical)
```

If the diff is non-empty, something has perturbed JAX's PRNG state
(e.g. accidental `jnp.zeros` of unspecified dtype before x64 was
enabled). The engine's `__init__.py` enables x64 first thing on
import, so this should not happen unless the user pre-imports JAX.

---

## 9. Performance notes

CPU JAX, default settings, on a modern laptop:

| Stage | Wall-clock |
|:---|:---:|
| First-time import + JIT compile | ~5s |
| One Adam step (after compile) | ~10ms |
| 2000 Adam steps | ~20s |
| Closed-loop verification (256 particles, 14 days) | ~3s |
| Full SWAT pipeline (optimise + verify + compare) | ~40s |

If too slow:

* Reduce `n_particles` to 128 (gradient variance roughly $\propto 1/N$;
  64 starts to be noisy).
* Reduce `dt_days` resolution from 0.05 to 0.1 days (SWAT remains
  stable; absolute amplitude values shift slightly).
* Tighten convergence: `convergence_tol=1e-3` and the optimiser
  typically stops in ~500 steps.

GPU JAX is supported by the engine but untested in v1.1; expect 5-10×
speedup on the simulator inner loop, no speedup on the Adam state
manipulation.

---

## 10. Writing a new experiment

The minimum scaffold for a new experiment is:

```python
# experiments/run_my_experiment.py
import argparse, jax
from ot_engine import (PiecewiseConstant, run_ot_pipeline)
from adapters.swat import make_swat_problem

def main():
    p = make_swat_problem(
        scenario='recovery',
        horizon_days=14,
        # ... whatever you want to vary
    )
    pol = PiecewiseConstant(p.horizon_days, p.n_controls)
    sch, tr, cl = run_ot_pipeline(p, pol, jax.random.PRNGKey(0))
    # Save / plot / analyse
    ...

if __name__ == '__main__':
    main()
```

Run it:

```bash
PYTHONPATH=. python experiments/run_my_experiment.py
```

That's it. Everything else is optional polish.

---

## 11. Adding a new model adapter

See `how_to_add_a_new_model_adapter/` (in the repo root). The
two-paragraph version:

1. Vendor the model's JAX drift / diffusion / parameters into
   `_vendored_models/<your_model>/`.
2. Write `adapters/<your_model>/adapter.py` that exposes a
   `make_<model>_problem(...)` function returning a `BridgeProblem`.
   Implement `sample_initial_state`, `sample_target_amplitude`,
   `amplitude_of`, `state_clip_fn` (optional), `basin_indicator_fn`
   (optional).

The engine's contract — `BridgeProblem` — is the only API. The same
`run_ot_pipeline` works for any adapter that satisfies the contract.

---

## 12. Where to ask questions

* Engine bugs: open an issue against this repo.
* SWAT model questions: open against
  `Python-Model-Development-Simulation`.
* Mathematical questions: see `docs/Mathematical_Specification.md`
  and the references in `SWAT_Basic_Documentation.md`.
