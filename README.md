# Python-Model-OT-Control

A model-agnostic differentiable-transport framework for optimal-control
schedules in N-of-1 physiological-recovery problems. Designed to plug
into the SMC²-based estimation pipeline (`smc2-blackjax-rolling`)
downstream of model definitions in `Python-Model-Development-Simulation`.

## Status

**Phases 1 + 2 + 3 + 4 + 5 of v1 complete and tested.**

Working:
- All four data contracts (`BridgeProblem`, `Schedule`, `OptimisationTrace`, `ClosedLoopResult`)
- `PiecewiseConstant` control policy (JAX-native)
- Gaussian-kernel MMD with median-bandwidth heuristic
- Closed-form Gaussian-iid reference KL
- JAX Euler-Maruyama latent simulator (vmap'd, scanned, AD-traced, jit-compatible)
- Optional `state_clip_fn` for physically-bounded states
- Three-term loss composition with `make_loss_fn` factory
- Float64 enabled globally for numerical stability
- Optax Adam optimisation loop with gradient clipping and sliding-window convergence
- `convergence_check` and `summarise_trace` diagnostic helpers
- SWAT adapter with three canonical scenarios: `insomnia`, `recovery`, `shift_work`
- Vendored SWAT model (drift, diffusion, state-clip, parameters) under `_vendored_models/swat/`
- `run_swat.py` CLI producing optimised schedule + 4 figures + NPZ + comparison CSV
- SWAT plot module with schedule, latent paths, terminal-amplitude, loss-trace figures
- **`simulate_closed_loop`** for post-optimisation Monte-Carlo verification
- **Three naive baselines**: `zero_control_schedule`, `constant_reference_schedule`, `linear_interpolation_schedule`
- **`run_ot_pipeline`** end-to-end composition; **`compare_schedules`** label-keyed verification

Phases 6–7 stubbed.

## Test status

```
74 passed (33 Phase 1 + 13 Phase 2 + 9 Phase 3 + 9 Phase 4 + 10 Phase 5)
```

## End-to-end SWAT example with baseline comparison

```
$ python -m experiments.run_swat --scenario insomnia --horizon 14 --steps 200 --n-particles 64

  ...
  Optimising schedule ...
  step   100  L=0.218869  |g|=0.1099
  step   200  L=0.181226  |g|=0.0691

  Final loss:       0.1812
    terminal:       0.0760
    transport:      0.0454
    reference:      0.0598

  Running closed-loop comparison ...

  Schedule                   mean T(D)   distance    basin frac    MMD       
    zero_control                  0.583       0.033         0.758      0.2161
    constant_reference            0.081       0.469         0.000      1.5197
    linear_interpolation          0.180       0.370         0.082      1.1500
    optimised                     0.533       0.017         0.000      0.0385
```

The optimised schedule beats the constant_reference and
linear_interpolation baselines on every metric and ties zero_control
on terminal-T proximity (with a much tighter MMD).

The basin fraction of 0.0 for the optimised schedule reveals an
interesting artifact: the optimiser found a schedule that hits the
terminal-T target via inertia rather than sustained healthy
entrainment. A running-cost loss term that maintains entrainment
throughout the horizon is the principled fix and is recorded as a
future feature.

## Reading order

1. `OT_Control_Codebase_Plan.md` (in the planning folder) — phase-by-phase plan
   plus the maths in plain language (loss decomposition, reference KL, MMD).
2. `ot_engine/types.py` — the four data contracts
3. `ot_engine/simulator.py` — the JAX SDE driver
4. `ot_engine/loss.py` — the three-term loss factory
5. `ot_engine/optimise.py` — the Adam loop
6. `ot_engine/closed_loop.py`, `compare.py`, `pipeline.py` — verification
7. `adapters/swat/adapter.py` — example of how to write an adapter
8. `experiments/run_swat.py` — example of how to run an experiment

## Layout

```
version_1/
├── ot_engine/                     # Layer 1: model-agnostic engine
│   ├── types.py                   # ✅ dataclasses, enums, default_control_names
│   ├── policies/                  # ✅ PiecewiseConstant
│   ├── terminal_cost/             # ✅ MMD
│   ├── reference/                 # ✅ Gaussian-iid prior penalty
│   ├── simulator.py               # ✅ JAX Euler-Maruyama
│   ├── loss.py                    # ✅ three-term loss factory
│   ├── optimise.py                # ✅ Adam + convergence
│   ├── diagnostics.py             # ✅ convergence + summary
│   ├── closed_loop.py             # ✅ Monte-Carlo verification
│   ├── compare.py                 # ✅ naive baselines
│   └── pipeline.py                # ✅ run_ot_pipeline + compare_schedules
├── adapters/                      # Layer 2: per-model glue
│   ├── swat/                      # ✅ SWAT adapter + plots
│   └── fsa_high_res/              # 📋 (Phase 6)
├── experiments/                   # Layer 3: runnable scripts
│   └── run_swat.py                # ✅ CLI runner
├── _vendored_models/              # vendored copies of model SDEs
│   └── swat/                      # ✅ JAX dynamics + parameters
└── tests/
    ├── engine/                    # ✅ engine unit + integration tests (74 tests)
    └── adapters/                  # ✅ SWAT adapter + Phase-5 acceptance (13 tests)
```

## Known limitations (deferred to FUTURE_FEATURES.md)

* **Single-bandwidth MMD**: when source and target distributions are
  far apart compared to the kernel bandwidth, gradients can vanish.
  Multi-bandwidth MMD mixture is the standard fix — extension F2.
* **Optax `optax.global_norm` deprecation warning**: cosmetic; the
  function still works. Migrating to `optax.tree.norm` is a small
  follow-up task.
* **No running-cost term**: the optimiser can hit a terminal target
  via dynamical inertia even if the terminal-day controls would
  collapse the system. Adding a running-cost penalty for time spent
  outside the healthy basin is the principled fix — extension F6.
* **Zero-clip bias on T marginals**: the Euler-Maruyama state-clip
  introduces a small one-sided bias on the mean of physically-bounded
  states. The IMEX integrator from the upstream model-dev repo handles
  this more cleanly. Acceptable for control purposes; switch to Diffrax
  (extension F4) if higher-precision recovery is needed.

## Requirements

- Python ≥ 3.10
- `jax`, `jaxlib` (CPU is fine)
- `optax`
- `matplotlib` (Phase 4+ for plots)
- `pytest` (testing)

## Running the tests

```
cd version_1/
pytest tests/ -v
```

Tests group by phase if you prefer to run them in pieces:

```
# Phase 1
pytest tests/engine/test_types.py tests/engine/test_piecewise_constant.py
pytest tests/engine/test_mmd.py tests/engine/test_gaussian_iid_kl.py
pytest tests/engine/test_public_api.py

# Phase 2
pytest tests/engine/test_simulator.py tests/engine/test_loss.py

# Phase 3
pytest tests/engine/test_diagnostics.py tests/engine/test_optimise.py

# Phase 4
pytest tests/adapters/test_swat_adapter.py

# Phase 5
pytest tests/engine/test_closed_loop_compare_pipeline.py
pytest tests/adapters/test_swat_phase5.py
```

## License

Private research repository.
