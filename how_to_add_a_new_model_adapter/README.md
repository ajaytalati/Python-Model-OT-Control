# How to add a new model adapter

This folder explains how to plug a new physiological model into the
OT-Control engine. The engine is **model-agnostic** — it knows nothing
about SWAT specifically. An adapter is the (small) glue layer that
translates a model's SDE definition into a `BridgeProblem` the engine
can consume.

If you are adding a model that follows the project's standard pattern
(controlled SDE on a latent state, scalar amplitude variable, clinical
target on the amplitude marginal at terminal time), expect to write
about 200–400 lines of adapter code plus a vendored copy of the
model's JAX dynamics. The SWAT adapter is the canonical worked example.

## Read these in order

1. **`01_overview.md`** — what an adapter is, what it doesn't have to do,
   the dependency direction. 5-minute read.
2. **`02_bridge_problem_contract.md`** — the precise input/output
   contract every adapter must satisfy. The `BridgeProblem` dataclass
   field-by-field, plus shape and signature constraints. Reference
   document — read once, then come back.
3. **`03_step_by_step_guide.md`** — the procedure end-to-end:
   vendoring the model code, writing the adapter module, exposing
   scenarios, basin indicators, target distributions, plotting helpers,
   and tests. The guide assumes the reader has already skimmed (1) and
   (2).
4. **`04_worked_example_FSA.md`** — a complete walkthrough using the
   FSA-high-res model (Phase 6, deferred but specced) to show the
   pattern when the controls are different (`T_B`, `Φ` instead of
   SWAT's `V_h`, `V_n`, `V_c`) and the amplitude variable is fitness
   rather than testosterone.

## What an adapter is NOT

* It is not a model implementation. The model lives upstream
  (`Python-Model-Development-Simulation`), is vendored into
  `version_1/_vendored_models/<your_model>/`, and the adapter only
  *references* it.
* It is not part of the engine. The engine (`ot_engine/`) is
  one-way: adapters depend on the engine, the engine never imports
  from any adapter.
* It is not where you put estimation, observation models, or anything
  that isn't directly needed to set up the control problem.

## What an adapter MUST provide

Five callables and a few configuration fields, packaged into a
`BridgeProblem`:

| Required | Field | Purpose |
|:---:|:---|:---|
| ✓ | `drift_fn_jax` | The model's deterministic drift, JAX-native |
| ✓ | `diffusion_fn_jax` | The model's diffusion (per-component noise), JAX-native |
| ✓ | `sample_initial_state` | Sampler from $\rho_0$ over latent states |
| ✓ | `sample_target_amplitude` | Sampler from $\mu_D^A$ over the amplitude variable at $t = D$ |
| ✓ | `amplitude_of` | Projector $\mathbb{R}^{d_x} \to \mathbb{R}$ extracting the amplitude component from the latent state |
|  | `state_clip_fn` | Optional: physical-bounds clip applied after each EM step |
|  | `basin_indicator_fn` | Optional: indicator of "patient is in healthy attractor at terminal time" — drives the basin-fraction metric in closed-loop verification |
|  | `bifurcation_surface_fn` | Optional: plotting helper for adapter-specific figures |

Plus configuration: `n_controls`, `control_bounds`, `control_names`,
`horizon_days`, `reference_schedule`, `reference_sigma`, and the
solver hyperparameters (`n_particles`, `dt_days`, `optim_steps`,
`learning_rate`).

## Standard adapter layout

```
version_1/
├── _vendored_models/
│   └── your_model/
│       ├── __init__.py
│       ├── README_vendored.md      # provenance, conversion notes
│       ├── dynamics_jax.py         # drift, diffusion, state_clip
│       └── parameters.py           # default parameter dictionary
├── adapters/
│   └── your_model/
│       ├── __init__.py             # re-exports make_X_problem etc.
│       ├── adapter.py              # the BridgeProblem factory
│       └── plots.py                # optional adapter-specific plots
├── experiments/
│   └── run_your_model.py           # CLI runner (clone of run_swat.py)
└── tests/
    └── adapters/
        ├── test_your_model_adapter.py   # smoke + unit tests
        └── test_your_model_phase5.py    # acceptance test
```

## Common pitfalls (skim before starting)

* **Time-unit consistency.** The engine works in **days**. If your
  upstream model is in hours, do the conversion once at the boundary
  (in `parameters.py`); never mix units inside `dynamics_jax.py`.
  See SWAT's `README_vendored.md` for the conventions.
* **Float dtype.** The engine enables JAX's float64 mode globally on
  import. Don't `.astype(jnp.float32)` anywhere inside the adapter or
  vendored model.
* **Reference must be inside `control_bounds`.** Adapters supply both;
  the validator only checks shapes. If the reference falls outside the
  bounds, the optimiser will start from a clipped value and the
  reference penalty will pull θ toward an unreachable point.
* **Target must be reachable.** A hardcoded target like
  `T = N(0.55, 0.05²)` is fine *if the model can reach it*. When the
  model's own predicted distribution under healthy controls is
  different, the optimiser will hunt in counterintuitive directions
  trying to bridge an impossible gap. The SWAT adapter solves this by
  constructing the target from a one-off simulation of the model under
  idealised healthy controls (see §4 of `02_bridge_problem_contract.md`).
* **Validation runs at construction.** `BridgeProblem.__post_init__`
  rejects missing callables, shape mismatches, non-positive integers,
  and `reference_sigma <= 0` elementwise. Fix these at the adapter,
  not by working around the validator.

## When you're done

Your adapter is complete when:

1. `make_<your_model>_problem('<scenario>')` returns a valid
   `BridgeProblem` for at least one scenario.
2. `python experiments/run_<your_model>.py --scenario <scenario>`
   produces an optimised schedule, four standard plots, and a
   comparison CSV without errors.
3. `pytest tests/adapters/test_<your_model>_adapter.py` is green.
4. The optimised schedule is **clinically sensible** for the scenario
   (controls in correct directions, no boundary saturation pinned at
   bounds for the entire horizon).

## See also

* `docs/Mathematical_Specification.md` — what the engine is doing
  with your `BridgeProblem`.
* `docs/Clinical_Specification.md` — what the output is supposed to
  mean clinically.
* `docs/Future_Features.md` — extension hooks (alternative policies,
  alternative target metrics, alternative reference measures) that
  your adapter may want to make use of.
