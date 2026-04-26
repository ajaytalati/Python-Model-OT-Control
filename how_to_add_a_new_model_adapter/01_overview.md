# 01 — Overview

## What an adapter is

An adapter is a thin glue layer with one job: **construct a
`BridgeProblem` for a specific physiological model**.

```
┌─────────────────────────┐
│   Upstream model repo   │   Python-Model-Development-Simulation
│   (drift, diffusion,    │   ─ source of truth for the SDE
│    parameters, spec)    │
└────────────┬────────────┘
             │ vendored (copy + light JAX rewrite)
             ▼
┌─────────────────────────┐
│   _vendored_models/     │
│   your_model/           │   self-contained JAX dynamics
│     dynamics_jax.py     │
│     parameters.py       │
└────────────┬────────────┘
             │ imported by
             ▼
┌─────────────────────────┐
│   adapters/             │
│   your_model/           │   THIS IS YOUR ADAPTER
│     adapter.py          │   ─ make_<X>_problem(scenario, ...)
│     plots.py            │     returns BridgeProblem
└────────────┬────────────┘
             │ consumed by
             ▼
┌─────────────────────────┐
│   ot_engine/            │   model-agnostic engine
│   (never imports        │   ─ optimise_schedule
│    your adapter)        │   ─ simulate_closed_loop
└─────────────────────────┘
```

The dependency direction is strictly top-to-bottom. The engine knows
nothing about any specific model. Your adapter knows about the engine
(it produces a `BridgeProblem` for the engine to consume) and about
the upstream model (it references the vendored copy).

## What "model-agnostic" actually means

The engine treats the model as a black box defined by four functions:

* `drift_fn_jax(t, x, u, model_params) -> jnp.ndarray of shape (dim_state,)`
* `diffusion_fn_jax(x, model_params) -> jnp.ndarray of shape (dim_state,)`
* `sample_initial_state(rng, n) -> jnp.ndarray of shape (n, dim_state)`
* `sample_target_amplitude(rng, n) -> jnp.ndarray of shape (n,)`

Plus a fifth pure-functional helper:

* `amplitude_of(x: jnp.ndarray) -> jnp.ndarray` (scalar) — extracts the
  scalar amplitude variable from a latent state vector. For SWAT this
  picks T (testosterone). For FSA-high-res this picks A (fitness).

Anywhere the engine needs to "know" something about your model, it
calls one of these functions or reads a field on the `BridgeProblem`.
Nothing else.

## The clinical story your adapter encodes

A `BridgeProblem` represents one specific clinical situation:

* a **patient phenotype** at $t = 0$ — encoded by `sample_initial_state`
  (e.g. "insomnia patient: T_0 ≈ 0.05 with bad sleep architecture");
* a **clinical target** at $t = D$ — encoded by
  `sample_target_amplitude` (e.g. "testosterone restored to a healthy
  pulsatility distribution");
* an **uncontrolled baseline** ("the do-nothing schedule") — encoded by
  `reference_schedule`, against which the optimiser is regularised;
* a **control budget and shape** — encoded by `n_controls`,
  `control_bounds`, `horizon_days`, and the choice of policy.

Once those are filled in, the engine's job — minimising the loss in
§2 of `docs/Mathematical_Specification.md` — is the same regardless of
what specific dynamics live inside `drift_fn_jax`.

## Why this layering matters

Three reasons.

**1. Reproducibility.** Vendoring the model means an adapter can be
checked out at any historical commit and still build a valid
`BridgeProblem`, even if the upstream model has moved on. The
`_vendored_models/<your_model>/README_vendored.md` records the
provenance.

**2. Fault isolation.** A bug in the engine (loss, optimiser,
simulator) is a single fix that propagates to every adapter. A bug in
the SWAT model is fixed in the upstream repo, then re-vendored.
Neither category of fix touches the adapter.

**3. Engine-side guarantees survive.** When the engine validates that
`reference_sigma > 0` everywhere, or that `control_bounds` are
respected by the policy, it is enforcing those guarantees for *every*
adapter, including ones not yet written. Adapter authors do not need
to re-implement these checks.

## The contract you are signing up to

By writing an adapter you are committing to:

* JAX-native dynamics (the engine uses `jax.grad` / `jax.vmap` /
  `jax.jit` end-to-end; numpy or scipy inside the drift will break
  gradients);
* shape and signature conformance for all five required callables (see
  `02_bridge_problem_contract.md`);
* time-unit consistency (engine works in days);
* float dtype consistency (engine enables `jax_enable_x64` globally);
* no mutation of `model_params` after `BridgeProblem` construction;
* the "do-nothing reference" interpretation of `reference_schedule`
  (i.e. the controls that represent the patient's pre-intervention
  status; the schedule deviation from this is what the optimiser is
  prescribing).

Anything else (how you parameterise scenarios, how many of them you
expose, what your basin indicator looks like, how you generate the
target distribution) is a design decision, not a contract.

## What you do NOT need to do

* You do not implement the optimiser. The engine has Adam.
* You do not implement the simulator. The engine has Euler-Maruyama
  via `simulate_latent`.
* You do not implement the loss. The engine has `make_loss_fn` and
  the three-term composition (terminal MMD + transport regulariser +
  reference penalty).
* You do not implement bounds enforcement. The engine's
  `PiecewiseConstant` policy clips to `control_bounds` automatically
  when constructed via `from_problem`.
* You do not implement validation. `BridgeProblem.__post_init__`
  catches all the shape and value-range mistakes.
* You do not implement plotting infrastructure. The engine ships
  `closed_loop.py`, `compare.py`, and `pipeline.py`; your adapter only
  needs adapter-specific figures (e.g. SWAT's `latent_paths.png`
  showing W, Z, T trajectories).

## Sizing

A typical adapter is ~300 lines of Python plus a vendored model of
maybe 200 lines. The SWAT adapter is the canonical reference:

```
$ wc -l version_1/adapters/swat/*.py
   343 version_1/adapters/swat/adapter.py
   229 version_1/adapters/swat/plots.py
    35 version_1/adapters/swat/__init__.py
   607 total

$ wc -l version_1/_vendored_models/swat/*.py
   215 version_1/_vendored_models/swat/dynamics_jax.py
    98 version_1/_vendored_models/swat/parameters.py
    18 version_1/_vendored_models/swat/__init__.py
   331 total
```

Plan on a few days of work for a model you understand well, longer if
the dynamics need significant translation from the upstream
implementation.

## Next

Read `02_bridge_problem_contract.md` for the precise field-by-field
contract, then `03_step_by_step_guide.md` for the procedure.
