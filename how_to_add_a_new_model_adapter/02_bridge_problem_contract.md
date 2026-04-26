# 02 — The `BridgeProblem` contract

This is the reference document. Read once, refer back as needed.

Every field of `BridgeProblem` is described below, with the precise
shape / signature / value constraints. The dataclass itself lives at
`ot_engine/types.py`; this doc explains what to put in each field.

## Required: identity

### `name: str`

Human-readable identifier, e.g. `'swat_insomnia'` or
`'fsa_unfit_recovery'`. Used in metadata and CSV outputs. No engine
behaviour depends on the name.

## Required: SDE specification

### `drift_fn_jax: Callable`

Signature:

```python
drift_fn_jax(t: jnp.ndarray,         # scalar, days
             x: jnp.ndarray,         # shape (dim_state,)
             u: jnp.ndarray,         # shape (n_controls,)
             model_params: dict,
            ) -> jnp.ndarray         # shape (dim_state,)
```

Returns the drift vector $f(t, x, u; \theta_{\text{model}})$ at one
$(t, x, u)$. Must use only `jnp.*` operations — `jax.grad` traces
through this function.

`t` is in **days** (engine convention). If the upstream model expresses
its drift in hours, convert at vendoring time, not inside the drift.

### `diffusion_fn_jax: Callable`

Signature:

```python
diffusion_fn_jax(x: jnp.ndarray,         # shape (dim_state,)
                 model_params: dict,
                ) -> jnp.ndarray          # shape (dim_state,)
```

Returns the per-component noise amplitudes $\sigma_i(x)$ for diagonal
diffusion. Engine assumes diagonal — multiplicative noise via a square
root matrix is not in v1.

For state-independent (additive) noise, ignore `x` (use
`del x`) and return constants. For state-dependent (multiplicative)
noise, use `x`.

### `model_params: Dict[str, Any]`

Treated as opaque by the engine. Passed verbatim to
`drift_fn_jax(... , model_params)` and
`diffusion_fn_jax(x, model_params)`.

Should contain everything the model's drift/diffusion needs:
timescales, coupling constants, bifurcation parameters,
diffusion-temperature scalars. Do **not** put per-particle initial
conditions, scenario-specific controls, or RNG state in here.

## Required: endpoint distributions

### `sample_initial_state: Callable`

Signature:

```python
sample_initial_state(rng: jax.Array,
                     n_particles: int,
                    ) -> jnp.ndarray         # shape (n_particles, dim_state)
```

Draws $n$ samples from the initial-state distribution $\rho_0$. Must
return a 2-D array; the engine checks this.

The scenario phenotype lives here: an "insomnia" patient has a
different `sample_initial_state` than a "shift_work" patient even
though both use the same `drift_fn_jax`.

### `sample_target_amplitude: Callable`

Signature:

```python
sample_target_amplitude(rng: jax.Array,
                        n_samples: int,
                       ) -> jnp.ndarray      # shape (n_samples,)
```

Draws $n$ samples from the clinical target distribution $\mu_D^A$ on
the amplitude variable at terminal time.

**Critical**: the target must be reachable. If the model's own
predicted distribution under healthy controls cannot reach this
target, the optimiser will hunt in counterintuitive directions trying
to bridge the gap. The recommended pattern is to construct the target
from a one-off model simulation under "idealised healthy" controls;
see SWAT's `_build_healthy_target_sampler` for a worked example.

### `amplitude_of: Callable`

Signature:

```python
amplitude_of(x: jnp.ndarray,        # shape (dim_state,)
            ) -> jnp.ndarray         # scalar
```

Pure projection from the latent state vector to the scalar amplitude
variable of clinical interest. For SWAT this picks $T$ (the
testosterone-pulsatility component, index 3 of the state vector). For
FSA-high-res this picks $A$ (fitness amplitude). For a 1-D model this
is `lambda x: x[0]`.

The engine uses this to compute the terminal-time amplitude marginal
$A(x_D)$ that gets MMD-matched to `sample_target_amplitude`.

## Required: control specification

### `n_controls: int`

Dimension of the control vector $u_t$. Must be ≥ 1.
SWAT: 3 (`V_h`, `V_n`, `V_c`).
FSA-high-res: 2 (`T_B`, `Φ`).

### `control_bounds: Tuple[Tuple[float, float], ...]`

Length must equal `n_controls`. Each entry is `(lo, hi)` with
`lo < hi`. Enforced by the policy when constructed via
`PiecewiseConstant.from_problem(problem)` — the policy clips its
output to these bounds. Adam's sub-gradient at a binding bound is zero
so the optimiser stops pushing past it.

Note: the bounds describe *physical / clinical* bounds on the control
variables. They are not soft penalties; the engine enforces them via
clipping, which means the optimiser cannot recommend a schedule that
violates them.

Common pitfall: signed vs unsigned controls.
* Vitality reserves, loads, and similar non-negative quantities should
  have `lo = 0`.
* Phase shifts and other genuinely-signed quantities should have
  `lo < 0 < hi`.

### `horizon_days: int`

Schedule length $D$ in days. Must be ≥ 1. The engine works in days
throughout.

### `policy_kind: PolicyKind = PolicyKind.PIECEWISE_CONSTANT`

v1 only supports `PIECEWISE_CONSTANT`. Future kinds (B-spline, neural
net) are deferred per `docs/Future_Features.md` (F1).

## Required: reference path measure

### `reference_schedule: jnp.ndarray`

Shape `(horizon_days, n_controls)`. The "uncontrolled baseline"
schedule — typically the patient's pre-intervention controls held
constant. Plays two roles:

1. **Centring** of the Gaussian reference penalty in the loss
   (the optimiser is regularised toward this);
2. **Initial value of θ** — `policy.init_params(reference_schedule)`
   returns this as the starting point for Adam.

Should lie inside `control_bounds` elementwise. The validator does not
check this — if the reference is out-of-bounds, the policy will clip
its starting point and the optimiser will pull θ toward an
unreachable target.

### `reference_sigma: jnp.ndarray`

Shape `(horizon_days, n_controls)`. Per-(day, control) standard
deviation of the Gaussian reference. Strictly positive elementwise
(validator checks this).

Smaller σ → tighter reference → optimiser stays close to baseline.
Larger σ → looser reference → optimiser free to deviate.
Default 1.0 is reasonable for SWAT-magnitude controls.

### `reference_kind: ReferenceKind = ReferenceKind.GAUSSIAN_IID`

v1 only supports `GAUSSIAN_IID` (independent Gaussian per (day,
control) entry). Other reference families are deferred per F3.

## Optional: control labelling

### `control_names: Optional[Tuple[str, ...]] = None`

Length must equal `n_controls` if supplied. Used in CSV outputs and
plot legends. Defaults to `('u_0', 'u_1', ...)` if not supplied; the
adapter is encouraged to set explicit names like
`('V_h', 'V_n', 'V_c')` for SWAT or `('T_B', 'Phi')` for FSA-high-res.

## Loss weights (have sensible defaults; rarely override)

### `alpha_terminal: float = 1.0`

Weight on the terminal MMD term. The "match the target distribution"
pressure. Default 1.0 sets the scale.

### `alpha_transport: float = 0.1`

Weight on the quadratic regulariser $\frac{1}{2} \|u\|^2$. Default 0.1
gives a mild preference for low-magnitude schedules. Increase if the
optimiser produces over-aggressive schedules.

### `alpha_reference: float = 0.1`

Weight on the reference (Gaussian iid) penalty. Default 0.1 is loose;
the optimiser will deviate substantially from the reference if the
terminal MMD demands it. Increase if you want the schedule to stay
close to baseline.

### `terminal_cost_kind: TerminalCostKind = TerminalCostKind.MMD`

v1 only supports MMD. Other terminal costs are deferred per F2.

## Solver hyperparameters

### `n_particles: int = 256`

Monte-Carlo particle count. Trade-off:
* MMD bias scales as 1/N
* Compute scales as O(N · K · dim_state)
* Gradient variance scales as 1/N

256 is fine for SWAT. Use 128 for fast iteration; 512+ for production
runs where MMD precision matters.

### `dt_days: float = 0.1`

EM step size in days. Choose based on the fastest timescale in your
drift:
* SWAT's $\tau_W = 2$ h ≈ 0.083 days → `dt_days = 0.05` resolves it.
* If your model has a sub-hour timescale, you'll need `dt_days ≈
  0.01`.

### `optim_steps: int = 2000`

Maximum Adam steps. Convergence detection (sliding-window relative
change) typically stops earlier.

### `learning_rate: float = 1e-2`

Adam learning rate. 1e-2 is a good starting point for the magnitudes
of θ that arise from typical reference schedules.

## Optional: adapter overlays

### `bifurcation_surface_fn: Optional[Callable] = None`

Plotting helper used by adapter-specific figures. Not consumed by the
engine. Default `None`.

### `basin_indicator_fn: Optional[Callable] = None`

Verification helper consumed by `simulate_closed_loop` for the basin-
fraction metric.

Signature:

```python
basin_indicator_fn(x: jnp.ndarray,             # shape (dim_state,)
                   u_terminal: jnp.ndarray,    # shape (n_controls,)
                   model_params: dict,
                  ) -> jnp.ndarray              # scalar bool
```

Returns `True` iff the latent state $x$ at terminal time, under the
schedule's terminal-day controls $u_D$, is in the "healthy" region.

If `None`, `simulate_closed_loop` returns `NaN` for the
`fraction_in_healthy_basin` metric without failing.

For SWAT the basin condition is "entrainment-quality at terminal
controls is super-critical AND $T$ is inside the empirical target
pool's central 80% interval". For another model, the equivalent is
typically "the model's deterministic dynamics under terminal controls
have a healthy stable fixed point AND the patient's amplitude is near
that fixed point".

### `state_clip_fn: Optional[Callable] = None`

Element-wise state bounds enforced after each Euler-Maruyama step.

Signature:

```python
state_clip_fn(x: jnp.ndarray,           # shape (dim_state,)
             ) -> jnp.ndarray            # shape (dim_state,)
```

Used for physically-bounded state components: $W \in [0, 1]$, $T \geq
0$, $Z \in [0, A_{\text{scale}}]$, etc. Applied unconditionally inside
the simulator if non-`None`. Captures `None` at trace time when
absent (no per-step branch, so JIT is happy either way).

If your `model_params` include a relevant scale parameter (like SWAT's
`A_scale`), wrap `state_clip_fn` with a closure that binds the params
at adapter-build time so the clip stays consistent with whatever
parameter dictionary was actually used.

## Validation — what the engine checks at construction

`BridgeProblem.__post_init__` raises `ValueError` if:

* any of the five required callables is `None`;
* `n_controls`, `horizon_days`, `n_particles`, or `optim_steps` is
  not ≥ 1;
* `dt_days` or `learning_rate` is not > 0;
* `control_bounds` length ≠ `n_controls`;
* any `control_bounds[i]` has `lo ≥ hi`;
* `reference_schedule` shape ≠ `(horizon_days, n_controls)`;
* `reference_sigma` shape ≠ `(horizon_days, n_controls)`;
* `reference_sigma` has any non-positive element;
* any `alpha_*` is negative;
* `control_names` is supplied but length ≠ `n_controls`.

Engine-side validation; you don't need to re-implement these in your
adapter.

## What the engine does NOT validate

* Whether `reference_schedule` is inside `control_bounds`. (It should
  be; if not, the policy will clip the start point.)
* Whether `sample_target_amplitude` is reachable from the
  `sample_initial_state` distribution under the model. (It should
  be; otherwise the optimiser hunts in counterintuitive directions.)
* Whether `drift_fn_jax` and `diffusion_fn_jax` are JAX-traceable. (If
  they aren't, you'll get a `jax.grad`-time error, not a construction-
  time error.)
* Whether your model is identifiable, controllable, or
  scientifically-meaningful in any other way.

These are the adapter author's responsibility. Choose your scenarios
and target distributions thoughtfully.

## Field summary table

| Field | Type | Required | Default | Engine-validated |
|:---|:---|:---:|:---|:---:|
| `name` | str | ✓ | — | ✗ |
| `drift_fn_jax` | Callable | ✓ | — | ✓ (non-None) |
| `diffusion_fn_jax` | Callable | ✓ | — | ✓ (non-None) |
| `model_params` | dict | ✓ | — | ✗ |
| `sample_initial_state` | Callable | ✓ | — | ✓ (non-None) |
| `sample_target_amplitude` | Callable | ✓ | — | ✓ (non-None) |
| `amplitude_of` | Callable | ✓ | — | ✓ (non-None) |
| `n_controls` | int | ✓ | — | ✓ (≥ 1) |
| `control_bounds` | tuple of (lo, hi) | ✓ | — | ✓ (length, lo < hi) |
| `horizon_days` | int | ✓ | — | ✓ (≥ 1) |
| `policy_kind` | enum | | `PIECEWISE_CONSTANT` | ✗ |
| `reference_schedule` | jnp array | ✓ | — | ✓ (shape) |
| `reference_sigma` | jnp array | ✓ | — | ✓ (shape, > 0 elementwise) |
| `reference_kind` | enum | | `GAUSSIAN_IID` | ✗ |
| `control_names` | tuple of str | | `None` | ✓ (length if supplied) |
| `alpha_terminal` | float | | `1.0` | ✓ (≥ 0) |
| `alpha_transport` | float | | `0.1` | ✓ (≥ 0) |
| `alpha_reference` | float | | `0.1` | ✓ (≥ 0) |
| `terminal_cost_kind` | enum | | `MMD` | ✗ |
| `n_particles` | int | | `256` | ✓ (≥ 1) |
| `dt_days` | float | | `0.1` | ✓ (> 0) |
| `optim_steps` | int | | `2000` | ✓ (≥ 1) |
| `learning_rate` | float | | `1e-2` | ✓ (> 0) |
| `bifurcation_surface_fn` | Callable | | `None` | ✗ |
| `basin_indicator_fn` | Callable | | `None` | ✗ |
| `state_clip_fn` | Callable | | `None` | ✗ |

## Next

Now you know what to fill in. Read `03_step_by_step_guide.md` for
how to fill it in.
