# Python-Model-OT-Control — Changelog

## 1.1.0 — 2026-04-26

### Code review hardening (31 findings addressed)

**Critical bug fixes**

- `policies/piecewise_constant.py`: `init_params` no longer downcasts
  float64 -> float32 (was silently breaking JAX x64 mode).
- `optimise.py`, `diagnostics.py`: empty optimisation traces no longer
  crash with `IndexError`; metadata returns NaN scalars instead.
- `types.py`: `BridgeProblem` now validates required fields, shape
  consistency, positivity of `n_controls`/`horizon_days`/`n_particles`/
  `optim_steps`/`dt_days`/`learning_rate`, and strictly-positive
  `reference_sigma` at construction time via `__post_init__`.

**API additions**

- `BridgeProblem.control_names: Optional[Tuple[str, ...]]` — adapters
  can now pass through human-readable control labels (e.g.
  `('V_h', 'V_n', 'V_c')` for SWAT) which propagate to `Schedule`
  outputs and CSV/JSON metadata.
- `default_control_names(n_controls)` helper exported.
- `gaussian_iid_log_prior_penalty` exported as a clearer alias for
  `gaussian_iid_kl` (both names point at the same function).

**Closed-loop / pipeline hardening**

- `simulate_closed_loop` validates `schedule.horizon_days` and
  `schedule.n_controls` against the problem.
- `simulate_closed_loop` raises on `n_realisations=0` instead of
  silently falling back to `problem.n_particles`.

**Documentation fixes**

- Removed stale references to a non-existent
  `docs/Mathematical_Background.md` (in `loss.py` and `README.md`).
- `__init__.py`: removed the "Phase 3+ stubbed" line.
- `types.py`: corrected `basin_indicator_fn` description (it IS used by
  the closed-loop module for the basin-fraction metric).
- `types.py`: corrected `ClosedLoopResult.t` shape to `(n_steps + 1,)`.
- `types.py`: documented dual role of `reference_schedule` (KL centring
  + `init_params` seed).
- `loss.py`: clarified `transport_cost_piecewise_constant` is a
  quadratic regulariser when controls have heterogeneous units.
- `gaussian_iid.py`: clarified the function is a negative-log-density
  penalty (a MAP regulariser equivalent to a Gaussian prior on theta),
  not a true KL between distributions.
- `plots.py`: header now says "Four standard figures" (was "Three").

**Cosmetic cleanup**

- `simulator.py`: tightened initial-state shape validation; removed
  dangling `dim_state` variable.
- `mmd.py`: removed dead `mask` / `flat_mask` variables.
- `_vendored_models/swat/dynamics_jax.py`: `swat_state_clip` now reads
  `A_scale` from `params` (was hardcoded to 6.0); SWAT adapter wraps it
  with a closure binding the configured params.
- `experiments/run_swat.py`: removed unused `simulate_latent` import.
- `adapters/swat/plots.py`: removed unused `Tuple` import; removed dead
  `ax` parameter from `plot_latent_paths`.
- `policies/_abstract.py`: `_validate_attrs()` enforces required
  `horizon_days` / `n_controls` attributes are set by subclasses.

### Upstream parameter sync (Python-Model-Development-Simulation)

Re-pulled `_vendored_models/swat/parameters.py` from upstream main:

- `beta_Z`: 2.5 -> 4.0  (closes upstream #5 / #7)
- `T_Z`:    0.01 -> 0.05 per hour  (matches upstream PARAM_SET_A)
- `c_tilde`: 3.0 -> 2.5  (closes upstream #8; observation channel only)
- Initial `Z_0`: 3.0 -> 3.5  (matches upstream INIT_STATE_A)

Vendored parameters file bumped to v1.1.0.

### Tests

- 87 tests passing (was 74). 13 new tests cover the validation paths
  added in this release: `BridgeProblem.__post_init__` rejection cases,
  `init_params` float64 preservation, `summarise_trace` empty-trace
  handling, `simulate_closed_loop` horizon and n_realisations checks,
  `default_control_names` helper, explicit `control_names` propagation
  through the SWAT adapter.

### Public API surface

```
ot_engine
├── BridgeProblem, Schedule, OptimisationTrace, ClosedLoopResult
├── PolicyKind, TerminalCostKind, ReferenceKind
├── default_control_names
├── ControlPolicy, PiecewiseConstant
├── mmd_squared, median_bandwidth
├── gaussian_iid_kl, gaussian_iid_log_prior_penalty
├── simulate_latent
├── make_loss_fn, transport_cost_piecewise_constant
├── optimise_schedule
├── convergence_check, summarise_trace
├── simulate_closed_loop
├── zero_control_schedule, constant_reference_schedule,
│   linear_interpolation_schedule
└── run_ot_pipeline, compare_schedules
```


## 1.0.0 — 2026-04-25

Initial release. Phase 1+2+3+5 of the codebase plan complete:

- Generic OT/control engine (`ot_engine/`) — model-agnostic, JAX-native
- SWAT adapter (`adapters/swat/`) — first concrete model
- Vendored SWAT dynamics (`_vendored_models/swat/`)
- CLI runner (`experiments/run_swat.py`)
- 74 tests passing across engine + adapter
