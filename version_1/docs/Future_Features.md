# Future Features — Python-Model-OT-Control

**Version**: 1.1.0
**Date**: 26 April 2026
**Status**: Backlog of deferred extensions. None of these are implemented
in v1; they are recorded as a forward-compatibility map for the engine's
abstract interfaces.

The locking principle behind v1 was: ship a small, correct, model-agnostic
engine — `BridgeProblem` + `ControlPolicy` + `simulate_latent` + `make_loss_fn`
+ `optimise_schedule` + `simulate_closed_loop` — and defer everything else
to this list. Each F-numbered item below identifies an abstract interface
or a swap point in the engine where the extension plugs in without engine
surgery.

---

## F1 — Alternative control parameterisations

The engine's only policy in v1 is `PiecewiseConstant` (daily values,
$\theta \in \mathbb{R}^{D \times n_c}$). Other parameterisations:

* **`BSplineBasis`** — schedule as a B-spline in time, $K$ knots per
  control. Smoother schedules with fewer parameters; can hit
  intermediate-day values without locking into per-day step changes.
* **`NeuralNet`** — a small MLP $\mathbb{R} \to \mathbb{R}^{n_c}$ taking
  $t$ (and optionally model state) to controls. For long horizons or
  rich control structures.
* **`PiecewiseLinear`** — daily *anchor points* connected by linear
  ramps. Half the smoothness of B-splines, twice the parameter count of
  piecewise-constant.

**Implementation route**: subclass `ot_engine.policies._abstract.ControlPolicy`,
implement `init_params`, `evaluate(t, theta)`, `evaluate_daily(theta)`,
and `n_params`. Add to the `PolicyKind` enum. No engine surgery required.

---

## F2 — Alternative terminal-cost surrogates

The engine's only terminal cost in v1 is `mmd_squared` with median-bandwidth
heuristic. Other choices:

* **Sliced Wasserstein distance** — closer to "transport cost" in
  spirit; differentiable; better tail handling than MMD.
* **Gaussian KL to fitted parametric target** — simplest, loses tail
  information; useful when the target is genuinely Gaussian.
* **Energy distance / Cramér distance** — kernel-free alternatives.
* **Multi-bandwidth MMD mixture** — sum of Gaussian-kernel MMDs at
  different bandwidths; addresses the gradient-vanishing pathology
  when source and target are far apart.

**Implementation route**: add a module under `ot_engine/terminal_cost/`,
register in the `TerminalCostKind` enum, dispatch in `make_loss_fn`.

---

## F3 — Alternative reference path measures

The engine's only reference in v1 is `gaussian_iid` (independent
Gaussian per (day, control)). Other choices:

* **Smooth Gaussian random walk** — adjacent days correlated; encodes
  "schedule should not change abruptly" as a soft constraint.
* **Ornstein-Uhlenbeck reference** — mean-reverting around the
  baseline; encourages the schedule to drift back to baseline if it
  deviates.
* **Data-driven empirical reference** — empirical measure on schedules
  of healthy subjects; useful for population-level priors.
* **Hierarchical reference** — population-level prior with patient-level
  adjustment; pairs naturally with Bayesian estimation upstream.

**Implementation route**: add a module under `ot_engine/reference/`,
expose a function with the `gaussian_iid_kl` signature, register in
the `ReferenceKind` enum.

---

## F4 — Alternative SDE solvers

The engine uses a hand-rolled JAX Euler-Maruyama in v1. For stiff
systems (where the fastest timescale dictates the timestep) we may
need:

* **Diffrax integration with adjoint sensitivity** — Diffrax is the
  JAX-native equivalent of `torchdiffeq`; supports adaptive-step
  solvers and full adjoint sensitivity for memory-efficient
  backpropagation through long horizons.
* **Implicit-explicit (IMEX) schemes** — separate the stiff (linear)
  drift component from the nonlinear; treat stiff implicitly. The
  upstream SWAT simulator uses IMEX; the OT engine uses explicit EM
  for simplicity.
* **Strong-order-1.5 Itô-Taylor scheme** — better convergence than
  EM at fixed timestep; useful when the diffusion is state-dependent
  in a non-trivial way.

**Implementation route**: replace `simulator.py`'s EM core with a
solver-dispatching wrapper; introduce a `SolverKind` enum.

**When this matters**: SWAT's $\tau_T = 48$h timescale combined with
$\tau_W = 2$h is *not* stiff at sub-daily resolution, but if a future
adapter has $\tau_\text{fast} = 30$ min and $\tau_\text{slow} = 7$ d,
EM will require very small timesteps and the wall-clock cost will
become noticeable.

---

## F5 — Alternative barrier types (the user's principal research area)

Closed-loop verification in v1 uses an adapter-supplied
`basin_indicator_fn` — a Boolean test on the terminal state. This is
the simplest possible barrier. The user has flagged richer alternatives:

* **Lyapunov sublevel sets** — barrier as $\{x : V(x) < c_d\}$ where
  $V$ is a model-supplied Lyapunov function. Allows a *time-varying*
  barrier $c_d$ that shrinks as $d \to D$, certifying convergence
  along the trajectory rather than only at terminal time.
* **Reachability tubes** — Hamilton-Jacobi backward reachability from
  $\mu_D^A$. Computes the set of states at each $t$ from which the
  target is reachable under bounded disturbance; the schedule keeps
  the trajectory inside the tube.
* **Stochastic barrier functions** — $E[V(x_t)] \leq c_t$
  in expectation; certifies probabilistic safety. The right tool when
  the user genuinely cares about *distribution* of trajectories
  staying in a safe region, not only the terminal point.
* **Covariance ellipsoids** — Mahalanobis ellipsoids derived from
  closed-loop empirical moments; cheap surrogate for the more rigorous
  reachability tube.
* **Schrödinger-bridge barrier schedules** — barriers derived from
  the Schrödinger bridge between $\rho_0$ and $\mu_D$; this is the
  user's research direction recorded in the Optimal Transport
  Schrödinger Bridge documents.

**Implementation route**: add a `BarrierKind` enum on `BridgeProblem`;
implement evaluation in `closed_loop.py`; the closed-loop simulator
returns per-day barrier-violation flags in addition to the terminal
basin metric.

This is the most-flagged extension. Several specs in the project
folder (`Stochastic_Barrier_Functions_General_Methodology.md`,
`Sleep_Wake_Adenosine_Barrier_Application.md`,
`Barrier_Schedule_Construction_Algorithm.md`) describe the targeted
infrastructure.

---

## F6 — Alternative cost functionals

The v1 loss is:

$$
L = \alpha_1 \mathrm{MMD}^2 + \alpha_2 \tfrac{1}{2}\|u\|^2 \Delta t + \alpha_3 P(\theta).
$$

Other formulations:

* **Hamilton-Jacobi-Bellman (HJB) regulariser** as in OT-Flow —
  penalises violations of the HJB equation along the trajectory;
  requires potential-flow parameterisation $u = -\nabla\Phi$.
* **Mayer-only terminal cost** — drop $\alpha_2$; simpler optimisation,
  less regularised solutions. Useful when the schedule's "smoothness"
  is enforced by the policy (e.g. low-knot B-splines).
* **Running state cost** — $\int_0^D L_\text{run}(x_t) \, \mathrm{d}t$
  penalty for time spent outside a "soft" healthy region. Maintains
  entrainment throughout the horizon, not only at terminal time.
* **Risk-sensitive cost** — replace $E[L]$ with $\frac{1}{\beta}\log E[\exp(\beta L)]$;
  penalises high-loss tail outcomes.

**Implementation route**: each is a swap of the `make_loss_fn` factory.
The simulator and policy abstractions are unchanged.

---

## F7 — Alternative differentiable-transport methods

The v1 method is simulation-based MMD-matching with a fixed control
policy. Other members of the differentiable-transport family:

* **OT-Flow potential parameterisation**: $u = -\nabla\Phi$ for a
  scalar potential $\Phi$ (Onken et al. 2021). Pairs naturally with
  the HJB regulariser (F6). Requires reframing the policy abstraction
  as a potential rather than a direct control.
* **Schrödinger-Föllmer SDE-bridge** via differentiable particle
  filter — the user's research direction. Documented in
  `OT_Schrodinger_Bridge_Barrier_Design.md`,
  `OT_Differentiable_PF_MCLMC_Specification.md`.
* **RNODE-style transport-cost-only regularisation** — drop the
  reference penalty $\alpha_3$, rely on transport cost alone for
  regularisation.
* **Continuous Normalising Flow (CNF) bridge** — parameterise the
  transport as a CNF; train via likelihood-matching on $\mu_D$.

**Implementation route**: each is a swap of the `make_loss_fn`
factory plus the policy parameterisation. The `BridgeProblem`
contract is unchanged.

---

## F8 — Future model adapters

After SWAT (v1) and FSA-high-res (planned for Phase 6), the next
adapters to add:

* **Sleep-Wake (SW)** with full 4-channel Garmin observation model —
  drops the slow-manifold reduction the legacy `ot_schrodinger_bridge_sleep_wake_v2.py`
  used; provides direct testability against real wearable data.
* **Bistable-controlled** — pedagogical adapter useful for unit testing
  extension features. Smallest possible model that exhibits
  bifurcation under control.
* **3-state HPA-HPG / 4-state HPA-HPG-R** — the family flagged in
  `3_State_HPA_HPG_Model_Specification_and_Analysis.md` and the
  identifiability proofs.
* **24-parameter and 44-parameter combined SWAT extensions** —
  `Spec_24_Parameter_Sleep_Wake_Adenosine_Testosterone_Model.md`,
  `Identifiability_Proof_44_Parameter_Combined_Model.md`.

All future models share the Stuart-Landau-on-slow-manifold structure
(per the project's design decisions). The engine is built to absorb
this without re-architecture.

**Implementation route**: vendor the model's JAX dynamics into
`_vendored_models/<model>/`, write the adapter under `adapters/<model>/`,
add a CLI runner under `experiments/run_<model>.py`. The `BridgeProblem`
contract is the only API.

---

## F9 — Bayesian schedule distribution

The v1 engine treats `model_params` as a single fixed dictionary. In
practice, the upstream SMC² estimator returns *samples* from the
parameter posterior. Plumbing these through:

* The engine vmaps across parameter samples → a *distribution over
  schedules* — credible bands on each daily control value.
* This is what makes the pipeline genuinely Bayesian control.
* `BridgeProblem.model_params` extends to accept either a dict (v1
  behaviour) or a `(n_samples,) -> params` callable (new path).

**Implementation route**: `make_loss_fn` and `simulate_latent` both
add a vmap over a parameter axis; `optimise_schedule` returns a
`Schedule` whose `theta` has a leading $n_\text{samples}$ axis. Plot
helpers extend to render credible bands.

---

## F10 — Receding-horizon (model-predictive) control

The v1 engine optimises a single $D$-day schedule and stops. In
production:

* Optimise schedule for $[0, D]$ days.
* Apply only the first $k$ days (typically $k = 1$ or 7).
* Re-estimate model parameters on the new data window.
* Re-optimise for $[k, D + k]$ days.
* Repeat.

**Implementation route**: a new module `experiments/receding_horizon.py`
that loops over the existing engine. The engine itself is unchanged —
each iteration is a self-contained `run_ot_pipeline` call.

The schedule produced for window $i$ feeds into window $i+1$'s
estimation as a known exogenous input. The model-dev repo already
handles `exogenous_inputs` in its `SDEModel` dataclass; the OT repo
writes the schedule to a CSV the SMC² rolling-window estimator reads.

---

## F11 — Constrained controls

The engine's `control_bounds` field is currently *informational only*;
the simulator does not clip controls. Hard-bounded controls would
require:

* **Policy-side bounding** — bake the bounds into the policy
  parameterisation (e.g. logit-transformed PiecewiseConstant). No
  engine change.
* **Penalty barrier** — soft bound penalty that diverges at the
  boundary. Keeps the simulator and gradient flow simple.
* **Lagrangian constrained optimisation** — augmented Lagrangian on
  the bounds; introduces dual variables to the optimiser. Heavier
  machinery.

**Implementation route**: F11a — new policies under `policies/`.
F11b — extension to the loss factory. F11c — replacement of the Adam
loop with a constrained optimiser (e.g. Optax projected Adam).

---

## F12 — Adversarial robustness / worst-case scenarios

The v1 schedule is optimal in expectation under the model. A clinically
risk-averse user might want:

* **Worst-case parameter scenarios** — optimise for the worst element
  of a credible parameter set (min-max). Combine with F9.
* **Robust to model mis-specification** — penalise the schedule's
  performance under perturbed dynamics.
* **Tail-risk-sensitive cost** — F6's risk-sensitive cost specialised
  to clinical safety constraints.

**Implementation route**: a new outer loop wrapping `optimise_schedule`,
with a worst-case sampler over `model_params`. The engine's existing
gradient infrastructure handles the inner problem.

---

## F13 — Multi-objective / Pareto-front optimisation

A clinician may want to trade off:

* terminal MMD (efficacy);
* transport cost (intervention burden);
* basin fraction (clinical-success probability);
* time-to-recovery (urgency).

**Implementation route**: weighted-sum scalarisation across a grid of
$\alpha$-vectors; collect the Pareto front. No engine change required —
this is a sweep over the existing API.

---

## F14 — Real-time / online optimisation

The v1 optimiser takes ~20s for 2000 Adam steps. For real-time clinical
use (re-optimise on each new patient measurement):

* **Warm-start** the schedule from the previous run's $\theta$.
* **Online streaming** of the simulator (replace JIT'd batch with
  scan-based rollout that accepts new state mid-horizon).
* **Truncated optimisation** — fewer Adam steps, accept worse but
  fast solutions.

**Implementation route**: extend `optimise_schedule` with a
`warm_start_theta` argument; add a `simulate_streaming` helper.

---

## F15 — UI / dashboard

Out-of-scope for v1 (pure library), but flagged:

* Interactive schedule editor (drag bars, see closed-loop response
  update live).
* Patient phenotype selector (link to upstream SMC² posterior).
* Comparison-with-baseline visualiser.
* Schedule export to clinical EHR formats.

**Implementation route**: a separate frontend repo consuming the
engine's CSV outputs.

---

## Acceptance criteria for promoting a feature out of this doc

A feature graduates from this list to the engine when:

1. A specification document exists in the project folder describing
   the precise mathematical formulation.
2. An adapter or test case demonstrates a concrete use.
3. The engine extension passes the existing 87-test regression suite
   plus its own dedicated tests.
4. The `BridgeProblem` API is either unchanged or extended (never
   replaced — backward compatibility is non-negotiable).

---

## Where this list is maintained

This doc lives at `docs/Future_Features.md`. PRs adding entries are
welcome; PRs removing entries (because the feature has been built)
should reference the corresponding test suite that proves it works.
