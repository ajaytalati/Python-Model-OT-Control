# Mathematical Specification — Python-Model-OT-Control

**Version**: 1.1.0
**Date**: 26 April 2026
**Status**: Specification of the engine as implemented in `version_1/`.

---

## 1. The control problem

Each adapter (SWAT, FSA-high-res, …) supplies a controlled stochastic
differential equation on a latent state $x_t \in \mathbb{R}^{d_x}$:

$$
\mathrm{d}x_t \;=\; f(t, x_t, u_t; \theta_{\text{model}}) \, \mathrm{d}t
              \;+\; \sigma(x_t; \theta_{\text{model}}) \, \mathrm{d}W_t,
\qquad t \in [0, D],
$$

with adapter-supplied initial-condition prior

$$
x_0 \sim \rho_0,
$$

and adapter-supplied projection $A: \mathbb{R}^{d_x} \to \mathbb{R}$ that
extracts the *amplitude* component of clinical interest (testosterone
$T$ for SWAT, fitness $A$ for FSA).

The **clinical target** is supplied as a sampler from a 1-D distribution
on the amplitude marginal at terminal time:

$$
\mu_D^A: \quad a^\star \sim \mu_D^A.
$$

The control $u_t \in \mathbb{R}^{n_c}$ is parameterised by $\theta$ —
piecewise-constant per day in v1:

$$
u(t; \theta) \;=\; \theta_d \quad\text{for}\quad t \in [d, d+1), \qquad
\theta \in \mathbb{R}^{D \times n_c}.
$$

The job of the engine is to choose $\theta$ such that the simulated
distribution of $A(x_D)$ matches $\mu_D^A$, while paying a small
quadratic regularisation cost and staying near an adapter-supplied
reference baseline.

---

## 2. The loss

The engine minimises a three-term loss in $\theta$:

$$
L(\theta) \;=\;
  \alpha_1 \cdot \mathrm{MMD}^2\!\bigl(A(x_D)_{\text{simulated}},\, \mu_D^A\bigr)
\;+\; \alpha_2 \cdot \tfrac{1}{2} \sum_{d=0}^{D-1} \|\theta_d\|^2 \, \Delta t_d
\;+\; \alpha_3 \cdot P\!\bigl(\theta \,\big|\, \mu_{\text{ref}}, \sigma_{\text{ref}}^2\bigr).
$$

### 2.1 Terminal cost — squared MMD

Given $N$ simulated samples $\{a_i\}_{i=1}^N$ at terminal time and $M$
target samples $\{a^\star_j\}_{j=1}^M$, the squared maximum mean
discrepancy with Gaussian kernel $k_h(a, a') = \exp(-(a-a')^2 / (2 h^2))$
is

$$
\mathrm{MMD}^2 \;=\;
\frac{1}{N^2} \sum_{i,i'} k_h(a_i, a_{i'})
\;+\; \frac{1}{M^2} \sum_{j,j'} k_h(a^\star_j, a^\star_{j'})
\;-\; \frac{2}{NM} \sum_{i,j} k_h(a_i, a^\star_j).
$$

Bandwidth $h$ is set adaptively to the median of pairwise distances on
the pooled sample (median heuristic). Differentiable in $\theta$ through
$a_i = A(x_D(\theta))$ via reparameterisation of the noise (see §4).

### 2.2 Transport cost — quadratic regulariser

For piecewise-constant controls:

$$
\frac{1}{2} \int_0^D \|u(t)\|^2 \, \mathrm{d}t
\;=\;
\frac{1}{2} \sum_{d=0}^{D-1} \|\theta_d\|^2 \, \Delta t_d.
$$

When all controls share units this is the Benamou-Brenier $L^2$
kinetic-energy cost. When they don't (e.g. SWAT mixes dimensionless
$V_h, V_n$ with phase-shift $V_c$ in hours) it acts as a generic
quadratic regulariser pulling each control component toward zero.

### 2.3 Reference penalty

Adapter supplies a baseline schedule $\mu_{\text{ref}} \in \mathbb{R}^{D \times n_c}$
and per-entry standard deviation $\sigma_{\text{ref}}$ (also shape
$D \times n_c$). The penalty is

$$
P(\theta \mid \mu_{\text{ref}}, \sigma_{\text{ref}}^2)
\;=\;
\sum_{d, c} \frac{(\theta_{d,c} - \mu_{\text{ref}, d, c})^2}
                  {2 \, \sigma_{\text{ref}, d, c}^2},
$$

which is the negative-log-density (up to a $\theta$-independent
constant) of $\theta$ under an iid Gaussian prior. Equivalent to a MAP
regulariser. The function is exposed as both
`gaussian_iid_kl` (legacy name) and
`gaussian_iid_log_prior_penalty` (clearer alias).

### 2.4 Default loss weights

$\alpha_1 = 1.0$, $\alpha_2 = 0.1$, $\alpha_3 = 0.1$. Adapters override
on a per-problem basis.

---

## 3. The forward simulator

### 3.1 Euler-Maruyama discretisation

Time grid: $t_k = k \cdot \Delta t$ for $k = 0, \ldots, K$ where
$K = \mathrm{round}(D / \Delta t)$. Default $\Delta t = 0.1$ days
(2.4 h); SWAT uses $\Delta t = 0.05$ days (1.2 h).

Update for particle $i$ at step $k$:

$$
x_{i, k+1}
\;=\;
\mathrm{clip}\!\Bigl(
x_{i, k} \;+\; f\bigl(t_k, x_{i, k}, u(t_k; \theta); \theta_{\text{model}}\bigr) \, \Delta t
\;+\; \sigma\bigl(x_{i, k}; \theta_{\text{model}}\bigr) \, \xi_{i, k} \, \sqrt{\Delta t}
\Bigr),
$$

with $\xi_{i, k} \sim \mathcal{N}(0, I)$ pre-sampled (see §4). The
clip is adapter-optional: SWAT clips to $W \in [0,1]$, $\tilde Z \in [0, A_{\text{scale}}]$,
$a, T \geq 0$.

### 3.2 Particle ensemble

Default $N = 256$ particles. Trade-off:

* MMD bias scales as $1/N$ (median heuristic estimator).
* Memory and compute scale as $\mathcal{O}(N \cdot K \cdot d_x)$.
* Gradient variance scales as $1/N$.

Closed-loop verification uses a separate, independent draw of $M$
particles (default $M = N$).

---

## 4. Differentiability — reparameterisation

The Brownian increments $\xi_{i, k}$ are pre-sampled at the start of
each loss call from the rng key passed in. Inside the simulator they
are constants from JAX's perspective, so

$$
x_K \;=\; x_K(\theta; \xi)
\quad\text{is differentiable in }\theta\text{ with }\xi\text{ fixed}.
$$

This is the standard pathwise / reparameterisation trick. The gradient
$\nabla_\theta L(\theta)$ is obtained via `jax.value_and_grad`. JIT is
applied once at problem-build time; subsequent gradient evaluations are
fast.

**Gradient pathology — saturated state clip**: if a trajectory hits the
state clip on every step (e.g. $W$ pinned at $0$), the clip is
piecewise-constant in $x$ and the gradient through that step is zero.
The optimiser then sees no signal pushing $\theta$ away from the wall.
This explains rare cases of "stuck at boundary" optimisations and is
the principal failure mode of the current EM + clip scheme.

---

## 5. The optimiser

Adam with default learning rate $10^{-2}$, betas $(0.9, 0.999)$, eps
$10^{-8}$. Default 2000 steps, with early-stopping convergence check.

**Convergence rule**: at step $k$, compare the mean of the last $W$
losses with the mean of the previous $W$ losses (default $W = 50$).
Declare converged when

$$
\frac{|\bar L_{k-W:k} - \bar L_{k-2W:k-W}|}{\bar L_{k-2W:k-W}} \;<\; \mathrm{tol},
$$

default $\mathrm{tol} = 10^{-4}$. The check fires no earlier than step
$2W$.

**Stability tooling**:
* JAX float64 enabled globally.
* Gradient norms recorded per step; included in trace metadata.
* Single rng per gradient call (deterministic gradients given rng and
  $\theta$).
* Optional `resample_every` parameter resamples the rng every $k$
  steps to reduce overfitting to one MC sample (defaults to 0 = never).

---

## 6. Closed-loop verification

After optimisation, simulate the latent SDE under the optimised
schedule with a fresh, independent rng and (optionally) larger particle
count. Two metrics are reported:

* `mmd_target` — terminal MMD$^2$ to $\mu_D^A$.
* `fraction_in_healthy_basin` — fraction of trajectories where the
  adapter's basin indicator returns true at terminal time. NaN if the
  adapter does not supply a basin function. For SWAT the basin
  indicator checks whether the entrainment quality $E$ at the terminal
  controls exceeds $E_\text{crit}$ AND $T \geq 0.4$.

Schedules from baselines (zero control, constant reference, linear
interpolation) are run through the same closed-loop simulator and
compared head-to-head.

---

## 7. Public API summary

```
ot_engine
├── BridgeProblem, Schedule, OptimisationTrace, ClosedLoopResult
├── PolicyKind, TerminalCostKind, ReferenceKind
├── ControlPolicy (abstract), PiecewiseConstant
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

`BridgeProblem` is the contract. An adapter constructs one; the engine
consumes it. Every numerical detail above is encoded in the fields of
`BridgeProblem` and the methods of `ControlPolicy`.

---

## 8. Notation

| Symbol | Meaning | Code |
|:---:|:---|:---|
| $d_x$ | latent state dimension (4 for SWAT) | `dim_state` |
| $n_c$ | control vector dimension (3 for SWAT) | `n_controls` |
| $D$ | horizon in days | `horizon_days` |
| $N$ | training particle count | `n_particles` |
| $M$ | verification particle count | `n_realisations` |
| $K$ | discrete EM steps over horizon | `n_steps` |
| $\Delta t$ | EM step size in days | `dt_days` |
| $\theta$ | control parameters $(D \times n_c)$ | `theta` |
| $\theta_{\text{model}}$ | fixed model parameters | `model_params` |
| $A$ | amplitude projection $\mathbb{R}^{d_x} \to \mathbb{R}$ | `amplitude_of` |
| $\mu_D^A$ | target distribution on $A$ at $t=D$ | `sample_target_amplitude` |
| $\rho_0$ | initial-condition prior on $x_0$ | `sample_initial_state` |
| $\mathrm{MMD}^2$ | squared MMD with median bandwidth | `mmd_squared` |
| $h$ | kernel bandwidth | `median_bandwidth` |
| $\mu_{\text{ref}}, \sigma_{\text{ref}}$ | Gaussian reference centring + width | `reference_schedule, reference_sigma` |

---

## 9. Validation hooks (engine-side)

The `BridgeProblem` dataclass runs a `__post_init__` validator at
construction time. It rejects:

* missing required callables;
* non-positive `n_controls`, `horizon_days`, `n_particles`, `optim_steps`,
  `dt_days`, `learning_rate`;
* `control_bounds` of wrong length or with $\text{lo} \geq \text{hi}$;
* `reference_schedule` / `reference_sigma` of wrong shape;
* `reference_sigma` not strictly positive elementwise;
* negative loss-weight $\alpha$;
* `control_names` of wrong length.

These checks fail loudly at the adapter call site rather than producing
inscrutable JAX trace errors deep inside the loss.
