"""
ot_engine/types.py — Core dataclasses and enums for the OT/control engine.
==========================================================================
Date:    26 April 2026
Version: 1.1.0
Status:  Phase 1 deliverable + Phase 5 wiring + post-review hardening.

Defines the four data contracts that hold the engine together:

    BridgeProblem        — adapter output / engine input
    Schedule             — engine output (the optimised control schedule)
    OptimisationTrace    — diagnostics from the gradient-descent loop
    ClosedLoopResult     — closed-loop verification output

Plus enums that select among engine variants:

    PolicyKind           — control parameterisation (PIECEWISE_CONSTANT in v1)
    TerminalCostKind     — terminal-distribution mismatch metric (MMD in v1)
    ReferenceKind        — reference path measure family (GAUSSIAN_IID in v1)

This module has NO numerical logic. It is the type contract that every other
module of the engine and every adapter must respect. Per the development
plan, all numerical code is in dedicated modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple

import jax.numpy as jnp


# =========================================================================
# ENUMS
# =========================================================================

class PolicyKind(Enum):
    """Control parameterisation kind.

    PIECEWISE_CONSTANT: u(t) = theta_d for t in [d, d+1) days.
    BSPLINE / NEURAL_NET: deferred to FUTURE_FEATURES.md.
    """
    PIECEWISE_CONSTANT = "piecewise_constant"


class TerminalCostKind(Enum):
    """Terminal-distribution mismatch metric.

    MMD: Maximum Mean Discrepancy with Gaussian kernel + median bandwidth.
    SLICED_W2 / GAUSSIAN_KL: deferred to FUTURE_FEATURES.md.
    """
    MMD = "mmd"


class ReferenceKind(Enum):
    """Reference path measure family.

    GAUSSIAN_IID: independent Gaussian on each (day, control) entry,
                  centred on adapter-supplied baseline.
    GAUSSIAN_RW / OU / EMPIRICAL: deferred to FUTURE_FEATURES.md.
    """
    GAUSSIAN_IID = "gaussian_iid"


# =========================================================================
# BRIDGE PROBLEM (adapter -> engine)
# =========================================================================

@dataclass(frozen=True)
class BridgeProblem:
    """Complete specification of a differentiable-transport control problem.

    An adapter constructs one of these. The engine consumes it.

    The four callables (drift, diffusion, sample_initial_state,
    sample_target_amplitude) are JAX-native: they must use jnp.* operations
    so jax.grad / jax.vmap / jax.jit work end-to-end.

    Construction is validated by `__post_init__`: shapes, positivity, and
    presence of required callables are checked at problem-build time so
    bugs surface loudly at the adapter rather than mysteriously inside
    the JIT'd loss.

    Attributes:
        name: Adapter-supplied identifier, e.g. "swat_insomnia".

        drift_fn_jax: Latent SDE drift. Signature
            (t: float, x: jnp.ndarray, u: jnp.ndarray, model_params: dict)
            -> jnp.ndarray of shape (dim_state,).

        diffusion_fn_jax: Latent SDE diffusion. Signature
            (x: jnp.ndarray, model_params: dict) -> jnp.ndarray of shape
            (dim_state,) for diagonal noise. Returns the per-dim sigma_i(x).

        model_params: Fixed model-parameter dictionary (theta_model from
            upstream SMC2 estimation). Treated as opaque by the engine.

        sample_initial_state: (rng_key, n_particles) -> jnp.ndarray of shape
            (n_particles, dim_state). Draws from rho_0.

        sample_target_amplitude: (rng_key, n_samples) -> jnp.ndarray of shape
            (n_samples,). Draws from the clinical target mu_D^A.

        amplitude_of: (x: jnp.ndarray) -> jnp.ndarray scalar. Projects a
            full latent state onto its amplitude component. For SWAT this
            picks T; for FSA it picks A.

        n_controls: Dimension of the control vector u_t. Must be >= 1.

        control_bounds: ((lo_1, hi_1), ..., (lo_n, hi_n)) physical bounds
            per control. Documented for callers; NOT enforced by the
            engine (the simulator does not clip controls). Adapters that
            need hard bounds should bake them into the policy or the
            drift function.

        horizon_days: Schedule length D in days. Must be >= 1.

        policy_kind: Which control parameterisation to use. v1 only
            supports PIECEWISE_CONSTANT.

        reference_schedule: Adapter's "uncontrolled" baseline, shape
            (D, n_controls). Plays two distinct roles:
                1) Centring of the Gaussian reference KL term in the loss
                   (mu_ref in gaussian_iid_kl).
                2) Initial value of theta in policy.init_params (the
                   optimisation starts at the reference).

        reference_sigma: Gaussian reference width per (day, control), same
            shape as reference_schedule. Smaller = tighter constraint.
            Validated to be strictly positive elementwise.

        control_names: Human-readable names for each control component,
            e.g. ("V_h", "V_n", "V_c") for SWAT. Default None; if None,
            the engine generates ("u_0", "u_1", ...) when constructing
            Schedule objects.

        alpha_terminal, alpha_transport, alpha_reference: Loss-term
            weights. Defaults match the development plan; adapter can
            override.

        n_particles: Monte-Carlo particle count for the forward simulator.
            Must be >= 1.

        dt_days: Euler-Maruyama timestep in days. Must be > 0 and ideally
            small relative to horizon_days (the simulator silently runs 0
            steps if dt_days >= horizon_days).

        optim_steps: Maximum Adam steps in the optimisation loop. Must be
            >= 1 (use 1 if you only want a single forward pass + loss).

        learning_rate: Adam learning rate. Must be > 0.

        bifurcation_surface_fn: Optional plotting helper used by adapter
            figures. Not consumed by the engine. Default None.

        basin_indicator_fn: Optional verification helper consumed by
            `simulate_closed_loop` for the basin-fraction metric.
            Signature (x, u_terminal, model_params) -> bool, where x is
            the per-particle terminal state (shape dim_state), u_terminal
            is the schedule's terminal-day control vector (shape n_controls),
            and model_params is the BridgeProblem.model_params dict.
            If None, simulate_closed_loop returns NaN for the metric.

        state_clip_fn: Optional state-bound enforcement applied
            elementwise after each Euler-Maruyama step. Used for
            physically-bounded states (W in [0,1], T >= 0, etc.).
            Default None = no clipping. Recommended for models with
            bounded state components — Euler-Maruyama noise can push
            states out of physical range otherwise.
    """

    name: str

    # ── SDE specification (JAX-native) ──
    drift_fn_jax: Callable
    diffusion_fn_jax: Callable
    model_params: Dict[str, Any]

    # ── Endpoint distributions ──
    sample_initial_state: Callable
    sample_target_amplitude: Callable
    amplitude_of: Callable

    # ── Control specification ──
    n_controls: int
    control_bounds: Tuple[Tuple[float, float], ...]
    horizon_days: int
    policy_kind: PolicyKind = PolicyKind.PIECEWISE_CONSTANT

    # ── Reference path measure ──
    reference_schedule: jnp.ndarray = field(default=None)  # type: ignore
    reference_sigma: jnp.ndarray = field(default=None)     # type: ignore
    reference_kind: ReferenceKind = ReferenceKind.GAUSSIAN_IID

    # ── Optional control-name labelling (review fix H-3) ──
    control_names: Optional[Tuple[str, ...]] = None

    # ── Loss weights ──
    alpha_terminal: float = 1.0
    alpha_transport: float = 0.1
    alpha_reference: float = 0.1
    terminal_cost_kind: TerminalCostKind = TerminalCostKind.MMD

    # ── Solver hyperparameters ──
    n_particles: int = 256
    dt_days: float = 0.1
    optim_steps: int = 2000
    learning_rate: float = 1e-2

    # ── Optional adapter overlays (plotting + verification) ──
    bifurcation_surface_fn: Optional[Callable] = None
    basin_indicator_fn: Optional[Callable] = None

    # ── Optional state clipping (applied after each Euler-Maruyama step) ──
    state_clip_fn: Optional[Callable] = None

    # =====================================================================
    # Validation
    # =====================================================================
    def __post_init__(self) -> None:
        """Validate required fields and inter-field consistency.

        Frozen-dataclass-compatible (raises before the instance is
        considered fully-constructed). Raises ValueError on any
        violation; the message names the offending field.
        """
        # --- Required callables present ---
        for fname in ('drift_fn_jax', 'diffusion_fn_jax',
                       'sample_initial_state', 'sample_target_amplitude',
                       'amplitude_of'):
            if getattr(self, fname) is None:
                raise ValueError(
                    f"BridgeProblem field '{fname}' is required (got None)."
                )

        # --- Positive integers ---
        if int(self.n_controls) < 1:
            raise ValueError(
                f"n_controls must be >= 1, got {self.n_controls}."
            )
        if int(self.horizon_days) < 1:
            raise ValueError(
                f"horizon_days must be >= 1, got {self.horizon_days}."
            )
        if int(self.n_particles) < 1:
            raise ValueError(
                f"n_particles must be >= 1, got {self.n_particles}."
            )
        if int(self.optim_steps) < 1:
            raise ValueError(
                f"optim_steps must be >= 1, got {self.optim_steps}."
            )

        # --- Positive floats ---
        if float(self.dt_days) <= 0.0:
            raise ValueError(f"dt_days must be > 0, got {self.dt_days}.")
        if float(self.learning_rate) <= 0.0:
            raise ValueError(
                f"learning_rate must be > 0, got {self.learning_rate}."
            )

        # --- control_bounds shape ---
        if len(self.control_bounds) != int(self.n_controls):
            raise ValueError(
                f"control_bounds has {len(self.control_bounds)} entries, "
                f"but n_controls = {self.n_controls}."
            )
        for i, (lo, hi) in enumerate(self.control_bounds):
            if float(lo) >= float(hi):
                raise ValueError(
                    f"control_bounds[{i}] = ({lo}, {hi}) — lo must be < hi."
                )

        # --- reference_schedule and reference_sigma shape ---
        if self.reference_schedule is None:
            raise ValueError("reference_schedule is required (got None).")
        if self.reference_sigma is None:
            raise ValueError("reference_sigma is required (got None).")
        ref_shape = tuple(jnp.asarray(self.reference_schedule).shape)
        sigma_shape = tuple(jnp.asarray(self.reference_sigma).shape)
        expected_shape = (int(self.horizon_days), int(self.n_controls))
        if ref_shape != expected_shape:
            raise ValueError(
                f"reference_schedule shape {ref_shape} != expected "
                f"{expected_shape}."
            )
        if sigma_shape != expected_shape:
            raise ValueError(
                f"reference_sigma shape {sigma_shape} != expected "
                f"{expected_shape}."
            )
        # sigma_ref must be strictly positive elementwise (review fix C-5).
        sigma_arr = jnp.asarray(self.reference_sigma)
        if bool(jnp.any(sigma_arr <= 0.0)):
            raise ValueError(
                "reference_sigma must be strictly positive elementwise; "
                f"min value is {float(jnp.min(sigma_arr))}."
            )

        # --- Loss weights non-negative ---
        for wname in ('alpha_terminal', 'alpha_transport', 'alpha_reference'):
            if float(getattr(self, wname)) < 0.0:
                raise ValueError(
                    f"{wname} must be >= 0, got {getattr(self, wname)}."
                )

        # --- control_names shape if supplied ---
        if self.control_names is not None:
            if len(self.control_names) != int(self.n_controls):
                raise ValueError(
                    f"control_names has {len(self.control_names)} entries, "
                    f"but n_controls = {self.n_controls}."
                )


# =========================================================================
# SCHEDULE (engine output)
# =========================================================================

@dataclass(frozen=True)
class Schedule:
    """The optimised control schedule, plus metadata.

    Attributes:
        theta: Optimised raw policy parameters; shape depends on policy
            kind. For PIECEWISE_CONSTANT, shape (D, n_controls).

        daily_values: u(t_d; theta) evaluated at the start of each day.
            Shape (D, n_controls). For a piecewise-constant policy this
            equals theta exactly; for B-splines it differs.

        horizon_days: Schedule length D.

        n_controls: Control-vector dimension.

        control_names: Human-readable names for each control component,
            e.g. ("V_h", "V_n", "V_c") for SWAT. Required field — callers
            supply explicit names. The engine modules use the helper
            `default_control_names(n)` to generate ("u_0", "u_1", ...)
            when no explicit names are available from the BridgeProblem.

        metadata: Free-form dictionary for diagnostic info (final loss,
            converged flag, etc.).
    """
    theta: jnp.ndarray
    daily_values: jnp.ndarray
    horizon_days: int
    n_controls: int
    control_names: Tuple[str, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)


def default_control_names(n_controls: int) -> Tuple[str, ...]:
    """Generate generic ('u_0', 'u_1', ...) names for n controls.

    Used by engine modules that build Schedule objects when no adapter-
    supplied names are available on the BridgeProblem.

    Args:
        n_controls: Number of control components.

    Returns:
        Tuple of length n_controls with strings 'u_0', 'u_1', ...
    """
    return tuple(f"u_{i}" for i in range(int(n_controls)))


# =========================================================================
# OPTIMISATION TRACE
# =========================================================================

@dataclass(frozen=True)
class OptimisationTrace:
    """Per-step diagnostics from the Adam loop.

    All arrays have length equal to the number of steps actually run
    (which may be less than BridgeProblem.optim_steps if convergence is
    detected early).

    Attributes:
        losses_total: Total loss per step.
        losses_terminal: alpha_terminal * MMD term per step.
        losses_transport: alpha_transport * transport-cost term per step.
        losses_reference: alpha_reference * reference-KL term per step.
        grad_norms: ||grad_theta L||_2 per step.
        converged: True if the convergence check fired before optim_steps.
        n_steps_run: Steps actually executed.
    """
    losses_total: jnp.ndarray
    losses_terminal: jnp.ndarray
    losses_transport: jnp.ndarray
    losses_reference: jnp.ndarray
    grad_norms: jnp.ndarray
    converged: bool
    n_steps_run: int


# =========================================================================
# CLOSED-LOOP RESULT
# =========================================================================

@dataclass(frozen=True)
class ClosedLoopResult:
    """Output of running the latent SDE under the optimised schedule.

    Attributes:
        t: Time grid (days), shape (n_steps + 1,) — includes both
            endpoints, matching simulator output convention.
        trajectories: Latent state trajectories, shape
            (n_realisations, n_steps + 1, dim_state). Includes the
            initial state at index [:, 0, :].
        amplitude_at_D: Amplitude at terminal time, shape (n_realisations,).
        target_samples: Samples from mu_D^A used as the reference
            distribution. Shape (n_realisations,) — the closed-loop
            simulator draws the same number of target samples as
            verification realisations.
        mmd_target: MMD^2 between amplitude_at_D and target_samples
            (scalar Python float).
        fraction_in_healthy_basin: Fraction of trajectories where the
            adapter-supplied basin indicator is True at t = D. NaN if
            the adapter did not supply a basin indicator.
    """
    t: jnp.ndarray
    trajectories: jnp.ndarray
    amplitude_at_D: jnp.ndarray
    target_samples: jnp.ndarray
    mmd_target: float
    fraction_in_healthy_basin: float
