"""ot_engine — Generic OT/control engine for the SMC2 estimation-control pipeline.

Phase 1+2+3+5 public API:
    BridgeProblem, Schedule, OptimisationTrace, ClosedLoopResult — data contracts
    PolicyKind, TerminalCostKind, ReferenceKind                   — variant enums
    default_control_names                                         — naming helper
    PiecewiseConstant                                             — control policy
    mmd_squared, median_bandwidth                                 — terminal cost
    gaussian_iid_kl, gaussian_iid_log_prior_penalty               — reference penalty
    simulate_latent                                               — JAX EM simulator
    make_loss_fn, transport_cost_piecewise_constant               — loss
    optimise_schedule                                             — Adam loop
    convergence_check, summarise_trace                            — diagnostics
    simulate_closed_loop                                          — verification
    zero_control_schedule, constant_reference_schedule,
    linear_interpolation_schedule                                 — naive baselines
    run_ot_pipeline, compare_schedules                            — top-level

NUMERICAL PRECISION
-------------------
The engine enables JAX's float64 mode globally on import. The optimisation
loop accumulates many small Adam steps and the loss involves median-based
kernel bandwidths whose numerical conditioning is poor in float32. CPU JAX
runs float64 at ~half the speed of float32 — acceptable given the modest
problem sizes (theta has ~30-60 entries; trajectories ~10^4 floats).

If you need float32 (e.g. for GPU runs), unset this BEFORE importing
ot_engine: `from jax import config; config.update("jax_enable_x64", False)`.
"""

# Enable float64 globally before any jnp arrays are constructed.
from jax import config as _jax_config
_jax_config.update("jax_enable_x64", True)

from ot_engine.types import (
    BridgeProblem,
    Schedule,
    OptimisationTrace,
    ClosedLoopResult,
    PolicyKind,
    TerminalCostKind,
    ReferenceKind,
    default_control_names,
)
from ot_engine.policies import ControlPolicy, PiecewiseConstant
from ot_engine.terminal_cost import mmd_squared, median_bandwidth
from ot_engine.reference import gaussian_iid_kl, gaussian_iid_log_prior_penalty
from ot_engine.simulator import simulate_latent
from ot_engine.loss import make_loss_fn, transport_cost_piecewise_constant
from ot_engine.diagnostics import convergence_check, summarise_trace
from ot_engine.optimise import optimise_schedule
from ot_engine.closed_loop import simulate_closed_loop
from ot_engine.compare import (
    zero_control_schedule,
    constant_reference_schedule,
    linear_interpolation_schedule,
)
from ot_engine.pipeline import run_ot_pipeline, compare_schedules

__version__ = "1.1.0"

__all__ = [
    # Data contracts
    "BridgeProblem",
    "Schedule",
    "OptimisationTrace",
    "ClosedLoopResult",
    # Enums
    "PolicyKind",
    "TerminalCostKind",
    "ReferenceKind",
    # Naming helper
    "default_control_names",
    # Policies
    "ControlPolicy",
    "PiecewiseConstant",
    # Terminal cost
    "mmd_squared",
    "median_bandwidth",
    # Reference
    "gaussian_iid_kl",
    "gaussian_iid_log_prior_penalty",
    # Simulator
    "simulate_latent",
    # Loss
    "make_loss_fn",
    "transport_cost_piecewise_constant",
    # Optimisation
    "optimise_schedule",
    # Diagnostics
    "convergence_check",
    "summarise_trace",
    # Closed-loop verification
    "simulate_closed_loop",
    # Naive baselines
    "zero_control_schedule",
    "constant_reference_schedule",
    "linear_interpolation_schedule",
    # Top-level pipeline
    "run_ot_pipeline",
    "compare_schedules",
]
