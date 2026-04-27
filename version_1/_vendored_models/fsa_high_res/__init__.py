"""FSA-high-res vendored dynamics for the OT-Control engine."""

from _vendored_models.fsa_high_res.dynamics_jax import (
    fsa_drift,
    fsa_diffusion,
    fsa_state_clip,
    amplitude_of_fsa,
    healthy_attractor_check,
)
from _vendored_models.fsa_high_res.parameters import (
    default_fsa_parameters,
)

__all__ = [
    "fsa_drift",
    "fsa_diffusion",
    "fsa_state_clip",
    "amplitude_of_fsa",
    "healthy_attractor_check",
    "default_fsa_parameters",
]
