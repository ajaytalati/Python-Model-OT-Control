"""Vendored minimal SWAT model for OT-Control engine."""

from _vendored_models.swat.dynamics_jax import (
    swat_drift, swat_diffusion, amplitude_of_swat, swat_state_clip,
    entrainment_quality,
)
from _vendored_models.swat.parameters import default_swat_parameters

__all__ = [
    "swat_drift",
    "swat_diffusion",
    "amplitude_of_swat",
    "swat_state_clip",
    "entrainment_quality",
    "default_swat_parameters",
]
