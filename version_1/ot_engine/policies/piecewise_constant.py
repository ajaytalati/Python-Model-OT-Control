"""
ot_engine/policies/piecewise_constant.py — Piecewise-constant control schedule.
================================================================================
Date:    25 April 2026
Version: 1.0.0

Schedule parameterised as one constant value per day per control:

    u(t; theta) = theta[d, :]   for t in [d, d+1) days

Parameter shape: theta has shape (D, n_controls) and contains the control
value applied on each of D days.

This is the v1 default. Smoother parameterisations (B-splines, neural
nets) are deferred per FUTURE_FEATURES.md.
"""

from __future__ import annotations

import jax.numpy as jnp

from ot_engine.policies._abstract import ControlPolicy


class PiecewiseConstant(ControlPolicy):
    """Piecewise-constant control schedule with daily resolution.

    Attributes:
        horizon_days: Schedule length D in days.
        n_controls: Control-vector dimension.
    """

    def __init__(self, horizon_days: int, n_controls: int):
        if horizon_days <= 0:
            raise ValueError(f"horizon_days must be positive, got {horizon_days}")
        if n_controls <= 0:
            raise ValueError(f"n_controls must be positive, got {n_controls}")
        self.horizon_days = int(horizon_days)
        self.n_controls = int(n_controls)
        self._validate_attrs()

    def init_params(self, reference_schedule: jnp.ndarray) -> jnp.ndarray:
        """Initialise theta to the reference schedule.

        Args:
            reference_schedule: Adapter's baseline, shape (D, n_controls).

        Returns:
            theta_0 of the same shape, equal to the reference. Preserves
            the input dtype — important when JAX x64 is enabled (the
            engine default), so theta stays in float64 throughout the
            optimisation.

        Raises:
            ValueError: If reference_schedule has the wrong shape.
        """
        expected = (self.horizon_days, self.n_controls)
        ref = jnp.asarray(reference_schedule)
        if ref.shape != expected:
            raise ValueError(
                f"reference_schedule shape mismatch: expected {expected}, "
                f"got {ref.shape}"
            )
        return ref

    def evaluate(self, t: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """Evaluate u(t; theta) at scalar time t in days.

        For t = D exactly (end of horizon), returns the last day's value.
        For t > D, behaviour is undefined and clipped to the last day's value.

        Args:
            t: Scalar time in days (jnp scalar or 0-d array).
            theta: Schedule parameters, shape (D, n_controls).

        Returns:
            Control vector of shape (n_controls,).
        """
        # Floor to integer day index, clipped to [0, D-1].
        d = jnp.clip(jnp.floor(t).astype(jnp.int32), 0, self.horizon_days - 1)
        return theta[d]

    def evaluate_daily(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Daily values are theta itself for piecewise-constant policy.

        Args:
            theta: Schedule parameters, shape (D, n_controls).

        Returns:
            Daily values, shape (D, n_controls).
        """
        return theta

    @property
    def n_params(self) -> int:
        """D * n_controls scalar parameters."""
        return self.horizon_days * self.n_controls
