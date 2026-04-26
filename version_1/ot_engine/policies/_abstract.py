"""
ot_engine/policies/_abstract.py — Control-policy interface.
============================================================
Date:    26 April 2026
Version: 1.1.0

Defines the abstract contract that every control parameterisation must
honour. v1 ships PiecewiseConstant; future kinds (B-spline, neural net)
plug in by subclassing ControlPolicy.

A control policy maps schedule parameters theta to a time-varying control
function u(t; theta). The engine treats it as a black box exposing four
methods plus two required attributes:

    horizon_days, n_controls          — set in __init__
    init_params(reference_schedule)   -> theta_0
    evaluate(t, theta)                -> u(t)
    evaluate_daily(theta)             -> u sampled at start of each day
    n_params                          -> total scalar parameter count
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax.numpy as jnp


class ControlPolicy(ABC):
    """Abstract control policy. Subclasses provide concrete parameterisations.

    Subclasses MUST set `self.horizon_days` and `self.n_controls` in
    their `__init__`. The base class enforces this with a check after
    subclass construction (run via `_validate_attrs`); subclasses
    should call this at the end of their own __init__ for an early
    failure on misuse.

    Attributes:
        horizon_days: Schedule length D.
        n_controls: Control-vector dimension.
    """

    horizon_days: int
    n_controls: int

    def _validate_attrs(self) -> None:
        """Verify the required attributes have been set by the subclass.

        Subclasses should call this at the end of their __init__ to fail
        loudly if they forgot to set horizon_days or n_controls.
        """
        for attr in ('horizon_days', 'n_controls'):
            if not hasattr(self, attr):
                raise AttributeError(
                    f"{type(self).__name__} must set self.{attr} in __init__."
                )
            val = getattr(self, attr)
            if not isinstance(val, int) or val < 1:
                raise ValueError(
                    f"{type(self).__name__}.{attr} must be a positive int, "
                    f"got {val!r}."
                )

    @abstractmethod
    def init_params(self, reference_schedule: jnp.ndarray) -> jnp.ndarray:
        """Initialise theta from the reference baseline.

        Args:
            reference_schedule: Adapter's baseline schedule, shape
                (horizon_days, n_controls).

        Returns:
            theta_0 with the policy's natural shape.
        """

    @abstractmethod
    def evaluate(self, t: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """Evaluate u(t; theta) at a single time t (in days).

        Args:
            t: Scalar time in days, in [0, horizon_days).
            theta: Policy parameters with the policy's natural shape.

        Returns:
            Control vector of shape (n_controls,).
        """

    @abstractmethod
    def evaluate_daily(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Evaluate u at the start of each day.

        Args:
            theta: Policy parameters.

        Returns:
            Daily control values, shape (horizon_days, n_controls).
        """

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Total number of scalar parameters in theta."""
