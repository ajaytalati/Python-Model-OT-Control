"""
ot_engine/policies/piecewise_constant.py — Piecewise-constant control schedule.
================================================================================
Date:    26 April 2026
Version: 1.2.0

Schedule parameterised as one constant value per day per control:

    u(t; theta) = clip(theta[d, :], bounds_lo, bounds_hi)
                  for t in [d, d+1) days

Parameter shape: theta has shape (D, n_controls) and contains the control
value applied on each of D days. When `control_bounds` is supplied at
construction time, the policy clips the *applied* control to the bounds.
Adam updates outside the box have a sub-gradient of zero past the
boundary, so the optimiser sees the bound binding and stops pushing
past it. Clipping in the policy (rather than only in the simulator)
ensures `evaluate_daily(theta)` returns what the simulator actually
applies — the displayed schedule matches the executed schedule.

This is the v1 default. Smoother parameterisations (B-splines, neural
nets) are deferred per docs/Future_Features.md (F1).
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax.numpy as jnp

from ot_engine.policies._abstract import ControlPolicy


class PiecewiseConstant(ControlPolicy):
    """Piecewise-constant control schedule with daily resolution.

    Attributes:
        horizon_days: Schedule length D in days.
        n_controls: Control-vector dimension.
        control_bounds: Optional per-control box bounds. If supplied,
            the applied control is clipped to these bounds in
            `evaluate` and `evaluate_daily`.
    """

    def __init__(
        self,
        horizon_days: int,
        n_controls: int,
        control_bounds: Optional[Tuple[Tuple[float, float], ...]] = None,
    ):
        if horizon_days <= 0:
            raise ValueError(f"horizon_days must be positive, got {horizon_days}")
        if n_controls <= 0:
            raise ValueError(f"n_controls must be positive, got {n_controls}")
        self.horizon_days = int(horizon_days)
        self.n_controls = int(n_controls)
        self.control_bounds = control_bounds
        if control_bounds is not None:
            if len(control_bounds) != self.n_controls:
                raise ValueError(
                    f"control_bounds has {len(control_bounds)} entries "
                    f"but n_controls = {self.n_controls}."
                )
            for i, (lo, hi) in enumerate(control_bounds):
                if float(lo) >= float(hi):
                    raise ValueError(
                        f"control_bounds[{i}] = ({lo}, {hi}) — lo must be < hi."
                    )
            self._bounds_lo = jnp.asarray(
                [b[0] for b in control_bounds], dtype=jnp.float64
            )
            self._bounds_hi = jnp.asarray(
                [b[1] for b in control_bounds], dtype=jnp.float64
            )
        else:
            self._bounds_lo = None
            self._bounds_hi = None
        self._validate_attrs()

    @classmethod
    def from_problem(cls, problem) -> "PiecewiseConstant":
        """Convenience constructor that pulls horizon and bounds from a problem.

        Args:
            problem: A BridgeProblem.

        Returns:
            A PiecewiseConstant configured from the problem's
            horizon_days, n_controls, and control_bounds.
        """
        return cls(
            horizon_days=problem.horizon_days,
            n_controls=problem.n_controls,
            control_bounds=problem.control_bounds,
        )

    def _clip(self, u: jnp.ndarray) -> jnp.ndarray:
        """Clip u to control_bounds if bounds were supplied; identity otherwise."""
        if self._bounds_lo is None:
            return u
        return jnp.clip(u, self._bounds_lo, self._bounds_hi)

    def init_params(self, reference_schedule: jnp.ndarray) -> jnp.ndarray:
        """Initialise theta to the reference schedule.

        Args:
            reference_schedule: Adapter's baseline, shape (D, n_controls).

        Returns:
            theta_0 of the same shape, equal to the reference. Preserves
            the input dtype — important when JAX x64 is enabled (the
            engine default), so theta stays in float64 throughout the
            optimisation. The reference is NOT pre-clipped to bounds;
            the adapter is expected to provide a reference inside the
            bounds.

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
        Result is then clipped to control_bounds if bounds were supplied.

        Args:
            t: Scalar time in days (jnp scalar or 0-d array).
            theta: Schedule parameters, shape (D, n_controls).

        Returns:
            Control vector of shape (n_controls,).
        """
        d = jnp.clip(jnp.floor(t).astype(jnp.int32), 0, self.horizon_days - 1)
        return self._clip(theta[d])

    def evaluate_daily(self, theta: jnp.ndarray) -> jnp.ndarray:
        """Daily values are clip(theta) for piecewise-constant policy.

        Args:
            theta: Schedule parameters, shape (D, n_controls).

        Returns:
            Daily values, shape (D, n_controls). Clipped to
            control_bounds if bounds were supplied — the returned array
            is what the simulator actually applies as the control,
            consistent with `evaluate(t, theta)`.
        """
        return self._clip(theta)

    @property
    def n_params(self) -> int:
        """D * n_controls scalar parameters."""
        return self.horizon_days * self.n_controls
