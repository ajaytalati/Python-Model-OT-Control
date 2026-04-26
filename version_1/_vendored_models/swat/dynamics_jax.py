"""
_vendored_models/swat/dynamics_jax.py — SWAT SDE in JAX.
=========================================================
Date:    26 April 2026
Version: 1.0.0
Source:  SWAT_Basic_Documentation.md §4 (SDE system) and §5 (E_dyn).

JAX-native drift and diffusion functions for the four-state SWAT SDE,
plus the entrainment-quality helper E_dyn.

State vector convention used throughout
---------------------------------------
    x[0] = W       wakefulness                in [0, 1]
    x[1] = Z       sleep depth (rescaled)     in [0, A_scale = 6]
    x[2] = a       adenosine                  >= 0
    x[3] = T       testosterone amplitude     >= 0

Control vector convention
-------------------------
    u[0] = V_h     vitality reserve           dimensionless
    u[1] = V_n     chronic load               dimensionless, >= 0
    u[2] = V_c     phase shift                hours, in [-12, 12]

Time convention
---------------
    t in DAYS throughout. The circadian formula sin(2*pi*t + phi_0) has
    period 1 day. Any V_c in hours is converted to days inside the drift.
"""

from __future__ import annotations

from typing import Dict, Optional

import jax.numpy as jnp


# =========================================================================
# Helpers
# =========================================================================

def _sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable logistic sigmoid."""
    return 1.0 / (1.0 + jnp.exp(-x))


def _circadian(t_days: jnp.ndarray, V_c_hours: jnp.ndarray,
               phi_0: float) -> jnp.ndarray:
    """C_eff(t) = sin(2*pi*(t - V_c/24) + phi_0).

    With t in days and V_c in hours, the V_c/24 conversion turns it into
    days for the phase shift.

    Args:
        t_days: Scalar time in days.
        V_c_hours: Scalar phase shift in hours (control u[2]).
        phi_0: Baseline circadian phase, typically -pi/3.

    Returns:
        Scalar in [-1, 1].
    """
    return jnp.sin(2.0 * jnp.pi * (t_days - V_c_hours / 24.0) + phi_0)


def entrainment_quality(W: jnp.ndarray, Z: jnp.ndarray, a: jnp.ndarray,
                         T: jnp.ndarray, V_h: jnp.ndarray, V_n: jnp.ndarray,
                         V_c: jnp.ndarray, params: Dict[str, float]
                         ) -> jnp.ndarray:
    """E_dyn from §5.1 of the SWAT spec.

    E_dyn = amp_W * amp_Z * phase(V_c)
    where amp_i = 4 * sigma(mu_i^slow) * (1 - sigma(mu_i^slow))
    and   phase(V_c) = max(cos(2*pi*V_c/24), 0).

    Args:
        W, Z, a, T: Latent state components (scalars or matching arrays).
        V_h, V_n, V_c: Control values (scalars or matching arrays).
        params: SWAT parameter dictionary.

    Returns:
        Scalar (or matching-shape array) in [0, 1].
    """
    alpha_T = params['alpha_T']
    beta_Z = params['beta_Z']

    mu_W_slow = V_h + V_n - a + alpha_T * T
    mu_Z_slow = -V_n + beta_Z * a
    s_W = _sigmoid(mu_W_slow)
    s_Z = _sigmoid(mu_Z_slow)
    amp_W = 4.0 * s_W * (1.0 - s_W)
    amp_Z = 4.0 * s_Z * (1.0 - s_Z)
    phase = jnp.maximum(jnp.cos(2.0 * jnp.pi * V_c / 24.0), 0.0)
    return amp_W * amp_Z * phase


# =========================================================================
# Drift and diffusion
# =========================================================================

def swat_drift(t: jnp.ndarray, x: jnp.ndarray, u: jnp.ndarray,
               params: Dict[str, float]) -> jnp.ndarray:
    """SWAT drift, JAX-native.

    Args:
        t: Scalar time in days.
        x: Latent state, shape (4,) = (W, Z, a, T).
        u: Control vector, shape (3,) = (V_h, V_n, V_c).
        params: SWAT parameter dictionary.

    Returns:
        dx/dt, shape (4,).
    """
    W, Z, a, T = x[0], x[1], x[2], x[3]
    V_h, V_n, V_c = u[0], u[1], u[2]

    kappa = params['kappa']
    lmbda = params['lmbda']
    gamma_3 = params['gamma_3']
    beta_Z = params['beta_Z']
    A_scale = params['A_scale']
    phi_0 = params['phi_0']
    tau_W = params['tau_W']
    tau_Z = params['tau_Z']
    tau_a = params['tau_a']
    tau_T = params['tau_T']
    mu_0 = params['mu_0']
    mu_E = params['mu_E']
    eta = params['eta']
    alpha_T = params['alpha_T']

    # Wakefulness sigmoid argument.
    C_eff = _circadian(t, V_c, phi_0)
    u_W = lmbda * C_eff + V_h + V_n - a - kappa * Z + alpha_T * T
    u_Z = -gamma_3 * W - V_n + beta_Z * a

    dW = (_sigmoid(u_W) - W) / tau_W
    dZ = (A_scale * _sigmoid(u_Z) - Z) / tau_Z
    da = (W - a) / tau_a

    # Stuart-Landau testosterone.
    E_dyn = entrainment_quality(W, Z, a, T, V_h, V_n, V_c, params)
    mu = mu_0 + mu_E * E_dyn
    dT = (mu * T - eta * T ** 3) / tau_T

    return jnp.stack([dW, dZ, da, dT])


def swat_diffusion(x: jnp.ndarray,
                    params: Dict[str, float]) -> jnp.ndarray:
    """SWAT diffusion (per-component noise amplitudes), JAX-native.

    Returns sigma_i(x) such that the Euler-Maruyama update is
        x_{t+dt}[i] = x_t[i] + drift_i * dt + sigma_i(x) * sqrt(dt) * xi
    with xi ~ N(0, 1) per component.

    Args:
        x: Latent state, shape (4,).
        params: SWAT parameter dictionary.

    Returns:
        Diagonal noise vector, shape (4,).
    """
    del x  # Diffusion is state-independent in SWAT.
    return jnp.array([
        jnp.sqrt(2.0 * params['T_W']),
        jnp.sqrt(2.0 * params['T_Z']),
        jnp.sqrt(2.0 * params['T_a']),
        jnp.sqrt(2.0 * params['T_T']),
    ])


def amplitude_of_swat(x: jnp.ndarray) -> jnp.ndarray:
    """Project the latent state onto its amplitude scalar (testosterone T).

    Args:
        x: Latent state, shape (4,).

    Returns:
        Scalar testosterone amplitude.
    """
    return x[3]


def swat_state_clip(x: jnp.ndarray,
                     params: Optional[Dict[str, float]] = None) -> jnp.ndarray:
    """Clip the SWAT latent state to its physical domain.

    The SDE solver can push states outside the physical range due to
    Euler-Maruyama discretisation noise; this function restores them.
    Used as BridgeProblem.state_clip_fn in the SWAT adapter.

    Constraints (from SWAT spec):
        W in [0, 1]
        Z in [0, A_scale]   (A_scale = 6.0 in default parameters)
        a >= 0
        T >= 0

    Args:
        x: Latent state, shape (4,).
        params: SWAT parameter dictionary. If None, defaults A_scale=6.0
            for backward compatibility with the BridgeProblem.state_clip_fn
            signature, which only takes the state. The adapter wraps this
            function in a closure that pre-binds the params dict so the
            clip stays consistent with the model's configured A_scale.

    Returns:
        Clipped state, shape (4,).
    """
    A_scale = 6.0 if params is None else float(params['A_scale'])
    return jnp.array([
        jnp.clip(x[0], 0.0, 1.0),         # W
        jnp.clip(x[1], 0.0, A_scale),     # Z
        jnp.maximum(x[2], 0.0),           # a
        jnp.maximum(x[3], 0.0),           # T
    ])
