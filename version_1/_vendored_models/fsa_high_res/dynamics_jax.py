"""
_vendored_models/fsa_high_res/dynamics_jax.py
==============================================
JAX-native dynamics for the 3-state FSA-high-res model.

Source:  Python-Model-Development-Simulation @ main,
         version_1/models/fsa_high_res/simulation.py
         (drift_jax: lines 176-195;  noise_scale_fn_jax: lines 218-226).
Vendored: 26 April 2026
Spec:     model_documentation/fsa_high_res/fsa_high_res_Documentation.md

Differences from upstream
-------------------------
The upstream `drift_jax` reads T_B(t) and Phi(t) from per-bin arrays
(96 bins/day) via integer-index lookup. For the OT-Control engine the
controls are scalar daily values supplied by the policy
(`PiecewiseConstant.evaluate(t, theta)`), so this vendored copy takes
`u = (T_B, Phi)` directly and skips the bin-lookup machinery. The high-
resolution aspect of the upstream simulator is for the observation
channel (15-min HR / sleep / stress / steps), not the control. The
control schedule is naturally daily.

Time is in days throughout (engine convention). The upstream simulator
also uses days, so no unit conversion is needed.

State convention: x = (B, F, A) where
    B in [0, 1]   fitness                (Jacobi diffusion)
    F >= 0        strain                 (CIR diffusion)
    A >= 0        endocrine amplitude    (regularised Landau)

Control convention: u = (T_B, Phi) where
    T_B in [0, 1] training-load target
    Phi >= 0      training intensity / strain production
"""

from __future__ import annotations

from typing import Dict, Optional

import jax.numpy as jnp


# Frozen non-absorbing-boundary regularisers. Match upstream
# (simulation.py line 42-43). Not estimable; not configurable.
EPS_A_FROZEN = 1.0e-4
EPS_B_FROZEN = 1.0e-4


def _bifurcation_parameter(B: jnp.ndarray, F: jnp.ndarray,
                            params: Dict[str, float]) -> jnp.ndarray:
    """mu(B, F) — Stuart-Landau bifurcation parameter for the A equation.

    From spec eq. (mu):  mu = mu_0 + mu_B*B - mu_F*F - mu_FF*F^2.
    Positive mu => super-critical Hopf, A is attracted to A* = sqrt(mu/eta).
    Negative mu => sub-critical, A decays to 0.

    Args:
        B: fitness, jnp scalar.
        F: strain, jnp scalar.
        params: model parameter dict.

    Returns:
        mu, jnp scalar.
    """
    return (params['mu_0']
            + params['mu_B'] * B
            - params['mu_F'] * F
            - params['mu_FF'] * (F ** 2))


def fsa_drift(t: jnp.ndarray,
              x: jnp.ndarray,
              u: jnp.ndarray,
              params: Dict[str, float]) -> jnp.ndarray:
    """Drift f(t, x, u) for the FSA-high-res SDE.

    Verbatim from upstream `drift_jax` (simulation.py lines 187-194)
    except controls come from `u` directly rather than bin-array lookup.

    Args:
        t: scalar time in days. Drift is autonomous in t (model has no
            explicit time dependence), so this is unused.
        x: latent state (B, F, A), shape (3,).
        u: control vector (T_B, Phi), shape (2,).
        params: model parameter dict.

    Returns:
        Drift vector (dB, dF, dA), shape (3,).
    """
    del t                                    # autonomous in t
    B, F, A = x[0], x[1], x[2]
    T_B, Phi = u[0], u[1]

    # Effective fitness adaptation rate boosted by amplitude:
    #   1/tau_B^eff = (1 + alpha_A * A) / tau_B
    inv_tau_B_eff = (1.0 + params['alpha_A'] * A) / params['tau_B']
    dB = inv_tau_B_eff * (T_B - B)

    # Effective strain recovery rate boosted by fitness and amplitude:
    #   1/tau_F^eff = (1 + lambda_B * B + lambda_A * A) / tau_F
    inv_tau_F_eff = ((1.0 + params['lambda_B'] * B
                      + params['lambda_A'] * A) / params['tau_F'])
    dF = Phi - inv_tau_F_eff * F

    # Stuart-Landau amplitude with cubic regularisation
    mu = _bifurcation_parameter(B, F, params)
    dA = mu * A - params['eta'] * (A ** 3)

    return jnp.array([dB, dF, dA])


def fsa_diffusion(x: jnp.ndarray,
                  params: Dict[str, float]) -> jnp.ndarray:
    """Per-component diffusion sigma_i(x) for diagonal noise.

    Adapted from upstream `noise_scale_fn_jax` (simulation.py lines
    218-226). The upstream form is forward-only:

        sigma_B = sigma_B * sqrt(B*(1-B))     with B clipped to [eps, 1-eps]
        sigma_F = sigma_F * sqrt(F)           with F clipped to >= 0
        sigma_A = sigma_A * sqrt(A + eps_A)   no clip needed

    The clip-then-sqrt form has a +inf gradient at the boundary
    (d/dB[sqrt(B)] = 1/(2*sqrt(B))), which crashes JAX-AD when the
    state hits the boundary at any point during the trajectory. The
    OT-Control engine traces gradients through `simulate_latent`, so
    we need a smooth boundary regularisation. Hence:

        sigma_B = sigma_B * sqrt(B*(1-B) + eps_B)
        sigma_F = sigma_F * sqrt(F + eps_B)
        sigma_A = sigma_A * sqrt(A + eps_A)        (matches upstream)

    The eps regularisation keeps the argument of sqrt strictly positive,
    so the gradient is bounded. eps = 1e-4 (the same constant upstream
    uses for A) is small enough to be physiologically negligible and
    large enough to keep gradients finite.

    State-dependent forms:
      B: Jacobi sqrt(B*(1-B) + eps_B)
      F: CIR    sqrt(F + eps_B)
      A: Landau sqrt(A + eps_A)

    Args:
        x: latent state, shape (3,).
        params: model parameter dict.

    Returns:
        Per-component sigma, shape (3,).
    """
    B = x[0]
    F = jnp.maximum(x[1], 0.0)
    A = jnp.maximum(x[2], 0.0)
    sigma_B = params['sigma_B'] * jnp.sqrt(B * (1.0 - B) + EPS_B_FROZEN)
    sigma_F = params['sigma_F'] * jnp.sqrt(F + EPS_B_FROZEN)
    sigma_A = params['sigma_A'] * jnp.sqrt(A + EPS_A_FROZEN)
    return jnp.array([sigma_B, sigma_F, sigma_A])


def fsa_state_clip(x: jnp.ndarray,
                   params: Optional[Dict[str, float]] = None) -> jnp.ndarray:
    """Physical-bounds clip applied after each Euler-Maruyama step.

    B in [0, 1] (Jacobi domain), F >= 0 (CIR domain), A >= 0 (Landau
    with non-absorbing boundary). Clipping after each EM step prevents
    occasional Euler excursions across the boundaries.

    Args:
        x: latent state, shape (3,).
        params: unused, kept for signature consistency with SWAT's
            state_clip_fn that does need params.

    Returns:
        Clipped state, shape (3,).
    """
    del params
    return jnp.array([
        jnp.clip(x[0], 0.0, 1.0),    # B
        jnp.maximum(x[1], 0.0),      # F
        jnp.maximum(x[2], 0.0),      # A
    ])


def amplitude_of_fsa(x: jnp.ndarray) -> jnp.ndarray:
    """Project the latent state onto the scalar amplitude variable.

    A is index 2 in the (B, F, A) state vector. This is the variable
    the OT engine MMD-matches against the clinical target distribution.

    Args:
        x: latent state, shape (3,).

    Returns:
        A, jnp scalar.
    """
    return x[2]


def healthy_attractor_check(B: jnp.ndarray, F: jnp.ndarray,
                             params: Dict[str, float]) -> jnp.ndarray:
    """True iff mu(B, F) > 0 — the Stuart-Landau bifurcation is super-critical.

    Used by the basin indicator. When mu > 0 the deterministic
    A-dynamics have a stable fixed point A* > 0; the patient's
    physiology supports a healthy endocrine amplitude.

    Args:
        B: fitness, jnp scalar.
        F: strain, jnp scalar.
        params: model parameter dict.

    Returns:
        Boolean jnp scalar.
    """
    return _bifurcation_parameter(B, F, params) > 0.0
