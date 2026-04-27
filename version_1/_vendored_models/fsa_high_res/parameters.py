"""
_vendored_models/fsa_high_res/parameters.py
============================================
Default parameters for FSA-high-res, all in DAYS.

Source:  Python-Model-Development-Simulation @ main,
         version_1/models/fsa_high_res/simulation.py line 463 onwards.
Vendored: 26 April 2026

Only the dynamics block is vendored here. The observation-model
parameters (HR, sleep, stress, steps) live in the upstream module and
are not used by the OT-Control engine — control is on latent dynamics,
not on observations.

Tuning note (from upstream)
---------------------------
The defaults are tuned for the 14-day proof-of-principle horizon: mu_0
is positive (+0.02) so mu(B, F) stays positive across the recovery
trajectory, putting A near a stable fixed point A* = sqrt(mu/eta) of
roughly 0.7 mid-run. This is *less negative* than the v4.1 spec's
mu_0 = -0.3, which would require a bifurcation crossing to develop
A > 0; under that spec a 14-day horizon is too short to see recovery.
"""

from typing import Dict


def default_fsa_parameters() -> Dict[str, float]:
    """Healthy-baseline 13-parameter dynamics dictionary.

    Returns:
        A dict with these keys (all values float):

          tau_B, alpha_A          -- fitness block (2)
          tau_F, lambda_B,        -- strain block (3)
          lambda_A
          mu_0, mu_B, mu_F,       -- amplitude block (5)
          mu_FF, eta
          sigma_B, sigma_F,       -- frozen process noises (3)
          sigma_A

        Total: 13 parameters. epsilon_A is not in the dict because it
        is hardcoded inside dynamics_jax.fsa_diffusion (frozen at 1e-4
        per the upstream convention).
    """
    return {
        # Fitness block
        'tau_B':    14.0,    # days
        'alpha_A':   1.0,    # 1/amplitude — A boosts adaptation rate

        # Strain block
        'tau_F':     7.0,    # days
        'lambda_B':  3.0,    # B-enhanced recovery
        'lambda_A':  1.5,    # A-enhanced recovery (1/amplitude)

        # Amplitude block (Landau bifurcation parameter mu(B, F))
        'mu_0':      0.02,   # 1/day — baseline (positive, see tuning note)
        'mu_B':      0.30,   # 1/day — fitness drives mu up
        'mu_F':      0.10,   # 1/(day*strain) — strain pulls mu down (linear)
        'mu_FF':     0.40,   # 1/(day*strain^2) — strain pulls mu down (quadratic, the overtraining cliff)
        'eta':       0.20,   # 1/(day*amp^2) — amplitude saturation

        # Frozen process noises (per spec §2 / upstream lines 478-480)
        'sigma_B':   0.01,
        'sigma_F':   0.005,
        'sigma_A':   0.02,
    }
