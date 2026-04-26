"""
ot_engine/reference/gaussian_iid.py — Independent Gaussian reference penalty.
==============================================================================
Date:    26 April 2026
Version: 1.1.0

Reference path measure: each (day d, control c) entry of the schedule
has an independent Gaussian prior centred on the adapter's baseline.

    u_{d, c} | reference  ~  N(mu_ref[d, c], sigma_ref[d, c]^2)

Strictly speaking the optimiser commits to a single deterministic
schedule theta, so the "KL divergence" between a delta-function chosen
schedule and a Gaussian prior is infinite. The standard reformulation —
and what we compute — is the negative log-density of theta under the
Gaussian prior, which is equivalent to a MAP penalty:

    -log p(theta | mu_ref, sigma_ref^2)
        = sum_{d, c} (theta[d, c] - mu_ref[d, c])^2 / (2 sigma_ref[d, c]^2)
          + theta-independent normalising constants

Only the quadratic term is theta-dependent; we drop the constants. The
resulting `gaussian_iid_kl` function is therefore a quadratic
regulariser pulling theta toward mu_ref with strength 1/sigma_ref^2.

This is the simplest possible reference and is what v1 uses. Smoother
references (Gaussian random walk, Ornstein-Uhlenbeck, empirical from
healthy populations) are deferred per FUTURE_FEATURES.md.

The function name `gaussian_iid_kl` is retained for API stability; an
alias `gaussian_iid_log_prior_penalty` is exported for callers who want
the more accurate name.
"""

from __future__ import annotations

import jax.numpy as jnp


def gaussian_iid_kl(
    theta: jnp.ndarray,
    mu_ref: jnp.ndarray,
    sigma_ref: jnp.ndarray,
) -> jnp.ndarray:
    """Quadratic Gaussian-prior penalty (a.k.a. iid Gaussian reference KL).

    Returns

        sum_{d, c} (theta[d, c] - mu_ref[d, c])^2 / (2 sigma_ref[d, c]^2)

    which is the only theta-dependent term in the negative log-density
    of theta under the iid Gaussian reference. This is sufficient for
    optimisation; the theta-independent normalising constants drop out
    of jax.grad.

    Pre-conditions (callers must ensure):
        * shapes match
        * sigma_ref is strictly positive elementwise

    BridgeProblem.__post_init__ enforces both at problem-construction
    time, so a problem built through the standard adapter cannot reach
    this function with bad inputs. The shape check here is a defence
    in depth for callers using the function directly.

    Args:
        theta: Candidate schedule, shape (D, n_controls).
        mu_ref: Reference centring, same shape as theta.
        sigma_ref: Reference per-(day, control) standard deviation, same
            shape as theta. Must be strictly positive elementwise.

    Returns:
        Scalar quadratic penalty.

    Raises:
        ValueError: If shapes do not match.
    """
    theta = jnp.asarray(theta)
    mu_ref = jnp.asarray(mu_ref)
    sigma_ref = jnp.asarray(sigma_ref)
    if theta.shape != mu_ref.shape or theta.shape != sigma_ref.shape:
        raise ValueError(
            "Shape mismatch in gaussian_iid_kl: "
            f"theta {theta.shape}, mu_ref {mu_ref.shape}, "
            f"sigma_ref {sigma_ref.shape}"
        )
    diff = theta - mu_ref
    return 0.5 * jnp.sum(diff * diff / (sigma_ref * sigma_ref))


# More accurate alias — same function under a clearer name.
gaussian_iid_log_prior_penalty = gaussian_iid_kl
