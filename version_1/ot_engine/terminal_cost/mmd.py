"""
ot_engine/terminal_cost/mmd.py — Maximum Mean Discrepancy.
==========================================================
Date:    25 April 2026
Version: 1.0.0

Sample-based MMD with Gaussian kernel:

    MMD^2(X, Y) = 1/N^2 sum_{i,i'} k(x_i, x_i')
                + 1/M^2 sum_{j,j'} k(y_j, y_j')
                - 2/(NM) sum_{i,j} k(x_i, y_j)

with k(x, y) = exp(-(x - y)^2 / (2 h^2)) for scalar arguments, and the
sum-of-squared-distances generalisation for vector arguments.

Bandwidth h defaults to the **median heuristic**: h = median pairwise
distance among the union {X, Y}. This is parameter-free in practice and
robust across orders of magnitude.

The implementation is designed so jax.grad propagates cleanly: every
operation is differentiable in X (the simulated terminal samples).
The target samples Y are treated as constants.

References:
    Gretton et al. (2012). "A Kernel Two-Sample Test." JMLR.
    Onken et al. (2021). "OT-Flow." (Uses MMD for evaluation.)
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp


def _pairwise_sq_dists(X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
    """Pairwise squared Euclidean distances ||X_i - Y_j||^2.

    Works for either scalar samples (1-D arrays) or vector samples
    (2-D arrays of shape (N, d)).

    Args:
        X: Shape (N,) or (N, d).
        Y: Shape (M,) or (M, d). Must match X's rank.

    Returns:
        Pairwise distances, shape (N, M).
    """
    # Promote 1-D scalars to (N, 1) for uniform handling.
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    # ||X_i - Y_j||^2 = ||X_i||^2 + ||Y_j||^2 - 2 <X_i, Y_j>
    X_sq = jnp.sum(X * X, axis=1, keepdims=True)        # (N, 1)
    Y_sq = jnp.sum(Y * Y, axis=1, keepdims=True).T      # (1, M)
    cross = X @ Y.T                                      # (N, M)
    return X_sq + Y_sq - 2.0 * cross


def median_bandwidth(X: jnp.ndarray, Y: jnp.ndarray) -> jnp.ndarray:
    """Median pairwise distance among the union {X, Y}.

    The Gaussian kernel bandwidth h is set so that the typical
    inter-sample distance corresponds to ~one bandwidth, giving a kernel
    matrix with entries spanning the dynamic range of exp(-1/2) ≈ 0.6.

    Floored at a small positive value to avoid h = 0 when X and Y are
    near-identical (which would zero out the kernel and the gradient).

    Args:
        X: Shape (N,) or (N, d).
        Y: Shape (M,) or (M, d).

    Returns:
        Bandwidth scalar (jnp scalar).
    """
    Z = jnp.concatenate([X.ravel() if X.ndim == 1 else X,
                         Y.ravel() if Y.ndim == 1 else Y], axis=0)
    if Z.ndim == 1:
        Z = Z[:, None]
    # Pairwise squared distances; take square-root and median.
    sq = _pairwise_sq_dists(Z, Z)
    n = Z.shape[0]
    flat = sq.ravel()
    sorted_dists = jnp.sort(flat)
    # The diagonal contributes n zeros at sorted positions [0, n). The
    # off-diagonal entries are at sorted positions [n, n*n). The median
    # of those n*n - n entries lands close to (n*n + n) // 2 of the
    # sorted-all array. This is one of the two median entries — close
    # enough as a kernel-bandwidth heuristic.
    idx = (n * n + n) // 2
    med_sq = sorted_dists[idx]
    # Floor to keep numerically well-behaved when X ~ Y.
    return jnp.sqrt(jnp.maximum(med_sq, 1e-8))


def mmd_squared(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    bandwidth: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Biased empirical MMD^2 with Gaussian kernel.

    The biased estimator (sum over all i, i' including i = i') is used
    rather than the unbiased one because it is non-negative and matches
    the OT-Flow paper's evaluation. The bias is O(1/N) and vanishes as
    N -> infinity.

    Args:
        X: Simulated samples, shape (N,) or (N, d). The differentiable
            input — gradients flow through here.
        Y: Target samples, shape (M,) or (M, d). Treated as constant.
        bandwidth: Gaussian-kernel bandwidth h. If None, set to the
            median heuristic over {X, Y}.

    Returns:
        MMD^2 scalar (jnp scalar). Non-negative.
    """
    if bandwidth is None:
        bandwidth = median_bandwidth(X, Y)

    h_sq = bandwidth ** 2
    # Three pairwise-kernel sums.
    K_XX = jnp.exp(-_pairwise_sq_dists(X, X) / (2.0 * h_sq))
    K_YY = jnp.exp(-_pairwise_sq_dists(Y, Y) / (2.0 * h_sq))
    K_XY = jnp.exp(-_pairwise_sq_dists(X, Y) / (2.0 * h_sq))

    n = K_XX.shape[0]
    m = K_YY.shape[0]
    return (jnp.sum(K_XX) / (n * n)
            + jnp.sum(K_YY) / (m * m)
            - 2.0 * jnp.sum(K_XY) / (n * m))
