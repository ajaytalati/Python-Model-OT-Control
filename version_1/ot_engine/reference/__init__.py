"""ot_engine.reference — reference path measure variants."""

from ot_engine.reference.gaussian_iid import (
    gaussian_iid_kl,
    gaussian_iid_log_prior_penalty,
)

__all__ = ["gaussian_iid_kl", "gaussian_iid_log_prior_penalty"]
