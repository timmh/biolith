"""Helper functions for linear predictors used in occupancy models."""
from __future__ import annotations

from typing import Optional

import jax.numpy as jnp


def occupancy_linear(beta: jnp.ndarray, site_covs: jnp.ndarray, w: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """Compute the occupancy linear predictor.

    Parameters
    ----------
    beta : jnp.ndarray
        Regression coefficients of shape (n_covs + 1,).
    site_covs : jnp.ndarray
        Site covariate matrix of shape (n_covs, n_sites).
    w : Optional[jnp.ndarray]
        Optional spatial random effect of shape (n_sites,).

    Returns
    -------
    jnp.ndarray
        Linear predictor of shape (n_sites,).
    """
    lin = jnp.tile(beta[0], (site_covs.shape[-1],)) + jnp.dot(beta[1:], site_covs)
    if w is not None:
        lin = lin + w
    return lin


def detection_linear(alpha: jnp.ndarray, obs_covs: jnp.ndarray) -> jnp.ndarray:
    """Compute the detection linear predictor.

    Parameters
    ----------
    alpha : jnp.ndarray
        Regression coefficients of shape (n_covs + 1,).
    obs_covs : jnp.ndarray
        Observation covariate tensor of shape (n_covs, time_periods, n_sites).

    Returns
    -------
    jnp.ndarray
        Linear predictor of shape (time_periods, n_sites).
    """
    return jnp.tile(alpha[0], (obs_covs.shape[1], obs_covs.shape[2])) + jnp.sum(
        alpha[1:, None, None] * obs_covs, axis=0
    )
