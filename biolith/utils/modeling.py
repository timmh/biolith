from contextlib import contextmanager
import math

import jax.numpy as jnp
from numpyro.handlers import mask


@contextmanager
def mask_missing_obs(obs=None):
    """Context manager to mask missing observations (NaNs) in numpyro models.

    Args:
        obs: Observation array with possible NaNs representing missing data.
    """
    if obs is not None:
        with mask(mask=jnp.isfinite(obs)):
            yield
    else:
        yield


def flatten_covariates(covs: jnp.ndarray) -> tuple[jnp.ndarray, tuple[int, ...]]:
    """Flatten covariates from (n_covs, *obs_shape) to (n_obs, n_covs)."""
    if covs.ndim < 2:
        raise ValueError(f"Covariates must be at least 2D, got shape {covs.shape}.")
    obs_shape = covs.shape[1:]
    covs_flat = covs.reshape(covs.shape[0], -1).T
    return covs_flat, obs_shape


def reshape_predictions(preds: jnp.ndarray, obs_shape: tuple[int, ...]) -> jnp.ndarray:
    """Reshape predictions from (n_obs, *batch_shape) back to (*obs_shape, *batch_shape)."""
    batch_shape = preds.shape[1:] if preds.ndim > 1 else ()
    if preds.shape[0] != math.prod(obs_shape):
        raise ValueError(
            f"Prediction length {preds.shape[0]} does not match obs_shape {obs_shape}."
        )
    return preds.reshape(obs_shape + batch_shape)
