from contextlib import contextmanager

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
