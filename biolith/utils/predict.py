from typing import Callable

import jax
from numpyro.infer import Predictive

from .data import dataframes_to_arrays, rename_samples


def predict(
    model_fn: Callable,
    mcmc,
    site_covs=None,
    obs_covs=None,
    obs=None,
    session_duration=None,
    num_samples: int = 1000,
    random_seed: int = 0,
    infer_discrete: bool = False,
    timeout: int | None = None,
    **kwargs,
) -> dict:
    """Generate posterior predictive samples using a fitted model.

    Parameters
    ----------
    model_fn:
        The model function used for inference.
    mcmc:
        An ``MCMC`` object returned by :func:`fit`.
    site_covs:
        Array or Pandas DataFrame containing site-level covariates.
    obs_covs:
        Array or Pandas DataFrame containing observation-level covariates.
    obs:
        Array or Pandas DataFrame containing the observed data.
    session_duration:
        Array or Pandas DataFrame containing the session duration for each observation.
    num_samples:
        Number of predictive draws to generate.
    random_seed:
        Seed used for the random number generator.
    infer_discrete:
        Whether to only sample discrete latent variables.
    timeout:
        Optional timeout (in seconds) for the prediction step.
    **kwargs:
        Additional keyword arguments passed to ``model_fn``.

    Returns
    -------
    dict
        Dictionary of posterior predictive samples.

    Examples
    --------
    >>> from biolith.models import simulate, occu
    >>> from biolith.utils import fit, predict
    >>> data, _ = simulate()
    >>> results = fit(occu, **data, num_samples=10, num_warmup=10, num_chains=1)
    >>> preds = predict(occu, results.mcmc, **data, num_samples=5)
    """

    site_covs, obs_covs, obs, session_duration, site_covs_names, obs_covs_names = (
        dataframes_to_arrays(site_covs, obs_covs, obs, session_duration)
    )

    predictive = Predictive(
        model_fn,
        posterior_samples=mcmc.get_samples(),
        num_samples=num_samples,
        infer_discrete=infer_discrete,
    )
    arguments = dict(
        site_covs=site_covs,
        obs_covs=obs_covs,
        obs=obs,
        session_duration=session_duration,
    )
    valid_arguments = {
        k: v for k, v in arguments.items() if v is not None and k != "obs"
    }
    rng_key = jax.random.PRNGKey(random_seed)

    if timeout is not None:
        from .misc import time_limit

        with time_limit(timeout):
            samples = predictive(rng_key, **valid_arguments, **kwargs)
    else:
        samples = predictive(rng_key, **valid_arguments, **kwargs)

    samples = rename_samples(samples, site_covs_names, obs_covs_names)

    return samples
