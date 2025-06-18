import jax
from numpyro.infer import Predictive

from .data import dataframes_to_arrays, rename_samples


def predict(
        model_fn: callable,
        mcmc,
        site_covs=None,
        obs_covs=None,
        obs=None,
        session_duration=None,
        num_samples=1000,
        random_seed=0,
        infer_discrete=False,
        timeout=None,
        **kwargs,
    ):
    
    site_covs, obs_covs, obs, session_duration, site_covs_names, obs_covs_names = dataframes_to_arrays(site_covs, obs_covs, obs, session_duration)

    predictive = Predictive(model_fn, posterior_samples=mcmc.get_samples(), num_samples=num_samples, infer_discrete=infer_discrete)
    arguments = dict(site_covs=site_covs, obs_covs=obs_covs, obs=obs, session_duration=session_duration)
    valid_arguments = {k: v for k, v in arguments.items() if v is not None and k != "obs"}
    rng_key = jax.random.PRNGKey(random_seed)

    if timeout is not None:
        from .misc import time_limit
        with time_limit(timeout):
            samples = predictive(rng_key, **valid_arguments, **kwargs)
    else:
        samples = predictive(rng_key, **valid_arguments, **kwargs)

    samples = rename_samples(samples, site_covs_names, obs_covs_names)

    return samples