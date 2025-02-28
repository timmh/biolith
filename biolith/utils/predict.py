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
        **kwargs,
    ):
    
    site_covs, obs_covs, obs, session_duration, site_covs_names, obs_covs_names = dataframes_to_arrays(site_covs, obs_covs, obs, session_duration)

    params = {k: v.mean(axis=0) for k, v in mcmc.get_samples().items() if k.startswith("beta") or k.startswith("alpha")}
    predictive = Predictive(model_fn, params=params, num_samples=num_samples)
    arguments = dict(site_covs=site_covs, obs_covs=obs_covs, obs=obs, session_duration=session_duration)
    samples = predictive(jax.random.PRNGKey(random_seed), **{k: v for k, v in arguments.items() if v is not None}, **kwargs)
    samples = rename_samples(samples, site_covs_names, obs_covs_names)

    return samples