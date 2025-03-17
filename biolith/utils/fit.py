from collections import namedtuple
import pandas as pd
import jax
from numpyro.infer import MCMC, HMC, NUTS, MixedHMC, DiscreteHMCGibbs

from .data import dataframes_to_arrays, rename_samples


FitResult = namedtuple("FitResult", ["samples", "mcmc"])


def fit(
        model_fn: callable,
        site_covs=None,
        obs_covs=None,
        obs=None,
        session_duration=None,
        num_samples=1000,
        num_warmup=1000,
        random_seed=0,
        num_chains=5,
        **kwargs,
    ):

    site_covs, obs_covs, obs, session_duration, site_covs_names, obs_covs_names = dataframes_to_arrays(site_covs, obs_covs, obs, session_duration)

    kernel = NUTS(model_fn)
    # kernel = HMC(model_fn)
    # kernel = MixedHMC(HMC(model_fn))
    # kernel = DiscreteHMCGibbs(NUTS(model_fn))
    mcmc = MCMC(kernel, num_samples=num_samples, num_warmup=num_warmup, num_chains=num_chains, chain_method='parallel' if num_chains <= jax.local_device_count() else 'sequential')

    arguments = dict(site_covs=site_covs, obs_covs=obs_covs, obs=obs, session_duration=session_duration)
    mcmc.run(jax.random.PRNGKey(random_seed), **{k: v for k, v in arguments.items() if v is not None}, **kwargs)
    samples = mcmc.get_samples()
    samples = rename_samples(samples, site_covs_names, obs_covs_names)

    return FitResult(samples, mcmc)