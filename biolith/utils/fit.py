from collections import namedtuple

import jax
from numpyro.infer import HMC, HMCECS, MCMC, NUTS, DiscreteHMCGibbs, MixedHMC

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
    kernel="nuts",
    timeout=None,
    **kwargs,
):

    site_covs, obs_covs, obs, session_duration, site_covs_names, obs_covs_names = (
        dataframes_to_arrays(site_covs, obs_covs, obs, session_duration)
    )

    kernel_inst = dict(
        nuts=lambda: NUTS(model_fn),
        hmc=lambda: HMC(model_fn),
        mixed_hmc=lambda: MixedHMC(HMC(model_fn)),
        discrete_hmc_gibbs=lambda: DiscreteHMCGibbs(NUTS(model_fn)),
        hmcecs=lambda: HMCECS(NUTS(model_fn)),
    )[kernel]()
    mcmc = MCMC(
        kernel_inst,
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_chains=num_chains,
        chain_method=(
            "parallel" if num_chains <= jax.local_device_count() else "sequential"
        ),
    )

    arguments = dict(
        site_covs=site_covs,
        obs_covs=obs_covs,
        obs=obs,
        session_duration=session_duration,
    )
    valid_arguments = {k: v for k, v in arguments.items() if v is not None}
    rng_key = jax.random.PRNGKey(random_seed)

    if timeout is not None:
        from .misc import time_limit

        with time_limit(timeout):
            mcmc.run(rng_key, **valid_arguments, **kwargs)
    else:
        mcmc.run(rng_key, **valid_arguments, **kwargs)

    samples = mcmc.get_samples()
    samples = rename_samples(samples, site_covs_names, obs_covs_names)

    return FitResult(samples, mcmc)
