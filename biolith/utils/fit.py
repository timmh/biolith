from collections import namedtuple
import pandas as pd
import jax
from numpyro.infer import MCMC, HMC, NUTS, MixedHMC, DiscreteHMCGibbs


FitResult = namedtuple("FitResult", ["samples", "mcmc"])

def fit(model_fn, site_covs=None, obs_covs=None, obs=None, session_duration=None, num_samples=1000, num_warmup=1000, random_seed=0, num_chains=1, **kwargs):
    
    kernel = NUTS(model_fn)

    # kernel = HMC(model_fn)
    
    # kernel = MixedHMC(HMC(model_fn))

    # kernel = DiscreteHMCGibbs(NUTS(model_fn))
    
    mcmc = MCMC(kernel, num_samples=num_samples, num_warmup=num_warmup, num_chains=num_chains, chain_method='parallel' if num_chains <= jax.local_device_count() else 'sequential')

    # convert dataframes to numpy arrays
    site_covs_names = None
    obs_covs_names = None
    if isinstance(site_covs, pd.DataFrame):
        site_covs_names = ["intercept"] + [c for c in site_covs.columns]
        site_covs = site_covs.sort_index().to_numpy()
    if isinstance(obs_covs, pd.DataFrame):
        if not isinstance(obs_covs.columns, pd.MultiIndex):
            obs_covs = obs_covs.sort_index().to_numpy()
        else:
            assert len(obs_covs.columns.levels) == 2, "obs_covs with MultiIndex columns must have columns of exactly two levels"
            obs_covs_names = ["intercept"] + [c for c in obs_covs.columns.levels[0]]
            obs_covs = obs_covs.sort_index().to_numpy().reshape(obs_covs.shape[0], len(obs_covs.columns.levels[0]), len(obs_covs.columns.levels[1])).transpose(0, 2, 1)
    if isinstance(session_duration, pd.DataFrame):
        session_duration = session_duration.sort_index().to_numpy()
    if isinstance(obs, pd.DataFrame):
        obs = obs.sort_index().to_numpy()

    if site_covs_names is None:
        site_covs_names = [str(0)] + [str(i + 1) for i in range(site_covs.shape[1])]
    if obs_covs_names is None:
        obs_covs_names = [str(0)] + [str(i + 1) for i in range(obs_covs.shape[2])]
    
    arguments = dict(site_covs=site_covs, obs_covs=obs_covs, obs=obs, session_duration=session_duration)
    mcmc.run(jax.random.PRNGKey(random_seed), **{k: v for k, v in arguments.items() if v is not None}, **kwargs)
    samples = mcmc.get_samples()

    # give samples descriptive names
    for i in range(site_covs.shape[1] + 1):
        samples[f'cov_state_{site_covs_names[i]}'] = samples.pop(f'beta_{i}')
    for i in range(obs_covs.shape[2] + 1):
        samples[f'cov_det_{obs_covs_names[i]}'] = samples.pop(f'alpha_{i}')

    return FitResult(samples, mcmc)