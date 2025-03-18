import copy
import pandas as pd


def dataframes_to_arrays(site_covs=None, obs_covs=None, obs=None, session_duration=None):

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

    if site_covs_names is None and site_covs is not None:
        site_covs_names = [str(0)] + [str(i + 1) for i in range(site_covs.shape[1])]
    if obs_covs_names is None and obs_covs is not None:
        obs_covs_names = [str(0)] + [str(i + 1) for i in range(obs_covs.shape[2])]

    return site_covs, obs_covs, obs, session_duration, site_covs_names, obs_covs_names


def rename_samples(samples, site_covs_names=None, obs_covs_names=None):
    samples = copy.deepcopy(samples)
    if site_covs_names is not None:
        for i in range(len(site_covs_names)):
            samples[f'cov_state_{site_covs_names[i]}'] = samples.pop(f'beta_{i}')
    if obs_covs_names is not None:
        for i in range(len(obs_covs_names)):
            samples[f'cov_det_{obs_covs_names[i]}'] = samples.pop(f'alpha_{i}')
    return samples