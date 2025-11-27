import copy
from typing import Any, List, Optional, Tuple, cast

import jax.numpy as jnp
import numpy as np
import pandas as pd


def dataframes_to_arrays(
    site_covs=None, obs_covs=None, obs=None, session_duration=None
):

    site_covs_names = None
    obs_covs_names = None

    # Use the first dataframe we encounter as the reference ordering for the rest.
    reference_index = None
    for df in (obs, site_covs, obs_covs, session_duration):
        if isinstance(df, pd.DataFrame):
            reference_index = df.index
            break

    def _align_to_reference(df):
        if not isinstance(df, pd.DataFrame) or reference_index is None:
            return df
        ref_index = pd.Index(reference_index)
        df_index = pd.Index(df.index)
        # Only reorder when the two indexes contain the same elements (ignoring order).
        if (
            len(ref_index) == len(df_index)
            and ref_index.isin(df_index).all()
            and df_index.isin(ref_index).all()
            and not df_index.equals(ref_index)
        ):
            return df.loc[reference_index]
        return df

    site_covs = _align_to_reference(site_covs)
    obs_covs = _align_to_reference(obs_covs)
    session_duration = _align_to_reference(session_duration)
    obs = _align_to_reference(obs)

    if isinstance(site_covs, pd.DataFrame):
        site_covs_names = ["intercept"] + [c for c in site_covs.columns]
        site_covs = site_covs.to_numpy()
    if isinstance(obs_covs, pd.DataFrame):
        if not isinstance(obs_covs.columns, pd.MultiIndex):
            obs_covs = obs_covs.to_numpy()
            if obs_covs.ndim == 2:
                obs_covs = obs_covs[:, :, None]
            obs_covs = cast(np.ndarray[Tuple[int, int, int], Any], obs_covs)
        else:
            assert (
                len(obs_covs.columns.levels) == 2
            ), "obs_covs with MultiIndex columns must have columns of exactly two levels"
            obs_covs_names = ["intercept"] + [c for c in obs_covs.columns.levels[0]]
            obs_covs = (
                obs_covs.to_numpy()
                .reshape(
                    obs_covs.shape[0],
                    len(obs_covs.columns.levels[0]),
                    len(obs_covs.columns.levels[1]),
                )
                .transpose(0, 2, 1)
            )
    if isinstance(session_duration, pd.DataFrame):
        session_duration = session_duration.to_numpy()
    if isinstance(obs, pd.DataFrame):
        obs = obs.to_numpy()

    if site_covs_names is None and site_covs is not None:
        site_covs_names = [str(0)] + [str(i + 1) for i in range(site_covs.shape[1])]
    if obs_covs_names is None and obs_covs is not None:
        obs_covs_names = [str(0)] + [str(i + 1) for i in range(obs_covs.shape[2])]

    site_covs = jnp.array(site_covs) if site_covs is not None else None
    obs_covs = jnp.array(obs_covs) if obs_covs is not None else None
    obs = jnp.array(obs) if obs is not None else None
    session_duration = (
        jnp.array(session_duration) if session_duration is not None else None
    )

    return site_covs, obs_covs, obs, session_duration, site_covs_names, obs_covs_names


def rename_samples(
    samples, site_covs_names=None, obs_covs_names: Optional[List[str]] = None
):
    samples = copy.copy(samples)
    if site_covs_names is not None:
        for i in range(len(site_covs_names)):
            if f"beta_{i}" in samples:
                samples[f"cov_state_{site_covs_names[i]}"] = samples.pop(f"beta_{i}")
        if "beta" in samples:
            beta = samples.pop("beta")
            for i in range(len(site_covs_names)):
                samples[f"cov_state_{site_covs_names[i]}"] = beta[..., i]
    if obs_covs_names is not None:
        for i in range(len(obs_covs_names)):
            if f"alpha_{i}" in samples:
                samples[f"cov_det_{obs_covs_names[i]}"] = samples.pop(f"alpha_{i}")
        if "alpha" in samples:
            alpha = samples.pop("alpha")
            for i in range(len(obs_covs_names)):
                samples[f"cov_det_{obs_covs_names[i]}"] = alpha[..., i]
    return samples
