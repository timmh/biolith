import unittest
from typing import Optional, Type

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from biolith.regression import AbstractRegression, LinearRegression
from biolith.utils.distributions import RightTruncatedPoisson
from biolith.utils.spatial import sample_spatial_effects, simulate_spatial_effects


def occu_rn(
    site_covs: jnp.ndarray,
    obs_covs: jnp.ndarray,
    coords: Optional[jnp.ndarray] = None,
    ell: float = 1.0,
    false_positives_constant: bool = False,
    max_abundance: int = 100,
    obs: Optional[jnp.ndarray] = None,
    prior_beta: dist.Distribution = dist.Normal(),
    prior_alpha: dist.Distribution = dist.Normal(),
    regressor_abu: Type[AbstractRegression] = LinearRegression,
    regressor_det: Type[AbstractRegression] = LinearRegression,
    prior_prob_fp_constant: dist.Distribution = dist.Beta(2, 5),
    prior_gp_sd: dist.Distribution = dist.HalfNormal(1.0),
    prior_gp_length: dist.Distribution = dist.HalfNormal(1.0),
) -> None:
    """Occupancy model inspired by Royle and Nichols (2003), relating observations to
    the number of individuals present at a site.

    References
    ----------
        - Royle, J. A. and Nichols, J. D. (2003) Estimating Abundance from Repeated Presence-Absence Data or Point Counts. Ecology, 84(3) pp. 777–790.

    Parameters
    ----------
    site_covs : jnp.ndarray
        An array of site-level covariates of shape (n_sites, n_site_covs).
    obs_covs : jnp.ndarray
        An array of observation-level covariates of shape (n_sites, n_revisits, n_obs_covs).
    coords : jnp.ndarray, optional
        Coordinates for a spatial random effect when provided.
    ell : float
        Spatial kernel length scale used if coords is provided.
    false_positives_constant : bool
        Flag indicating whether to model a constant false positive rate.
    max_abundance : int
        Maximum abundance cutoff for the Poisson distribution.
    obs : jnp.ndarray, optional
        Observation matrix of shape (n_sites, n_revisits) or None.
    prior_beta : numpyro.distributions.Distribution
        Prior distribution for the site-level regression coefficients.
    prior_alpha : numpyro.distributions.Distribution
        Prior distribution for the observation-level regression coefficients.
    regressor_abu : Type[AbstractRegression]
        Class for the abundance regression model, defaults to LinearRegression.
    regressor_det : Type[AbstractRegression]
        Class for the detection regression model, defaults to LinearRegression.
    prior_prob_fp_constant : numpyro.distributions.Distribution
        Prior distribution for the constant false positive rate.
    prior_prob_fp_unoccupied : numpyro.distributions.Distribution
        Prior distribution for the false positive rate in unoccupied sites.
    prior_gp_sd : numpyro.distributions.Distribution
        Prior distribution for the spatial random effect scale.
    prior_gp_length : numpyro.distributions.Distribution
        Prior distribution for the spatial kernel length scale.

    Examples
    --------
    >>> from biolith.models import occu_rn, simulate_rn
    >>> from biolith.utils import fit
    >>> data, _ = simulate_rn()
    >>> results = fit(occu_rn, **data)
    >>> print(results.samples['abundance'].mean())
    """

    # Check input data
    assert site_covs.ndim == 2, "site_covs must be (n_sites, n_site_covs)"
    assert obs_covs.ndim == 3, "obs_covs must be (n_sites, time_periods, n_obs_covs)"

    n_sites = site_covs.shape[0]
    time_periods = obs_covs.shape[1]
    n_site_covs = site_covs.shape[1]
    n_obs_covs = obs_covs.shape[2]

    # # TODO: re-enable
    # if obs is not None:
    #     assert obs.shape == (n_sites, time_periods), "obs must have shape (n_sites, time_periods)"
    #     assert (obs[np.isfinite(obs)] >= 0).all() and (obs[np.isfinite(obs)] <= 1).all(), "Detections (obs) must be in {0,1} (or NaN for missing)."

    # Mask observations where covariates are missing
    obs_mask = jnp.isnan(obs_covs).any(axis=-1) | jnp.tile(
        jnp.isnan(site_covs).any(axis=-1)[:, None], (1, time_periods)
    )
    obs = jnp.where(obs_mask, jnp.nan, obs) if obs is not None else None
    obs_covs = jnp.nan_to_num(obs_covs)
    site_covs = jnp.nan_to_num(site_covs)

    # Priors
    # Model false positive rate for both occupied and unoccupied sites
    prob_fp_constant = (
        numpyro.sample("prob_fp_constant", prior_prob_fp_constant)
        if false_positives_constant
        else 0
    )

    # Abundance and detection regression models
    reg_abu = regressor_abu("beta", n_site_covs, prior=prior_beta)
    reg_det = regressor_det("alpha", n_obs_covs, prior=prior_alpha)

    if coords is not None:
        w = sample_spatial_effects(
            coords,
            ell=ell,
            prior_gp_sd=prior_gp_sd,
            prior_gp_length=prior_gp_length,
        )
    else:
        w = jnp.zeros(n_sites)

    # Transpose in order to fit NumPyro's plate structure
    site_covs = site_covs.transpose((1, 0))
    obs_covs = obs_covs.transpose((2, 1, 0))
    obs = obs.transpose((1, 0)) if obs is not None else None

    with numpyro.plate("site", n_sites, dim=-1):

        # Occupancy process
        abundance = numpyro.deterministic(
            "abundance",
            jnp.exp(reg_abu(site_covs) + w),
        )

        N_i = numpyro.sample(
            "N_i",
            RightTruncatedPoisson(abundance, max_cutoff=max_abundance),  # type: ignore
            infer={"enumerate": "parallel"},
        )

        with numpyro.plate("time", time_periods, dim=-2):

            # Detection process
            r_it = numpyro.deterministic(
                f"prob_detection",
                jax.nn.sigmoid(reg_det(obs_covs)),
            )
            p_it = 1.0 - (1.0 - r_it) ** N_i[None, :]  # type: ignore

            if obs is not None:
                with numpyro.handlers.mask(mask=jnp.isfinite(obs)):
                    numpyro.sample(
                        "y",
                        dist.Bernoulli(1 - (1 - p_it) * (1 - prob_fp_constant)),  # type: ignore
                        obs=jnp.nan_to_num(obs),
                        infer={"enumerate": "parallel"},
                    )
            else:
                numpyro.sample(
                    "y",
                    dist.Bernoulli(1 - (1 - p_it) * (1 - prob_fp_constant)),  # type: ignore
                    infer={"enumerate": "parallel"},
                )


def simulate_rn(
    n_site_covs: int = 1,
    n_obs_covs: int = 1,
    n_sites: int = 100,
    deployment_days_per_site: int = 365,
    session_duration: int = 7,
    prob_fp: float = 0.0,
    simulate_missing: bool = False,
    min_occupancy: float = 0.25,
    max_occupancy: float = 0.75,
    min_observation_rate: float = 0.1,
    max_observation_rate: float = 0.5,
    random_seed: int = 0,
    spatial: bool = False,
    gp_sd: float = 1.0,
    gp_l: float = 0.2,
) -> tuple[dict, dict]:
    """Simulate data for :func:`occu_rn`.

    Returns ``(data, true_params)`` for :func:`fit`.

    Examples
    --------
    >>> from biolith.models import simulate_rn
    >>> data, params = simulate_rn()
    >>> sorted(data.keys())
    ['coords', 'ell', 'obs', 'obs_covs', 'site_covs']
    """

    # Initialize random number generator
    rng = np.random.default_rng(random_seed)
    if spatial:
        coords = rng.uniform(0, 1, size=(n_sites, 2))
    else:
        coords = None

    # Make sure occupancy and detection are not too close to 0 or 1
    N_i = None
    while (
        N_i is None
        or (N_i > 0).mean() < min_occupancy
        or (N_i > 0).mean() > max_occupancy
        or np.mean(obs[np.isfinite(obs)]) < min_observation_rate
        or np.mean(obs[np.isfinite(obs)]) > max_observation_rate
    ):

        # Generate intercept and slopes
        beta = rng.normal(
            size=n_site_covs + 1
        )  # intercept and slopes for occupancy logistic regression
        alpha = rng.normal(
            size=n_obs_covs + 1
        )  # intercept and slopes for detection logistic regression

        # Generate occupancy and site-level covariates
        site_covs = rng.normal(size=(n_sites, n_site_covs))

        if spatial and coords is not None:
            w, ell = simulate_spatial_effects(coords, gp_sd=gp_sd, gp_l=gp_l, rng=rng)
        else:
            w, ell = np.zeros(n_sites), 0.0
        abundance = np.exp(
            beta[0].repeat(n_sites)
            + np.sum(
                [beta[i + 1] * site_covs[..., i] for i in range(n_site_covs)], axis=0
            )
            + w
        )
        N_i = rng.poisson(
            abundance, size=n_sites
        )  # vector of latent occupancy status for each site

        # Generate detection data
        time_periods = round(deployment_days_per_site / session_duration)

        # Create matrix of detection covariates
        obs_covs = rng.normal(size=(n_sites, time_periods, n_obs_covs))
        r_it = 1 / (
            1
            + np.exp(
                -(
                    alpha[0].repeat(n_sites)[:, None]
                    + np.sum(
                        [alpha[i + 1] * obs_covs[..., i] for i in range(n_obs_covs)],
                        axis=0,
                    )
                )
            )
        )

        # Create matrix of detections
        obs = np.zeros((n_sites, time_periods))

        for i in range(n_sites):
            # According to the Royle model in unmarked, false positives are generated only if the site is unoccupied
            # Note this is different than how we think about false positives being a random occurrence per image.
            # For now, this is generating positive/negative per time period, which is different than per image.
            p_it = 1.0 - (1.0 - r_it[i]) ** N_i[i]
            obs[i, :] = rng.binomial(
                n=1, p=1 - (1 - p_it) * (1 - prob_fp), size=time_periods
            )

        # Convert counts into observed occupancy
        obs = (obs >= 1) * 1.0

        if simulate_missing:
            # Simulate missing data:
            obs[rng.choice([True, False], size=obs.shape, p=[0.2, 0.8])] = np.nan
            obs_covs[rng.choice([True, False], size=obs_covs.shape, p=[0.05, 0.95])] = (
                np.nan
            )
            site_covs[
                rng.choice([True, False], size=site_covs.shape, p=[0.05, 0.95])
            ] = np.nan

    print(f"True occupancy: {np.mean(N_i > 0):.4f}")
    print(f"True abundance: {np.mean(abundance):.4f}")
    print(
        f"Proportion of timesteps with observation: {np.mean(obs[np.isfinite(obs)]):.4f}"
    )

    return dict(
        site_covs=site_covs,
        obs_covs=obs_covs,
        obs=obs,
        coords=coords,
        ell=ell,
    ), dict(
        abundance=abundance,
        beta=beta,
        alpha=alpha,
        w=w,
        gp_sd=gp_sd,
        gp_l=gp_l,
    )


class TestOccuRN(unittest.TestCase):

    def test_occu(self):
        data, true_params = simulate_rn(simulate_missing=True)

        from biolith.utils import fit

        results = fit(occu_rn, **data, timeout=600)

        self.assertTrue(
            np.allclose(
                results.samples["abundance"].mean(),
                true_params["abundance"].mean(),
                rtol=0.1,
            )
        )
        self.assertTrue(
            np.allclose(
                [
                    results.samples[k].mean()
                    for k in [f"cov_state_{i}" for i in range(len(true_params["beta"]))]
                ],
                true_params["beta"],
                atol=0.5,
            )
        )
        self.assertTrue(
            np.allclose(
                [
                    results.samples[k].mean()
                    for k in [f"cov_det_{i}" for i in range(len(true_params["alpha"]))]
                ],
                true_params["alpha"],
                atol=0.5,
            )
        )

    # TODO: fix this test
    @unittest.skip("Skipping test for spatial model")
    def test_occu_spatial(self):
        data, true_params = simulate_rn(simulate_missing=True, spatial=True)

        from biolith.utils import fit

        results = fit(occu_rn, **data, timeout=600)

        self.assertTrue(
            np.allclose(
                results.samples["abundance"].mean(),
                true_params["abundance"].mean(),
                rtol=0.1,
            )
        )
        self.assertTrue(
            np.allclose(results.samples["gp_sd"].mean(), true_params["gp_sd"], atol=1.0)
        )
        self.assertTrue(
            np.allclose(results.samples["gp_l"].mean(), true_params["gp_l"], atol=0.5)
        )


if __name__ == "__main__":
    unittest.main()
