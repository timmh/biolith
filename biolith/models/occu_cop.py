import unittest
from typing import Type, Optional

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from biolith.regression import AbstractRegression, LinearRegression
from biolith.utils.spatial import sample_spatial_effects, simulate_spatial_effects


def occu_cop(
    site_covs: jnp.ndarray,
    obs_covs: jnp.ndarray,
    coords: Optional[jnp.ndarray] = None,
    ell: float = 1.0,
    session_duration: Optional[jnp.ndarray] = None,
    false_positives_constant: bool = False,
    false_positives_unoccupied: bool = False,
    obs: Optional[jnp.ndarray] = None,
    prior_beta: dist.Distribution = dist.Normal(),
    prior_alpha: dist.Distribution = dist.Normal(),
    regressor_occ: Type[AbstractRegression] = LinearRegression,
    regressor_det: Type[AbstractRegression] = LinearRegression,
    prior_rate_fp_constant: dist.Distribution = dist.Exponential(),
    prior_rate_fp_unoccupied: dist.Distribution = dist.Exponential(),
    prior_gp_sd: dist.Distribution = dist.HalfNormal(1.0),
    prior_gp_length: dist.Distribution = dist.HalfNormal(1.0),
) -> None:
    """
    Count occupancy model using a Poisson detection process, inspired by Pautrel et al. (2024).

    References
    ----------
        - Pautrel, L., Moulherat, S., Gimenez, O., & Etienne, M.-P. (2024). Analysing biodiversity observation data collected in continuous time: Should we use discrete- or continuous-time occupancy models? Methods in Ecology and Evolution, 15, 935–950.

    Parameters
    ----------
    site_covs : jnp.ndarray
        Per-site covariates, shape (n_sites, n_site_covs).
    obs_covs : jnp.ndarray
        Observation covariates, shape (n_sites, time_periods, n_obs_covs).
    coords : Optional[jnp.ndarray], optional
        Site coordinates for spatial effects, shape (n_sites, 2).
    ell : float
        Spatial correlation length for the GP prior.
    session_duration : Optional[jnp.ndarray], optional
        Duration of each sampling session, shape (n_sites, time_periods).
    false_positives_constant : bool
        If True, model a constant false positive rate for all sites.
    false_positives_unoccupied : bool
        If True, model a false positive rate only for unoccupied sites.
    obs : Optional[jnp.ndarray], optional
        Observed counts, shape (n_sites, time_periods).
    prior_beta : dist.Distribution
        Prior distribution for occupancy coefficients.
    prior_alpha : dist.Distribution
        Prior distribution for detection coefficients.
    regressor_occ : Type[AbstractRegression]
        Class for the occupancy regression model, defaults to LinearRegression.
    regressor_det : Type[AbstractRegression]
        Class for the detection regression model, defaults to LinearRegression.
    prior_rate_fp_constant : dist.Distribution
        Prior for the constant false positive rate parameter.
    prior_rate_fp_unoccupied : dist.Distribution
        Prior for the false positive rate parameter in unoccupied sites.
    prior_gp_sd : dist.Distribution
        Prior distribution for the spatial random effect scale.
    prior_gp_length : dist.Distribution
        Prior distribution for the spatial kernel length scale.

    Examples
    --------
    >>> from biolith.models import occu_cop, simulate_cop
    >>> from biolith.utils import fit
    >>> data, _ = simulate_cop()
    >>> results = fit(occu_cop, **data)
    >>> print(results.samples['psi'].mean())
    """

    # Check input data
    assert (
        obs is None or obs.ndim == 2
    ), "obs must be None or of shape (n_sites, time_periods)"
    assert site_covs.ndim == 2, "site_covs must be of shape (n_sites, n_site_covs)"
    assert (
        obs_covs.ndim == 3
    ), "obs_covs must be of shape (n_sites, time_periods, n_obs_covs)"
    assert (
        session_duration is None or session_duration.ndim == 2
    ), "session_duration must be None or of shape (n_sites, time_periods)"
    assert not (
        false_positives_constant and false_positives_unoccupied
    ), "false_positives_constant and false_positives_unoccupied cannot both be True"

    n_sites = site_covs.shape[0]
    time_periods = obs_covs.shape[1]
    n_site_covs = site_covs.shape[1]
    n_obs_covs = obs_covs.shape[2]

    assert (
        n_sites == site_covs.shape[0] == obs_covs.shape[0]
    ), "site_covs and obs_covs must have the same number of sites"
    assert (
        time_periods == obs_covs.shape[1]
    ), "obs_covs must have the same number of time periods as obs"
    if obs is not None:
        assert n_sites == obs.shape[0], "obs must have n_sites rows"
        assert time_periods == obs.shape[1], "obs must have time_periods columns"
    if session_duration is not None:
        assert (
            n_sites == session_duration.shape[0]
        ), "session_duration must have n_sites rows"
        assert (
            time_periods == session_duration.shape[1]
        ), "session_duration must have time_periods columns"

    # Mask observations where covariates are missing
    obs_mask = jnp.isnan(obs_covs).any(axis=-1) | jnp.tile(
        jnp.isnan(site_covs).any(axis=-1)[:, None], (1, time_periods)
    )
    obs = jnp.where(obs_mask, jnp.nan, obs) if obs is not None else None
    obs_covs = jnp.nan_to_num(obs_covs)
    site_covs = jnp.nan_to_num(site_covs)

    # Model false positive rate for both occupied and unoccupied sites
    rate_fp_constant = (
        numpyro.sample("rate_fp_constant", prior_rate_fp_constant)
        if false_positives_constant
        else 0
    )

    # Model false positive rate only for unoccupied sites
    rate_fp_unoccupied = (
        numpyro.sample("rate_fp_unoccupied", prior_rate_fp_unoccupied)
        if false_positives_unoccupied
        else 0
    )

    # Occupancy and detection regression models
    reg_occ = regressor_occ("beta", n_site_covs, prior=prior_beta)
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
    session_duration = session_duration.transpose((1, 0))
    obs = obs.transpose((1, 0))

    with numpyro.plate("site", n_sites, dim=-1):

        # Occupancy process
        psi = numpyro.deterministic(
            "psi",
            jax.nn.sigmoid(reg_occ(site_covs) + w),
        )
        z = numpyro.sample(
            "z", dist.Bernoulli(probs=psi), infer={"enumerate": "parallel"}
        )

        with numpyro.plate("time_periods", time_periods, dim=-2):

            # Detection process
            rate_detection = numpyro.deterministic(
                f"rate_detection",
                jnp.exp(reg_det(obs_covs)),
            )
            l_det = z * rate_detection + (1 - z) * rate_fp_unoccupied + rate_fp_constant

            if obs is not None:
                with numpyro.handlers.mask(mask=jnp.isfinite(obs)):
                    numpyro.sample(
                        f"y",
                        dist.Poisson(jnp.nan_to_num(session_duration * l_det)),
                        obs=jnp.nan_to_num(obs),
                    )
            else:
                numpyro.sample(
                    f"y", dist.Poisson(jnp.nan_to_num(session_duration * l_det))
                )


def simulate_cop(
    n_site_covs: int = 1,
    n_obs_covs: int = 1,
    n_sites: int = 100,
    deployment_days_per_site: int = 365,
    session_duration: int = 7,
    simulate_missing: bool = False,
    min_occupancy: float = 0.25,
    max_occupancy: float = 0.75,
    min_observation_rate: float = 0.5,
    max_observation_rate: float = 10.0,
    random_seed: int = 0,
    spatial: bool = False,
    gp_sd: float = 1.0,
    gp_l: float = 0.2,
) -> tuple[dict, dict]:
    """Simulate data for :func:`occu_cop`.

    Returns ``(data, true_params)`` for use with :func:`fit`.

    Examples
    --------
    >>> from biolith.models import simulate_cop
    >>> data, params = simulate_cop()
    >>> sorted(data.keys())
    ['coords', 'ell', 'obs', 'obs_covs', 'session_duration', 'site_covs']
    """

    # Initialize random number generator
    rng = np.random.default_rng(random_seed)
    if spatial:
        coords = rng.uniform(0, 1, size=(n_sites, 2))
    else:
        coords = None

    # Make sure occupancy and detection are not too close to 0 or 1
    z = None
    while (
        z is None
        or z.mean() < min_occupancy
        or z.mean() > max_occupancy
        or np.mean(obs[np.isfinite(obs)]) < min_observation_rate
        or np.mean(obs[np.isfinite(obs)]) > max_observation_rate
    ):

        # Generate false positive rate
        rate_fp = rng.uniform(0.05, 0.2)

        # Generate intercept and slopes
        beta = rng.normal(
            size=n_site_covs + 1
        )  # intercept and slopes for occupancy logistic regression
        alpha = rng.normal(
            size=n_obs_covs + 1
        )  # intercept and slopes for detection logistic regression

        # Generate occupancy and site-level covariates
        site_covs = rng.normal(size=(n_sites, n_site_covs))
        if spatial:
            w, ell = simulate_spatial_effects(coords, gp_sd=gp_sd, gp_l=gp_l, rng=rng)
        else:
            w, ell = np.zeros(n_sites), 0.0
        psi = 1 / (
            1
            + np.exp(
                -(
                    beta[0].repeat(n_sites)
                    + np.sum(
                        [beta[i + 1] * site_covs[..., i] for i in range(n_site_covs)],
                        axis=0,
                    )
                    + w
                )
            )
        )
        z = rng.binomial(
            n=1, p=psi, size=n_sites
        )  # vector of latent occupancy status for each site

        # Generate detection data
        time_periods = round(deployment_days_per_site / session_duration)

        # Create matrix of detection covariates
        obs_covs = rng.normal(size=(n_sites, time_periods, n_obs_covs))
        detection_rate = np.exp(
            alpha[0].repeat(n_sites)[:, None]
            + np.sum(
                [alpha[i + 1] * obs_covs[..., i] for i in range(n_obs_covs)], axis=0
            )
        )

        # Create matrix of detections
        obs = np.zeros((n_sites, time_periods))

        for i in range(n_sites):
            # Similar to the Royle model in unmarked, false positives are generated only if the site is unoccupied
            # Note this is different than how we think about false positives being a random occurrence per image.
            obs[i, :] = rng.poisson(
                lam=(
                    session_duration
                    * (detection_rate[i, :] * z[i] + rate_fp * (1 - z[i]))
                ),
                size=time_periods,
            )

        if simulate_missing:
            # Simulate missing data:
            obs[rng.choice([True, False], size=obs.shape, p=[0.2, 0.8])] = np.nan
            obs_covs[rng.choice([True, False], size=obs_covs.shape, p=[0.05, 0.95])] = (
                np.nan
            )
            site_covs[
                rng.choice([True, False], size=site_covs.shape, p=[0.05, 0.95])
            ] = np.nan

    print(f"True occupancy: {np.mean(z):.4f}")
    print(
        f"Fraction of observations with at least one observation: {np.mean(obs[np.isfinite(obs)] >= 1):.4f}"
    )
    print(f"Mean rate: {np.mean(obs[np.isfinite(obs)]):.4f}")

    # session duration is assumed to be constant over all time periods
    session_duration = np.full((n_sites, time_periods), session_duration)

    return dict(
        site_covs=site_covs,
        obs_covs=obs_covs,
        session_duration=session_duration,
        obs=obs,
        false_positives_constant=True,
        coords=coords,
        ell=ell,
    ), dict(
        z=z,
        beta=beta,
        alpha=alpha,
        w=w,
        gp_sd=gp_sd,
        gp_l=gp_l,
    )


class TestOccuCOP(unittest.TestCase):

    def test_occu(self):
        data, true_params = simulate_cop(simulate_missing=True)

        from biolith.utils import fit

        results = fit(occu_cop, **data, timeout=600)

        self.assertTrue(
            np.allclose(
                results.samples["psi"].mean(), true_params["z"].mean(), atol=0.1
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

    def test_occu_spatial(self):
        data, true_params = simulate_cop(simulate_missing=True, spatial=True)

        from biolith.utils import fit

        results = fit(occu_cop, **data, timeout=600)

        self.assertTrue(
            np.allclose(
                results.samples["psi"].mean(), true_params["z"].mean(), atol=0.1
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
