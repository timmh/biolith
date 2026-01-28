from typing import Optional, Tuple, Type

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from biolith.regression import AbstractRegression, LinearRegression
from biolith.utils.modeling import flatten_covariates, mask_missing_obs, reshape_predictions
from biolith.utils.spatial import sample_spatial_effects, simulate_spatial_effects


def occu_cs(
    site_covs: jnp.ndarray,
    obs_covs: jnp.ndarray,
    coords: Optional[jnp.ndarray] = None,
    ell: float = 1.0,
    obs: Optional[jnp.ndarray] = None,
    n_species: int = 1,
    prior_beta: dist.Distribution = dist.Normal(),
    prior_alpha: dist.Distribution = dist.Normal(),
    regressor_occ: Type[AbstractRegression] = LinearRegression,
    regressor_det: Type[AbstractRegression] = LinearRegression,
    prior_mu: dist.Distribution | Tuple[dist.Distribution] = dist.Normal(0, 10),
    prior_sigma: dist.Distribution | Tuple[dist.Distribution] = dist.Gamma(5, 1),
    prior_gp_sd: dist.Distribution = dist.HalfNormal(1.0),
    prior_gp_length: dist.Distribution = dist.HalfNormal(1.0),
    site_random_effects: bool = False,
    obs_random_effects: bool = False,
    prior_site_re_sd: dist.Distribution = dist.HalfNormal(1.0),
    prior_obs_re_sd: dist.Distribution = dist.HalfNormal(1.0),
) -> None:
    """Continuous-score occupancy model inspired by Rhinehart et al. (2022), modeling
    classification scores as being drawn from true or false positive distributions.

    References
    ----------
        - Rhinehart, T. A., Turek, D., & Kitzes, J. (2022). A continuous-score occupancy model that incorporates uncertain machine learning output from autonomous biodiversity surveys. Methods in Ecology and Evolution, 13, 1778–1789.

    Parameters
    ----------
    site_covs : jnp.ndarray
        Site-level covariates (n_sites, n_site_covs).
    obs_covs : jnp.ndarray
        Observation covariates (n_sites, n_periods, n_replicates, n_obs_covs).
    coords : Optional[jnp.ndarray]
        Site coordinates for spatial effects (n_sites, 2) or None.
    ell : float
        Optional distance matrix parameter.
    obs : Optional[jnp.ndarray]
        Observations (n_species, n_sites, n_periods, n_replicates) or None.
    n_species : int
        Number of species. Used when ``obs`` is None.
    prior_beta : dist.Distribution
        Prior for occupancy coefficients.
    prior_alpha : dist.Distribution
        Prior for detection coefficients.
    regressor_occ : Type[AbstractRegression]
        Class for the occupancy regression model, defaults to LinearRegression.
    regressor_det : Type[AbstractRegression]
        Class for the detection regression model, defaults to LinearRegression.
    prior_mu : dist.Distribution or Tuple[dist.Distribution]
        Prior for mean continuous scores.
    prior_sigma : dist.Distribution or Tuple[dist.Distribution]
        Prior for standard deviations of continuous scores.
    prior_gp_sd : dist.Distribution
        Prior distribution for the spatial random effect scale.
    prior_gp_length : dist.Distribution
        Prior distribution for the spatial kernel length scale.
    site_random_effects : bool
        Flag indicating whether to include site-level random effects.
    obs_random_effects : bool
        Flag indicating whether to include observation-level random effects.
    prior_site_re_sd : dist.Distribution
        Prior distribution for the site-level random effect standard deviation.
    prior_obs_re_sd : dist.Distribution
        Prior distribution for the observation-level random effect standard deviation.

    Examples
    --------
    >>> from biolith.models import occu_cs, simulate_cs
    >>> from biolith.utils import fit
    >>> data, _ = simulate_cs()
    >>> results = fit(occu_cs, **data)
    >>> print(results.samples['psi'].mean())
    """

    # Check input data
    assert (
        obs is None or obs.ndim == 4
    ), "obs must be None or of shape (n_species, n_sites, n_periods, n_replicates)"
    assert site_covs.ndim == 2, "site_covs must be of shape (n_sites, n_site_covs)"
    assert (
        obs_covs.ndim == 4
    ), "obs_covs must be of shape (n_sites, n_periods, n_replicates, n_obs_covs)"

    n_sites = site_covs.shape[0]
    n_periods = obs_covs.shape[1]
    n_replicates = obs_covs.shape[2]
    n_site_covs = site_covs.shape[1]
    n_obs_covs = obs_covs.shape[3]
    if obs is not None:
        n_species = obs.shape[0]

    assert (
        n_sites == site_covs.shape[0] == obs_covs.shape[0]
    ), "site_covs and obs_covs must have the same number of sites"
    assert (
        n_periods == obs_covs.shape[1]
    ), "obs_covs must have the same number of periods as obs"
    if obs is not None:
        assert n_sites == obs.shape[1], "obs must have n_sites rows"
        assert n_periods == obs.shape[2], "obs must have n_periods columns"
        assert n_replicates == obs.shape[3], "obs must have n_replicates columns"

    # Mask observations where covariates are missing
    obs_mask = jnp.isnan(obs_covs).any(axis=-1) | jnp.isnan(site_covs).any(axis=-1)[
        :, None, None
    ]
    obs = jnp.where(obs_mask[None, ...], jnp.nan, obs) if obs is not None else None
    obs_covs = jnp.nan_to_num(obs_covs)
    site_covs = jnp.nan_to_num(site_covs)

    if coords is not None:
        w = sample_spatial_effects(
            coords,
            ell=ell,
            prior_gp_sd=prior_gp_sd,
            prior_gp_length=prior_gp_length,
        )
    else:
        w = jnp.zeros(n_sites)

    # Random effects standard deviations (sampled before plates)
    if site_random_effects:
        site_re_sd = numpyro.sample("site_re_sd", prior_site_re_sd)
    if obs_random_effects:
        obs_re_sd = numpyro.sample("obs_re_sd", prior_obs_re_sd)

    # Continuous score parameters
    prior_mus = prior_mu if isinstance(prior_mu, tuple) else (prior_mu, prior_mu)
    mu0 = numpyro.sample("mu0", prior_mus[0])
    mu1 = numpyro.sample("mu1", dist.TruncatedDistribution(prior_mus[1], low=mu0))  # type: ignore
    prior_sigmas = (
        prior_sigma if isinstance(prior_sigma, tuple) else (prior_sigma, prior_sigma)
    )
    sigma0 = numpyro.sample("sigma0", prior_sigmas[0])  # type: ignore
    sigma1 = numpyro.sample("sigma1", prior_sigmas[1])  # type: ignore

    # Transpose in order to fit NumPyro's plate structure
    site_covs = site_covs.transpose((1, 0))
    obs_covs = obs_covs.transpose((3, 2, 1, 0))
    obs = obs.transpose((3, 2, 1, 0)) if obs is not None else None
    site_covs_flat, site_shape = flatten_covariates(site_covs)
    obs_covs_flat, obs_shape = flatten_covariates(obs_covs)

    with numpyro.plate("species", n_species, dim=-1):

        # Occupancy and detection regression models
        reg_occ = regressor_occ("beta", n_site_covs, prior=prior_beta)
        reg_det = regressor_det("alpha", n_obs_covs, prior=prior_alpha)

        with numpyro.plate("site", n_sites, dim=-2):

            # Site-level random effects
            if site_random_effects:
                site_re_occ = numpyro.sample("site_re_occ", dist.Normal(0, site_re_sd))  # type: ignore
                site_re_det = numpyro.sample("site_re_det", dist.Normal(0, site_re_sd))  # type: ignore
            else:
                site_re_occ = 0.0
                site_re_det = 0.0

            occ_linear = (
                reshape_predictions(reg_occ(site_covs_flat), site_shape)
                + w[:, None]
                + site_re_occ
            )

            with numpyro.plate("period", n_periods, dim=-3):

                # Occupancy process
                psi = numpyro.deterministic("psi", jax.nn.sigmoid(occ_linear))
                z = numpyro.sample(
                    "z", dist.Bernoulli(probs=psi), infer={"enumerate": "parallel"}  # type: ignore
                )

                with numpyro.plate("replicate", n_replicates, dim=-4):

                    # Observation-level random effects
                    if obs_random_effects:
                        obs_re = numpyro.sample("obs_re", dist.Normal(0, obs_re_sd))  # type: ignore
                    else:
                        obs_re = 0.0

                    # Detection process
                    f = numpyro.sample(
                        "f",
                        dist.Bernoulli(
                            z
                            * jax.nn.sigmoid(
                                reshape_predictions(reg_det(obs_covs_flat), obs_shape)
                                + site_re_det
                                + obs_re
                            )
                        ),  # type: ignore
                        infer={"enumerate": "parallel"},
                    )

                    with mask_missing_obs(obs):
                        numpyro.sample(
                            "s",
                            dist.Normal(
                                (1 - f) * mu0 + f * mu1,
                                (1 - f) * sigma0 + f * sigma1,  # type: ignore
                            ),  # type: ignore
                            obs=obs,
                        )


def simulate_cs(
    n_site_covs: int = 1,
    n_obs_covs: int = 1,
    n_sites: int = 100,
    n_periods: int = 1,
    n_species: int = 1,
    deployment_days_per_site: int = 365,
    session_duration: int = 7,
    simulate_missing: bool = False,
    min_occupancy: float = 0.25,
    max_occupancy: float = 0.75,
    random_seed: int = 0,
    spatial: bool = False,
    gp_sd: float = 1.0,
    gp_l: float = 0.2,
) -> tuple[dict, dict]:
    """Simulate data for :func:`occu_cs`.

    Returns ``(data, true_params)`` for :func:`fit`.

    Examples
    --------
    >>> from biolith.models import simulate_cs
    >>> data, params = simulate_cs()
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
    z = None
    while z is None or z.mean() < min_occupancy or z.mean() > max_occupancy:

        # Generate intercept and slopes
        beta = rng.normal(
            size=(n_species, n_site_covs + 1)
        )  # intercept and slopes for occupancy logistic regression
        alpha = rng.normal(
            size=(n_species, n_obs_covs + 1)
        )  # intercept and slopes for detection logistic regression

        # Generate occupancy and site-level covariates
        site_covs = rng.normal(size=(n_sites, n_site_covs))
        if spatial and coords is not None:
            w, ell = simulate_spatial_effects(coords, gp_sd=gp_sd, gp_l=gp_l, rng=rng)
        else:
            w, ell = np.zeros(n_sites), 0.0
        psi = 1 / (
            1
            + np.exp(
                -(
                    beta[:, 0][:, None]
                    + np.tensordot(beta[:, 1:], site_covs, axes=([1], [1]))
                    + w[None, :]
                )
            )
        )
        z = rng.binomial(
            n=1, p=psi[:, None, :], size=(n_species, n_periods, n_sites)
        )  # matrix of latent occupancy status for each site and period

        # Generate detection data
        n_replicates = round(deployment_days_per_site / session_duration)

        # Create matrix of detection covariates
        obs_covs = rng.normal(size=(n_sites, n_periods, n_replicates, n_obs_covs))
        p = 1 / (
            1
            + np.exp(
                -(
                    alpha[:, 0][:, None, None, None]
                    + np.tensordot(alpha[:, 1:], obs_covs, axes=([1], [3]))
                )
            )
        )

        # Create matrix of detections
        obs = np.zeros((n_species, n_sites, n_periods, n_replicates))

        # Generate score distributions
        mu0 = 0
        sigma0 = 10
        mu1 = 10
        sigma1 = 5

        # Create matrix of detections
        f = np.zeros((n_species, n_sites, n_periods, n_replicates))
        obs = np.zeros((n_species, n_sites, n_periods, n_replicates))

        z_site = z.transpose(0, 2, 1)
        f = rng.binomial(
            n=1,
            p=p * z_site[..., None],
            size=(n_species, n_sites, n_periods, n_replicates),
        )
        obs = rng.normal(
            loc=np.where(f == 1, mu1, mu0),
            scale=np.where(f == 1, sigma1, sigma0),
            size=(n_species, n_sites, n_periods, n_replicates),
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

    return dict(
        site_covs=site_covs,
        obs_covs=obs_covs,
        obs=obs,
        coords=coords,
        ell=ell,
    ), dict(
        z=z,
        beta=beta,
        alpha=alpha,
        mu0=mu0,
        sigma0=sigma0,
        mu1=mu1,
        sigma1=sigma1,
        w=w,
        gp_sd=gp_sd,
        gp_l=gp_l,
    )



def test_occu():
    data, true_params = simulate_cs(simulate_missing=True)

    from biolith.utils import fit

    results = fit(occu_cs, **data, timeout=600)

    assert (
        np.allclose(
            results.samples["psi"].mean(), true_params["z"].mean(), atol=0.1
        )
    )
    assert (
        np.allclose(
            [
                results.samples[k].mean()
                for k in [
                    f"cov_state_{i}"
                    for i in range(true_params["beta"].shape[1])
                ]
            ],
            true_params["beta"].mean(axis=0),
            atol=0.5,
        )
    )
    assert (
        np.allclose(
            [
                results.samples[k].mean()
                for k in [
                    f"cov_det_{i}"
                    for i in range(true_params["alpha"].shape[1])
                ]
            ],
            true_params["alpha"].mean(axis=0),
            atol=0.5,
        )
    )
    assert (
        np.allclose(results.samples["mu0"].mean(), true_params["mu0"], atol=1)
    )
    assert (
        np.allclose(results.samples["mu1"].mean(), true_params["mu1"], atol=1)
    )
    assert (
        np.allclose(results.samples["sigma0"].mean(), true_params["sigma0"], atol=1)
    )
    assert (
        np.allclose(results.samples["sigma1"].mean(), true_params["sigma1"], atol=1)
    )

def test_occu_multi_season():
    data, true_params = simulate_cs(simulate_missing=True, n_periods=3)

    from biolith.utils import fit

    results = fit(
        occu_cs,
        **data,
        num_chains=1,
        num_samples=300,
        num_warmup=300,
        timeout=600,
    )

    assert (
        np.allclose(
            results.samples["psi"].mean(), true_params["z"].mean(), atol=0.15
        )
    )

def test_occu_multi_species():
    data, _ = simulate_cs(simulate_missing=True, n_species=2, n_sites=30)

    from biolith.utils import fit

    results = fit(occu_cs, **data, num_chains=1, num_samples=200, timeout=600)

    assert results.samples["psi"].shape[-1] == 2

def test_occu_spatial():
    data, true_params = simulate_cs(simulate_missing=True, spatial=True)

    from biolith.utils import fit

    results = fit(occu_cs, **data, timeout=600)

    assert (
        np.allclose(
            results.samples["psi"].mean(), true_params["z"].mean(), atol=0.1
        )
    )
    assert (
        np.allclose(results.samples["gp_sd"].mean(), true_params["gp_sd"], atol=1.0)
    )
    assert (
        np.allclose(results.samples["gp_l"].mean(), true_params["gp_l"], atol=0.5)
    )

def test_site_random_effects():
    data, true_params = simulate_cs(simulate_missing=True)

    from biolith.utils import fit

    results = fit(
        occu_cs,
        **data,
        site_random_effects=True,
        num_chains=1,
        num_samples=500,
        timeout=600,
    )

    assert ("site_re_sd" in results.samples)
    assert ("site_re_occ" in results.samples)
    assert ("site_re_det" in results.samples)
    assert (results.samples["site_re_sd"].mean() > 0)
    assert (
        np.allclose(
            results.samples["psi"].mean(), true_params["z"].mean(), atol=0.15
        )
    )

def test_obs_random_effects():
    data, true_params = simulate_cs(simulate_missing=True)

    from biolith.utils import fit

    results = fit(
        occu_cs,
        **data,
        obs_random_effects=True,
        num_chains=1,
        num_samples=500,
        timeout=600,
    )

    assert ("obs_re_sd" in results.samples)
    assert ("obs_re" in results.samples)
    assert (results.samples["obs_re_sd"].mean() > 0)
    assert (
        np.allclose(
            results.samples["psi"].mean(), true_params["z"].mean(), atol=0.15
        )
    )

def test_combined_random_effects():
    data, true_params = simulate_cs(simulate_missing=True)

    from biolith.utils import fit

    results = fit(
        occu_cs,
        **data,
        site_random_effects=True,
        obs_random_effects=True,
        num_chains=1,
        num_samples=500,
        timeout=600,
    )

    assert ("site_re_sd" in results.samples)
    assert ("site_re_occ" in results.samples)
    assert ("site_re_det" in results.samples)
    assert ("obs_re_sd" in results.samples)
    assert ("obs_re" in results.samples)
    assert (
        np.allclose(
            results.samples["psi"].mean(), true_params["z"].mean(), atol=0.15
        )
    )

