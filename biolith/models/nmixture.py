from typing import Optional, Type

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
import numpyro
import numpyro.distributions as dist

from biolith.regression import AbstractRegression, LinearRegression
from biolith.utils.modeling import flatten_covariates, mask_missing_obs, reshape_predictions
from biolith.utils.spatial import sample_spatial_effects, simulate_spatial_effects


def nmixture(
    site_covs: jnp.ndarray,
    obs_covs: jnp.ndarray,
    coords: Optional[jnp.ndarray] = None,
    ell: float = 1.0,
    max_abundance: int = 100,
    obs: Optional[jnp.ndarray] = None,
    n_species: int = 1,
    prior_beta: dist.Distribution = dist.Normal(),
    prior_alpha: dist.Distribution = dist.Normal(),
    regressor_abu: Type[AbstractRegression] = LinearRegression,
    regressor_det: Type[AbstractRegression] = LinearRegression,
    prior_gp_sd: dist.Distribution = dist.HalfNormal(1.0),
    prior_gp_length: dist.Distribution = dist.HalfNormal(1.0),
    site_random_effects: bool = False,
    obs_random_effects: bool = False,
    prior_site_re_sd: dist.Distribution = dist.HalfNormal(1.0),
    prior_obs_re_sd: dist.Distribution = dist.HalfNormal(1.0),
) -> None:
    """N-mixture model for repeated count data (Royle 2004).

    References
    ----------
        - Royle, J. A. (2004). N-mixture models for estimating population size from spatially replicated counts. Biometrics, 60(1), 108-115.

    Parameters
    ----------
    site_covs : jnp.ndarray
        An array of site-level covariates of shape (n_sites, n_site_covs).
    obs_covs : jnp.ndarray
        An array of observation-level covariates of shape
        (n_sites, n_periods, n_replicates, n_obs_covs).
    coords : jnp.ndarray, optional
        Coordinates for a spatial random effect when provided.
    ell : float
        Spatial kernel length scale used if coords is provided.
    max_abundance : int
        Maximum abundance cutoff for the Poisson distribution.
    obs : jnp.ndarray, optional
        Observation matrix of shape (n_species, n_sites, n_periods, n_replicates) or None.
    n_species : int
        Number of species. Used when ``obs`` is None.
    prior_beta : numpyro.distributions.Distribution
        Prior distribution for the abundance regression coefficients.
    prior_alpha : numpyro.distributions.Distribution
        Prior distribution for the detection regression coefficients.
    regressor_abu : Type[AbstractRegression]
        Class for the abundance regression model, defaults to LinearRegression.
    regressor_det : Type[AbstractRegression]
        Class for the detection regression model, defaults to LinearRegression.
    prior_gp_sd : numpyro.distributions.Distribution
        Prior distribution for the spatial random effect scale.
    prior_gp_length : numpyro.distributions.Distribution
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
    >>> from biolith.models import nmixture, simulate_nmixture
    >>> from biolith.utils import fit
    >>> data, _ = simulate_nmixture()
    >>> results = fit(nmixture, **data)
    >>> print(results.samples['abundance'].mean())
    """

    # Check input data
    assert (
        obs is None or obs.ndim == 4
    ), "obs must be None or of shape (n_species, n_sites, n_periods, n_replicates)"
    assert site_covs.ndim == 2, "site_covs must be (n_sites, n_site_covs)"
    assert (
        obs_covs.ndim == 4
    ), "obs_covs must be (n_sites, n_periods, n_replicates, n_obs_covs)"

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

    # Transpose in order to fit NumPyro's plate structure
    site_covs = site_covs.transpose((1, 0))
    obs_covs = obs_covs.transpose((3, 2, 1, 0))
    obs = obs.transpose((3, 2, 1, 0)) if obs is not None else None
    site_covs_flat, site_shape = flatten_covariates(site_covs)
    obs_covs_flat, obs_shape = flatten_covariates(obs_covs)

    min_counts = None
    if obs is not None:
        obs_max = jnp.nanmax(jnp.nan_to_num(obs, nan=-jnp.inf), axis=0)
        obs_max = jnp.where(jnp.isfinite(obs_max), obs_max, 0)
        min_counts = obs_max.astype(int)

    with numpyro.plate("species", n_species, dim=-1):

        # Abundance and detection regression models
        reg_abu = regressor_abu("beta", n_site_covs, prior=prior_beta)
        reg_det = regressor_det("alpha", n_obs_covs, prior=prior_alpha)

        with numpyro.plate("site", n_sites, dim=-2):

            # Site-level random effects
            if site_random_effects:
                site_re_abu = numpyro.sample("site_re_abu", dist.Normal(0, site_re_sd))  # type: ignore
                site_re_det = numpyro.sample("site_re_det", dist.Normal(0, site_re_sd))  # type: ignore
            else:
                site_re_abu = 0.0
                site_re_det = 0.0

            abu_linear = (
                reshape_predictions(reg_abu(site_covs_flat), site_shape)
                + w[:, None]
                + site_re_abu
            )

            with numpyro.plate("period", n_periods, dim=-3):

                # Abundance process
                abundance = numpyro.deterministic("abundance", jnp.exp(abu_linear))
                support = jnp.arange(max_abundance + 1)
                logits = dist.Poisson(abundance[..., None]).log_prob(support)
                if min_counts is not None:
                    logits = jnp.where(
                        support < min_counts[..., None], -jnp.inf, logits
                    )
                numpyro.factor("N_i_trunc_norm", jsp.logsumexp(logits, axis=-1))
                N_i = numpyro.sample(
                    "N_i",
                    dist.Categorical(logits=logits),
                    infer={"enumerate": "parallel"},
                )

                with numpyro.plate("replicate", n_replicates, dim=-4):

                    # Observation-level random effects
                    if obs_random_effects:
                        obs_re = numpyro.sample("obs_re", dist.Normal(0, obs_re_sd))  # type: ignore
                    else:
                        obs_re = 0.0

                    prob_detection = numpyro.deterministic(
                        "prob_detection",
                        jax.nn.sigmoid(
                            reshape_predictions(reg_det(obs_covs_flat), obs_shape)
                            + site_re_det
                            + obs_re
                        ),
                    )

                    with mask_missing_obs(obs):
                        numpyro.sample(
                            "y",
                            dist.Binomial(
                                total_count=N_i[None, ...], probs=prob_detection
                            ),  # type: ignore
                            obs=obs,
                        )


def simulate_nmixture(
    n_site_covs: int = 1,
    n_obs_covs: int = 1,
    n_sites: int = 100,
    n_periods: int = 1,
    n_species: int = 1,
    deployment_days_per_site: int = 365,
    session_duration: int = 7,
    simulate_missing: bool = False,
    min_abundance: float = 0.5,
    max_abundance: float = 6.0,
    min_observation_rate: float = 0.5,
    max_observation_rate: float = 4.0,
    random_seed: int = 0,
    spatial: bool = False,
    gp_sd: float = 1.0,
    gp_l: float = 0.2,
    site_random_effects: bool = False,
    obs_random_effects: bool = False,
    site_re_sd: float = 0.5,
    obs_re_sd: float = 0.3,
) -> tuple[dict, dict]:
    """Simulate data for :func:`nmixture`.

    Returns ``(data, true_params)`` for :func:`fit`.

    Examples
    --------
    >>> from biolith.models import simulate_nmixture
    >>> data, params = simulate_nmixture()
    >>> sorted(data.keys())
    ['coords', 'ell', 'obs', 'obs_covs', 'site_covs']
    """

    # Initialize random number generator
    rng = np.random.default_rng(random_seed)
    if spatial:
        coords = rng.uniform(0, 1, size=(n_sites, 2))
    else:
        coords = None

    N_i = None
    while (
        N_i is None
        or np.mean(N_i) < min_abundance
        or np.mean(N_i) > max_abundance
        or np.mean(obs[np.isfinite(obs)]) < min_observation_rate
        or np.mean(obs[np.isfinite(obs)]) > max_observation_rate
    ):

        # Generate intercept and slopes
        beta = rng.normal(size=(n_species, n_site_covs + 1))
        alpha = rng.normal(size=(n_species, n_obs_covs + 1))

        # Generate site-level covariates and spatial effects
        site_covs = rng.normal(size=(n_sites, n_site_covs))
        if spatial and coords is not None:
            w, ell = simulate_spatial_effects(coords, gp_sd=gp_sd, gp_l=gp_l, rng=rng)
        else:
            w, ell = np.zeros(n_sites), 0.0

        # Generate random effects
        if site_random_effects:
            site_re_abu = rng.normal(0, site_re_sd, size=(n_species, n_sites))
            site_re_det = rng.normal(0, site_re_sd, size=(n_species, n_sites))
        else:
            site_re_abu = np.zeros((n_species, n_sites))
            site_re_det = np.zeros((n_species, n_sites))

        abundance = np.exp(
            beta[:, 0][:, None]
            + np.tensordot(beta[:, 1:], site_covs, axes=([1], [1]))
            + w[None, :]
            + site_re_abu
        )
        N_i = rng.poisson(
            abundance[:, None, :], size=(n_species, n_periods, n_sites)
        )

        # Generate detection data
        n_replicates = round(deployment_days_per_site / session_duration)
        obs_covs = rng.normal(size=(n_sites, n_periods, n_replicates, n_obs_covs))

        if obs_random_effects:
            obs_re = rng.normal(
                0, obs_re_sd, size=(n_species, n_sites, n_periods, n_replicates)
            )
        else:
            obs_re = np.zeros((n_species, n_sites, n_periods, n_replicates))

        prob_detection = 1 / (
            1
            + np.exp(
                -(
                    alpha[:, 0][:, None, None, None]
                    + np.tensordot(alpha[:, 1:], obs_covs, axes=([1], [3]))
                    + site_re_det[:, :, None, None]
                    + obs_re
                )
            )
        )

        N_i_site = N_i.transpose(0, 2, 1)
        obs = rng.binomial(n=N_i_site[..., None], p=prob_detection).astype(float)

        if simulate_missing:
            obs[rng.choice([True, False], size=obs.shape, p=[0.2, 0.8])] = np.nan
            obs_covs[rng.choice([True, False], size=obs_covs.shape, p=[0.05, 0.95])] = (
                np.nan
            )
            site_covs[
                rng.choice([True, False], size=site_covs.shape, p=[0.05, 0.95])
            ] = np.nan

    print(f"True abundance: {np.mean(N_i):.4f}")
    print(f"Mean count: {np.mean(obs[np.isfinite(obs)]):.4f}")

    true_params = dict(
        N_i=N_i,
        abundance=abundance,
        beta=beta,
        alpha=alpha,
        w=w,
        gp_sd=gp_sd,
        gp_l=gp_l,
    )

    if site_random_effects:
        true_params.update(
            {
                "site_re_abu": site_re_abu,
                "site_re_det": site_re_det,
                "site_re_sd": site_re_sd,
            }
        )

    if obs_random_effects:
        true_params.update(
            {
                "obs_re": obs_re,
                "obs_re_sd": obs_re_sd,
            }
        )

    return (
        dict(
            site_covs=site_covs,
            obs_covs=obs_covs,
            obs=obs,
            coords=coords,
            ell=ell,
        ),
        true_params,
    )


def test_nmixture():
    data, true_params = simulate_nmixture(
        simulate_missing=True,
        deployment_days_per_site=70,
        session_duration=7,
        min_abundance=1.0,
        min_observation_rate=1.0,
        max_observation_rate=6.0,
    )
    max_abundance = int(np.nanmax(data["obs"]))

    from biolith.utils import fit

    results = fit(
        nmixture,
        **data,
        max_abundance=max_abundance,
        num_chains=1,
        num_samples=300,
        num_warmup=300,
        timeout=600,
    )

    assert (
        np.allclose(
            results.samples["abundance"].mean(),
            true_params["abundance"].mean(),
            rtol=0.2,
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


def test_nmixture_multi_season():
    data, true_params = simulate_nmixture(
        simulate_missing=True,
        n_periods=3,
        deployment_days_per_site=70,
        session_duration=7,
        min_abundance=1.0,
        min_observation_rate=1.0,
        max_observation_rate=6.0,
    )
    max_abundance = int(np.nanmax(data["obs"]))

    from biolith.utils import fit

    results = fit(
        nmixture,
        **data,
        max_abundance=max_abundance,
        num_chains=1,
        num_samples=200,
        num_warmup=200,
        timeout=600,
    )

    assert (
        np.allclose(
            results.samples["abundance"].mean(),
            true_params["abundance"].mean(),
            rtol=0.25,
        )
    )


def test_nmixture_multi_species():
    data, _ = simulate_nmixture(
        simulate_missing=True,
        n_species=2,
        n_sites=30,
        deployment_days_per_site=70,
        session_duration=7,
        min_abundance=1.0,
        min_observation_rate=1.0,
        max_observation_rate=6.0,
    )
    max_abundance = int(np.nanmax(data["obs"]))

    from biolith.utils import fit

    results = fit(
        nmixture,
        **data,
        max_abundance=max_abundance,
        num_chains=1,
        num_samples=150,
        timeout=600,
    )

    assert results.samples["abundance"].shape[-1] == 2


def test_nmixture_spatial():
    data, true_params = simulate_nmixture(
        simulate_missing=True,
        spatial=True,
        deployment_days_per_site=70,
        session_duration=7,
        min_abundance=1.0,
        min_observation_rate=1.0,
        max_observation_rate=6.0,
    )
    max_abundance = int(np.nanmax(data["obs"]))

    from biolith.utils import fit

    results = fit(
        nmixture,
        **data,
        max_abundance=max_abundance,
        num_chains=1,
        num_samples=200,
        num_warmup=200,
        timeout=600,
    )

    assert (
        np.allclose(
            results.samples["abundance"].mean(),
            true_params["abundance"].mean(),
            rtol=0.25,
        )
    )
    assert ("gp_sd" in results.samples)
    assert ("gp_l" in results.samples)
    assert (results.samples["gp_sd"].mean() > 0)
    assert (results.samples["gp_l"].mean() > 0)


def test_nmixture_site_random_effects():
    data, true_params = simulate_nmixture(
        simulate_missing=True,
        site_random_effects=True,
        obs_random_effects=False,
        deployment_days_per_site=140,
        session_duration=7,
        min_abundance=1.0,
        min_observation_rate=1.0,
        max_observation_rate=6.0,
    )
    max_abundance = int(np.nanmax(data["obs"]))

    from biolith.utils import fit

    results = fit(
        nmixture,
        **data,
        site_random_effects=True,
        obs_random_effects=False,
        max_abundance=max_abundance,
        num_chains=1,
        num_samples=300,
        timeout=600,
    )

    assert ("site_re_sd" in results.samples)
    assert ("site_re_abu" in results.samples)
    assert ("site_re_det" in results.samples)
    assert (results.samples["site_re_sd"].mean() > 0)
    assert (
        np.allclose(
            results.samples["abundance"].mean(),
            true_params["abundance"].mean(),
            rtol=0.25,
        )
    )


def test_nmixture_obs_random_effects():
    data, true_params = simulate_nmixture(
        simulate_missing=True,
        deployment_days_per_site=70,
        session_duration=7,
        min_abundance=1.0,
        min_observation_rate=1.0,
        max_observation_rate=6.0,
    )
    max_abundance = int(np.nanmax(data["obs"]))

    from biolith.utils import fit

    results = fit(
        nmixture,
        **data,
        obs_random_effects=True,
        max_abundance=max_abundance,
        num_chains=1,
        num_samples=300,
        timeout=600,
    )

    assert ("obs_re_sd" in results.samples)
    assert ("obs_re" in results.samples)
    assert (results.samples["obs_re_sd"].mean() > 0)
    assert (
        np.allclose(
            results.samples["abundance"].mean(),
            true_params["abundance"].mean(),
            rtol=0.25,
        )
    )


def test_nmixture_combined_random_effects():
    data, _ = simulate_nmixture(
        simulate_missing=True,
        deployment_days_per_site=70,
        session_duration=7,
        min_abundance=1.0,
        min_observation_rate=1.0,
        max_observation_rate=6.0,
    )
    max_abundance = int(np.nanmax(data["obs"]))

    from biolith.utils import fit

    results = fit(
        nmixture,
        **data,
        site_random_effects=True,
        obs_random_effects=True,
        max_abundance=max_abundance,
        num_chains=1,
        num_warmup=10,
        num_samples=10,
        timeout=600,
    )

    assert ("site_re_sd" in results.samples)
    assert ("site_re_abu" in results.samples)
    assert ("site_re_det" in results.samples)
    assert ("obs_re_sd" in results.samples)
    assert ("obs_re" in results.samples)
