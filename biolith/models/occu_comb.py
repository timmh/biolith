import unittest
from typing import Optional, Tuple, Type

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from biolith.regression import AbstractRegression, LinearRegression
from biolith.utils.modeling import (
    flatten_covariates,
    mask_missing_obs,
    reshape_predictions,
)
from biolith.utils.spatial import sample_spatial_effects, simulate_spatial_effects


def occu_comb(
    site_covs: jnp.ndarray,
    PC_obs_covs: jnp.ndarray,
    ARU_obs_covs: jnp.ndarray,
    scores_obs: jnp.ndarray,
    coords: Optional[jnp.ndarray] = None,
    ell: float = 1.0,
    PC_obs: Optional[jnp.ndarray] = None,
    ARU_obs: Optional[jnp.ndarray] = None,
    n_species: int = 1,
    prior_beta: dist.Distribution = dist.Normal(),
    prior_alpha: dist.Distribution = dist.Normal(),
    regressor_occ: Type[AbstractRegression] = LinearRegression,
    regressor_PC_det: Type[AbstractRegression] = LinearRegression,
    regressor_ARU_det: Type[AbstractRegression] = LinearRegression,
    prior_ARU_prob_fp_constant: dist.Distribution = dist.Beta(2, 5),
    prior_ARU_prob_fp_unoccupied: dist.Distribution = dist.Beta(2, 5),
    prior_mu: dist.Distribution | Tuple[dist.Distribution] = dist.Normal(0, 10),
    prior_sigma: dist.Distribution | Tuple[dist.Distribution] = dist.Gamma(5, 1),
    prior_gp_sd: dist.Distribution = dist.HalfNormal(1.0),
    prior_gp_length: dist.Distribution = dist.HalfNormal(1.0),
    site_random_effects: bool = False,
    PC_obs_random_effects: bool = False,
    ARU_obs_random_effects: bool = False,
    prior_site_re_sd: dist.Distribution = dist.HalfNormal(1.0),
    prior_obs_re_sd: dist.Distribution = dist.HalfNormal(1.0),
) -> None:
    """Combined continuous-score occupancy model, combining point count and ARU
    detections. Supports multiple species and multiple seasons (periods).

    Parameters
    ----------
    site_covs : jnp.ndarray
        Site-level covariates of shape (n_sites, n_site_covs).
    PC_obs_covs : jnp.ndarray
        Observation covariates of shape (n_sites, n_periods, PC_replicates,
        n_PC_obs_covs) for point count data.
    ARU_obs_covs : jnp.ndarray
        Observation covariates of shape (n_sites, n_periods, ARU_replicates,
        n_ARU_obs_covs) for ARU data.
    scores_obs : jnp.ndarray
        Continuous score observations of shape (n_species, n_sites, n_periods,
        scores_replicates).
    coords : jnp.ndarray, optional
        Coordinates for a spatial random effect when provided, shape (n_sites, 2).
    ell : float
        Spatial kernel length scale used if coords is provided.
    PC_obs : jnp.ndarray, optional
        Point count observations of shape (n_species, n_sites, n_periods,
        PC_replicates) or None.
    ARU_obs : jnp.ndarray, optional
        ARU binary observations of shape (n_species, n_sites, n_periods,
        ARU_replicates) or None.
    n_species : int
        Number of species. Inferred from ``scores_obs`` when provided.
    prior_beta : dist.Distribution
        Prior for occupancy coefficients.
    prior_alpha : dist.Distribution
        Prior for detection coefficients.
    regressor_occ : Type[AbstractRegression]
        Class for the occupancy regression model, defaults to LinearRegression.
    regressor_PC_det : Type[AbstractRegression]
        Class for the point count detection regression model, defaults to
        LinearRegression.
    regressor_ARU_det : Type[AbstractRegression]
        Class for the ARU detection regression model, defaults to LinearRegression.
    prior_ARU_prob_fp_constant : dist.Distribution
        Prior for the per-species ARU constant false positive rate.
    prior_ARU_prob_fp_unoccupied : dist.Distribution
        Prior for the per-species ARU false positive rate at unoccupied sites.
    prior_mu : dist.Distribution | Tuple[dist.Distribution]
        Prior for the per-species mean continuous scores.
    prior_sigma : dist.Distribution | Tuple[dist.Distribution]
        Prior for the per-species standard deviation of continuous scores.
    prior_gp_sd : dist.Distribution
        Prior distribution for the spatial random effects scale.
    prior_gp_length : dist.Distribution
        Prior distribution for the spatial kernel length scale.
    site_random_effects : bool
        Flag indicating whether to include site-level random effects.
    PC_obs_random_effects : bool
        Flag indicating whether to include observation-level random effects for
        point counts.
    ARU_obs_random_effects : bool
        Flag indicating whether to include observation-level random effects for
        ARUs.
    prior_site_re_sd : dist.Distribution
        Prior for the standard deviation of site-level random effects.
    prior_obs_re_sd : dist.Distribution
        Prior for the standard deviation of observation-level random effects.
    """

    # Check input data
    assert (
        PC_obs is None or PC_obs.ndim == 4
    ), "PC_obs must be None or of shape (n_species, n_sites, n_periods, PC_replicates)"
    assert (
        ARU_obs is None or ARU_obs.ndim == 4
    ), "ARU_obs must be None or of shape (n_species, n_sites, n_periods, ARU_replicates)"
    assert (
        scores_obs.ndim == 4
    ), "scores_obs must be of shape (n_species, n_sites, n_periods, scores_replicates)"
    assert site_covs.ndim == 2, "site_covs must be of shape (n_sites, n_site_covs)"
    assert (
        PC_obs_covs.ndim == 4
    ), "PC_obs_covs must be of shape (n_sites, n_periods, PC_replicates, n_PC_obs_covs)"
    assert (
        ARU_obs_covs.ndim == 4
    ), "ARU_obs_covs must be of shape (n_sites, n_periods, ARU_replicates, n_ARU_obs_covs)"

    n_sites = site_covs.shape[0]
    n_periods = PC_obs_covs.shape[1]
    PC_replicates = PC_obs_covs.shape[2]
    ARU_replicates = ARU_obs_covs.shape[2]
    scores_replicates = scores_obs.shape[3]
    n_species = scores_obs.shape[0]

    n_site_covs = site_covs.shape[1]
    n_PC_obs_covs = PC_obs_covs.shape[3]
    n_ARU_obs_covs = ARU_obs_covs.shape[3]

    assert (
        n_sites == site_covs.shape[0] == PC_obs_covs.shape[0] == ARU_obs_covs.shape[0]
    ), "site_covs, PC_obs_covs, and ARU_obs_covs must have the same number of sites"
    assert (
        PC_obs_covs.shape[1] == ARU_obs_covs.shape[1]
    ), "PC_obs_covs and ARU_obs_covs must have the same number of periods"
    assert scores_obs.shape[1] == n_sites, "scores_obs must have n_sites in dimension 1"
    assert (
        scores_obs.shape[2] == n_periods
    ), "scores_obs must have n_periods in dimension 2"

    if PC_obs is not None:
        assert PC_obs.shape[0] == n_species
        assert PC_obs.shape[1] == n_sites
        assert PC_obs.shape[2] == n_periods
        assert PC_obs.shape[3] == PC_replicates
    if ARU_obs is not None:
        assert ARU_obs.shape[0] == n_species
        assert ARU_obs.shape[1] == n_sites
        assert ARU_obs.shape[2] == n_periods
        assert ARU_obs.shape[3] == ARU_replicates

    # Mask observations where covariates are missing
    site_missing = jnp.isnan(site_covs).any(axis=-1)  # (n_sites,)

    PC_obs_mask = jnp.isnan(PC_obs_covs).any(axis=-1) | site_missing[:, None, None]
    # (n_sites, n_periods, PC_replicates)
    PC_obs = (
        jnp.where(PC_obs_mask[None, ...], jnp.nan, PC_obs)
        if PC_obs is not None
        else None
    )

    ARU_obs_mask = jnp.isnan(ARU_obs_covs).any(axis=-1) | site_missing[:, None, None]
    # (n_sites, n_periods, ARU_replicates)
    ARU_obs = (
        jnp.where(ARU_obs_mask[None, ...], jnp.nan, ARU_obs)
        if ARU_obs is not None
        else None
    )

    scores_obs = jnp.where(site_missing[None, :, None, None], jnp.nan, scores_obs)

    PC_obs_covs = jnp.nan_to_num(PC_obs_covs)
    ARU_obs_covs = jnp.nan_to_num(ARU_obs_covs)
    site_covs = jnp.nan_to_num(site_covs)

    # Spatial random effects (shared across species)
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
    if PC_obs_random_effects:
        PC_obs_re_sd = numpyro.sample("PC_obs_re_sd", prior_obs_re_sd)
    if ARU_obs_random_effects:
        ARU_obs_re_sd = numpyro.sample("ARU_obs_re_sd", prior_obs_re_sd)

    # Transpose covariate arrays to (n_covs, *obs_shape) for flatten_covariates
    # site_covs: (n_sites, n_site_covs) -> (n_site_covs, n_sites)
    # PC_obs_covs: (n_sites, n_periods, PC_replicates, n_PC_obs_covs)
    #           -> (n_PC_obs_covs, PC_replicates, n_periods, n_sites)
    site_covs = site_covs.transpose((1, 0))
    PC_obs_covs = PC_obs_covs.transpose((3, 2, 1, 0))
    ARU_obs_covs = ARU_obs_covs.transpose((3, 2, 1, 0))

    # Transpose obs arrays to (replicates, n_periods, n_sites, n_species)
    PC_obs = PC_obs.transpose((3, 2, 1, 0)) if PC_obs is not None else None
    ARU_obs = ARU_obs.transpose((3, 2, 1, 0)) if ARU_obs is not None else None
    scores_obs = scores_obs.transpose((3, 2, 1, 0))

    # Flatten covariates for regression: (n_obs, n_covs)
    site_covs_flat, site_shape = flatten_covariates(site_covs)
    PC_obs_covs_flat, PC_obs_shape = flatten_covariates(PC_obs_covs)
    ARU_obs_covs_flat, ARU_obs_shape = flatten_covariates(ARU_obs_covs)

    with numpyro.plate("species", n_species, dim=-1):
        # Occupancy and detection regression models — one set per species
        reg_occ = regressor_occ("beta", n_site_covs, prior=prior_beta)
        reg_PC_det = regressor_PC_det("alpha_PC", n_PC_obs_covs, prior=prior_alpha)
        reg_ARU_det = regressor_ARU_det("alpha_ARU", n_ARU_obs_covs, prior=prior_alpha)

        # Per-species ARU false positive parameters
        ARU_prob_fp_constant = numpyro.sample(
            "ARU_prob_fp_constant", prior_ARU_prob_fp_constant
        )
        ARU_prob_fp_unoccupied = numpyro.sample(
            "ARU_fp_unoccupied", prior_ARU_prob_fp_unoccupied
        )

        # Per-species scores distribution parameters
        prior_mus = prior_mu if isinstance(prior_mu, tuple) else (prior_mu, prior_mu)
        mu0 = numpyro.sample("mu0", prior_mus[0])
        mu1 = numpyro.sample("mu1", dist.TruncatedDistribution(prior_mus[1], low=mu0))  # type: ignore
        prior_sigmas = (
            prior_sigma
            if isinstance(prior_sigma, tuple)
            else (prior_sigma, prior_sigma)
        )
        sigma0 = numpyro.sample("sigma0", prior_sigmas[0])  # type: ignore
        sigma1 = numpyro.sample("sigma1", prior_sigmas[1])  # type: ignore

        with numpyro.plate("site", n_sites, dim=-2):
            # Site-level random effects
            if site_random_effects:
                site_re_occ = numpyro.sample("site_re_occ", dist.Normal(0, site_re_sd))  # type: ignore
                site_re_det = numpyro.sample("site_re_det", dist.Normal(0, site_re_sd))  # type: ignore
            else:
                site_re_occ = 0.0
                site_re_det = 0.0

            # Occupancy linear predictor — computed once, broadcast across periods
            occ_linear = (
                reshape_predictions(reg_occ(site_covs_flat), site_shape)
                + w[:, None]
                + site_re_occ
            )

            with numpyro.plate("period", n_periods, dim=-3):
                # Occupancy process
                psi = numpyro.deterministic("psi", jax.nn.sigmoid(occ_linear))
                z = numpyro.sample(
                    "z",
                    dist.Bernoulli(probs=psi),  # type: ignore
                    infer={"enumerate": "parallel"},
                )

                with numpyro.plate("PC_replicate", PC_replicates, dim=-4):
                    # Observation-level random effects for PC
                    if PC_obs_random_effects:
                        PC_obs_re = numpyro.sample(
                            "PC_obs_re",
                            dist.Normal(0, PC_obs_re_sd),  # type: ignore
                        )
                    else:
                        PC_obs_re = 0.0

                    # PC detection process — no false positives
                    PC_prob_detection = numpyro.deterministic(
                        "PC_prob_detection",
                        jax.nn.sigmoid(
                            reshape_predictions(
                                reg_PC_det(PC_obs_covs_flat), PC_obs_shape
                            )
                            + site_re_det
                            + PC_obs_re
                        ),
                    )

                    with mask_missing_obs(PC_obs):
                        numpyro.sample(
                            "y_pc",
                            dist.Bernoulli(z * PC_prob_detection),  # type: ignore
                            obs=PC_obs,
                        )

                with numpyro.plate("ARU_replicate", ARU_replicates, dim=-4):
                    # Observation-level random effects for ARU
                    if ARU_obs_random_effects:
                        ARU_obs_re = numpyro.sample(
                            "ARU_obs_re",
                            dist.Normal(0, ARU_obs_re_sd),  # type: ignore
                        )
                    else:
                        ARU_obs_re = 0.0

                    # ARU detection process — with false positives
                    ARU_prob_detection = numpyro.deterministic(
                        "ARU_prob_detection",
                        jax.nn.sigmoid(
                            reshape_predictions(
                                reg_ARU_det(ARU_obs_covs_flat), ARU_obs_shape
                            )
                            + site_re_det
                            + ARU_obs_re
                        ),
                    )
                    ARU_prob_detection_fp = numpyro.deterministic(
                        "ARU_prob_detection_fp",
                        1
                        - (1 - z * ARU_prob_detection)
                        * (1 - ARU_prob_fp_constant)
                        * (1 - (1 - z) * ARU_prob_fp_unoccupied),
                    )

                    with mask_missing_obs(ARU_obs):
                        numpyro.sample(
                            "y_aru",
                            dist.Bernoulli(ARU_prob_detection_fp),  # type: ignore
                            obs=ARU_obs,
                        )

                with numpyro.plate("scores_replicate", scores_replicates, dim=-4):
                    with mask_missing_obs(scores_obs):
                        numpyro.sample(
                            "scores",
                            dist.Normal(  # type: ignore
                                (1 - z) * mu0 + z * mu1,
                                (1 - z) * sigma0 + z * sigma1,
                            ),
                            obs=scores_obs,
                        )


def simulate_comb(
    n_site_covs: int = 1,
    n_PC_covs: int = 1,
    n_ARU_covs: int = 1,
    n_sites: int = 100,
    n_species: int = 1,
    n_periods: int = 1,
    PC_replicates: int = 3,
    ARU_replicates: int = 24,
    scores_replicates: int = 24,
    ARU_prob_fp_constant: float = 0.0,
    ARU_prob_fp_unoccupied: float = 0.0,
    min_occupancy: float = 0.25,
    max_occupancy: float = 0.75,
    min_PC_observation_rate: float = 0.1,
    max_PC_observation_rate: float = 0.9,
    simulate_missing: bool = False,
    random_seed: int = 0,
    spatial: bool = False,
    gp_sd: float = 1.0,
    gp_l: float = 0.2,
    site_random_effects: bool = False,
    PC_obs_random_effects: bool = False,
    ARU_obs_random_effects: bool = False,
    site_re_sd: float = 0.5,
    obs_re_sd: float = 0.3,
) -> tuple[dict, dict]:
    """Generate a synthetic dataset for the :func:`occu_comb` model.

    Returns ``(data, true_params)`` suitable for :func:`fit`.

    Examples
    --------
    >>> from biolith.models.occu_comb import simulate_comb
    >>> data, params = simulate_comb()
    >>> list(data.keys())
    ['site_covs', 'PC_obs_covs', 'ARU_obs_covs', 'PC_obs', 'ARU_obs', 'scores_obs', 'coords', 'ell']
    """

    rng = np.random.default_rng(random_seed)

    if spatial:
        coords = rng.uniform(0, 1, size=(n_sites, 2))
    else:
        coords = None

    z = None
    while (
        z is None
        or z.mean() < min_occupancy
        or z.mean() > max_occupancy
        or np.mean(PC_obs[np.isfinite(PC_obs)]) < min_PC_observation_rate
        or np.mean(PC_obs[np.isfinite(PC_obs)]) > max_PC_observation_rate
    ):
        # ------------------------------------------------------------------ #
        # Occupancy model                                                      #
        # ------------------------------------------------------------------ #

        # beta: (n_species, n_site_covs + 1) — intercept + slopes per species
        beta = rng.normal(size=(n_species, n_site_covs + 1))
        site_covs = rng.normal(size=(n_sites, n_site_covs))

        if spatial and coords is not None:
            w, ell = simulate_spatial_effects(coords, gp_sd=gp_sd, gp_l=gp_l, rng=rng)
        else:
            w, ell = np.zeros(n_sites), 0.0

        # Site-level random effects: (n_species, n_sites)
        if site_random_effects:
            site_re_occ = rng.normal(0, site_re_sd, size=(n_species, n_sites))
            site_re_det = rng.normal(0, site_re_sd, size=(n_species, n_sites))
        else:
            site_re_occ = np.zeros((n_species, n_sites))
            site_re_det = np.zeros((n_species, n_sites))

        # psi: (n_species, n_sites)
        psi = 1 / (
            1
            + np.exp(
                -(
                    beta[:, 0][:, None]  # (n_species, 1)
                    + np.tensordot(
                        beta[:, 1:], site_covs, axes=([1], [1])
                    )  # (n_species, n_sites)
                    + w[None, :]  # (1, n_sites)
                    + site_re_occ  # (n_species, n_sites)
                )
            )
        )

        # z: (n_species, n_periods, n_sites)
        z = rng.binomial(n=1, p=psi[:, None, :], size=(n_species, n_periods, n_sites))

        # z_site: (n_species, n_sites, n_periods) — for broadcasting with replicates
        z_site = z.transpose(0, 2, 1)

        # ------------------------------------------------------------------ #
        # PC detection model                                                   #
        # ------------------------------------------------------------------ #

        # alpha_PC: (n_species, n_PC_covs + 1)
        alpha_PC = rng.normal(size=(n_species, n_PC_covs + 1))
        # PC_obs_covs: (n_sites, n_periods, PC_replicates, n_PC_covs)
        PC_obs_covs = rng.normal(size=(n_sites, n_periods, PC_replicates, n_PC_covs))

        # PC observation-level random effects: (n_species, n_sites, n_periods, PC_replicates)
        if PC_obs_random_effects:
            PC_obs_re = rng.normal(
                0, obs_re_sd, size=(n_species, n_sites, n_periods, PC_replicates)
            )
        else:
            PC_obs_re = np.zeros((n_species, n_sites, n_periods, PC_replicates))

        # lin_PC: (n_species, n_sites, n_periods, PC_replicates)
        lin_PC = (
            alpha_PC[:, 0][:, None, None, None]  # (n_species, 1, 1, 1)
            + np.tensordot(
                alpha_PC[:, 1:], PC_obs_covs, axes=([1], [3])
            )  # (n_species, n_sites, n_periods, PC_replicates)
            + site_re_det[:, :, None, None]  # (n_species, n_sites, 1, 1)
            + PC_obs_re
        )
        PC_prob_detection = 1 / (1 + np.exp(-lin_PC))

        # PC_obs: (n_species, n_sites, n_periods, PC_replicates)
        PC_obs = rng.binomial(
            1,
            z_site[..., None]
            * PC_prob_detection,  # z_site[..., None]: (n_species, n_sites, n_periods, 1)
            size=(n_species, n_sites, n_periods, PC_replicates),
        ).astype(float)

        # ------------------------------------------------------------------ #
        # ARU detection model (with false positives)                          #
        # ------------------------------------------------------------------ #

        # alpha_ARU: (n_species, n_ARU_covs + 1)
        alpha_ARU = rng.normal(size=(n_species, n_ARU_covs + 1))
        # ARU_obs_covs: (n_sites, n_periods, ARU_replicates, n_ARU_covs)
        ARU_obs_covs = rng.normal(size=(n_sites, n_periods, ARU_replicates, n_ARU_covs))

        # ARU observation-level random effects: (n_species, n_sites, n_periods, ARU_replicates)
        if ARU_obs_random_effects:
            ARU_obs_re = rng.normal(
                0, obs_re_sd, size=(n_species, n_sites, n_periods, ARU_replicates)
            )
        else:
            ARU_obs_re = np.zeros((n_species, n_sites, n_periods, ARU_replicates))

        # lin_ARU: (n_species, n_sites, n_periods, ARU_replicates)
        lin_ARU = (
            alpha_ARU[:, 0][:, None, None, None]
            + np.tensordot(alpha_ARU[:, 1:], ARU_obs_covs, axes=([1], [3]))
            + site_re_det[:, :, None, None]
            + ARU_obs_re
        )
        ARU_prob_detection = 1 / (1 + np.exp(-lin_ARU))

        ARU_prob_detection_fp = 1 - (
            (1 - z_site[..., None] * ARU_prob_detection)
            * (1 - ARU_prob_fp_constant)
            * (1 - (1 - z_site[..., None]) * ARU_prob_fp_unoccupied)
        )

        # ARU_obs: (n_species, n_sites, n_periods, ARU_replicates)
        ARU_obs = rng.binomial(
            1,
            ARU_prob_detection_fp,
            size=(n_species, n_sites, n_periods, ARU_replicates),
        ).astype(float)

        # ------------------------------------------------------------------ #
        # Scores model                                                         #
        # ------------------------------------------------------------------ #

        mu0, sigma0 = -3.0, 5.0
        mu1, sigma1 = 2.0, 3.0

        # scores_obs: (n_species, n_sites, n_periods, scores_replicates)
        scores_obs = rng.normal(
            loc=(1 - z_site[..., None]) * mu0 + z_site[..., None] * mu1,
            scale=(1 - z_site[..., None]) * sigma0 + z_site[..., None] * sigma1,
            size=(n_species, n_sites, n_periods, scores_replicates),
        )

    print(f"True occupancy: {z.mean():.4f}")
    print(
        f"Proportion of PC timesteps with detection: {np.mean(PC_obs[np.isfinite(PC_obs)]):.4f}"
    )

    if simulate_missing:
        PC_obs[rng.choice([True, False], size=PC_obs.shape, p=[0.2, 0.8])] = np.nan
        ARU_obs[rng.choice([True, False], size=ARU_obs.shape, p=[0.2, 0.8])] = np.nan
        scores_obs[rng.choice([True, False], size=scores_obs.shape, p=[0.2, 0.8])] = (
            np.nan
        )
        PC_obs_covs[
            rng.choice([True, False], size=PC_obs_covs.shape, p=[0.05, 0.95])
        ] = np.nan
        ARU_obs_covs[
            rng.choice([True, False], size=ARU_obs_covs.shape, p=[0.05, 0.95])
        ] = np.nan
        site_covs[rng.choice([True, False], size=site_covs.shape, p=[0.05, 0.95])] = (
            np.nan
        )

    # Build true_params dict
    true_params = dict(
        z=z,
        beta=beta,
        alpha_PC=alpha_PC,
        alpha_ARU=alpha_ARU,
        mu0=mu0,
        sigma0=sigma0,
        mu1=mu1,
        sigma1=sigma1,
        w=w,
        gp_sd=gp_sd,
        gp_l=gp_l,
    )

    if site_random_effects:
        true_params.update(
            {
                "site_re_occ": site_re_occ,
                "site_re_det": site_re_det,
                "site_re_sd": site_re_sd,
            }
        )

    if PC_obs_random_effects:
        true_params.update({"PC_obs_re": PC_obs_re, "obs_re_sd": obs_re_sd})

    if ARU_obs_random_effects:
        true_params.update({"ARU_obs_re": ARU_obs_re, "obs_re_sd": obs_re_sd})

    return (
        dict(
            site_covs=site_covs,
            PC_obs_covs=PC_obs_covs,
            ARU_obs_covs=ARU_obs_covs,
            PC_obs=PC_obs,
            ARU_obs=ARU_obs,
            scores_obs=scores_obs,
            coords=coords,
            ell=ell,
        ),
        true_params,
    )


class TestOccuCOMB(unittest.TestCase):
    def test_occu_comb(self):
        data, true_params = simulate_comb(simulate_missing=True)

        from biolith.utils import fit

        results = fit(occu_comb, **data, timeout=600, num_chains=1)

        self.assertTrue(
            np.allclose(
                results.samples["psi"].mean(),
                true_params["z"].mean(),
                atol=0.1,
            )
        )

    def test_occu_comb_multi_season(self):
        data, true_params = simulate_comb(simulate_missing=True, n_periods=3)

        from biolith.utils import fit

        results = fit(
            occu_comb,
            **data,
            num_chains=1,
            num_samples=300,
            num_warmup=300,
            timeout=600,
        )

        self.assertTrue(
            np.allclose(
                results.samples["psi"].mean(), true_params["z"].mean(), atol=0.15
            )
        )

    def test_occu_comb_multi_species(self):
        data, _ = simulate_comb(simulate_missing=True, n_species=2, n_sites=30)

        from biolith.utils import fit

        results = fit(
            occu_comb,
            **data,
            num_chains=1,
            num_samples=100,
            num_warmup=100,
            timeout=600,
        )

        self.assertTrue(results.samples["psi"].shape[-1] == 2)

    def test_occu_comb_spatial(self):
        data, true_params = simulate_comb(simulate_missing=True, spatial=True)

        from biolith.utils import fit

        results = fit(occu_comb, **data, timeout=600, num_chains=1)

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

    def test_site_random_effects(self):
        data, true_params = simulate_comb(
            site_random_effects=True,
            PC_obs_random_effects=False,
            ARU_obs_random_effects=False,
        )

        from biolith.utils import fit

        results = fit(
            occu_comb,
            **data,
            site_random_effects=True,
            num_chains=1,
            num_samples=500,
            timeout=600,
        )

        self.assertTrue("site_re_sd" in results.samples)
        self.assertTrue("site_re_occ" in results.samples)
        self.assertTrue("site_re_det" in results.samples)
        self.assertTrue(results.samples["site_re_sd"].mean() > 0)
        self.assertTrue(
            np.allclose(
                results.samples["psi"].mean(), true_params["z"].mean(), atol=0.15
            )
        )

    def test_obs_random_effects(self):
        data, true_params = simulate_comb(
            PC_obs_random_effects=True,
            ARU_obs_random_effects=True,
        )

        from biolith.utils import fit

        results = fit(
            occu_comb,
            **data,
            PC_obs_random_effects=True,
            ARU_obs_random_effects=True,
            num_chains=1,
            num_samples=500,
            timeout=600,
        )

        self.assertTrue("PC_obs_re_sd" in results.samples)
        self.assertTrue("PC_obs_re" in results.samples)
        self.assertTrue("ARU_obs_re_sd" in results.samples)
        self.assertTrue("ARU_obs_re" in results.samples)
        self.assertTrue(
            np.allclose(
                results.samples["psi"].mean(), true_params["z"].mean(), atol=0.15
            )
        )
