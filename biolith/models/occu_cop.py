from typing import Optional, Type

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from biolith.regression import AbstractRegression, LinearRegression
from biolith.utils.modeling import flatten_covariates, mask_missing_obs, reshape_predictions
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
    n_species: int = 1,
    prior_beta: dist.Distribution = dist.Normal(),
    prior_alpha: dist.Distribution = dist.Normal(),
    regressor_occ: Type[AbstractRegression] = LinearRegression,
    regressor_det: Type[AbstractRegression] = LinearRegression,
    prior_rate_fp_constant: dist.Distribution = dist.Exponential(),
    prior_rate_fp_unoccupied: dist.Distribution = dist.Exponential(),
    prior_gp_sd: dist.Distribution = dist.HalfNormal(1.0),
    prior_gp_length: dist.Distribution = dist.HalfNormal(1.0),
    site_random_effects: bool = False,
    obs_random_effects: bool = False,
    prior_site_re_sd: dist.Distribution = dist.HalfNormal(1.0),
    prior_obs_re_sd: dist.Distribution = dist.HalfNormal(1.0),
) -> None:
    """Count occupancy model using a Poisson detection process, inspired by Pautrel et
    al. (2024).

    References
    ----------
        - Pautrel, L., Moulherat, S., Gimenez, O., & Etienne, M.-P. (2024). Analysing biodiversity observation data collected in continuous time: Should we use discrete- or continuous-time occupancy models? Methods in Ecology and Evolution, 15, 935–950.

    Parameters
    ----------
    site_covs : jnp.ndarray
        Per-site covariates, shape (n_sites, n_site_covs).
    obs_covs : jnp.ndarray
        Observation covariates, shape (n_sites, n_periods, n_replicates, n_obs_covs).
    coords : Optional[jnp.ndarray], optional
        Site coordinates for spatial effects, shape (n_sites, 2).
    ell : float
        Spatial correlation length for the GP prior.
    session_duration : Optional[jnp.ndarray], optional
        Duration of each sampling session, shape (n_sites, n_periods, n_replicates).
    false_positives_constant : bool
        If True, model a constant false positive rate for all sites.
    false_positives_unoccupied : bool
        If True, model a false positive rate only for unoccupied sites.
    obs : Optional[jnp.ndarray], optional
        Observed counts, shape (n_species, n_sites, n_periods, n_replicates).
    n_species : int
        Number of species. Used when ``obs`` is None.
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
    >>> from biolith.models import occu_cop, simulate_cop
    >>> from biolith.utils import fit
    >>> data, _ = simulate_cop()
    >>> results = fit(occu_cop, **data)
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
    assert (
        session_duration is None or session_duration.ndim == 3
    ), "session_duration must be None or of shape (n_sites, n_periods, n_replicates)"
    assert not (
        false_positives_constant and false_positives_unoccupied
    ), "false_positives_constant and false_positives_unoccupied cannot both be True"

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
    if session_duration is not None:
        assert (
            n_sites == session_duration.shape[0]
        ), "session_duration must have n_sites rows"
        assert (
            n_periods == session_duration.shape[1]
        ), "session_duration must have n_periods columns"
        assert (
            n_replicates == session_duration.shape[2]
        ), "session_duration must have n_replicates columns"

    if session_duration is None:
        # If session_duration is not provided, assume a constant duration of 1 for each time period
        session_duration = jnp.ones((n_sites, n_periods, n_replicates))

    # Mask observations where covariates are missing
    obs_mask = jnp.isnan(obs_covs).any(axis=-1) | jnp.isnan(site_covs).any(axis=-1)[
        :, None, None
    ]
    obs = jnp.where(obs_mask[None, ...], jnp.nan, obs) if obs is not None else None
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
    session_duration = session_duration.transpose((2, 1, 0))[..., None]
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
                    rate_detection = numpyro.deterministic(
                        "rate_detection",
                        jnp.exp(
                            reshape_predictions(reg_det(obs_covs_flat), obs_shape)
                            + site_re_det
                            + obs_re
                        ),
                    )
                    l_det = (
                        z * rate_detection
                        + (1 - z) * rate_fp_unoccupied
                        + rate_fp_constant
                    )

                    with mask_missing_obs(obs):
                        numpyro.sample(
                            "y",
                            dist.Poisson(session_duration * l_det),
                            obs=obs,
                        )


def simulate_cop(
    n_site_covs: int = 1,
    n_obs_covs: int = 1,
    n_sites: int = 100,
    n_species: int = 1,
    n_periods: int = 1,
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
        )  # matrix of latent occupancy status for each species, period, and site

        # Generate detection data
        n_replicates = round(deployment_days_per_site / session_duration)

        # Create matrix of detection covariates
        obs_covs = rng.normal(size=(n_sites, n_periods, n_replicates, n_obs_covs))
        detection_rate = np.exp(
            alpha[:, 0][:, None, None, None]
            + np.tensordot(alpha[:, 1:], obs_covs, axes=([1], [3]))
        )

        # Create matrix of detections
        obs = np.zeros((n_species, n_sites, n_periods, n_replicates))
        z_site = z.transpose(0, 2, 1)

        obs = rng.poisson(
            lam=(
                session_duration
                * (
                    detection_rate * z_site[..., None]
                    + rate_fp * (1 - z_site[..., None])
                )
            ),
            size=(n_species, n_sites, n_periods, n_replicates),
        )
        obs = obs.astype(float)

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
    session_duration_arr = np.full((n_sites, n_periods, n_replicates), session_duration)

    return dict(
        site_covs=site_covs,
        obs_covs=obs_covs,
        session_duration=session_duration_arr,
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



def test_occu():
    data, true_params = simulate_cop(simulate_missing=True)

    from biolith.utils import fit

    results = fit(occu_cop, **data, timeout=600)

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

def test_occu_multi_season():
    data, true_params = simulate_cop(simulate_missing=True, n_periods=3)

    from biolith.utils import fit

    results = fit(
        occu_cop,
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
    data, _ = simulate_cop(simulate_missing=True, n_species=2, n_sites=30)

    from biolith.utils import fit

    results = fit(
        occu_cop,
        **data,
        num_chains=1,
        num_samples=100,
        num_warmup=100,
        timeout=600,
    )

    assert results.samples["psi"].shape[-1] == 2

def test_occu_spatial():
    data, true_params = simulate_cop(simulate_missing=True, spatial=True)

    from biolith.utils import fit

    results = fit(occu_cop, **data, timeout=600)

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
    data, true_params = simulate_cop(simulate_missing=True)

    from biolith.utils import fit

    results = fit(
        occu_cop,
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
    data, true_params = simulate_cop(simulate_missing=True)

    from biolith.utils import fit

    results = fit(
        occu_cop,
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
    data, true_params = simulate_cop(simulate_missing=True)

    from biolith.utils import fit

    results = fit(
        occu_cop,
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

