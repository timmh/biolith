from typing import Optional, Type

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from biolith.regression import AbstractRegression, LinearRegression
from biolith.utils.modeling import flatten_covariates, mask_missing_obs, reshape_predictions
from biolith.utils.spatial import sample_spatial_effects, simulate_spatial_effects


def occu(
    site_covs: jnp.ndarray,
    obs_covs: jnp.ndarray,
    coords: Optional[jnp.ndarray] = None,
    ell: float = 1.0,
    false_positives_constant: bool = False,
    false_positives_unoccupied: bool = False,
    obs: Optional[jnp.ndarray] = None,
    n_species: int = 1,
    prior_beta: dist.Distribution = dist.Normal(),
    prior_alpha: dist.Distribution = dist.Normal(),
    regressor_occ: Type[AbstractRegression] = LinearRegression,
    regressor_det: Type[AbstractRegression] = LinearRegression,
    prior_prob_fp_constant: dist.Distribution = dist.Beta(2, 5),
    prior_prob_fp_unoccupied: dist.Distribution = dist.Beta(2, 5),
    prior_gp_sd: dist.Distribution = dist.HalfNormal(1.0),
    prior_gp_length: dist.Distribution = dist.HalfNormal(1.0),
    site_random_effects: bool = False,
    obs_random_effects: bool = False,
    prior_site_re_sd: dist.Distribution = dist.HalfNormal(1.0),
    prior_obs_re_sd: dist.Distribution = dist.HalfNormal(1.0),
) -> None:
    """Bernoulli occupancy model inspired by MacKenzie et al. (2002) with optional false
    positives inspired by Royle and Link (2006).

    References
    ----------
        - MacKenzie, D. I., J. D. Nichols, G. B. Lachman, S. Droege, J. Andrew Royle, and C. A. Langtimm. 2002. Estimating Site Occupancy Rates When Detection Probabilities Are Less Than One. Ecology 83: 2248-2255.
        - Royle, J.A., and W.A. Link. 2006. Generalized site occupancy models allowing for false positive and false negative errors. Ecology 87:835-841.

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
    false_positives_constant : bool
        Flag indicating whether to model a constant false positive rate.
    false_positives_unoccupied : bool
        Flag indicating whether to model false positives in unoccupied sites.
    obs : jnp.ndarray, optional
        Observation matrix of shape (n_species, n_sites, n_periods, n_replicates) or None.
    n_species : int
        Number of species. Used when ``obs`` is None.
    prior_beta : numpyro.distributions.Distribution
        Prior distribution for the site-level regression coefficients.
    prior_alpha : numpyro.distributions.Distribution
        Prior distribution for the observation-level regression coefficients.
    regressor_occ : Type[AbstractRegression]
        Class for the occupancy regression model, defaults to LinearRegression.
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
    site_random_effects : bool
        Flag indicating whether to include site-level random effects.
    obs_random_effects : bool
        Flag indicating whether to include observation-level random effects.
    prior_site_re_sd : numpyro.distributions.Distribution
        Prior distribution for the site-level random effect standard deviation.
    prior_obs_re_sd : numpyro.distributions.Distribution
        Prior distribution for the observation-level random effect standard deviation.

    Examples
    --------
    >>> from biolith.models import occu, simulate
    >>> from biolith.utils import fit
    >>> data, _ = simulate()
    >>> results = fit(occu, **data)
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
    # assert obs is None or (obs[np.isfinite(obs)] >= 0).all(), "observations must be non-negative"  # TODO: re-enable
    # assert obs is None or (obs[np.isfinite(obs)] <= 1).all(), "observations must be binary"  # TODO: re-enable
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

    # Mask observations where covariates are missing
    obs_mask = jnp.isnan(obs_covs).any(axis=-1) | jnp.isnan(site_covs).any(axis=-1)[
        :, None, None
    ]
    obs = jnp.where(obs_mask[None, ...], jnp.nan, obs) if obs is not None else None
    obs_covs = jnp.nan_to_num(obs_covs)
    site_covs = jnp.nan_to_num(site_covs)

    # Priors
    # Model false positive rate for both occupied and unoccupied sites
    prob_fp_constant = (
        numpyro.sample("prob_fp_constant", prior_prob_fp_constant)
        if false_positives_constant
        else 0
    )

    # Model false positive rate only for occupied sites
    prob_fp_unoccupied = (
        numpyro.sample("prob_fp_unoccupied", prior_prob_fp_unoccupied)
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
                    prob_detection = numpyro.deterministic(
                        "prob_detection",
                        jax.nn.sigmoid(
                            reshape_predictions(reg_det(obs_covs_flat), obs_shape)
                            + site_re_det
                            + obs_re
                        ),
                    )
                    prob_detection_fp = numpyro.deterministic(
                        "prob_detection_fp",
                        1
                        - (1 - z * prob_detection)
                        * (1 - prob_fp_constant)
                        * (1 - (1 - z) * prob_fp_unoccupied),
                    )

                    with mask_missing_obs(obs):
                        numpyro.sample(
                            "y",
                            dist.Bernoulli(prob_detection_fp),  # type: ignore
                            obs=obs,
                        )


def simulate(
    n_site_covs: int = 1,
    n_obs_covs: int = 1,
    n_sites: int = 100,
    n_species: int = 1,
    n_periods: int = 1,
    deployment_days_per_site: int = 365,
    session_duration: int = 7,
    prob_fp_unoccupied: float = 0.0,
    prob_fp_constant: float = 0.0,
    simulate_missing: bool = False,
    min_occupancy: float = 0.25,
    max_occupancy: float = 0.75,
    min_observation_rate: float = 0.1,
    max_observation_rate: float = 0.5,
    random_seed: int = 0,
    spatial: bool = False,
    gp_sd: float = 1.0,
    gp_l: float = 0.2,
    site_random_effects: bool = False,
    obs_random_effects: bool = False,
    site_re_sd: float = 0.5,
    obs_re_sd: float = 0.3,
) -> tuple[dict, dict]:
    """Generate a synthetic dataset for the :func:`occu` model.

    Returns ``(data, true_params)`` suitable for :func:`fit`.

    Examples
    --------
    >>> from biolith.models import simulate
    >>> data, params = simulate()
    >>> list(data.keys())
    ['site_covs', 'obs_covs', 'obs', 'coords', 'ell']
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

        # Generate random effects
        if site_random_effects:
            site_re_occ = rng.normal(0, site_re_sd, size=(n_species, n_sites))
            site_re_det = rng.normal(0, site_re_sd, size=(n_species, n_sites))
        else:
            site_re_occ = np.zeros((n_species, n_sites))
            site_re_det = np.zeros((n_species, n_sites))
        psi = 1 / (
            1
            + np.exp(
                -(
                    beta[:, 0][:, None]
                    + np.tensordot(beta[:, 1:], site_covs, axes=([1], [1]))
                    + w[None, :]
                    + site_re_occ
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

        # Generate observation-level random effects
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

        # Create matrix of detections
        obs = np.zeros((n_species, n_sites, n_periods, n_replicates))
        prob_detection_fp = np.zeros((n_species, n_sites, n_periods, n_replicates))

        z_site = z.transpose(0, 2, 1)
        prob_detection_fp = (
            1
            - (1 - (z_site[..., None] * prob_detection))
            * (1 - prob_fp_constant)
            * (1 - ((1 - z_site[..., None]) * prob_fp_unoccupied))
        )
        obs = rng.binomial(
            n=1,
            p=prob_detection_fp,
            size=(n_species, n_sites, n_periods, n_replicates),
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

    print(f"True occupancy: {np.mean(z):.4f}")
    print(
        f"Proportion of timesteps with observation: {np.mean(obs[np.isfinite(obs)]):.4f}"
    )

    # Build true_params dict
    true_params = dict(
        z=z,
        beta=beta,
        alpha=alpha,
        w=w,
        gp_sd=gp_sd,
        gp_l=gp_l,
    )

    # Add random effects to true_params if they were simulated
    if site_random_effects:
        true_params.update(
            {
                "site_re_occ": site_re_occ,
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



def test_occu():
    data, true_params = simulate(simulate_missing=True)

    from biolith.utils import fit

    results = fit(occu, **data, timeout=600)

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
    data, true_params = simulate(simulate_missing=True, n_periods=3)

    from biolith.utils import fit

    results = fit(
        occu,
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
    data, _ = simulate(simulate_missing=True, n_species=2, n_sites=30)

    from biolith.utils import fit

    results = fit(
        occu,
        **data,
        num_chains=1,
        num_samples=100,
        num_warmup=100,
        timeout=600,
    )

    assert results.samples["psi"].shape[-1] == 2

def test_occu_fp_constant():
    prob_fp_constant = 0.1
    data, true_params = simulate(
        simulate_missing=True, prob_fp_constant=prob_fp_constant
    )

    from biolith.utils import fit

    results = fit(occu, **data, false_positives_constant=True, timeout=600)

    assert (
        np.allclose(
            results.samples["psi"].mean(), true_params["z"].mean(), atol=0.1
        )
    )
    assert (
        np.allclose(
            results.samples["prob_fp_constant"].mean(), prob_fp_constant, atol=0.1
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

# TODO: fix this test
@pytest.mark.skip(reason="Skipping test for false positives at unoccupied sites")
def test_occu_fp_unoccupied():
    prob_fp_unoccupied = 0.1
    data, true_params = simulate(
        simulate_missing=True, prob_fp_unoccupied=prob_fp_unoccupied
    )

    from biolith.utils import fit

    results = fit(occu, **data, false_positives_unoccupied=True, timeout=600)

    assert (
        np.allclose(
            results.samples["psi"].mean(), true_params["z"].mean(), atol=0.1
        )
    )
    assert (
        np.allclose(
            results.samples["prob_fp_unoccupied"].mean(),
            prob_fp_unoccupied,
            atol=0.1,
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

def test_occu_spatial():
    data, true_params = simulate(simulate_missing=True, spatial=True)

    from biolith.utils import fit

    results = fit(occu, **data, timeout=600)

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

def test_vs_spoccupancy():
    data, true_params = simulate(simulate_missing=True)

    num_samples = 1000

    # fit the biolith model
    try:
        from rpy2.robjects.packages import importr
    except ImportError:
        pytest.skip("rpy2 is not installed.")

    from biolith.utils import fit

    results = fit(
        occu,
        **data,
        timeout=600,
        num_chains=1,
        num_samples=num_samples,
        num_warmup=100,
    )

    # fit the spOccupancy model
    import rpy2.robjects as ro
    import rpy2.robjects.numpy2ri as numpy2ri_module

    # Import R packages
    base_r = importr("base")
    stats_r = importr("stats")
    spOccupancy_r = importr("spOccupancy")

    # Prepare data for R
    # Observations (y): use first species and first period -> shape (n_sites, n_replicates).
    y_py = data["obs"][0, :, 0, :].copy()
    y_r = numpy2ri_module.py2rpy(y_py)  # Converts to R matrix, np.nan to R NA

    # Occupancy covariates (site-level)
    # Replace NaNs with 0, as spOccupancy expects complete covariate data.
    occ_covs_py = np.nan_to_num(
        data["site_covs"].copy()
    )  # Shape: (n_sites, n_site_covs)
    n_sites, n_site_covs = occ_covs_py.shape

    occ_covs_r_elements = {}
    occ_formula_parts = []
    if n_site_covs > 0:
        for i in range(n_site_covs):
            cov_name = f"site_cov{i+1}"
            occ_covs_r_elements[cov_name] = numpy2ri_module.py2rpy(
                occ_covs_py[:, i]
            )
            occ_formula_parts.append(cov_name)
    occ_covs_r_df = ro.DataFrame(occ_covs_r_elements)  # R DataFrame
    occ_formula_str = (
        "~ " + " + ".join(occ_formula_parts) if occ_formula_parts else "~ 1"
    )

    # Detection covariates (observation-level)
    # Replace NaNs with 0.
    det_covs_py = np.nan_to_num(
        data["obs_covs"][:, 0, :, :].copy()
    )  # Shape: (n_sites, n_replicates, n_obs_covs)
    _, n_replicates, n_obs_covs = det_covs_py.shape
    assert y_py.shape[1] == n_replicates, (
        f"number of replicates in y ({y_py.shape[1]}) must match det.covs ({n_replicates})"
    )

    det_covs_r_elements = {}  # For the R named list
    det_formula_parts = []
    if n_obs_covs > 0:
        for i in range(n_obs_covs):
            cov_name = f"obs_cov{i+1}"
            # Convert each (n_sites, time_periods) slice to an R matrix
            det_covs_r_elements[cov_name] = numpy2ri_module.py2rpy(
                det_covs_py[:, :, i]
            )
            det_formula_parts.append(cov_name)
    det_covs_r_list = ro.ListVector(det_covs_r_elements)  # R named list
    det_formula_str = (
        "~ " + " + ".join(det_formula_parts) if det_formula_parts else "~ 1"
    )

    # Consolidate data into an R list for spOccupancy
    sp_data_r = ro.ListVector(
        {"y": y_r, "occ.covs": occ_covs_r_df, "det.covs": det_covs_r_list}
    )

    # Priors: Normal(0,1) for regression coefficients to match biolith's default
    n_beta_params = n_site_covs + 1  # site covariates + intercept
    n_alpha_params = n_obs_covs + 1  # observation covariates + intercept

    priors_list_r = ro.ListVector(
        {
            "beta.normal": ro.ListVector(
                {
                    "mean": base_r.rep(0, n_beta_params),
                    "var": base_r.rep(1, n_beta_params),
                }
            ),
            "alpha.normal": ro.ListVector(
                {
                    "mean": base_r.rep(0, n_alpha_params),
                    "var": base_r.rep(1, n_alpha_params),
                }
            ),
        }
    )

    pg_occ_results_r = spOccupancy_r.PGOcc(
        occ_formula=stats_r.as_formula(occ_formula_str),
        det_formula=stats_r.as_formula(det_formula_str),
        data=sp_data_r,
        priors=priors_list_r,
        n_samples=num_samples,
    )

    beta_samples_r_matrix = numpy2ri_module.rpy2py(
        pg_occ_results_r.rx2("beta.samples")
    )
    alpha_samples_r_matrix = numpy2ri_module.rpy2py(
        pg_occ_results_r.rx2("alpha.samples")
    )
    psi_samples_r_matrix = numpy2ri_module.rpy2py(
        pg_occ_results_r.rx2("psi.samples")
    )

    assert (
        np.allclose(
            psi_samples_r_matrix.mean(), results.samples["psi"].mean(), atol=0.1
        )
    )
    assert (
        np.allclose(
            beta_samples_r_matrix.mean(axis=0),
            [
                results.samples[k].mean()
                for k in [
                    f"cov_state_{i}"
                    for i in range(true_params["beta"].shape[1])
                ]
            ],
            atol=0.5,
        )
    )
    assert (
        np.allclose(
            alpha_samples_r_matrix.mean(axis=0),
            [
                results.samples[k].mean()
                for k in [
                    f"cov_det_{i}"
                    for i in range(true_params["alpha"].shape[1])
                ]
            ],
            atol=0.5,
        )
    )

def test_evaluation():
    data, _ = simulate(simulate_missing=True)

    from biolith.utils import fit, predict

    results = fit(occu, **data, timeout=600)
    posterior_samples = predict(occu, results.mcmc, **data)

    from biolith.evaluation import (
        deviance,
        diagnostics,
        lppd,
        posterior_predictive_check,
        residuals,
    )

    # Test log pointwise predictive density (lppd)
    lppd(occu, posterior_samples, **data)

    # Test posterior predictive checks
    for group_by in ["site", "revisit"]:
        for statistic in ["freeman-tukey", "chi-squared"]:
            posterior_predictive_check(
                posterior_samples,
                data["obs"],
                group_by=group_by,  # type: ignore
                statistic=statistic,  # type: ignore
            )

    # Test residuals
    residuals(posterior_samples, data["obs"])

    # Test deviance
    deviance(occu, posterior_samples, **data)

    # Test diagnostics
    diagnostics(results.mcmc)

def test_regression_methods():
    from biolith.regression import (
        BARTRegression,
        LinearRegression,
        MLPRegression,
    )
    from biolith.utils import fit

    data, true_params = simulate(simulate_missing=True)

    for reg_class in [LinearRegression, MLPRegression, BARTRegression]:
        results = fit(
            occu, **data, regressor_occ=reg_class, num_chains=1, timeout=600
        )
        assert (
            np.allclose(
                results.samples["psi"].mean(), true_params["z"].mean(), atol=0.1
            )
        )

def test_site_random_effects():

    data, true_params = simulate(
        site_random_effects=True,
        obs_random_effects=False,
        deployment_days_per_site=7000,  # to ensure enough data for random effects
    )

    from biolith.evaluation import lppd
    from biolith.utils import fit, predict

    results = fit(
        occu,
        **data,
        num_chains=1,
        num_samples=500,
        timeout=600,
    )
    posterior_samples = predict(occu, results.mcmc, **data)
    l = lppd(occu, posterior_samples, **data)

    results_re = fit(
        occu,
        **data,
        site_random_effects=True,
        obs_random_effects=False,
        num_chains=1,
        num_samples=500,
        timeout=600,
    )
    posterior_samples_re = predict(
        occu,
        results_re.mcmc,
        **data,
        site_random_effects=True,
        obs_random_effects=False,
    )
    l_re = lppd(occu, posterior_samples_re, **data)

    assert (l_re >= 0.95 * l)
    assert ("site_re_sd" in results_re.samples)
    assert ("site_re_occ" in results_re.samples)
    assert ("site_re_det" in results_re.samples)
    assert (results_re.samples["site_re_sd"].mean() > 0)
    assert (
        np.allclose(
            results_re.samples["psi"].mean(), true_params["z"].mean(), atol=0.15
        )
    )

def test_obs_random_effects():
    data, true_params = simulate(simulate_missing=True)

    from biolith.utils import fit

    results = fit(
        occu,
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
    data, true_params = simulate(simulate_missing=True)

    from biolith.utils import fit

    results = fit(
        occu,
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

