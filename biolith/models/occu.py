import unittest
from typing import Optional, Type

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from biolith.regression import AbstractRegression, LinearRegression
from biolith.utils.spatial import sample_spatial_effects, simulate_spatial_effects


def occu(
    site_covs: jnp.ndarray,
    obs_covs: jnp.ndarray,
    coords: Optional[jnp.ndarray] = None,
    ell: float = 1.0,
    false_positives_constant: bool = False,
    false_positives_unoccupied: bool = False,
    obs: Optional[jnp.ndarray] = None,
    prior_beta: dist.Distribution = dist.Normal(),
    prior_alpha: dist.Distribution = dist.Normal(),
    regressor_occ: Type[AbstractRegression] = LinearRegression,
    regressor_det: Type[AbstractRegression] = LinearRegression,
    prior_prob_fp_constant: dist.Distribution = dist.Beta(2, 5),
    prior_prob_fp_unoccupied: dist.Distribution = dist.Beta(2, 5),
    prior_gp_sd: dist.Distribution = dist.HalfNormal(1.0),
    prior_gp_length: dist.Distribution = dist.HalfNormal(1.0),
) -> None:
    """
    Bernoulli occupancy model inspired by MacKenzie et al. (2002) with optional false positives inspired by Royle and Link (2006).

    References
    ----------
        - MacKenzie, D. I., J. D. Nichols, G. B. Lachman, S. Droege, J. Andrew Royle, and C. A. Langtimm. 2002. Estimating Site Occupancy Rates When Detection Probabilities Are Less Than One. Ecology 83: 2248-2255.
        - Royle, J.A., and W.A. Link. 2006. Generalized site occupancy models allowing for false positive and false negative errors. Ecology 87:835-841.

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
    false_positives_unoccupied : bool
        Flag indicating whether to model false positives in unoccupied sites.
    obs : jnp.ndarray, optional
        Observation matrix of shape (n_sites, n_revisits) or None.
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
        obs is None or obs.ndim == 2
    ), "obs must be None or of shape (n_sites, time_periods)"
    assert site_covs.ndim == 2, "site_covs must be of shape (n_sites, n_site_covs)"
    assert (
        obs_covs.ndim == 3
    ), "obs_covs must be of shape (n_sites, time_periods, n_obs_covs)"
    # assert obs is None or (obs[np.isfinite(obs)] >= 0).all(), "observations must be non-negative"  # TODO: re-enable
    # assert obs is None or (obs[np.isfinite(obs)] <= 1).all(), "observations must be binary"  # TODO: re-enable
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

    # Model false positive rate only for occupied sites
    prob_fp_unoccupied = (
        numpyro.sample("prob_fp_unoccupied", prior_prob_fp_unoccupied)
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
    obs = obs.transpose((1, 0)) if obs is not None else None

    with numpyro.plate("site", n_sites, dim=-1):

        # Occupancy process
        psi = numpyro.deterministic(
            "psi",
            jax.nn.sigmoid(reg_occ(site_covs) + w),
        )
        z = numpyro.sample(
            "z", dist.Bernoulli(probs=psi), infer={"enumerate": "parallel"}  # type: ignore
        )

        with numpyro.plate("time_periods", time_periods, dim=-2):

            # Detection process
            prob_detection = numpyro.deterministic(
                f"prob_detection",
                jax.nn.sigmoid(reg_det(obs_covs)),
            )
            prob_detection_fp = numpyro.deterministic(
                "prob_detection_fp",
                1
                - (1 - z * prob_detection)
                * (1 - prob_fp_constant)
                * (1 - (1 - z) * prob_fp_unoccupied),
            )

            if obs is not None:
                with numpyro.handlers.mask(mask=jnp.isfinite(obs)):
                    numpyro.sample(
                        f"y",
                        dist.Bernoulli(prob_detection_fp),  # type: ignore
                        obs=jnp.nan_to_num(obs),
                        infer={"enumerate": "parallel"},
                    )
            else:
                numpyro.sample(
                    f"y",
                    dist.Bernoulli(prob_detection_fp),  # type: ignore
                    infer={"enumerate": "parallel"},
                )


def simulate(
    n_site_covs: int = 1,
    n_obs_covs: int = 1,
    n_sites: int = 100,
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
        prob_detection = 1 / (
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
        prob_detection_fp = np.zeros((n_sites, time_periods))

        for i in range(n_sites):
            # According to the Royle model in unmarked, false positives are generated only if the site is unoccupied
            # Note this is different than how we think about false positives being a random occurrence per image.
            # For now, this is generating positive/negative per time period, which is different than per image.
            prob_detection_fp[i, :] = 1 - (1 - (z[i] * prob_detection[i, :])) * (
                1 - prob_fp_constant
            ) * (1 - ((1 - z[i]) * prob_fp_unoccupied))
            obs[i, :] = rng.binomial(n=1, p=prob_detection_fp[i, :], size=time_periods)

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
        w=w,
        gp_sd=gp_sd,
        gp_l=gp_l,
    )


class TestOccu(unittest.TestCase):

    def test_occu(self):
        data, true_params = simulate(simulate_missing=True)

        from biolith.utils import fit

        results = fit(occu, **data, timeout=600)

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

    def test_occu_fp_constant(self):
        prob_fp_constant = 0.1
        data, true_params = simulate(
            simulate_missing=True, prob_fp_constant=prob_fp_constant
        )

        from biolith.utils import fit

        results = fit(occu, **data, false_positives_constant=True, timeout=600)

        self.assertTrue(
            np.allclose(
                results.samples["psi"].mean(), true_params["z"].mean(), atol=0.1
            )
        )
        self.assertTrue(
            np.allclose(
                results.samples["prob_fp_constant"].mean(), prob_fp_constant, atol=0.1
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
    @unittest.skip("Skipping test for false positives at unoccupied sites")
    def test_occu_fp_unoccupied(self):
        prob_fp_unoccupied = 0.1
        data, true_params = simulate(
            simulate_missing=True, prob_fp_unoccupied=prob_fp_unoccupied
        )

        from biolith.utils import fit

        results = fit(occu, **data, false_positives_unoccupied=True, timeout=600)

        self.assertTrue(
            np.allclose(
                results.samples["psi"].mean(), true_params["z"].mean(), atol=0.1
            )
        )
        self.assertTrue(
            np.allclose(
                results.samples["prob_fp_unoccupied"].mean(),
                prob_fp_unoccupied,
                atol=0.1,
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
        data, true_params = simulate(simulate_missing=True, spatial=True)

        from biolith.utils import fit

        results = fit(occu, **data, timeout=600)

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

    def test_vs_spoccupancy(self):
        data, true_params = simulate(simulate_missing=True)

        num_samples = 1000

        # fit the biolith model
        from rpy2.robjects.packages import importr

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
        # Observations (y) - can have NaNs, spOccupancy handles them.
        y_py = data["obs"].copy()  # Shape: (n_sites, time_periods)
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
            data["obs_covs"].copy()
        )  # Shape: (n_sites, time_periods, n_obs_covs)
        _, time_periods, n_obs_covs = det_covs_py.shape

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

        self.assertTrue(
            np.allclose(
                psi_samples_r_matrix.mean(), results.samples["psi"].mean(), atol=0.1
            )
        )
        self.assertTrue(
            np.allclose(
                beta_samples_r_matrix.mean(axis=0),
                [
                    results.samples[k].mean()
                    for k in [f"cov_state_{i}" for i in range(len(true_params["beta"]))]
                ],
                atol=0.5,
            )
        )
        self.assertTrue(
            np.allclose(
                alpha_samples_r_matrix.mean(axis=0),
                [
                    results.samples[k].mean()
                    for k in [f"cov_det_{i}" for i in range(len(true_params["alpha"]))]
                ],
                atol=0.5,
            )
        )

    def test_evaluation(self):
        data, _ = simulate(simulate_missing=True)

        from biolith.utils import fit, predict

        results = fit(occu, **data, timeout=600)
        posterior_samples = predict(occu, results.mcmc, **data)

        from biolith.evaluation import (
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

        # Test diagnostics
        diagnostics(results.mcmc)

    def test_regression_methods(self):
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
            self.assertTrue(
                np.allclose(
                    results.samples["psi"].mean(), true_params["z"].mean(), atol=0.1
                )
            )


if __name__ == "__main__":
    unittest.main()
