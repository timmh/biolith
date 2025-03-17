import unittest
from typing import Optional
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


def occu_cs(site_covs: np.ndarray, obs_covs: np.ndarray, obs: Optional[np.ndarray] = None):

    # Check input data
    assert obs is None or obs.ndim == 2, "obs must be None or of shape (n_sites, time_periods)"
    assert site_covs.ndim == 2, "site_covs must be of shape (n_sites, n_site_covs)"
    assert obs_covs.ndim == 3, "obs_covs must be of shape (n_sites, time_periods, n_obs_covs)"

    n_sites = site_covs.shape[0]
    time_periods = obs_covs.shape[1]
    n_site_covs = site_covs.shape[1]
    n_obs_covs = obs_covs.shape[2]

    assert n_sites == site_covs.shape[0] == obs_covs.shape[0], "site_covs and obs_covs must have the same number of sites"
    assert time_periods == obs_covs.shape[1], "obs_covs must have the same number of time periods as obs"
    if obs is not None:
        assert n_sites == obs.shape[0], "obs must have n_sites rows"
        assert time_periods == obs.shape[1], "obs must have time_periods columns"

    # Mask observations where covariates are missing
    obs_mask = jnp.isnan(obs_covs).any(axis=-1) | jnp.tile(jnp.isnan(site_covs).any(axis=-1)[:, None], (1, time_periods))
    obs = jnp.where(obs_mask, jnp.nan, obs) if obs is not None else None
    obs_covs = jnp.nan_to_num(obs_covs)
    site_covs = jnp.nan_to_num(site_covs)
    
    # Occupancy and detection covariates
    beta = jnp.array([numpyro.sample(f'beta_{i}', dist.Normal()) for i in range(n_site_covs + 1)])
    alpha = jnp.array([numpyro.sample(f'alpha_{i}', dist.Normal()) for i in range(n_obs_covs + 1)])

    # Continuous score parameters
    mu_dist = dist.Normal(0, 10)
    mu0 = numpyro.sample('mu0', mu_dist)
    mu1 = numpyro.sample('mu1', dist.TruncatedDistribution(mu_dist, low=mu0))
    sigma0 = numpyro.sample('sigma0', dist.HalfNormal())
    sigma1 = numpyro.sample('sigma1', dist.HalfNormal())

    # Transpose in order to fit NumPyro's plate structure
    site_covs = site_covs.transpose((1, 0))
    obs_covs = obs_covs.transpose((2, 1, 0))
    obs = obs.transpose((1, 0))

    with numpyro.plate('site', n_sites, dim=-1):

        # Occupancy process
        psi = numpyro.deterministic('psi', jax.nn.sigmoid(jnp.tile(beta[0], (n_sites,)) + jnp.sum(jnp.array([beta[i + 1] * site_covs[i, ...] for i in range(n_site_covs)]), axis=0)))
        z = numpyro.sample('z', dist.Bernoulli(probs=psi), infer={'enumerate': 'parallel'})

        with numpyro.plate('time_periods', time_periods, dim=-2):

            # Detection process
            f = numpyro.sample('f', dist.Bernoulli(z * jax.nn.sigmoid(jnp.tile(alpha[0], (time_periods, n_sites)) + jnp.sum(jnp.array([alpha[i + 1] * obs_covs[i, ...] for i in range(n_obs_covs)]), axis=0))), infer={'enumerate': 'parallel'})

            if obs is not None:
                with numpyro.handlers.mask(mask=jnp.isfinite(obs)):
                    numpyro.sample('s', dist.Normal((1 - f) * mu0 + f * mu1, (1 - f) * sigma0 + f * sigma1), obs=jnp.nan_to_num(obs))


def simulate_cs(
        n_site_covs=1,
        n_obs_covs=1,
        n_sites=100,  # number of sites
        deployment_days_per_site=365,  # number of days each site is monitored
        session_duration=7,  # 1, 7, or 30 days
        simulate_missing=False,  # whether to simulate missing data by setting some observations to NaN
        min_occupancy=0.25,  # minimum occupancy rate
        max_occupancy=0.75,  # maximum occupancy rate
        random_seed=0,
):

    # Initialize random number generator
    rng = np.random.default_rng(random_seed)

    # Make sure occupancy and detection are not too close to 0 or 1
    z = None
    while z is None or z.mean() < min_occupancy or z.mean() > max_occupancy:

        # Generate intercept and slopes
        beta = rng.normal(size=n_site_covs + 1)  # intercept and slopes for occupancy logistic regression
        alpha = rng.normal(size=n_obs_covs + 1)  # intercept and slopes for detection logistic regression

        # Generate occupancy and site-level covariates
        site_covs = rng.normal(size=(n_sites, n_site_covs))
        psi = 1 / (1 + np.exp(-(beta[0].repeat(n_sites) + np.sum([beta[i + 1] * site_covs[..., i] for i in range(n_site_covs)], axis=0))))
        z = rng.binomial(n=1, p=psi, size=n_sites)  # vector of latent occupancy status for each site

        # Generate detection data
        time_periods = round(deployment_days_per_site / session_duration)

        # Create matrix of detection covariates
        obs_covs = rng.normal(size=(n_sites, time_periods, n_obs_covs))
        p = 1 / (1 + np.exp(-(alpha[0].repeat(n_sites)[:, None] + np.sum([alpha[i + 1] * obs_covs[..., i] for i in range(n_obs_covs)], axis=0))))

        # Create matrix of detections
        obs = np.zeros((n_sites, time_periods))

        # Generate score distributions
        mu0 = 0
        sigma0 = 10
        mu1 = 10
        sigma1 = 5

        # Create matrix of detections
        f = np.zeros((n_sites, time_periods))
        obs = np.zeros((n_sites, time_periods))

        for i in range(n_sites):
            f[i] = rng.binomial(n=1, p=(p[i, :] * z[i]), size=time_periods)
            for t in range(time_periods):
                obs[i, t] = rng.normal(mu0 if f[i, t] == 0 else mu1, sigma0 if f[i, t] == 0 else sigma1)

        if simulate_missing:
            # Simulate missing data:
            obs[rng.choice([True, False], size=obs.shape, p=[0.2, 0.8])] = np.nan
            obs_covs[rng.choice([True, False], size=obs_covs.shape, p=[0.05, 0.95])] = np.nan
            site_covs[rng.choice([True, False], size=site_covs.shape, p=[0.05, 0.95])] = np.nan

    print(f"True occupancy: {np.mean(z):.4f}")

    return dict(
        site_covs=site_covs,
        obs_covs=obs_covs,
        obs=obs,
    ), dict(
        z=z,
        beta=beta,
        alpha=alpha,
        mu0=mu0,
        sigma0=sigma0,
        mu1=mu1,
        sigma1=sigma1,
    )


class TestOccuCS(unittest.TestCase):

    def test_occu(self):
        data, true_params = simulate_cs(
            simulate_missing=True,
            n_sites=200,  # TODO: occupancy covariates fail to fit with fewer sites, investigate why
        )

        from biolith.utils import fit
        results = fit(occu_cs, **data)

        self.assertTrue(np.allclose(results.samples["psi"].mean(), true_params["z"].mean(), atol=0.1))
        self.assertTrue(np.allclose([results.samples[k].mean() for k in [f"cov_state_{i}" for i in range(len(true_params["beta"]))]], true_params["beta"], atol=0.5))
        self.assertTrue(np.allclose([results.samples[k].mean() for k in [f"cov_det_{i}" for i in range(len(true_params["alpha"]))]], true_params["alpha"], atol=0.5))
        self.assertTrue(np.allclose(results.samples["mu0"].mean(), true_params["mu0"], atol=1))
        self.assertTrue(np.allclose(results.samples["mu1"].mean(), true_params["mu1"], atol=1))
        self.assertTrue(np.allclose(results.samples["sigma0"].mean(), true_params["sigma0"], atol=1))
        self.assertTrue(np.allclose(results.samples["sigma1"].mean(), true_params["sigma1"], atol=1))


if __name__ == '__main__':
    unittest.main()