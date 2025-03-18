import unittest
from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from biolith.utils.distributions import RightTruncatedPoisson


def occu_rn(site_covs: np.ndarray, obs_covs: np.ndarray, false_positives_constant: bool = False, max_abundance: int = 100, obs: Optional[np.ndarray] = None):

    # Check input data
    assert site_covs.ndim == 2, "site_covs must be (n_sites, n_site_covs)"
    assert obs_covs.ndim == 3, "obs_covs must be (n_sites, time_periods, n_obs_covs)"

    n_sites = site_covs.shape[0]
    time_periods = obs_covs.shape[1]
    n_site_covs = site_covs.shape[1]
    n_obs_covs = obs_covs.shape[2]

    if obs is not None:
        assert obs.shape == (n_sites, time_periods), "obs must have shape (n_sites, time_periods)"
        assert (obs[np.isfinite(obs)] >= 0).all() and (obs[np.isfinite(obs)] <= 1).all(), "Detections (obs) must be in {0,1} (or NaN for missing)."

    # Mask observations where covariates are missing
    obs_mask = jnp.isnan(obs_covs).any(axis=-1) | jnp.tile(jnp.isnan(site_covs).any(axis=-1)[:, None], (1, time_periods))
    obs = jnp.where(obs_mask, jnp.nan, obs) if obs is not None else None
    obs_covs = jnp.nan_to_num(obs_covs)
    site_covs = jnp.nan_to_num(site_covs)

    # Priors
    # Model false positive rate for both occupied and unoccupied sites
    prob_fp_constant = numpyro.sample('prob_fp_constant', dist.Beta(2, 5)) if false_positives_constant else 0

    # Abundance and detection covariates
    beta = jnp.array([numpyro.sample(f'beta_{i}', dist.Normal()) for i in range(n_site_covs + 1)])
    alpha = jnp.array([numpyro.sample(f'alpha_{i}', dist.Normal()) for i in range(n_obs_covs + 1)])

    # Transpose in order to fit NumPyro's plate structure
    site_covs = site_covs.transpose((1, 0))
    obs_covs = obs_covs.transpose((2, 1, 0))
    obs = obs.transpose((1, 0)) if obs is not None else None

    # Occupancy process
    abundance = numpyro.deterministic('abundance', jnp.exp(jnp.tile(beta[0], (n_sites,)) + jnp.sum(jnp.array([beta[i + 1] * site_covs[i, ...] for i in range(n_site_covs)]), axis=0)))

    with numpyro.plate('site', n_sites, dim=-1):

        N_i = numpyro.sample('N_i', RightTruncatedPoisson(abundance, max_cutoff=max_abundance), infer={"enumerate": "parallel"})

        with numpyro.plate('time', time_periods, dim=-2):

            # Detection process
            r_it = numpyro.deterministic(f'prob_detection', jax.nn.sigmoid(jnp.tile(alpha[0], (time_periods, n_sites)) + jnp.sum(jnp.array([alpha[i + 1] * obs_covs[i, ...] for i in range(n_obs_covs)]), axis=0)))
            p_it = 1.0 - (1.0 - r_it) ** N_i[None, :]

            if obs is not None:
                with numpyro.handlers.mask(mask=jnp.isfinite(obs)):
                    numpyro.sample("y", dist.Bernoulli(jnp.clip(p_it + prob_fp_constant, 0, 1)), obs=jnp.nan_to_num(obs), infer={'enumerate': 'parallel'})


def simulate_rn(
        n_site_covs=1,
        n_obs_covs=1,
        n_sites=100,  # number of sites
        deployment_days_per_site=365,  # number of days each site is monitored
        session_duration=7,  # 1, 7, or 30 days
        prob_fp=0,  # probability of a false positive for a given time point
        simulate_missing=False,  # whether to simulate missing data by setting some observations to NaN
        min_occupancy=0.25,  # minimum occupancy rate
        max_occupancy=0.75,  # maximum occupancy rate
        min_observation_rate=0.1,  # minimum proportion of timesteps with observation
        max_observation_rate=0.5,  # maximum proportion of timesteps with observation
        random_seed=0,
):

    # Initialize random number generator
    rng = np.random.default_rng(random_seed)

    # Make sure occupancy and detection are not too close to 0 or 1
    N_i = None
    while N_i is None or (N_i > 0).mean() < min_occupancy or (N_i > 0).mean() > max_occupancy or np.mean(obs[np.isfinite(obs)]) < min_observation_rate or np.mean(obs[np.isfinite(obs)]) > max_observation_rate:

        # Generate intercept and slopes
        beta = rng.normal(size=n_site_covs + 1)  # intercept and slopes for occupancy logistic regression
        alpha = rng.normal(size=n_obs_covs + 1)  # intercept and slopes for detection logistic regression

        # Generate occupancy and site-level covariates
        site_covs = rng.normal(size=(n_sites, n_site_covs))
        abundance = np.exp(beta[0].repeat(n_sites) + np.sum([beta[i + 1] * site_covs[..., i] for i in range(n_site_covs)], axis=0))
        N_i = rng.poisson(abundance, size=n_sites)  # vector of latent occupancy status for each site

        # Generate detection data
        time_periods = round(deployment_days_per_site / session_duration)

        # Create matrix of detection covariates
        obs_covs = rng.normal(size=(n_sites, time_periods, n_obs_covs))
        r_it = 1 / (1 + np.exp(-(alpha[0].repeat(n_sites)[:, None] + np.sum([alpha[i + 1] * obs_covs[..., i] for i in range(n_obs_covs)], axis=0))))

        # Create matrix of detections
        obs = np.zeros((n_sites, time_periods))

        for i in range(n_sites):
            # According to the Royle model in unmarked, false positives are generated only if the site is unoccupied
            # Note this is different than how we think about false positives being a random occurrence per image.
            # For now, this is generating positive/negative per time period, which is different than per image.
            p_it = 1.0 - (1.0 - r_it[i])**N_i[i]
            obs[i, :] = rng.binomial(n=1, p=np.clip(p_it + prob_fp, 0, 1), size=time_periods)

        # Convert counts into observed occupancy
        obs = (obs >= 1) * 1.

        if simulate_missing:
            # Simulate missing data:
            obs[rng.choice([True, False], size=obs.shape, p=[0.2, 0.8])] = np.nan
            obs_covs[rng.choice([True, False], size=obs_covs.shape, p=[0.05, 0.95])] = np.nan
            site_covs[rng.choice([True, False], size=site_covs.shape, p=[0.05, 0.95])] = np.nan

    print(f"True occupancy: {np.mean(N_i > 0):.4f}")
    print(f"True abundance: {np.mean(abundance):.4f}")
    print(f"Proportion of timesteps with observation: {np.mean(obs[np.isfinite(obs)]):.4f}")

    return dict(
        site_covs=site_covs,
        obs_covs=obs_covs,
        obs=obs,
    ), dict(
        abundance=abundance,
        beta=beta,
        alpha=alpha,
    )


class TestOccuRN(unittest.TestCase):

    def test_occu(self):
        data, true_params = simulate_rn(simulate_missing=True)

        from biolith.utils import fit
        results = fit(occu_rn, **data)

        self.assertTrue(np.allclose(results.samples["abundance"].mean(), true_params["abundance"].mean(), rtol=0.1))
        self.assertTrue(np.allclose([results.samples[k].mean() for k in [f"cov_state_{i}" for i in range(len(true_params["beta"]))]], true_params["beta"], atol=0.5))
        self.assertTrue(np.allclose([results.samples[k].mean() for k in [f"cov_det_{i}" for i in range(len(true_params["alpha"]))]], true_params["alpha"], atol=0.5))


if __name__ == '__main__':
    unittest.main()