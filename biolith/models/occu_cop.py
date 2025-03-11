import unittest
from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def occu_cop(site_covs: np.ndarray, obs_covs: np.ndarray, session_duration: Optional[np.ndarray] = None, false_positives_constant: bool = False, false_positives_unoccupied: bool = False, obs: Optional[np.ndarray] = None):

    # Check input data
    assert obs is None or obs.ndim == 2, "obs must be None or of shape (n_sites, time_periods)"
    assert site_covs.ndim == 2, "site_covs must be of shape (n_sites, n_site_covs)"
    assert obs_covs.ndim == 3, "obs_covs must be of shape (n_sites, time_periods, n_obs_covs)"
    assert session_duration is None or session_duration.ndim == 2, "session_duration must be None or of shape (n_sites, time_periods)"
    assert (obs[np.isfinite(obs)] >= 0).all(), "observations must be non-negative"
    assert not (false_positives_constant and false_positives_unoccupied), "false_positives_constant and false_positives_unoccupied cannot both be True"

    n_sites = site_covs.shape[0]
    time_periods = obs_covs.shape[1]
    n_site_covs = site_covs.shape[1]
    n_obs_covs = obs_covs.shape[2]

    assert n_sites == site_covs.shape[0] == obs_covs.shape[0], "site_covs and obs_covs must have the same number of sites"
    assert time_periods == obs_covs.shape[1], "obs_covs must have the same number of time periods as obs"
    if obs is not None:
        assert n_sites == obs.shape[0], "obs must have n_sites rows"
        assert time_periods == obs.shape[1], "obs must have time_periods columns"
    if session_duration is not None:
        assert n_sites == session_duration.shape[0], "session_duration must have n_sites rows"
        assert time_periods == session_duration.shape[1], "session_duration must have time_periods columns"

    # Mask observations where covariates are missing
    obs_mask = jnp.isnan(obs_covs).any(axis=-1) | jnp.tile(jnp.isnan(site_covs).any(axis=-1)[:, None], (1, time_periods))
    obs = jnp.where(obs_mask, jnp.nan, obs) if obs is not None else None
    obs_covs = jnp.nan_to_num(obs_covs)
    site_covs = jnp.nan_to_num(site_covs)

    # Model false positive rate for both occupied and unoccupied sites
    rate_fp_constant = numpyro.sample('rate_fp_constant', dist.Exponential()) if false_positives_constant else 0
    
    # Model false positive rate only for occupied sites
    rate_fp_unoccupied = numpyro.sample('rate_fp_unoccupied', dist.Exponential()) if false_positives_unoccupied else 0
    
    # Occupancy and detection covariates
    beta = jnp.array([numpyro.sample(f'beta_{i}', dist.Normal()) for i in range(n_site_covs + 1)])
    alpha = jnp.array([numpyro.sample(f'alpha_{i}', dist.Normal()) for i in range(n_obs_covs + 1)])

    # Transpose in order to fit NumPyro's plate structure
    site_covs = site_covs.transpose((1, 0))
    obs_covs = obs_covs.transpose((2, 1, 0))
    session_duration = session_duration.transpose((1, 0))
    obs = obs.transpose((1, 0))

    with numpyro.plate('site', n_sites, dim=-1):

        # Occupancy process
        psi = numpyro.deterministic('psi', jax.nn.sigmoid(jnp.tile(beta[0], (n_sites,)) + jnp.sum(jnp.array([beta[i + 1] * site_covs[i, ...] for i in range(n_site_covs)]), axis=0)))
        z = numpyro.sample('z', dist.Bernoulli(probs=psi), infer={'enumerate': 'parallel'})

        with numpyro.plate('time_periods', time_periods, dim=-2):

            # Detection process
            rate_detection = numpyro.deterministic(f'rate_detection', jnp.exp(jnp.tile(alpha[0], (time_periods, n_sites)) + jnp.sum(jnp.array([alpha[i + 1] * obs_covs[i, ...] for i in range(n_obs_covs)]), axis=0)))
            l_det = z * rate_detection + (1 - z) * rate_fp_unoccupied + rate_fp_constant

            if obs is not None:
                with numpyro.handlers.mask(mask=jnp.isfinite(obs)):
                    numpyro.sample(f'y', dist.Poisson(jnp.nan_to_num(session_duration * l_det)), obs=jnp.nan_to_num(obs))


def simulate_cop(
        n_site_covs=1,
        n_obs_covs=1,
        n_sites=100,  # number of sites
        deployment_days_per_site=365,  # number of days each site is monitored
        session_duration=7,  # 1, 7, or 30 days
        simulate_missing=False,  # whether to simulate missing data by setting some observations to NaN
        min_occupancy=0.25,  # minimum occupancy rate
        max_occupancy=0.75,  # maximum occupancy rate
        min_observation_rate=0.5,  # minimum proportion of timesteps with observation
        max_observation_rate=10,  # maximum proportion of timesteps with observation
        random_seed=0,
):

    # Initialize random number generator
    rng = np.random.default_rng(random_seed)

    # Make sure occupancy and detection are not too close to 0 or 1
    z = None
    while z is None or z.mean() < min_occupancy or z.mean() > max_occupancy or np.mean(obs[np.isfinite(obs)]) < min_observation_rate or np.mean(obs[np.isfinite(obs)]) > max_observation_rate:

        # Generate false positive rate
        rate_fp = rng.uniform(0.05, 0.2)

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
        detection_rate = np.exp(alpha[0].repeat(n_sites)[:, None] + np.sum([alpha[i + 1] * obs_covs[..., i] for i in range(n_obs_covs)], axis=0))

        # Create matrix of detections
        obs = np.zeros((n_sites, time_periods))

        for i in range(n_sites):
            # Similar to the Royle model in unmarked, false positives are generated only if the site is unoccupied
            # Note this is different than how we think about false positives being a random occurrence per image.
            obs[i, :] = rng.poisson(lam=(session_duration * (detection_rate[i, :] * z[i] + rate_fp * (1 - z[i]))), size=time_periods)

        if simulate_missing:
            # Simulate missing data:
            obs[rng.choice([True, False], size=obs.shape, p=[0.2, 0.8])] = np.nan
            obs_covs[rng.choice([True, False], size=obs_covs.shape, p=[0.05, 0.95])] = np.nan
            site_covs[rng.choice([True, False], size=site_covs.shape, p=[0.05, 0.95])] = np.nan

    print(f"True occupancy: {np.mean(z):.4f}")
    print(f"Fraction of observations with at least one observation: {np.mean(obs[np.isfinite(obs)] >= 1):.4f}")
    print(f"Mean rate: {np.mean(obs[np.isfinite(obs)]):.4f}")

    # session duration is assumed to be constant over all time periods
    session_duration = np.full((n_sites, time_periods), session_duration)

    return dict(
        site_covs=site_covs,
        obs_covs=obs_covs,
        session_duration=session_duration,
        obs=obs,
        false_positives_constant=True,
    ), dict(
        z=z,
        beta=beta,
        alpha=alpha,
    )


class TestOccuCOP(unittest.TestCase):

    def test_occu(self):
        data, true_params = simulate_cop(simulate_missing=True)

        from biolith.utils import fit
        results = fit(occu_cop, **data)

        # TODO: remove
        print("True Params:", true_params)
        print("Samples:", {k: results.samples[k].mean() for k in results.samples.keys()})

        self.assertTrue(np.allclose(results.samples["psi"].mean(), true_params["z"].mean(), atol=0.1))
        self.assertTrue(np.allclose([results.samples[k].mean() for k in [f"cov_state_{i}" for i in range(len(true_params["beta"]))]], true_params["beta"], atol=0.5))
        self.assertTrue(np.allclose([results.samples[k].mean() for k in [f"cov_det_{i}" for i in range(len(true_params["alpha"]))]], true_params["alpha"], atol=0.5))


if __name__ == '__main__':
    unittest.main()