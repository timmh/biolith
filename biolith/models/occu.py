import unittest
from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


def occu(site_covs: np.ndarray, obs_covs: np.ndarray, session_duration: Optional[np.ndarray] = None, false_positives_constant: bool = False, false_positives_unoccupied: bool = False, obs: Optional[np.ndarray] = None):

    # Check input data
    assert obs is None or obs.ndim == 2, "obs must be None or of shape (n_sites, time_periods)"
    assert site_covs.ndim == 2, "site_covs must be of shape (n_sites, n_site_covs)"
    assert obs_covs.ndim == 3, "obs_covs must be of shape (n_sites, time_periods, n_obs_covs)"
    assert session_duration is None or session_duration.ndim == 2, "session_duration must be None or of shape (n_sites, time_periods)"
    assert (obs[np.isfinite(obs)] >= 0).all(), "observations must be non-negative"
    assert (obs[np.isfinite(obs)] <= 1).all(), "observations must be binary"
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
    obs = jnp.where(obs_mask, jnp.nan, obs)
    obs_covs = jnp.nan_to_num(obs_covs)
    site_covs = jnp.nan_to_num(site_covs)

    # Priors
    # Model false positive rate for both occupied and unoccupied sites
    prob_fp_constant = numpyro.sample('prob_fp_constant', dist.Beta(2, 5)) if false_positives_constant else 0
    
    # Model false positive rate only for occupied sites
    prob_fp_unoccupied = numpyro.sample('prob_fp_unoccupied', dist.Beta(2, 5)) if false_positives_unoccupied else 0
    
    # Occupancy and detection covariates
    beta = jnp.array([numpyro.sample(f'beta_{i}', dist.Normal()) for i in range(n_site_covs + 1)])
    alpha = jnp.array([numpyro.sample(f'alpha_{i}', dist.Normal()) for i in range(n_obs_covs + 1)])

    # Transpose in order to fit NumPyro's plate structure
    site_covs = site_covs.transpose((1, 0))
    obs_covs = obs_covs.transpose((2, 1, 0))
    obs = obs.transpose((1, 0))

    with numpyro.plate('site', n_sites, dim=-1) as site:

        # Occupancy process
        psi = numpyro.deterministic('psi', jax.nn.sigmoid(jnp.tile(beta[0], (n_sites,)) + jnp.sum(jnp.array([beta[i + 1] * site_covs[i, ...] for i in range(n_site_covs)]), axis=0)))
        z = numpyro.sample('z', dist.Bernoulli(probs=psi[site]), infer={'enumerate': 'parallel'})

        with numpyro.plate('time_periods', time_periods, dim=-2):

            # Detection process
            prob_detection = numpyro.deterministic(f'prob_detection', jax.nn.sigmoid(jnp.tile(alpha[0], (time_periods, n_sites)) + jnp.sum(jnp.array([alpha[i + 1] * obs_covs[i, ...] for i in range(n_obs_covs)]), axis=0)))
            p_det = z * prob_detection + (1 - z) * prob_fp_unoccupied + prob_fp_constant
            p_det = jnp.clip(p_det, min=0, max=1)

            with numpyro.handlers.mask(mask=jnp.isfinite(obs)):
                y = numpyro.sample(f'y', dist.Bernoulli(p_det), obs=jnp.nan_to_num(obs))


    # Estimate proportion of occupied sites
    NOcc = numpyro.deterministic('NOcc', jnp.sum(z))
    PropOcc = numpyro.deterministic('PropOcc', NOcc / n_sites)


def simulate():
    # Initialize random number generator
    random_seed = 0
    rng = np.random.default_rng(random_seed)

    # Generate occupancy and site-level covariates
    n_sites = 100  # number of sites
    n_site_covs = 4
    site_covs = rng.normal(size=(n_sites, n_site_covs))
    beta = [1, -0.05, 0.02, 0.01, -0.02]  # intercept and slopes for occupancy logistic regression
    psi_cov = 1 / (1 + np.exp(-(beta[0] + np.sum([beta[i + 1] * site_covs[..., i] for i in range(n_site_covs)]))))
    z = rng.binomial(n=1, p=psi_cov, size=n_sites)  # vector of latent occupancy status for each site

    # Generate detection data
    deployment_days_per_site = 365
    
    prob_fp = 0.01  # probability of a false positive for a given time point
    session_duration = 7  # 1, 7, or 30
    time_periods = round(deployment_days_per_site / session_duration)

    # Create matrix of detection covariates
    n_obs_covs = 3
    obs_covs = rng.normal(size=(n_sites, time_periods, n_obs_covs))
    alpha = [-0.5, 0.5, -0.5, 0]  # intercept and slopes for detection logistic regression
    obs_reg = 1 / (1 + np.exp(-(alpha[0] + np.sum([alpha[i + 1] * obs_covs[..., i] for i in range(n_obs_covs)], axis=0))))

    # Create matrix of detections
    dfa = np.zeros((n_sites, time_periods))

    for i in range(n_sites):
        # According to the Royle model in unmarked, false positives are generated only if the site is unoccupied
        # Note this is different than how we think about false positives being a random occurrence per image.
        # For now, this is generating positive/negative per time period, which is different than per image.
        dfa[i, :] = rng.binomial(n=1, p=(obs_reg[i, :] * z[i] + prob_fp * (1 - z[i])), size=time_periods)

    # Convert counts into observed occupancy
    obs = (dfa >= 1) * 1.

    # Simulate missing data:
    obs[np.random.choice([True, False], size=obs.shape, p=[0.2, 0.8])] = np.nan
    obs_covs[np.random.choice([True, False], size=obs_covs.shape, p=[0.05, 0.95])] = np.nan
    site_covs[np.random.choice([True, False], size=site_covs.shape, p=[0.05, 0.95])] = np.nan

    print(f"True occupancy: {np.mean(z):.4f}")
    print(f"Proportion of timesteps with observation: {np.mean(obs[np.isfinite(obs)]):.4f}")

    return dict(
        site_covs=site_covs,
        obs_covs=obs_covs,
        false_positives_constant=True,
        obs=obs,
    ), dict(
        z=z,
        beta=beta,
        alpha=alpha,
    )


class TestOccu(unittest.TestCase):

    def test_occu(self):
        data, true_params = simulate()

        from biolith.utils import fit
        results = fit(occu, **data)

        self.assertTrue(np.allclose(results.samples["psi"].mean(), true_params["z"].mean(), atol=0.05))
        self.assertTrue(np.allclose([results.samples[k].mean() for k in [f"cov_state_{i}" for i in range(len(true_params["beta"]))]], true_params["beta"], atol=0.5))
        self.assertTrue(np.allclose([results.samples[k].mean() for k in [f"cov_det_{i}" for i in range(len(true_params["alpha"]))]], true_params["alpha"], atol=0.5))


if __name__ == '__main__':
    unittest.main()