import unittest
from typing import Optional, List
import numpy as np
import os
import tempfile
import subprocess
import json


def occu(
    site_covs: np.ndarray,
    obs_covs: np.ndarray,
    false_positives_constant: bool = False,
    false_positives_unoccupied: bool = False,
    obs: Optional[np.ndarray] = None,
    prior_beta_mean: float = 0,
    prior_beta_sd: float = 1,
    prior_alpha_mean: float = 0,
    prior_alpha_sd: float = 1,
    prior_prob_fp_a: float = 2,
    prior_prob_fp_b: float = 5,
    n_chains: int = 4,
    n_iter: int = 2000,
    n_burnin: int = 1000,
    n_thin: int = 1,
):
    # Check input data
    assert obs is None or obs.ndim == 2, "obs must be None or of shape (n_sites, time_periods)"
    assert site_covs.ndim == 2, "site_covs must be of shape (n_sites, n_site_covs)"
    assert obs_covs.ndim == 3, "obs_covs must be of shape (n_sites, time_periods, n_obs_covs)"
    assert obs is None or (obs[np.isfinite(obs)] >= 0).all(), "observations must be non-negative"
    assert obs is None or (obs[np.isfinite(obs)] <= 1).all(), "observations must be binary"
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

    # Create observation mask
    obs_mask = np.isnan(obs_covs).any(axis=-1) | np.tile(np.isnan(site_covs).any(axis=-1)[:, None], (1, time_periods))
    
    # Set up data for JAGS
    if obs is not None:
        obs_data = obs.copy()
        obs_data[np.isnan(obs_data) | obs_mask] = float('NaN')
    else:
        obs_data = np.full((n_sites, time_periods), float('NaN'))
    
    # JAGS can't have NaN values in covariates
    site_covs_data = np.nan_to_num(site_covs)
    obs_covs_data = np.nan_to_num(obs_covs)
    
    # Create JAGS model string
    model_string = """
    model {
      # Priors for occupancy model parameters
      beta0 ~ dnorm(beta_mean, pow(beta_sd, -2))
      for (c in 1:n_site_covs) {
        beta[c] ~ dnorm(beta_mean, pow(beta_sd, -2))
      }
      
      # Priors for detection model parameters
      alpha0 ~ dnorm(alpha_mean, pow(alpha_sd, -2))
      for (c in 1:n_obs_covs) {
        alpha[c] ~ dnorm(alpha_mean, pow(alpha_sd, -2))
      }
      
      # False positive parameters
      prob_fp_constant <- fp_constant * prob_fp_constant_raw
      prob_fp_unoccupied <- fp_unoccupied * prob_fp_unoccupied_raw
      
      prob_fp_constant_raw ~ dbeta(fp_a, fp_b)
      prob_fp_unoccupied_raw ~ dbeta(fp_a, fp_b)
      
      # Ecological process - site occupancy
      for (i in 1:n_sites) {
        # Occupancy linear predictor
        logit_psi[i] <- beta0
        for (c in 1:n_site_covs) {
          logit_psi[i] <- logit_psi[i] + beta[c] * site_covs[i, c]
        }
        psi[i] <- ilogit(logit_psi[i])
        z[i] ~ dbern(psi[i])
        
        for (j in 1:time_periods) {
          # Detection linear predictor
          logit_p[i, j] <- alpha0
          for (c in 1:n_obs_covs) {
            logit_p[i, j] <- logit_p[i, j] + alpha[c] * obs_covs[i, j, c]
          }
          prob_detection[i, j] <- ilogit(logit_p[i, j])
          
          # Detection probability considering false positives
          p_det[i, j] <- z[i] * prob_detection[i, j] + (1 - z[i]) * prob_fp_unoccupied + prob_fp_constant
          
          # Ensure p_det is within [0,1]
          p_det_constrained[i, j] <- max(0, min(1, p_det[i, j]))
          
          # Observation model
          y[i, j] ~ dbern(p_det_constrained[i, j])
        }
      }
    }
    """
    
    # Prepare data for JAGS
    jags_data = {
        'y': obs_data.tolist(),
        'site_covs': site_covs_data.tolist(),
        'obs_covs': obs_covs_data.tolist(),
        'n_sites': n_sites,
        'time_periods': time_periods,
        'n_site_covs': n_site_covs,
        'n_obs_covs': n_obs_covs,
        'beta_mean': prior_beta_mean,
        'beta_sd': prior_beta_sd,
        'alpha_mean': prior_alpha_mean,
        'alpha_sd': prior_alpha_sd,
        'fp_a': prior_prob_fp_a,
        'fp_b': prior_prob_fp_b,
        'fp_constant': 1 if false_positives_constant else 0,
        'fp_unoccupied': 1 if false_positives_unoccupied else 0
    }
    
    # Parameters to monitor
    monitor_params = ['beta0', 'beta', 'alpha0', 'alpha', 'psi', 'z', 'prob_detection']
    
    if false_positives_constant:
        monitor_params.append('prob_fp_constant')
    if false_positives_unoccupied:
        monitor_params.append('prob_fp_unoccupied')
    
    # Create temporary file for the model
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        f.write(model_string.encode())
        model_file = f.name
    
    try:
        # Prepare data file for JAGS
        with tempfile.NamedTemporaryFile("wt", suffix='.json', delete=False) as df:
            json.dump(jags_data, df)
            data_file = df.name

        # Prepare an output file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as of:
            output_file = of.name

        # Construct the command to run JAGS from the command line
        cmd = [
            "/data/vision/beery/scratch/timm/bin/jags",
            model_file,
            data_file,
            "-o", output_file,
            "-m", str(n_chains),  # number of chains
            "-n", str(n_iter),    # total number of samples
            "-b", str(n_burnin),  # burn-in
            "-t", str(n_thin)     # thinning
        ]

        # Run JAGS
        subprocess.run(cmd, check=True)

        # Roughly parse output (in real usage, parse results more carefully)
        # Here, just return the raw text
        with open(output_file, 'r') as f:
            results = f.read()

        return {"raw_output": results}
    
    finally:
        # Clean up temporary model file
        if os.path.exists(model_file):
            os.remove(model_file)


def simulate(
        n_site_covs=1,
        n_obs_covs=1,
        n_sites=100,
        deployment_days_per_site=365,
        session_duration=7,
        prob_fp=0,
        simulate_missing=False,
        min_occupancy=0.25,
        max_occupancy=0.75,
        min_observation_rate=0.1,
        max_observation_rate=0.5,
        random_seed=0,
):
    # Implementation unchanged
    # Initialize random number generator
    rng = np.random.default_rng(random_seed)

    # Make sure occupancy and detection are not too close to 0 or 1
    z = None
    while z is None or z.mean() < min_occupancy or z.mean() > max_occupancy or np.mean(obs[np.isfinite(obs)]) < min_observation_rate or np.mean(obs[np.isfinite(obs)]) > max_observation_rate:

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

        for i in range(n_sites):
            obs[i, :] = rng.binomial(n=1, p=(p[i, :] * z[i] + prob_fp * (1 - z[i])), size=time_periods)

        # Convert counts into observed occupancy
        obs = (obs >= 1) * 1.

        if simulate_missing:
            # Simulate missing data:
            obs[rng.choice([True, False], size=obs.shape, p=[0.2, 0.8])] = np.nan
            obs_covs[rng.choice([True, False], size=obs_covs.shape, p=[0.05, 0.95])] = np.nan
            site_covs[rng.choice([True, False], size=site_covs.shape, p=[0.05, 0.95])] = np.nan

    print(f"True occupancy: {np.mean(z):.4f}")
    print(f"Proportion of timesteps with observation: {np.mean(obs[np.isfinite(obs)]):.4f}")

    return dict(
        site_covs=site_covs,
        obs_covs=obs_covs,
        obs=obs,
    ), dict(
        z=z,
        beta=beta,
        alpha=alpha,
    )


class TestOccu(unittest.TestCase):
    def test_occu(self):
        data, true_params = simulate(simulate_missing=True)
        
        results = occu(**data, n_chains=2, n_iter=1000, n_burnin=500)
        
        # Check results
        self.assertTrue(np.allclose(results['psi'].mean(axis=0).mean(), true_params['z'].mean(), atol=0.1))
        self.assertTrue(np.allclose([results['beta0'].mean()] + [results['beta'].mean(axis=0)[i] for i in range(len(true_params['beta'])-1)], 
                                   true_params['beta'], atol=0.5))
        self.assertTrue(np.allclose([results['alpha0'].mean()] + [results['alpha'].mean(axis=0)[i] for i in range(len(true_params['alpha'])-1)], 
                                   true_params['alpha'], atol=0.5))


if __name__ == '__main__':
    unittest.main()