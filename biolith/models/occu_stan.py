import unittest
import pystan
import numpy as np

def fit_occu_stan(
    site_covs,
    obs_covs,
    obs,
    false_positives_constant=False,
    false_positives_unoccupied=False,
):
    """
    Fit the occupancy model using Stan (via pystan).
    
    site_covs: 2D numpy array of shape (n_sites, n_site_covs)
    obs_covs: 3D numpy array of shape (n_sites, time_periods, n_obs_covs)
    obs: 2D numpy array of shape (n_sites, time_periods) with 0/1 data
    false_positives_constant, false_positives_unoccupied: booleans
    """
    # Basic checks
    n_sites, n_site_covs = site_covs.shape
    _, time_periods, n_obs_covs = obs_covs.shape

    # Convert booleans into 0/1 so Stan can conditionally use or ignore these rates
    fp_constant_on = 1 if false_positives_constant else 0
    fp_unoccupied_on = 1 if false_positives_unoccupied else 0
    
    # Reshape data for Stan (flatten the (site, time) pairs)
    # We'll store all observations in 1D arrays. 
    # For missing data, we simply omit those from the likelihood here for brevity.
    obs_list = []
    obs_covar_list = []
    site_covar_list = []
    site_index = []
    
    for i in range(n_sites):
        for t in range(time_periods):
            if np.isfinite(obs[i, t]):  # keep only non-missing
                obs_list.append(obs[i, t])
                obs_covar_list.append(obs_covs[i, t, :])
                site_covar_list.append(site_covs[i, :])
                site_index.append(i)  # which site this observation belongs to
    
    obs_list = np.array(obs_list, dtype=int)
    obs_covar_list = np.array(obs_covar_list, dtype=float)
    site_covar_list = np.array(site_covar_list, dtype=float)
    site_index = np.array(site_index, dtype=int)
    
    # Stan code
    # We define:
    #   z[i] ~ Bernoulli(psi[i])
    #   psi[i] = inv_logit(beta_0 + beta * site_covs)
    #   For each observation k belonging to site s:
    #       p_det[k] = z[s]*inv_logit(alpha_0 + alpha*obs_covs[k]) 
    #                  + (1 - z[s])*prob_fp_unoccupied*fp_unoccupied_on 
    #                  + prob_fp_constant*fp_constant_on
    #   obs[k] ~ Bernoulli(p_det[k])
    #
    # Booleans turn prob_fp_constant or prob_fp_unoccupied off if set to false
    # by forcing them to stay at 0 in the model if the switch is off.
    
    stan_code = r"""
    data {
      int<lower=1> N_sites;
      int<lower=1> N_obs;
      int<lower=0> N_site_covs;
      int<lower=0> N_obs_covs;
      int<lower=0,upper=1> fp_constant_on;
      int<lower=0,upper=1> fp_unoccupied_on;
      int<lower=1,upper=N_sites> site_index[N_obs];
      matrix[N_obs, N_site_covs] site_covs_obs;
      matrix[N_obs, N_obs_covs] obs_covs_obs;
      int<lower=0,upper=1> y[N_obs];
    }
    
    parameters {
      // Occupancy intercept and slopes
      real beta_0;
      vector[N_site_covs] beta;
      
      // Detection intercept and slopes
      real alpha_0;
      vector[N_obs_covs] alpha;
      
      // Latent occupancy states
      // Note: For marginalizing z, we often do a discrete approach in Stan,
      // but we can just treat z as a parameter in a non-centered parameterization
      // with a Bernoulli prior if enumerations are small. Here we keep a Bernoulli approach.
      // Because Stan does not sample discrete parameters, we must marginalize z out 
      // or do something else. For simplicity, we store occupancy as continuous transforms
      // and approximate. A more rigorous approach is hierarchical with a custom lpmf.
      
      // We do a real occupancy_prob, then transform in model for log-likelihood.
      // That is the standard approach to marginalize out z.
      // occupancy_prob[i] = inv_logit(beta_0 + site_covar terms).
      // Probability of y[k] is occupancy_prob[site_index[k]]*(some detection) + ...
      // No discrete z needed in final model block. We handle it with log_mix.
      // 
      // If your sites are large, or you want an exact approach, you'd do big loops with log_mix.
      
      // False positive parameters
      real<lower=0,upper=1> prob_fp_constant;
      real<lower=0,upper=1> prob_fp_unoccupied;
    }
    
    transformed parameters {
      vector[N_sites] occupancy_prob;
      // site-level occupancy
      for(i in 1:N_sites){
        occupancy_prob[i] = inv_logit(beta_0 + dot_product(beta, rep_row_vector(0.0, N_site_covs))); 
      }
      // but we need site covariates in that expression
      // We only have site_covs_obs which is repeated for each observation,
      // so let's build a simpler aggregator outside. We'll rewrite:
      // We'll skip this block for clarity and do it in the model loop:
    }
    
    model {
      // Priors (simplified normal, can adjust as needed)
      beta_0 ~ normal(0, 5);
      beta ~ normal(0, 5);
      alpha_0 ~ normal(0, 5);
      alpha ~ normal(0, 5);
      
      // Beta(2,5) prior, but we keep it simpler here
      if(fp_constant_on == 1)
        prob_fp_constant ~ beta(2,5);
      else
        target += 0; // fixed at 0 if turned off
      
      if(fp_unoccupied_on == 1)
        prob_fp_unoccupied ~ beta(2,5);
      else
        target += 0; // fixed at 0 if turned off
      
      // Build model likelihood using marginalization over z for each observation
      // z ~ Bernoulli(occupancy_prob[site])
      // detection ~ Bernoulli(z * p_det + (1 - z)* prob_fp_unoccupied + prob_fp_constant)
      // We integrate out z: 
      // P(y=1) = occupancy_prob * p_det + (1-occupancy_prob) * (prob_fp_unoccupied*fp_unoccupied_on) + prob_fp_constant*fp_constant_on
      // P(y=0) = 1 - that quantity
      
      for(k in 1:N_obs){
        real occ_prob = inv_logit(
          beta_0 + dot_product(beta, site_covs_obs[k])
        );
        real p_det = inv_logit(
          alpha_0 + dot_product(alpha, obs_covs_obs[k])
        );
        real p_mixed = 
          occ_prob * p_det
          + (1 - occ_prob)*(prob_fp_unoccupied*fp_unoccupied_on)
          + (prob_fp_constant*fp_constant_on);
        // But note that adding prob_fp_constant again might not be the original logic
        // from the user code. The user code adds it outside z*(...) + (1-z)*(...).
        // Original code has p_det = z*p_det + (1-z)*prob_fp_unoccupied + prob_fp_constant
        // Then integrate out z: 
        // E[p_det] = occ_prob*p_det + (1 - occ_prob)*prob_fp_unoccupied + prob_fp_constant
        // We'll do it exactly:
        real p_true = occ_prob*p_det + (1 - occ_prob)* (prob_fp_unoccupied*fp_unoccupied_on) + (prob_fp_constant*fp_constant_on);
        p_true = fmin(fmax(p_true, 0), 1); // clip for safety
        y[k] ~ bernoulli(p_true);
      }
    }
    """
    
    # Build data dictionary
    stan_data = {
        'N_sites': n_sites,
        'N_obs': len(obs_list),
        'N_site_covs': n_site_covs,
        'N_obs_covs': n_obs_covs,
        'fp_constant_on': fp_constant_on,
        'fp_unoccupied_on': fp_unoccupied_on,
        'site_index': site_index + 1,  # Stan is 1-based
        'site_covs_obs': site_covar_list,
        'obs_covs_obs': obs_covar_list,
        'y': obs_list,
    }
    
    sm = pystan.StanModel(model_code=stan_code)
    fit_result = sm.sampling(data=stan_data, iter=1000, chains=4, control={'adapt_delta':0.9})
    
    return fit_result


def simulate_data_for_stan(
    n_sites=100,
    time_periods=10,
    n_site_covs=1,
    n_obs_covs=1,
    prob_fp_constant=0.0,
    prob_fp_unoccupied=0.0,
    seed=42
):
    """
    Simple simulator similar to the original, returning site_covs, obs_covs, obs.
    """
    rng = np.random.default_rng(seed)
    
    # site covs
    site_covs = rng.normal(size=(n_sites, n_site_covs))
    # occupancy
    beta_0 = 0.0
    beta = rng.normal(size=n_site_covs) * 0.5
    psi = 1 / (1 + np.exp(-(beta_0 + site_covs @ beta)))
    z = rng.binomial(1, psi)
    
    # obs covs
    obs_covs = rng.normal(size=(n_sites, time_periods, n_obs_covs))
    # detection
    alpha_0 = 0.0
    alpha = rng.normal(size=n_obs_covs) * 0.5
    
    p_det = 1 / (1 + np.exp(-(alpha_0 + np.einsum('ijk,k->ij', obs_covs, alpha))))
    
    # generate obs
    obs = np.zeros((n_sites, time_periods), dtype=int)
    for i in range(n_sites):
        for t in range(time_periods):
            # detection if occupied
            true_p = z[i]*p_det[i, t] + (1 - z[i])*prob_fp_unoccupied + prob_fp_constant
            obs[i, t] = rng.binomial(1, true_p)
    
    return site_covs, obs_covs, obs


class TestOccu(unittest.TestCase):
    def test_occu(self):
        data, true_params = simulate_data_for_stan(simulate_missing=True)
        
        results = fit_occu_stan(**data, n_chains=5, n_iter=1000, n_burnin=1000)
        
        # Check results
        self.assertTrue(np.allclose(results['psi'].mean(axis=0).mean(), true_params['z'].mean(), atol=0.1))
        self.assertTrue(np.allclose([results['beta0'].mean()] + [results['beta'].mean(axis=0)[i] for i in range(len(true_params['beta'])-1)], 
                                   true_params['beta'], atol=0.5))
        self.assertTrue(np.allclose([results['alpha0'].mean()] + [results['alpha'].mean(axis=0)[i] for i in range(len(true_params['alpha'])-1)], 
                                   true_params['alpha'], atol=0.5))


if __name__ == '__main__':
    unittest.main()