"""Run posterior predictive checks from occupancy models."""

import numpy as np
import jax
import jax.numpy as jnp
from numpyro.infer import Predictive
from biolith.utils import fit
from biolith.models import occu
from enum import Enum
import numpyro
from numpyro.infer import log_likelihood

class PPC_Stat(Enum):
    MEAN = 1
    CHI_SQUARED = 2


class OccModel(Enum):
    OCC = 1
    OCC_RN = 2
    OCC_CS = 3
    OCC_COP = 4


# def get_occupancy_model_type(occ_model:OccModel):
#     if occ
    
def ppc_mean(sample_replicates, obs_data, var_name = "y"):
    posterior_p_value = np.mean(sample_replicates[var_name].mean(axis=(0, 1)) > np.nanmean(obs_data, axis=1)).item()
    return posterior_p_value

def ppc_chi_squared(sample_replicates, obs_data, expected_var_name = "prob_detection_fp", sim_var_name = "y", visit_axis_idx = 1):
    y_obs_sum = obs_data.sum(axis=visit_axis_idx)
    y_site_expected_sum = sample_replicates[expected_var_name].sum(axis=visit_axis_idx)
    # this will take the difference within the same site of observed sum(y) and expected sum(y) for all n_sample replicates
    # We then sum over the sites
    obs_diff = np.square(np.subtract(y_obs_sum,y_site_expected_sum)) / y_site_expected_sum
    T_obs = obs_diff.sum(axis=1)

    y_sim_sum = sample_replicates[sim_var_name].sum(axis=visit_axis_idx)
    sim_diff = np.square(np.subtract(y_sim_sum,y_site_expected_sum)) / y_site_expected_sum
    T_sim = sim_diff.sum(axis=1)

    posterior_p_value = np.mean(T_sim >= T_obs).item()
    return posterior_p_value

def ppc_tukey_freeman(sample_replicates, obs_data, var_name = "y", visit_axis_idx = 1):
    # For each site [i], we compute T_obs[i] = (sqrt(y_sum[i]) - sqrt(expected_sum[i]))^2, summing over the visits
    # expected_sum[i] = sum_j (p_detection[i,j]*z[j])
    # For each replicate k, we do the same thing, computing T_sim[i,k] = (sqrt(y_sum[i,k]) - sqrt(expected_sum[i]))^2
    # Then we compute T_obs[i] = sum(T_obs[i]) and T_sim[k] = sum_i(T_sim[i,k])
    # The "p-value" is the proportion of the k replicate samples where T_sim[k] < T_obs[k]
    y_obs_sum = obs_data.sum(axis=visit_axis_idx)
    y_site_expected_sum = sample_replicates['prob_detection_fp'].sum(axis=visit_axis_idx)
    # this will take the difference within the same site of observed sum(y) and expected sum(y) for all n_sample replicates
    # We then sum over the sites
    obs_diff = np.subtract(np.sqrt(y_obs_sum),np.sqrt(y_site_expected_sum))
    T_obs = np.square(np.subtract(np.sqrt(y_obs_sum),np.sqrt(y_site_expected_sum))).sum(axis=1)

    y_sim_sum = sample_replicates['y'].sum(axis=visit_axis_idx)
    T_sim = np.square(np.subtract(np.sqrt(y_sim_sum),np.sqrt(y_site_expected_sum))).sum(axis=1)
    posterior_p_value = np.mean(T_sim >= T_obs).item()
    return posterior_p_value



    
def posterior_predictive_check(features: jnp.ndarray, 
                               observations: jnp.ndarray, 
                               taxon_idx: int, 
                               fit_result: fit.FitResult,
                               ppc_stat: PPC_Stat):
    predictive_gen = Predictive(occu, fit_result.mcmc.get_samples())
    posterior_pred_samples = predictive_gen(
        jax.random.PRNGKey(0),
        site_covs=features, obs_covs=np.zeros((observations.shape[0], observations.shape[1], 0), dtype=observations.dtype),
    )


def do_occupancy(features_train, observations_train, features_test, observations_test, taxon_id, taxon_idx):

    # Copy the species detection matrix for the target species
    # site dimension 0 (rows) visit dimension 1 (columns)
    y_train = observations_train[taxon_idx].copy()
    y_test = observations_test[taxon_idx].copy()

    # Turn count data into binary detection/non-detection
    y_train = np.where(np.isnan(y_train), np.nan, np.where(y_train >= 1, 1., 0.))
    y_test = np.where(np.isnan(y_test), np.nan, np.where(y_test >= 1, 1., 0.))

    # Fit occupancy model
    results = fit(occu, site_covs=features_train, obs_covs=np.zeros((y_train.shape[0], y_train.shape[1], 0), dtype=features_train.dtype), 
                  obs=y_train, num_chains=1, num_samples=500, num_warmup=500)
    
    # Sample from the posterior predictive distribution of the model
    predictive_train = Predictive(occu, results.mcmc.get_samples())
    posterior_pred_train = predictive_train(
        jax.random.PRNGKey(0),
        site_covs=features_train, obs_covs=np.zeros((y_train.shape[0], y_train.shape[1], 0), dtype=features_train.dtype),
    )

    p_value_train = np.mean(posterior_pred_train["y"].mean(axis=(0, 1)) > np.nanmean(y_train, axis=1)).item()

    predictive_test = Predictive(occu, results.mcmc.get_samples())
    posterior_pred_test = predictive_test(
        jax.random.PRNGKey(0),
        site_covs=features_test, obs_covs=np.zeros((y_test.shape[0], y_test.shape[1], 0), dtype=features_test.dtype),
    )
    p_value_test = np.mean(posterior_pred_test["y"].mean(axis=(0, 1)) > np.nanmean(y_test, axis=1)).item()


    features_test_jax = jnp.array(features_test)
    with numpyro.handlers.block(), numpyro.handlers.seed(rng_seed=jax.random.PRNGKey(0)):
      log_lik_test = log_likelihood(occu, posterior_pred_test,
                                  site_covs=features_test_jax,
                                  obs_covs=jnp.zeros((y_test.shape[0], y_test.shape[1], 0), dtype=features_test_jax.dtype),
                                  obs=jnp.array(y_test))
    elpd_test = jnp.sum(jnp.log(jnp.mean(jnp.exp(log_lik_test["y"]), axis=0))).item()
    # want: proportion of all samples where sum(log_lik_test sample) > sum(log_lik_observed for test data)
    
    return p_value_train, p_value_test

# Calculate ELPD for test set
