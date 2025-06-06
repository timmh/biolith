"""Run posterior predictive checks from occupancy models."""

import numpy as np
import jax
import jax.numpy as jnp
from numpyro.infer import Predictive
from biolith.utils import fit
from biolith.models import occu
from enum import Enum

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
    
def posterior_predictive_check(features: jnp.ndarray, 
                               observations: jnp.ndarray, 
                               taxon_idx: int, 
                               fit_result: fit.FitResult,
                               ppc_stat: PPC_Stat):
    predictive_gen = Predictive(occu, fit_result.mcmc.get_samples())
    posterior_pred = predictive_gen(
        jax.random.PRNGKey(0),
        site_covs=features, obs_covs=np.zeros((observations.shape[0], observations.shape[1], 0), dtype=observations.dtype),
    )


def do_occupancy(features_train, observations_train, features_test, observations_test, taxon_id, taxon_idx):

    # Copy the species detection matrix for the target species
    y_train = observations_train[taxon_idx].copy()
    y_test = observations_test[taxon_idx].copy()

    # Turn count data into binary detection/non-detection
    y_train = np.where(np.isnan(y_train), np.nan, np.where(y_train >= 1, 1., 0.))
    y_test = np.where(np.isnan(y_test), np.nan, np.where(y_test >= 1, 1., 0.))

    # Fit occupancy model
    results = fit(occu, site_covs=features_train, obs_covs=np.zeros((y_train.shape[0], y_train.shape[1], 0), dtype=features_train.dtype), obs=y_train, num_chains=1, num_samples=500, num_warmup=500)
    
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

    return p_value_train, p_value_test