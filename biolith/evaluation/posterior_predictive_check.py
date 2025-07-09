from typing import Dict, Literal

import jax.numpy as jnp


def _freeman_tukey_stat(obs: jnp.ndarray, exp: jnp.ndarray) -> jnp.ndarray:
    return (jnp.sqrt(obs) - jnp.sqrt(exp)) ** 2


def _chi_squared_stat(
    obs: jnp.ndarray, exp: jnp.ndarray, eps: float = 1e-10
) -> jnp.ndarray:
    return ((obs - exp) ** 2) / (exp + eps)


def posterior_predictive_check(
    posterior_samples: Dict[str, jnp.ndarray],
    obs: jnp.ndarray,
    group_by: Literal["site", "revisit"] = "site",
    statistic: Literal["freeman-tukey", "chi-squared"] = "freeman-tukey",
) -> float:
    """Performs posterior predictive checks of a Bayesian occupancy model.

    This function calculates a Bayesian p-value to assess the goodness-of-fit
    of the model. It compares the observed data with replicated data generated
    from the model's posterior predictive distribution using a specified
    discrepancy statistic.

    The discrepancy is calculated for both the real and replicated data for
    each posterior sample. The Bayesian p-value is the proportion of times
    the discrepancy of the replicated data exceeds that of the observed data.

    Parameters
    ----------
        posterior_samples (dict): A dictionary from a NumPyro Predictive
            call or a similar source. It must contain the following keys:
            - 'y': Replicated observation data from the posterior predictive
                   distribution. Shape: (num_samples, num_revisits, num_sites).
            - 'psi': Posterior samples for the occupancy probability.
                     Shape: (num_samples, num_sites).
            - 'prob_detection': Posterior samples for the detection probability. This is
                   necessary to compute the expected values for the GOF tests.
                   Shape: (num_samples, num_sites, num_revisits).
        obs (jnp.ndarray): Ground truth observations on an unobserved test set.
                           Shape: (num_sites, num_revisits).
        group_by (str): Specifies how to aggregate the data for the test
                        statistic. Must be either 'site' or 'revisit'.
        statistic (str): The discrepancy statistic to use for the comparison.
                         Must be either 'freeman-tukey' or 'chi-squared'.

    Returns
    -------
        float: The Bayesian p-value. A value close to 0.5 suggests a good
               model fit, while values near 0 or 1 suggest misfit.

    Examples
    --------
    >>> from biolith.models import simulate, occu
    >>> from biolith.utils import fit, predict
    >>> from biolith.evaluation import posterior_predictive_check
    >>> data, _ = simulate()
    >>> results = fit(occu, **data)
    >>> preds = predict(occu, results.mcmc, **data)
    >>> posterior_predictive_check(preds, data["obs"])

    Raises
    ------
        ValueError: If 'group_by' or 'statistic' are not valid options.
        KeyError: If 'posterior_samples' is missing the required keys
                  ('y', 'psi', 'prob_detection').
    """

    for key in ["y", "psi", "prob_detection"]:
        if key not in posterior_samples:
            raise KeyError(
                f"The `posterior_predictive` dictionary must contain a '{key}' key."
            )
    for key in ["prob_fp_constant", "prob_fp_unoccupied"]:
        if key in posterior_samples:
            raise KeyError(
                f"Models including false positives are not yet supported, but posterior samples for '{key}' were found."
            )
    y_rep = posterior_samples["y"]
    psi = posterior_samples["psi"]
    p = posterior_samples["prob_detection"]

    stat_funcs = {
        "freeman-tukey": _freeman_tukey_stat,
        "chi-squared": _chi_squared_stat,
    }
    if statistic not in stat_funcs:
        raise ValueError(f"`statistic` must be one of {list(stat_funcs.keys())}")
    stat_func = stat_funcs[statistic]

    # Transpose y_rep from (samples, revisits, sites) to (samples, sites, revisits)
    # to align with the dimensions of 'obs' and 'p'.
    if y_rep.ndim == 3:
        y_rep = jnp.transpose(y_rep, (0, 2, 1))

    # The expected value for an observation y_ij is the product of the
    # occupancy probability (psi_i) and the detection probability (p_ij).
    # TODO: This is only valid without false positives.
    # E has a shape of (num_samples, num_sites, num_revisits).
    E = psi[:, :, jnp.newaxis] * p.transpose((0, 2, 1))

    if group_by == "site":
        # Sum across the 'revisit' axis, summing only non-missing values.
        axis_to_agg = 2
        obs_grouped = jnp.nansum(obs, axis=1)
    elif group_by == "revisit":
        # Sum across the 'site' axis, summing only non-missing values.
        axis_to_agg = 1
        obs_grouped = jnp.nansum(obs, axis=0)
    else:
        raise ValueError("`group_by` must be either 'site' or 'revisit'")

    # Create a mask for non-missing observations
    obs_mask = jnp.isfinite(obs)
    
    if group_by == "site":
        # Sum across the 'revisit' axis, only including non-missing observations
        y_rep_grouped = jnp.where(obs_mask[None, :, :], y_rep, 0).sum(axis=axis_to_agg)
        E_grouped = jnp.where(obs_mask[None, :, :], E, 0).sum(axis=axis_to_agg)
    elif group_by == "revisit":
        # Sum across the 'site' axis, only including non-missing observations
        y_rep_grouped = jnp.where(obs_mask[None, :, :], y_rep, 0).sum(axis=axis_to_agg)
        E_grouped = jnp.where(obs_mask[None, :, :], E, 0).sum(axis=axis_to_agg)

    # Calculate the discrepancy for the observed data against the expected values
    # from each posterior sample. The result is averaged over the groups for each sample.
    d_obs = jnp.sum(stat_func(obs_grouped, E_grouped), axis=1)

    # Calculate the discrepancy for the replicated data against the expected values.
    d_rep = jnp.sum(stat_func(y_rep_grouped, E_grouped), axis=1)

    # This is the proportion of posterior samples where the replicated data
    # is more discrepant than the observed data.
    p_value = jnp.mean(d_rep > d_obs).item()

    return p_value
