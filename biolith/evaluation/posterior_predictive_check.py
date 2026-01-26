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
    r"""Performs posterior predictive checks of a Bayesian occupancy model.

    The Bayesian p-value is calculated as:

    .. math::
        p_B = P(T(y^{\text{rep}}, \theta) > T(y, \theta) | y)

    where :math:`T(\cdot, \cdot)` is the test statistic, :math:`y` is the observed data,
    :math:`y^{\text{rep}}` is replicated data from the posterior predictive distribution,
    and :math:`\theta` represents the model parameters.

    For Freeman-Tukey statistic:

    .. math::
        T_{\text{FT}} = \sum (\sqrt{y} - \sqrt{E})^2

    For Chi-squared statistic:

    .. math::
        T_{\chi^2} = \sum \frac{(y - E)^2}{E}

    where :math:`E` is the expected value and the sum is over the grouping dimension.

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
                   distribution. Shape: (num_samples, num_replicates, num_periods, num_sites).
            - 'psi': Posterior samples for the occupancy probability.
                     Shape: (num_samples, num_periods, num_sites).
            - 'prob_detection': Posterior samples for the detection probability. This is
                   necessary to compute the expected values for the GOF tests.
                   Shape: (num_samples, num_replicates, num_periods, num_sites).
        obs (jnp.ndarray): Ground truth observations of shape (n_sites, n_periods, n_replicates).
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

    # Transpose y_rep from (samples, replicates, periods, sites) to
    # (samples, sites, periods, replicates) to align with 'obs' and 'p'.
    if y_rep.ndim == 4:
        y_rep = jnp.transpose(y_rep, (0, 3, 2, 1))
    if p.ndim == 4:
        p = jnp.transpose(p, (0, 3, 2, 1))

    # The expected value for an observation y_ij is the product of the
    # occupancy probability (psi_i) and the detection probability (p_ij).
    # TODO: This is only valid without false positives.
    # E has a shape of (num_samples, num_sites, num_periods, num_replicates).
    if psi.ndim == 2:
        psi = psi[:, None, :]
    psi_by_site = psi.transpose((0, 2, 1))
    E = psi_by_site[:, :, :, None] * p

    # Create a mask for non-missing observations
    obs_mask = jnp.isfinite(obs)

    if group_by == "site":
        obs_grouped = jnp.nansum(obs, axis=(1, 2))
        y_rep_grouped = jnp.where(obs_mask[None, :, :, :], y_rep, 0).sum(axis=(2, 3))
        E_grouped = jnp.where(obs_mask[None, :, :, :], E, 0).sum(axis=(2, 3))

        # Calculate the discrepancy for the observed data against the expected values.
        d_obs = jnp.sum(stat_func(obs_grouped, E_grouped), axis=1)
        d_rep = jnp.sum(stat_func(y_rep_grouped, E_grouped), axis=1)
    elif group_by == "revisit":
        obs_grouped = jnp.nansum(obs, axis=0)
        y_rep_grouped = jnp.where(obs_mask[None, :, :, :], y_rep, 0).sum(axis=1)
        E_grouped = jnp.where(obs_mask[None, :, :, :], E, 0).sum(axis=1)

        d_obs = jnp.sum(stat_func(obs_grouped, E_grouped), axis=(1, 2))
        d_rep = jnp.sum(stat_func(y_rep_grouped, E_grouped), axis=(1, 2))
    else:
        raise ValueError("`group_by` must be either 'site' or 'revisit'")

    # This is the proportion of posterior samples where the replicated data
    # is more discrepant than the observed data.
    p_value = jnp.mean(d_rep > d_obs).item()

    return p_value
