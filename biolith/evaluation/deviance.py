from typing import Dict

import jax.numpy as jnp
from jax.scipy.special import logsumexp


def deviance(
    posterior_samples: Dict[str, jnp.ndarray],
    obs: jnp.ndarray,
) -> float:
    """Calculates deviance as a scoring rule following spOccupancy and Hooten and Hobbs (2015).

    References
    ----------
        - Hooten, M.B. and Hobbs, N.T. (2015), A guide to Bayesian model selection for ecologists. Ecological Monographs, 85: 3-28.

    Parameters
    ----------
        posterior_samples: A dictionary containing posterior samples from a fitted model.
                          Must include 'z' (latent occupancy state) and 'prob_detection'
                          (detection probability).
        obs: Ground truth observations of shape (n_sites, n_visits).

    Returns
    -------
        float: The deviance value as a scalar.

    Examples
    --------
    >>> from biolith.models import simulate, occu
    >>> from biolith.utils import fit, predict
    >>> from biolith.evaluation import deviance
    >>> data, _ = simulate()
    >>> results = fit(occu, **data)
    >>> preds = predict(occu, results.mcmc, **data)
    >>> dev = deviance(preds, data["obs"])
    """

    # Get posterior samples
    z_posterior = posterior_samples["z"]  # shape: (n_samples, n_sites)
    p_posterior = posterior_samples["prob_detection"].transpose(
        (0, 2, 1)
    )  # shape: (n_samples, n_sites, n_visits)

    # Calculate likelihood for each observation at each site and visit
    # TODO: This is only valid without false positives.
    prob_detection = p_posterior * z_posterior[:, :, None]

    # Bernoulli likelihood, using log-space for numerical stability
    obs = obs[None, :, :]
    log_likelihood = jnp.log(prob_detection + 1e-10) * obs + jnp.log(
        1 - prob_detection + 1e-10
    ) * (1 - obs)

    # Sum over all observations to get total log-likelihood for each sample
    log_likelihood_per_sample = jnp.nansum(log_likelihood, axis=(1, 2))

    # Calculate the mean likelihood across MCMC samples (in log space)
    log_mean_likelihood = logsumexp(log_likelihood_per_sample) - jnp.log(
        len(log_likelihood_per_sample)
    )

    # Calculate deviance: -2 * log(mean_likelihood)
    deviance_value = (-2 * log_mean_likelihood).item()

    return deviance_value
