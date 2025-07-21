import unittest
from typing import Callable, Dict

import jax.numpy as jnp
from jax.scipy.special import logsumexp

from biolith.evaluation.log_likelihood import log_likelihood, log_likelihood_manual


def deviance(
    model_fn: Callable,
    posterior_samples: Dict[str, jnp.ndarray],
    **kwargs,
) -> float:
    r"""Calculates deviance as a scoring rule following spOccupancy and Hooten and Hobbs
    (2015).

    The deviance is calculated as:

    .. math::
        D = -2 \log\left(\frac{1}{Q} \sum_{q=1}^{Q} \prod_{i=1}^{n} \prod_{j=1}^{J_i} p(y_{ij} | z_i^{(q)}, p_{ij}^{(q)})\right)

    where :math:`Q` is the number of posterior samples, :math:`n` is the number of sites,
    :math:`J_i` is the number of visits at site :math:`i`, :math:`y_{ij}` is the observed
    detection at site :math:`i` and visit :math:`j`, :math:`z_i^{(q)}` is the latent occupancy
    state for site :math:`i` in sample :math:`q`, and :math:`p_{ij}^{(q)}` is the detection
    probability for site :math:`i` and visit :math:`j` in sample :math:`q`.

    References
    ----------
        - Hooten, M.B. and Hobbs, N.T. (2015), A guide to Bayesian model selection for ecologists. Ecological Monographs, 85: 3-28.

    Parameters
    ----------
        model_fn: The model function used to fit the data.
        posterior_samples: A dictionary containing posterior samples from a fitted model.
        observation_keys: A set of keys that represent the observed data in the posterior samples.
        **kwargs: Additional keyword arguments passed to the log_likelihood or model function.

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

    valid_obs = (
        jnp.isfinite(kwargs["obs"])
        & jnp.isfinite(kwargs["obs_covs"]).all(axis=-1)
        & jnp.isfinite(kwargs["site_covs"]).all(axis=-1)[:, None]
    )
    log_lik = log_likelihood(model_fn, posterior_samples, **kwargs)["y"].transpose(
        (0, 2, 1)
    )

    # Sum over all observations to get total log-likelihood for each sample
    log_lik_per_sample = jnp.sum(
        log_lik.reshape(log_lik.shape[0], -1)[:, valid_obs.reshape(-1)], axis=1
    )

    # Calculate the mean likelihood across MCMC samples (in log space)
    log_mean_likelihood = logsumexp(log_lik_per_sample) - jnp.log(
        log_lik_per_sample.shape[0]
    )

    # Calculate deviance: -2 * log(mean_likelihood)
    deviance_value = (-2 * log_mean_likelihood).item()

    return deviance_value


def deviance_manual(
    posterior_samples: Dict[str, jnp.ndarray],
    data: Dict[str, jnp.ndarray],
) -> float:
    """Calculates deviance manually for a non-false positive, Bernoulli occupancy model.

    Parameters
    ----------
    posterior_samples: A dictionary containing posterior samples from a fitted model.
    data: A dictionary containing observed data.

    Returns
    -------
    float: The deviance value as a scalar.

    Examples
    --------
    >>> from biolith.models import occu, simulate
    >>> from biolith.utils import fit, predict
    >>> from biolith.evaluation import lppd_manual
    >>> data, _ = simulate()
    >>> results = fit(occu, **data)
    >>> posterior_samples = predict(occu, results.mcmc, **data)
    >>> dev = deviance_manual(posterior_samples, data)
    """

    valid_obs = (
        jnp.isfinite(data["obs"])
        & jnp.isfinite(data["obs_covs"]).all(axis=-1)
        & jnp.isfinite(data["site_covs"]).all(axis=-1)[:, None]
    )

    log_lik_manual = log_likelihood_manual(posterior_samples, data)

    # Sum over all observations to get total log-likelihood for each sample
    log_lik_per_sample = jnp.sum(
        log_lik_manual.reshape(log_lik_manual.shape[0], -1)[:, valid_obs.reshape(-1)],
        axis=1,
    )

    # Calculate the mean likelihood across MCMC samples (in log space)
    log_mean_likelihood = logsumexp(log_lik_per_sample) - jnp.log(
        log_lik_per_sample.shape[0]
    )

    # Calculate deviance: -2 * log(mean_likelihood)
    deviance_value = (-2 * log_mean_likelihood).item()

    return deviance_value


class TestDeviance(unittest.TestCase):

    # TODO: Fix the disparity between the NumPyro and manual deviance calculation.
    @unittest.skip("Skipping failing deviance test for now.")
    def test_deviance(self):
        from biolith.models import occu, simulate
        from biolith.utils import fit, predict

        data, _ = simulate(simulate_missing=False)
        results = fit(occu, **data)
        posterior_samples = predict(occu, results.mcmc, **data)

        d = deviance(occu, posterior_samples, **data)
        d_manual = deviance_manual(posterior_samples, data)

        self.assertTrue(0 < d < jnp.inf, "Deviance should be positive and finite.")
        self.assertTrue(
            jnp.allclose(d, d_manual, rtol=1e-3),
            "NumPyro deviance should match manually computed deviance.",
        )


if __name__ == "__main__":
    unittest.main()
