import unittest
from typing import Callable, Dict

import jax.numpy as jnp
from jax.scipy.special import logsumexp

from biolith.evaluation.log_likelihood import log_likelihood, log_likelihood_manual


def waic(
    model_fn: Callable,
    posterior_samples: Dict[str, jnp.ndarray],
    **kwargs,
) -> Dict[str, float]:
    r"""Calculates the Watanabe-Akaike Information Criterion (WAIC) for a fitted model.

    WAIC is calculated as:

    .. math::
        \text{WAIC} = -2(\text{lppd} - p_{\text{WAIC}})

    where the log pointwise predictive density (lppd) is:

    .. math::
        \text{lppd} = \sum_{i=1}^{n} \log\left(\frac{1}{Q} \sum_{q=1}^{Q} p(y_i | \theta^{(q)})\right)

    and the effective number of parameters :math:`p_{\text{WAIC}}` is:

    .. math::
        p_{\text{WAIC}} = \sum_{i=1}^{n} \text{Var}_q(\log p(y_i | \theta^{(q)}))

    where :math:`n` is the number of sites, :math:`Q` is the number of posterior samples,
    :math:`y_i` is the observed detection history for site :math:`i` across :math:`J_i` revisits,
    and :math:`\theta^{(q)}` represents the model parameters in posterior sample :math:`q`.

    This function uses an implementation based on the log likelihood of the model
    given the posterior samples and the observed data.

    Parameters
    ----------
    model_fn: The model function used to fit the data.
    posterior_samples: A dictionary containing posterior samples from a fitted model.
    **kwargs: Additional keyword arguments passed to the log_likelihood or model function.

    Returns
    -------
    Dict[str, float]: A dictionary containing 'waic', 'p_waic', and 'lppd' values.

    Examples
    --------
    >>> from biolith.models import simulate, occu
    >>> from biolith.utils import fit, predict
    >>> from biolith.evaluation import waic
    >>> data, _ = simulate()
    >>> results = fit(occu, **data)
    >>> preds = predict(occu, results.mcmc, **data)
    >>> waic_result = waic(occu, preds, **data)
    >>> print(f"WAIC: {waic_result['waic']:.2f}")
    """

    valid_obs = (
        jnp.isfinite(kwargs["obs"])
        & jnp.isfinite(kwargs["obs_covs"]).all(axis=-1)
        & jnp.isfinite(kwargs["site_covs"]).all(axis=-1)[:, None, None]
    )

    # Get log likelihood: shape (n_samples, n_visits, n_sites) - match LPPD implementation
    log_lik = log_likelihood(model_fn, posterior_samples, **kwargs)["y"].transpose(
        (0, 3, 2, 1)
    )

    # Calculate lppd for each valid observation - same as LPPD implementation
    lppd = jnp.sum(
        logsumexp(log_lik[:, valid_obs], axis=0) - jnp.log(log_lik.shape[0])
    ).item()

    # Calculate p_waic (effective number of parameters) - variance in log likelihood
    p_waic = jnp.sum(jnp.var(log_lik[:, valid_obs], axis=0, ddof=1)).item()

    # Calculate WAIC
    waic_value = -2 * (lppd - p_waic)

    return {"waic": waic_value, "p_waic": p_waic, "lppd": lppd}


def waic_manual(
    posterior_samples: Dict[str, jnp.ndarray],
    data: Dict[str, jnp.ndarray],
) -> Dict[str, float]:
    """Calculates the Watanabe-Akaike Information Criterion (WAIC) for a non-false
    positive, Bernoulli occupancy model using manual log likelihood calculations.

    Parameters
    ----------
    posterior_samples: A dictionary containing posterior samples from a fitted model.
    data: A dictionary containing observed data.

    Returns
    -------
    Dict[str, float]: A dictionary containing 'waic', 'p_waic', and 'lppd' values.

    Examples
    --------
    >>> from biolith.models import occu, simulate
    >>> from biolith.utils import fit, predict
    >>> from biolith.evaluation import waic_manual
    >>> data, _ = simulate()
    >>> results = fit(occu, **data)
    >>> posterior_samples = predict(occu, results.mcmc, **data)
    >>> waic_result = waic_manual(posterior_samples, data)
    >>> print(f"WAIC: {waic_result['waic']:.2f}")
    """

    valid_obs = (
        jnp.isfinite(data["obs"])
        & jnp.isfinite(data["obs_covs"]).all(axis=-1)
        & jnp.isfinite(data["site_covs"]).all(axis=-1)[:, None, None]
    )

    # Get manual log likelihood: shape (n_samples, n_sites, n_visits)
    log_lik_manual = log_likelihood_manual(posterior_samples, data)

    # Calculate lppd for each valid observation
    lppd = jnp.sum(
        logsumexp(log_lik_manual[:, valid_obs], axis=0)
        - jnp.log(log_lik_manual.shape[0])
    ).item()

    # Calculate p_waic (effective number of parameters) - variance in log likelihood
    p_waic = jnp.sum(jnp.var(log_lik_manual[:, valid_obs], axis=0, ddof=1)).item()

    # Calculate WAIC
    waic_value = -2 * (lppd - p_waic)

    return {"waic": waic_value, "p_waic": p_waic, "lppd": lppd}


class TestWAIC(unittest.TestCase):

    def test_waic(self):
        from biolith.models import occu, simulate
        from biolith.utils import fit, predict

        data, _ = simulate(simulate_missing=True, random_seed=1)
        results = fit(occu, **data)
        posterior_samples = predict(occu, results.mcmc, **data)

        waic_result = waic(occu, posterior_samples, **data)
        waic_manual_result = waic_manual(posterior_samples, data)

        for wr in [waic_result, waic_manual_result]:

            # Check that all values are finite
            self.assertTrue(jnp.isfinite(wr["waic"]), "WAIC should be finite.")
            self.assertTrue(jnp.isfinite(wr["p_waic"]), "p_WAIC should be finite.")
            self.assertTrue(jnp.isfinite(wr["lppd"]), "LPPD should be finite.")

            # Check that p_waic is positive
            self.assertTrue(wr["p_waic"] > 0, "p_WAIC should be positive.")


if __name__ == "__main__":
    unittest.main()
