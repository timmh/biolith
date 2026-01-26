import unittest
from typing import Callable, Dict

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from biolith.evaluation.log_likelihood import log_likelihood, log_likelihood_manual


def lppd(
    model_fn: Callable,
    posterior_samples: Dict[str, jnp.ndarray],
    **kwargs,
) -> float:
    r"""Calculates the log pointwise predictive density (lppd) for a fitted model.

    The lppd is calculated as:

    .. math::
        \text{lppd} = \sum_{i=1}^{n} \log\left(\frac{1}{Q} \sum_{q=1}^{Q} p(y_i | \theta^{(q)})\right)

    where :math:`n` is the number of sites, :math:`Q` is the number of posterior samples,
    :math:`y_i` is the observed detection history for site :math:`i` across :math:`J_i` revisits,
    and :math:`\theta^{(q)}` represents the model parameters in posterior sample :math:`q`.

    This function computes the lppd using the log likelihood of the model
    given the posterior samples and the observed data. It uses the
    `log_likelihood` function from numpyro to evaluate the model's
    likelihood for the provided observations.

    Parameters
    ----------
        model_fn: The model function used to fit the data.
        posterior_samples: A dictionary containing posterior samples from a fitted model.
        **kwargs: Additional keyword arguments passed to the log_likelihood or model function.
    Returns
    -------
        float: The log pointwise predictive density (lppd) value.

    Examples
    --------
    >>> from biolith.models import simulate, occu
    >>> from biolith.utils import fit, predict
    >>> from biolith.evaluation import lppd
    >>> data, _ = simulate()
    >>> results = fit(occu, **data)
    >>> preds = predict(occu, results.mcmc, **data)
    >>> lppd(occu, preds, **data)
    """

    valid_obs = (
        jnp.isfinite(kwargs["obs"])
        & jnp.isfinite(kwargs["obs_covs"]).all(axis=-1)
        & jnp.isfinite(kwargs["site_covs"]).all(axis=-1)[:, None, None]
    )
    log_lik = log_likelihood(model_fn, posterior_samples, **kwargs)["y"].transpose(
        (0, 3, 2, 1)
    )
    lppd = jnp.sum(
        logsumexp(log_lik[:, valid_obs], axis=0) - np.log(log_lik.shape[0])
    ).item()

    return lppd


def lppd_manual(
    posterior_samples: Dict[str, jnp.ndarray],
    data: Dict[str, jnp.ndarray],
) -> float:
    """Calculates the log pointwise predictive density (lppd) manually for a non-false
    positive, Bernoulli occupancy model.

    Parameters
    ----------
    posterior_samples: A dictionary containing posterior samples from a fitted model.
    data: A dictionary containing observed data.

    Returns
    -------
    float: The log pointwise predictive density (lppd) value.

    Examples
    --------
    >>> from biolith.models import occu, simulate
    >>> from biolith.utils import fit, predict
    >>> from biolith.evaluation import lppd_manual
    >>> data, _ = simulate()
    >>> results = fit(occu, **data)
    >>> posterior_samples = predict(occu, results.mcmc, **data)
    >>> lppd_manual(posterior_samples, data)
    """

    valid_obs = (
        jnp.isfinite(data["obs"])
        & jnp.isfinite(data["obs_covs"]).all(axis=-1)
        & jnp.isfinite(data["site_covs"]).all(axis=-1)[:, None, None]
    )

    log_lik_manual = log_likelihood_manual(posterior_samples, data)

    lppd_manual = jnp.sum(
        logsumexp(log_lik_manual[:, valid_obs], axis=0)
        - jnp.log(log_lik_manual.shape[0])
    ).item()

    return lppd_manual


class TestLPPD(unittest.TestCase):

    def test_lppd(self):
        from biolith.models import occu, simulate
        from biolith.utils import fit, predict

        data, _ = simulate(simulate_missing=True)
        results = fit(occu, **data)
        posterior_samples = predict(occu, results.mcmc, **data)

        l = lppd(occu, posterior_samples, **data)
        l_manual = lppd_manual(posterior_samples, data)

        self.assertTrue(-jnp.inf < l < 0, "LPPD should be negative and finite.")
        self.assertTrue(
            jnp.allclose(l, l_manual, rtol=1e-2),
            "NumPyro LPPD should match manually computed LPPD.",
        )


if __name__ == "__main__":
    unittest.main()
