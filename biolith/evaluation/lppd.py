from typing import Callable, Dict

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from jax.scipy.special import logsumexp
from numpyro.infer import log_likelihood


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

    with numpyro.handlers.block(), numpyro.handlers.seed(
        rng_seed=jax.random.PRNGKey(0)
    ):
        log_lik_test = log_likelihood(model_fn, posterior_samples, **kwargs)
        lppd_test = jnp.sum(
            logsumexp(log_lik_test["y"], axis=0) - np.log(log_lik_test["y"].shape[0])
        ).item()

    return lppd_test
