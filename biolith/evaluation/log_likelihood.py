import unittest
from typing import Callable, Dict

import jax
import jax.numpy as jnp
import numpyro
from jax.scipy.special import logsumexp
from numpyro.infer import log_likelihood as numpyro_log_likelihood


def log_likelihood(
    model_fn: Callable,
    posterior_samples: Dict[str, jnp.ndarray],
    observation_keys: set = {"y", "s"},
    **kwargs,
) -> dict[str, jnp.ndarray]:
    r"""Calculates the log likelihood of observations given a fitted model.

    Parameters
    ----------
        model_fn: The model function used to fit the data.
        posterior_samples: A dictionary containing posterior samples from a fitted model.
        observation_keys: A set of keys that represent the observed data in the posterior samples.
        **kwargs: Additional keyword arguments passed to the log_likelihood or model function.
    Returns
    -------
        dict[str, jnp.ndarray]: The log likelihood for each sample and observation.

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

    # Ensure posterior_samples does not contain observed samples,
    # because otherwise NumPyro will compute the likelihood for
    # the sampled values
    posterior_samples = {
        k: v for k, v in posterior_samples.items() if k not in observation_keys
    }

    with numpyro.handlers.block(), numpyro.handlers.seed(
        rng_seed=jax.random.PRNGKey(0)
    ):
        log_lik = numpyro_log_likelihood(model_fn, posterior_samples, **kwargs)

    return log_lik


class TestLogLikelihood(unittest.TestCase):

    def test_log_likelihood(self):
        from biolith.models import occu, simulate
        from biolith.utils import fit, predict

        data, _ = simulate()
        results = fit(occu, **data)
        posterior_samples = predict(occu, results.mcmc, **data)

        log_lik = log_likelihood(occu, posterior_samples, **data)["y"].transpose(
            (0, 2, 1)
        )

        log_lik_manual = jnp.log(
            posterior_samples["prob_detection"].transpose((0, 2, 1))
            * posterior_samples["psi"][:, :, None]
        ) * data["obs"][None, :, :] + jnp.log(
            1
            - posterior_samples["prob_detection"].transpose((0, 2, 1))
            * posterior_samples["psi"][:, :, None]
        ) * (
            1 - data["obs"][None, :, :]
        )

        log_lik_per_obs = logsumexp(
            log_lik[:, jnp.isfinite(data["obs"])], axis=0
        ) - jnp.log(log_lik.shape[0])
        log_lik_manual_per_obs = logsumexp(
            log_lik_manual[:, jnp.isfinite(data["obs"])], axis=0
        ) - jnp.log(log_lik_manual.shape[0])

        self.assertTrue(
            jnp.allclose(log_lik_per_obs, log_lik_manual_per_obs, rtol=1e-1)
        )


if __name__ == "__main__":
    unittest.main()
