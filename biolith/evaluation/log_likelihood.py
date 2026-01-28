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
    >>> log_likelihood(occu, preds, **data)
    """

    # Ensure posterior_samples does not contain observed samples,
    # because otherwise NumPyro will compute the likelihood for
    # the sampled values
    posterior_samples = {
        k: v for k, v in posterior_samples.items() if k not in observation_keys
    }

    with (
        numpyro.handlers.block(),
        numpyro.handlers.seed(rng_seed=jax.random.PRNGKey(0)),
    ):
        log_lik = numpyro_log_likelihood(model_fn, posterior_samples, **kwargs)

    return log_lik


def log_likelihood_manual(
    posterior_samples: Dict[str, jnp.ndarray], data: Dict[str, jnp.ndarray], eps=1e-10
) -> jnp.ndarray:
    """Calculates the log likelihood manually for a non-false positive, Bernoulli
    occupancy model, based on posterior samples and observed data.

    Parameters
    ----------
    posterior_samples: A dictionary containing posterior samples from a fitted model.
    data: A dictionary containing observed data.

    Returns
    -------
    jnp.ndarray: The manually computed log likelihood for each observation.

    Examples
    --------
    >>> from biolith.models import occu, simulate
    >>> from biolith.utils import fit, predict
    >>> data, _ = simulate()
    >>> results = fit(occu, **data)
    >>> posterior_samples = predict(occu, results.mcmc, **data)
    >>> log_likelihood_manual(posterior_samples, data
    """

    obs_transposed = data["obs"].transpose((3, 2, 1, 0))
    prob_detection = posterior_samples["prob_detection"]
    psi = posterior_samples["psi"]
    if psi.ndim == 2:
        psi = psi[:, None, :, None]
    elif psi.ndim == 3:
        psi = psi[:, None, ...]
    if psi.shape[1] != obs_transposed.shape[1]:
        psi = jnp.broadcast_to(
            psi, (psi.shape[0], obs_transposed.shape[1], psi.shape[2], psi.shape[3])
        )
    psi = psi[:, None, ...]

    prob_det_occ = jnp.clip(prob_detection * psi, min=eps, max=1 - eps)
    log_lik_manual = jnp.log(prob_det_occ) * obs_transposed[None, ...] + jnp.log(
        jnp.clip(1 - prob_detection * psi, min=eps, max=1 - eps)
    ) * (1 - obs_transposed[None, ...])

    return log_lik_manual.transpose((0, 4, 3, 2, 1))


def test_log_likelihood():
    from biolith.models import occu, simulate
    from biolith.utils import fit, predict

    data, _ = simulate(simulate_missing=True)
    valid_obs = (
        jnp.isfinite(data["obs"])
        & jnp.isfinite(data["obs_covs"]).all(axis=-1)[None, ...]
        & jnp.isfinite(data["site_covs"]).all(axis=-1)[None, :, None, None]
    )

    results = fit(occu, **data)
    posterior_samples = predict(occu, results.mcmc, **data)

    log_lik = log_likelihood(occu, posterior_samples, **data)["y"].transpose(
        (0, 4, 3, 2, 1)
    )

    log_lik_manual = log_likelihood_manual(posterior_samples, data)

    log_lik_per_obs = logsumexp(log_lik[:, valid_obs], axis=0) - jnp.log(
        log_lik.shape[0]
    )
    log_lik_manual_per_obs = logsumexp(
        log_lik_manual[:, valid_obs], axis=0
    ) - jnp.log(log_lik_manual.shape[0])

    assert jnp.allclose(log_lik_per_obs, log_lik_manual_per_obs, rtol=1e-1)
