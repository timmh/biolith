from collections import namedtuple
from typing import Callable, Literal, Optional

import jax
from numpyro.infer import HMC, HMCECS, MCMC, NUTS, DiscreteHMCGibbs, MixedHMC

from biolith.regression.bart import BARTRegression

from .data import dataframes_to_arrays, rename_samples

FitResult = namedtuple("FitResult", ["samples", "mcmc"])


def fit(
    model_fn: Callable,
    site_covs=None,
    obs_covs=None,
    obs=None,
    session_duration=None,
    num_samples: int = 1000,
    num_warmup: int = 1000,
    random_seed: int = 0,
    num_chains: int = 5,
    kernel: Optional[
        Literal["nuts", "hmc", "mixed_hmc", "discrete_hmc_gibbs", "hmcecs"]
    ] = None,
    timeout: int | None = None,
    **kwargs,
) -> FitResult:
    """Fit a NumPyro model using the provided data.

    Parameters
    ----------
    model_fn:
        The model function to fit.
    site_covs:
        Array or Pandas DataFrame containing site-level covariates.
    obs_covs:
        Array or Pandas DataFrame containing observation-level covariates.
    obs:
        Array or Pandas DataFrame containing the observed data.
    session_duration:
        Array or Pandas DataFrame containing the session duration for each observation.
    num_samples:
        Number of posterior samples to draw.
    num_warmup:
        Number of warmup steps for the sampler.
    random_seed:
        Seed used for the random number generator.
    num_chains:
        Number of MCMC chains to run.
    kernel:
        Name of the sampling kernel to use. Possible values include
        ``"nuts"``, ``"hmc"``, ``"mixed_hmc"``, ``"discrete_hmc_gibbs"``,
        or ``"hmcecs"``. Defaults to ``"nuts"`` for most models.
    timeout:
        Optional timeout (in seconds) for the sampling step.
    **kwargs:
        Additional keyword arguments passed to ``model_fn``.

    Returns
    -------
    FitResult
        A tuple-like object containing the posterior samples and the MCMC
        object itself.

    Examples
    --------
    >>> from biolith.models import simulate, occu
    >>> from biolith.utils import fit
    >>> data, _ = simulate()
    >>> results = fit(occu, **data)
    """

    site_covs, obs_covs, obs, session_duration, site_covs_names, obs_covs_names = (
        dataframes_to_arrays(site_covs, obs_covs, obs, session_duration)
    )

    if kernel is None:
        kernel = "nuts"

        # check if one of the arguments to the model is a RegressionModel that required discrete parameters
        if any([arg is BARTRegression for arg in kwargs.values()]):
            kernel = "discrete_hmc_gibbs"

    kernel_inst = dict(
        nuts=lambda: NUTS(model_fn),
        hmc=lambda: HMC(model_fn),
        mixed_hmc=lambda: MixedHMC(HMC(model_fn)),
        discrete_hmc_gibbs=lambda: DiscreteHMCGibbs(NUTS(model_fn)),
        hmcecs=lambda: HMCECS(NUTS(model_fn)),
    )[kernel]()
    mcmc = MCMC(
        kernel_inst,
        num_samples=num_samples,
        num_warmup=num_warmup,
        num_chains=num_chains,
        chain_method=(
            "parallel" if num_chains <= jax.local_device_count() else "sequential"
        ),
    )

    arguments = dict(
        site_covs=site_covs,
        obs_covs=obs_covs,
        obs=obs,
        session_duration=session_duration,
    )
    valid_arguments = {k: v for k, v in arguments.items() if v is not None}
    rng_key = jax.random.PRNGKey(random_seed)

    if timeout is not None:
        from .misc import time_limit

        with time_limit(timeout):
            mcmc.run(rng_key, **valid_arguments, **kwargs)
    else:
        mcmc.run(rng_key, **valid_arguments, **kwargs)

    samples = mcmc.get_samples()
    samples = rename_samples(samples, site_covs_names, obs_covs_names)

    return FitResult(samples, mcmc)
