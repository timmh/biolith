from collections import namedtuple
from multiprocessing import Process, Queue
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


def _fit_worker(queue, model_fn, fit_kwargs):
    """Worker function to fit a model in a separate process."""

    try:
        result = fit(model_fn, **fit_kwargs)
        queue.put(result)
    except Exception as e:
        queue.put(e)


def fit_multiprocess(
    model_fn: Callable,
    timeout: Optional[int] = None,
    **fit_kwargs,
) -> FitResult:
    """Fit a model in a separate process with timeout and error handling.

    This function runs the fit operation in a separate process to isolate
    memory usage.

    Parameters
    ----------
    model_fn : Callable
        The model function to fit.
    timeout : int, optional
        Timeout in seconds for the fitting process. If None, no timeout is applied.
    **fit_kwargs
        All other keyword arguments are passed directly to the fit function.

    Returns
    -------
    FitResult
        A tuple-like object containing the posterior samples and the MCMC object.

    Raises
    ------
    TimeoutError
        If the fitting process exceeds the specified timeout.
    Exception
        Any exception that occurred during the fitting process.

    Examples
    --------
    >>> from biolith.models import simulate, occu
    >>> from biolith.utils import fit_multiprocess
    >>> data, _ = simulate()
    >>> result = fit_multiprocess(occu, timeout=300, **data)
    """

    queue = Queue()
    process = Process(
        target=_fit_worker, args=(queue, model_fn, fit_kwargs), daemon=False
    )

    process.start()

    try:
        result = queue.get(timeout=timeout)
    except Exception:
        process.terminate()
        process.join()
        raise TimeoutError(f"Model fitting exceeded timeout of {timeout} seconds")

    process.join()

    # Check if the result is an exception
    if isinstance(result, Exception):
        raise result

    return result
