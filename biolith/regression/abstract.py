from abc import ABC, abstractmethod
from typing import Optional

import jax.numpy as jnp
from numpyro.distributions import Distribution


class AbstractRegression(ABC):
    """Abstract base class for regression models in occupancy modeling.

    This class defines the interface for regression models that can be used to compute
    predictors for occupancy or detection processes based on covariates.
    """

    def __init__(self, name: str, n_covs: int, prior: Optional[Distribution]):
        pass

    @abstractmethod
    def __call__(self, covs: jnp.ndarray) -> jnp.ndarray:
        pass
