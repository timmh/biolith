from typing import Optional
from abc import ABC, abstractmethod
from numpyro.distributions import Distribution
import jax.numpy as jnp


class AbstractRegression(ABC):
    def __init__(self, name: str, n_covs: int, prior: Optional[Distribution]):
        pass

    @abstractmethod
    def __call__(self, covs: jnp.ndarray) -> jnp.ndarray:
        pass
