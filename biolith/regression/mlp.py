import unittest
import numpyro
from numpyro.distributions import Distribution, Normal
import jax
import jax.numpy as jnp

from biolith.regression.abstract import AbstractRegression


class MLPRegression(AbstractRegression):
    """
    Multilayer perceptron model for occupancy or detection in an occupancy model.

    This model computes a potentially non-linear predictor based on covariates,
    which can be used for either occupancy or detection processes.
    """
    def __init__(self, name: str, n_covs: int, hidden_layer_sizes: list[int], prior: Distribution = Normal(0, 1)):
        self.weights = []
        self.biases = []
        in_features = n_covs
        for i, h in enumerate(hidden_layer_sizes):
            w = numpyro.sample(f"{name}_w_h{i}", prior.expand([in_features, h]))
            b = numpyro.sample(f"{name}_b_h{i}", prior.expand([h]))
            self.weights.append(w)
            self.biases.append(b)
            in_features = h
        w_out = numpyro.sample(f"{name}_w_out", prior.expand([in_features, 1]))
        b_out = numpyro.sample(f"{name}_b_out", prior.expand([1]))
        self.weights.append(w_out)
        self.biases.append(b_out)

    def __call__(self, covs: jnp.ndarray) -> jnp.ndarray:
        if covs.ndim not in [2, 3]:
            raise ValueError(f"Invalid covariate shape: {covs.shape}. Expected 2D or 3D array.")
        original_shape = covs.shape
        flattened = covs.reshape(original_shape[0], -1).T
        x = flattened
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = jnp.dot(x, w) + b
            x = jax.nn.relu(x)
        x = jnp.dot(x, self.weights[-1]) + self.biases[-1]
        x = jnp.squeeze(x, -1)
        return x.reshape(original_shape[1:])
    

class TestMLPRegression(unittest.TestCase):
    def test_mlp_regression(self):
        from numpyro.infer import MCMC, NUTS

        rng = jax.random.PRNGKey(0)
        x_data = jnp.linspace(-jnp.pi, jnp.pi, 50)
        y_true = jnp.sin(x_data)
        y_obs = y_true + 0.1 * jax.random.normal(rng, shape=y_true.shape)

        def model(x, y=None):
            lr = MLPRegression("mlp", n_covs=1, hidden_layer_sizes=[5, 5])
            mu = lr(x[None, :])
            numpyro.sample("obs", Normal(mu, 0.1), obs=y)

        mcmc = MCMC(NUTS(model), num_warmup=100, num_samples=100)
        mcmc.run(rng, x_data, y_obs)
        
        predictive = numpyro.infer.Predictive(model, mcmc.get_samples())
        samples = predictive(rng, x_data)
        
        preds = jnp.mean(samples["obs"], axis=0)
        self.assertTrue(jnp.mean(jnp.abs(preds - y_obs)) < 0.3)


if __name__ == "__main__":
    unittest.main()