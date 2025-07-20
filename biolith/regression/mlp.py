import unittest

import jax
import jax.numpy as jnp
import numpyro
from numpyro.distributions import Distribution, Normal

from biolith.regression.abstract import AbstractRegression


class MLPRegression(AbstractRegression):
    """Multilayer perceptron model for occupancy or detection in an occupancy model.

    This model computes a potentially non-linear predictor based on covariates, which
    can be used for either occupancy or detection processes.
    """

    def __init__(
        self,
        name: str,
        n_covs: int,
        hidden_layer_sizes: list[int] = [10, 10],
        prior: Distribution = Normal(0, 1),
    ):
        """Initialize the MLP regression model and sample its parameters.

        Parameters
        ----------
        name : str
            Name of the model, used for naming the parameters.
        n_covs : int
            Number of covariates.
        hidden_layer_sizes : list[int]
            List of integers specifying the number of neurons in each hidden layer.
        prior : Distribution, optional
            Prior distribution for the parameters, by default Normal(0, 1).
        """
        self.weights = []
        self.biases = []
        in_features = n_covs
        for i, h in enumerate(hidden_layer_sizes):
            w = numpyro.sample(
                f"{name}_w_h{i}", prior.expand([in_features, h]).to_event(2)
            )
            b = numpyro.sample(f"{name}_b_h{i}", prior.expand([h]).to_event(1))
            self.weights.append(w)
            self.biases.append(b)
            in_features = h
        w_out = numpyro.sample(
            f"{name}_w_out", prior.expand([in_features, 1]).to_event(2)
        )
        b_out = numpyro.sample(f"{name}_b_out", prior.expand([1]).to_event(1))
        self.weights.append(w_out)
        self.biases.append(b_out)

    def __call__(self, covs: jnp.ndarray) -> jnp.ndarray:
        """Compute the predictor for occupancy or detection.

        Parameters
        ----------
        covs : jnp.ndarray
            Site covariate matrix of shape (n_covs, n_sites) or observation covariate matrix of shape (n_covs, n_revisits, n_sites).

        Returns
        -------
        jnp.ndarray
            Predictor of shape (n_sites,) or of shape (n_revisits, n_sites).
        """
        if covs.ndim not in [2, 3]:
            raise ValueError(
                f"Invalid covariate shape: {covs.shape}. Expected 2D or 3D array."
            )
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
            numpyro.sample("obs", Normal(mu, 0.1), obs=y)  # type: ignore

        mcmc = MCMC(NUTS(model), num_warmup=100, num_samples=100)
        mcmc.run(rng, x_data, y_obs)

        predictive = numpyro.infer.Predictive(model, mcmc.get_samples())
        samples = predictive(rng, x_data)

        preds = jnp.mean(samples["obs"], axis=0)
        self.assertTrue(jnp.mean(jnp.abs(preds - y_obs)) < 0.3)

        # Plot results
        try:
            import matplotlib.pyplot as plt

            ci_lower = jnp.percentile(samples["obs"], 5, axis=0)
            ci_upper = jnp.percentile(samples["obs"], 95, axis=0)
            plt.figure(figsize=(10, 6))
            plt.scatter(x_data, y_obs, label="Simulated Data", alpha=0.7)
            plt.plot(x_data, preds, label="Mean Prediction", color="red")
            plt.fill_between(
                x_data,
                ci_lower,
                ci_upper,
                color="red",
                alpha=0.3,
                label="90% Confidence Interval",
            )
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("MLP Regression Fit")
            plt.legend()
            import os

            os.makedirs("figures", exist_ok=True)
            plt.savefig("figures/mlp_regression_test_plot.png")
            plt.close()
        except ImportError:
            print("Matplotlib is not installed, skipping plot generation.")


if __name__ == "__main__":
    unittest.main()
