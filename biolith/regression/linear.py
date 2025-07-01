import unittest

import jax
import jax.numpy as jnp
import numpyro
from numpyro.distributions import Distribution, Normal

from biolith.regression.abstract import AbstractRegression


class LinearRegression(AbstractRegression):
    """
    Linear regression model for occupancy or detection in an occupancy model.

    This model computes a linear predictor based on covariates, which can be used
    for either occupancy or detection processes.
    """

    def __init__(self, name: str, n_covs: int, prior: Distribution = Normal(0, 1)):
        """Initialize the linear regression model and sample its coefficients.

        Parameters
        ----------
        name : str
            Name of the model, used for naming the coefficients.
        n_covs : int
            Number of covariates.
        prior : Distribution, optional
            Prior distribution for the coefficients, by default Normal(0, 1).
        """
        self.coef = numpyro.sample(name, prior.expand([n_covs + 1]).to_event(1))

    def __call__(self, covs: jnp.ndarray) -> jnp.ndarray:
        """Compute the linear predictor for occupancy or detection.

        Parameters
        ----------
        covs : jnp.ndarray
            Site covariate matrix of shape (n_covs, n_sites) or observation covariate matrix of shape (n_covs, n_revisits, n_sites).

        Returns
        -------
        jnp.ndarray
            Linear predictor of shape (n_sites,) or of shape (n_revisits, n_sites).
        """
        if covs.ndim == 2:
            return jnp.tile(self.coef[0], (covs.shape[-1],)) + jnp.dot(
                self.coef[1:], covs
            )
        elif covs.ndim == 3:
            return jnp.tile(self.coef[0], (covs.shape[1], covs.shape[2])) + jnp.sum(
                self.coef[1:, None, None] * covs, axis=0
            )
        else:
            raise ValueError(
                f"Invalid covariate shape: {covs.shape}. Expected 2D or 3D array."
            )


class TestLinearRegression(unittest.TestCase):
    def test_linear_regression(self):
        from numpyro.infer import MCMC, NUTS

        rng = jax.random.PRNGKey(0)
        x_data = jnp.linspace(-1, 1, 50)
        true_params = jnp.array([1.0, 2.0])
        y_true = true_params[0] + true_params[1] * x_data
        y_obs = y_true + 0.1 * jax.random.normal(rng, shape=y_true.shape)

        def model(x, y=None):
            lr = LinearRegression("coef", n_covs=1)
            mu = lr(x[None, :])
            numpyro.sample("obs", Normal(mu, 0.1), obs=y)

        mcmc = MCMC(NUTS(model), num_warmup=100, num_samples=100)
        mcmc.run(rng, x_data, y_obs)

        predictive = numpyro.infer.Predictive(model, mcmc.get_samples())
        samples = predictive(rng, x_data)

        preds = jnp.mean(samples["obs"], axis=0)
        self.assertTrue(jnp.mean(jnp.abs(preds - y_obs)) < 0.3)

        # Plot results
        try:
            import matplotlib.pyplot as plt

            sorted_indices = jnp.argsort(x_data)
            x_sorted = x_data[sorted_indices]
            y_obs_sorted = y_obs[sorted_indices]
            preds_sorted = preds[sorted_indices]
            ci = jnp.percentile(samples["obs"], jnp.array([5.0, 95.0]), axis=0)
            ci_lower = ci[0, sorted_indices]
            ci_upper = ci[1, sorted_indices]
            plt.figure(figsize=(10, 6))
            plt.scatter(x_sorted, y_obs_sorted, label="Simulated Data", alpha=0.6)
            plt.plot(
                x_sorted,
                preds_sorted,
                label="Mean Prediction",
                color="red",
                linewidth=2,
            )
            plt.fill_between(
                x_sorted,
                ci_lower,
                ci_upper,
                color="red",
                alpha=0.2,
                label="90% Confidence Interval",
            )
            plt.xlabel("Covariate")
            plt.ylabel("Response")
            plt.title("Linear Regression Fit")
            plt.legend()
            plt.savefig("linear_regression_test_plot.png")
            plt.close()
        except ImportError:
            print("Matplotlib is not installed, skipping plot generation.")


if __name__ == "__main__":
    unittest.main()
