import unittest

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import lax

from biolith.regression.abstract import AbstractRegression


class BARTRegression(AbstractRegression):
    """Bayesian Additive Regression Trees (BART) in NumPyro.

    This model computes a potentially non-linear predictor based on covariates
    which are assumed to be standard normally distributed. The sum of the tree
    predictions follows a zero-mean normal distribution with user specified
    standard deviation via the ``scale`` parameter.
    """

    def __init__(
        self,
        name: str,
        n_covs: int,
        prior: dist.Distribution = dist.Normal(0, 1),
        n_trees: int = 50,
        max_depth: int = 2,
        k: float = 2.0,
        scale: float = 1.0,
        alpha: float = 0.95,
        beta: float = 2.0,
    ):
        """Initialize the BART model and sample its parameters.

        Parameters
        ----------
        name : str
            Name of the model, used for naming the parameters.
        n_covs : int
            Number of covariates.
        prior : dist.Distribution, optional
            Prior distribution for the split values, by default Normal(0, 1).
        n_trees : int, optional
            Number of trees in the ensemble, by default 50.
        max_depth : int, optional
            Maximum depth of each tree, by default 2.
        k : float, optional
            Scaling factor for the prior on leaf values, by default 2.0.
        scale : float, optional
            Target standard deviation for the sum of tree predictions,
            by default 1.0.
        alpha : float, optional
            Parameter for the prior on split probabilities, by default 0.95.
        beta : float, optional
            Parameter for the prior on split probabilities, by default 2.0.
        """
        self.name = name
        self.n_covs = n_covs
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.k = k
        self.scale = scale

        # Calculate dimensions of the full binary tree
        self.num_internal_nodes = 2**self.max_depth - 1
        self.num_nodes = 2 ** (self.max_depth + 1) - 1

        sigma_mu = self.scale / (self.k * jnp.sqrt(self.n_trees))

        with numpyro.plate(f"{self.name}_trees", self.n_trees, dim=-2):
            with numpyro.plate(f"{self.name}_nodes", self.num_nodes, dim=-1):
                self.leaf_values = numpyro.sample(
                    f"{self.name}_leaf_values", dist.Normal(0, sigma_mu)  # type: ignore
                )

        depths = jnp.floor(jnp.log2(jnp.arange(1, self.num_internal_nodes + 1)))
        split_probs = alpha * (1 + depths) ** (-beta)

        with numpyro.plate(f"{self.name}_trees", self.n_trees, dim=-2):
            with numpyro.plate(
                f"{self.name}_internal_nodes", self.num_internal_nodes, dim=-1
            ):
                self.is_split_node = numpyro.sample(
                    f"{self.name}_is_split",
                    dist.Bernoulli(split_probs),  # type: ignore
                    infer={"enumerate": None},
                )
                self.split_vars = numpyro.sample(
                    f"{self.name}_split_vars",
                    dist.Categorical(logits=jnp.zeros(self.n_covs)),  # type: ignore
                    infer={"enumerate": None},
                )
                self.split_values = numpyro.sample(f"{self.name}_split_values", prior)

        self.compute_feature_importances()

    def __call__(self, covs: jnp.ndarray) -> jnp.ndarray:
        """Compute the predictor for occupancy or detection as mean response.

        Parameters
        ----------
        covs : jnp.ndarray
            Site covariate matrix of shape (n_covs, n_sites) or observation covariate
            matrix of shape (n_covs, n_revisits, n_sites) or
            (n_covs, n_replicates, n_periods, n_sites).

        Returns
        -------
        jnp.ndarray
            Predictor of shape (n_sites,) or of shape (n_revisits, n_sites).
        """
        if covs.ndim not in [2, 3, 4]:
            raise ValueError(
                f"Invalid covariate shape: {covs.shape}. Expected 2D, 3D, or 4D array."
            )

        # change order of dimensions in covs to
        # (n_sites, n_covs), (n_sites, n_revisits, n_covs), or
        # (n_sites, n_periods, n_replicates, n_covs)
        if covs.ndim == 4:
            covs = covs.transpose((3, 2, 1, 0))
        elif covs.ndim == 3:
            covs = covs.transpose((2, 1, 0))
        else:
            covs = covs.transpose((1, 0))

        original_shape = covs.shape
        covs_flat = covs.reshape(-1, original_shape[-1]) if covs.ndim >= 3 else covs

        if covs_flat.shape[-1] != self.n_covs:
            raise ValueError(
                f"Covariate dim mismatch. Model has {self.n_covs}, got {covs_flat.shape[-1]}."
            )

        def get_leaf_idx_for_sample_and_tree(
            x, is_split_nodes_t, split_vars_t, split_values_t
        ):
            def body_fun(_, node_idx):
                is_split = is_split_nodes_t[node_idx]
                split_var = split_vars_t[node_idx]
                split_val = split_values_t[node_idx]
                go_left = x[split_var] <= split_val
                next_node_idx = jnp.where(go_left, 2 * node_idx + 1, 2 * node_idx + 2)
                return jnp.where(is_split, next_node_idx, node_idx)

            return lax.fori_loop(0, self.max_depth, body_fun, 0)

        # Use the original parameter arrays directly without trying to handle enumeration
        is_split_node = self.is_split_node
        split_vars = self.split_vars
        split_values = self.split_values

        leaf_indices = jax.vmap(
            lambda x: jax.vmap(
                get_leaf_idx_for_sample_and_tree, in_axes=(None, 0, 0, 0)
            )(x, is_split_node, split_vars, split_values)
        )(covs_flat)

        def gather_leaf_values(leaf_indices_for_sample):
            return jax.vmap(
                lambda tree_idx, leaf_idx: self.leaf_values[tree_idx, leaf_idx]  # type: ignore
            )(jnp.arange(self.n_trees), leaf_indices_for_sample)

        predictions_per_tree = jax.vmap(gather_leaf_values)(leaf_indices)
        final_prediction = self.k * jnp.sum(predictions_per_tree, axis=-1)

        if covs.ndim == 4:
            pred = final_prediction.reshape(
                (original_shape[0], original_shape[1], original_shape[2])
            )
            return pred.transpose((2, 1, 0))
        if covs.ndim == 3:
            pred = final_prediction.reshape((original_shape[0], original_shape[1]))
            return pred.transpose((1, 0))
        return final_prediction

    def compute_feature_importances(self) -> None:
        """Compute a simple feature importance score based on frequency of splits across
        all trees and internal nodes."""
        used_features = jnp.where(self.is_split_node == 1, self.split_vars, -1)
        one_hot_feats = jax.nn.one_hot(used_features, self.n_covs)
        counts = jnp.sum(one_hot_feats, axis=(0, 1))
        total_splits = jnp.sum(counts)
        feature_importances = counts / (total_splits + 1e-10)
        numpyro.deterministic(f"{self.name}_feature_importances", feature_importances)


class TestBARTRegression(unittest.TestCase):
    def test_bart_regression(self):
        from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs, Predictive

        rng_key = jax.random.PRNGKey(42)
        rng_key, rng_x, rng_obs, rng_inf = jax.random.split(rng_key, 4)

        n_samples = 50
        x_data = jax.random.normal(rng_x, (n_samples,))[None, :]
        y_true = jnp.sin(x_data[0, :] * jnp.pi)
        y_obs = y_true + 0.1 * jax.random.normal(rng_obs, shape=y_true.shape)

        x_train = x_data
        y_train = y_obs

        def model(x, y=None):
            bart = BARTRegression(
                "bart", n_covs=x.shape[0], n_trees=20, max_depth=2, scale=1.0
            )
            mu = bart(x)
            with numpyro.plate("data", x.shape[1]):
                numpyro.sample("obs", dist.Normal(mu, 0.1), obs=y)  # type: ignore

        kernel = DiscreteHMCGibbs(NUTS(model))
        mcmc = MCMC(kernel, num_warmup=500, num_samples=500)
        mcmc.run(rng_inf, x_train, y_train)

        predictive = Predictive(model, mcmc.get_samples())
        samples = predictive(rng_key, x_train)

        preds = jnp.mean(samples["obs"], axis=0)

        mae = jnp.mean(jnp.abs(preds - y_obs))
        self.assertTrue(
            mae < 0.3, f"Mean Absolute Error should be < 0.3, but was {mae:.4f}"
        )

        # Plot results
        try:
            import matplotlib.pyplot as plt

            ci_lower = jnp.percentile(samples["obs"], 5, axis=0)
            ci_upper = jnp.percentile(samples["obs"], 95, axis=0)
            sort_indices = jnp.argsort(x_data[0, :])
            x_data_sorted = x_data[0, sort_indices]
            y_obs_sorted = y_obs[sort_indices]
            y_true_sorted = y_true[sort_indices]
            preds_sorted = preds[sort_indices]
            ci_lower_sorted = ci_lower[sort_indices]
            ci_upper_sorted = ci_upper[sort_indices]
            plt.figure(figsize=(10, 6))
            plt.scatter(x_data_sorted, y_obs_sorted, label="Simulated Data", alpha=0.6)
            plt.plot(x_data_sorted, y_true_sorted, "g-", label="True Function")
            plt.plot(x_data_sorted, preds_sorted, "r-", label="Mean Prediction")
            plt.fill_between(
                x_data_sorted,
                ci_lower_sorted,
                ci_upper_sorted,
                color="r",
                alpha=0.2,
                label="90% Confidence Interval",
            )
            plt.title("BART Regression on Noisy Sine Wave")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            import os

            os.makedirs("figures", exist_ok=True)
            plt.savefig("figures/bart_regression_test_plot.png")
            plt.close()
        except ImportError:
            print("Matplotlib is not installed, skipping plot generation.")

    def test_bart_regression_feature_importance(self):
        from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs, Predictive

        rng_key = jax.random.PRNGKey(42)
        rng_key, rng_cov1, rng_cov2, rng_obs, rng_inf = jax.random.split(rng_key, 5)

        n_samples = 50
        x_data = jnp.stack(
            [
                jax.random.normal(rng_cov1, (n_samples,)),
                jax.random.normal(rng_cov2, (n_samples,)),
            ],
            axis=0,
        )
        y_true = jnp.sin(x_data[0, :] * jnp.pi)
        y_obs = y_true + 0.1 * jax.random.normal(rng_obs, shape=y_true.shape)

        x_train = x_data
        y_train = y_obs

        def model(x, y=None):
            bart = BARTRegression(
                "bart", n_covs=x.shape[0], n_trees=20, max_depth=2, scale=1.0
            )
            mu = bart(x)
            with numpyro.plate("data", x.shape[1]):
                numpyro.sample("obs", dist.Normal(mu, 0.1), obs=y)  # type: ignore

        kernel = DiscreteHMCGibbs(NUTS(model))
        mcmc = MCMC(kernel, num_warmup=500, num_samples=500)
        mcmc.run(rng_inf, x_train, y_train)

        predictive = Predictive(model, mcmc.get_samples())
        samples = predictive(rng_key, x_train)

        preds = jnp.mean(samples["obs"], axis=0)

        mae = jnp.mean(jnp.abs(preds - y_obs))
        self.assertTrue(
            mae < 0.3, f"Mean Absolute Error should be < 0.3, but was {mae:.4f}"
        )


if __name__ == "__main__":
    unittest.main()
