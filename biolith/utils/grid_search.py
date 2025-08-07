import gc
import itertools
import unittest
import warnings
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from sklearn.model_selection import StratifiedKFold

from biolith.evaluation.lppd import lppd
from biolith.utils.fit import FitResult, fit
from biolith.utils.predict import predict


class GridSearchResult(NamedTuple):
    best_result: FitResult
    best_params: Dict[str, Any]
    best_score: float
    cv_results: List[Dict[str, Any]]


def grid_search_priors(
    model_fn: Callable,
    site_covs: jnp.ndarray,
    obs_covs: jnp.ndarray,
    obs: jnp.ndarray,
    regressor_occ: Any,
    regressor_det: Any,
    prior_types: Optional[List[str]] = None,
    prior_params_occ: Union[Dict[str, Dict[str, List[float]]], bool, None] = None,
    prior_params_det: Union[Dict[str, Dict[str, List[float]]], bool, None] = None,
    cv_folds: int = 5,
    random_seed: int = 42,
    num_samples: int = 1000,
    num_warmup: int = 1000,
    num_chains: int = 5,
    timeout: Optional[int] = None,
    **kwargs,
) -> GridSearchResult:
    r"""Perform grid search for optimal priors using k-fold stratified cross-validation.

    This function performs a grid search over prior hyperparameters and distribution types
    for occupancy models, using stratified k-fold cross-validation based on whether sites
    have at least one detection. The method evaluates model performance using log pointwise
    predictive density (LPPD) on validation sets.

    The stratification ensures balanced representation of occupied (sites with ≥1 detection)
    and unoccupied (sites with no detections) across folds, which is important for
    occupancy models where detection histories are sparse.

    Parameters
    ----------
    model_fn : Callable
        The occupancy model function to fit (e.g., occu, occu_cop, occu_cs, occu_rn).
    site_covs : jnp.ndarray
        Site-level covariates of shape (n_sites, n_site_covs).
    obs_covs : jnp.ndarray
        Observation-level covariates of shape (n_sites, n_visits, n_obs_covs).
    obs : jnp.ndarray
        Observed detection data of shape (n_sites, n_visits).
    regressor_occ : Any
        Regression class for occupancy process (e.g., LinearRegression, MLPRegression, BARTRegression).
    regressor_det : Any
        Regression class for detection process (e.g., LinearRegression, MLPRegression, BARTRegression).
    prior_types : List[str], optional
        List of prior distribution types to test, e.g., ["normal", "laplace"].
        Default is ["normal", "laplace"] (tests both).
    prior_params_occ : Dict[str, Dict[str, List[float]]], bool, or None, optional
        Nested dictionary of grid parameters for occupancy priors by type.
        Structure: {"normal": {"loc": [...], "scale": [...]}, "laplace": {"loc": [...], "scale": [...]}}.
        If None, uses reasonable defaults for all prior types.
        If False, disables tuning of occupancy priors (uses default Normal(0, 1)).
    prior_params_det : Dict[str, Dict[str, List[float]]], bool, or None, optional
        Nested dictionary of grid parameters for detection priors by type.
        Same structure as prior_params_occ. If None, uses same defaults as occupancy prior.
        If False, disables tuning of detection priors (uses default Normal(0, 1)).
    cv_folds : int, optional
        Number of cross-validation folds. Default is 5.
    random_seed : int, optional
        Random seed for reproducibility. Default is 42.
    num_samples : int, optional
        Number of posterior samples per chain. Default is 1000.
    num_warmup : int, optional
        Number of warmup samples per chain. Default is 1000.
    num_chains : int, optional
        Number of MCMC chains. Default is 5.
    timeout : int, optional
        Timeout in seconds for each model fit. Default is None (no timeout).
    **kwargs
        Additional arguments passed to the model function.

    Returns
    -------
    GridSearchResult
        Named tuple containing:
        - best_result: FitResult object with best performing model
        - best_params: Dictionary of best hyperparameters
        - best_score: Best validation LPPD score
        - cv_results: List of dictionaries with all CV results

    Examples
    --------
    >>> from biolith.models import occu, simulate
    >>> from biolith.regression import LinearRegression
    >>> from biolith.utils import grid_search_priors
    >>>
    >>> # Simulate data
    >>> data, _ = simulate()
    >>>
    >>> # Grid search with default parameters (tests both normal and laplace)
    >>> result = grid_search_priors(
    ...     occu,
    ...     data["site_covs"],
    ...     data["obs_covs"],
    ...     data["obs"],
    ...     LinearRegression,
    ...     LinearRegression
    ... )
    >>>
    >>> # Only tune detection priors, keep occupancy at default Normal(0, 1)
    >>> result = grid_search_priors(
    ...     occu,
    ...     data["site_covs"],
    ...     data["obs_covs"],
    ...     data["obs"],
    ...     LinearRegression,
    ...     LinearRegression,
    ...     prior_params_occ=False
    ... )
    >>>
    >>> # Only tune occupancy priors, keep detection at default Normal(0, 1)
    >>> result = grid_search_priors(
    ...     occu,
    ...     data["site_covs"],
    ...     data["obs_covs"],
    ...     data["obs"],
    ...     LinearRegression,
    ...     LinearRegression,
    ...     prior_params_det=False
    ... )
    >>>
    >>> # Access best results
    >>> best_model = result.best_result
    >>> best_hyperparams = result.best_params
    >>> validation_score = result.best_score

    Notes
    -----
    The function assumes covariates are centered with unit variance as specified in
    the requirements. Cross-validation stratification is based on the binary
    indicator :math:`\mathbb{1}[\sum_j y_{ij} > 0]` for each site :math:`i`, where
    :math:`y_{ij}` represents the detection at site :math:`i` during visit :math:`j`.

    For numerical stability, the function filters out infinite or NaN LPPD scores
    and issues warnings when model fits fail.
    """

    # Set default prior types if not provided
    if prior_types is None:
        prior_types = ["normal", "laplace"]

    # Set default prior parameters if not provided
    if prior_params_occ is None:
        prior_params_occ = {
            "normal": {"loc": [0.0], "scale": [0.25, 0.5, 1.0, 2.0, 4.0]},
            "laplace": {"loc": [0.0], "scale": [0.25, 0.5, 1.0, 2.0, 4.0]},
        }
    elif prior_params_occ is False:
        # Disable tuning for occupancy priors - use single default
        prior_params_occ = {
            prior_type: {"loc": [0.0], "scale": [1.0]} for prior_type in prior_types
        }

    if prior_params_det is None:
        if isinstance(prior_params_occ, dict):
            prior_params_det = prior_params_occ.copy()
        else:
            # Fallback if prior_params_occ is False
            prior_params_det = {
                prior_type: {"loc": [0.0], "scale": [1.0]} for prior_type in prior_types
            }
    elif prior_params_det is False:
        # Disable tuning for detection priors - use single default
        prior_params_det = {
            prior_type: {"loc": [0.0], "scale": [1.0]} for prior_type in prior_types
        }

    # Validate prior types
    supported_types = ["normal", "laplace"]
    for ptype in prior_types:
        if ptype not in supported_types:
            raise ValueError(
                f"Unsupported prior type: {ptype}. Must be one of {supported_types}."
            )

    # Create stratification labels based on site detection history
    site_detections = jnp.nansum(obs, axis=1) > 0  # True if site has ≥1 detection
    stratify_labels = site_detections.astype(int)

    # Check if we have both occupied and unoccupied sites
    unique_labels = jnp.unique(stratify_labels)
    if len(unique_labels) == 1:
        warnings.warn(
            f"All sites have the same occupancy status ({unique_labels[0]}). "
            "Stratification will not be effective."
        )

    # Initialize cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)

    best_score = float(-jnp.inf)
    best_params: Dict[str, Any] = {}
    best_result = None
    cv_results = []

    n_sites = site_covs.shape[0]

    # Iterate over all prior types and parameter combinations
    for prior_type in prior_types:
        # Get parameters for this prior type
        if isinstance(prior_params_occ, dict):
            occ_params_for_type = prior_params_occ.get(prior_type, {})
        else:
            occ_params_for_type = {}

        if isinstance(prior_params_det, dict):
            det_params_for_type = prior_params_det.get(prior_type, {})
        else:
            det_params_for_type = {}

        if not occ_params_for_type and not det_params_for_type:
            warnings.warn(
                f"No parameters found for prior type '{prior_type}'. Skipping."
            )
            continue

        # If one of the parameter sets is disabled (False), use default values
        if not occ_params_for_type:
            occ_params_for_type = {"loc": [0.0], "scale": [1.0]}
        if not det_params_for_type:
            det_params_for_type = {"loc": [0.0], "scale": [1.0]}

        # Generate parameter combinations for this prior type
        occ_param_names = list(occ_params_for_type.keys())
        det_param_names = list(det_params_for_type.keys())

        occ_param_combinations = list(itertools.product(*occ_params_for_type.values()))
        det_param_combinations = list(itertools.product(*det_params_for_type.values()))

        type_combinations = list(
            itertools.product(occ_param_combinations, det_param_combinations)
        )

        for occ_params, det_params in type_combinations:
            # Create prior distributions
            occ_param_dict = dict(zip(occ_param_names, occ_params))
            det_param_dict = dict(zip(det_param_names, det_params))

            if prior_type == "normal":
                prior_occ = dist.Normal(occ_param_dict["loc"], occ_param_dict["scale"])
                prior_det = dist.Normal(det_param_dict["loc"], det_param_dict["scale"])
            elif prior_type == "laplace":
                prior_occ = dist.Laplace(occ_param_dict["loc"], occ_param_dict["scale"])
                prior_det = dist.Laplace(det_param_dict["loc"], det_param_dict["scale"])

            fold_scores = []

            # Perform cross-validation
            for fold_idx, (train_idx, val_idx) in enumerate(
                cv.split(np.arange(n_sites), stratify_labels)
            ):
                try:
                    # Split data
                    site_covs_train = site_covs[train_idx]
                    obs_covs_train = obs_covs[train_idx]
                    obs_train = obs[train_idx]

                    site_covs_val = site_covs[val_idx]
                    obs_covs_val = obs_covs[val_idx]
                    obs_val = obs[val_idx]

                    # Fit model on training data
                    train_result = fit(
                        model_fn,
                        site_covs=site_covs_train,
                        obs_covs=obs_covs_train,
                        obs=obs_train,
                        regressor_occ=regressor_occ,
                        regressor_det=regressor_det,
                        prior_beta=prior_occ,
                        prior_alpha=prior_det,
                        num_samples=num_samples,
                        num_warmup=num_warmup,
                        num_chains=num_chains,
                        random_seed=random_seed + fold_idx,
                        timeout=timeout,
                        **kwargs,
                    )

                    # Generate predictions for validation data
                    val_predictions = predict(
                        model_fn,
                        train_result.mcmc,
                        site_covs=site_covs_val,
                        obs_covs=obs_covs_val,
                        obs=obs_val,
                        regressor_occ=regressor_occ,
                        regressor_det=regressor_det,
                        prior_beta=prior_occ,
                        prior_alpha=prior_det,
                        **kwargs,
                    )

                    # Evaluate on validation data
                    val_lppd = lppd(
                        model_fn,
                        val_predictions,
                        site_covs=site_covs_val,
                        obs_covs=obs_covs_val,
                        obs=obs_val,
                        regressor_occ=regressor_occ,
                        regressor_det=regressor_det,
                        prior_beta=prior_occ,
                        prior_alpha=prior_det,
                        **kwargs,
                    )

                    # Check for valid score
                    if jnp.isfinite(val_lppd):
                        fold_scores.append(val_lppd)
                    else:
                        warnings.warn(
                            f"Invalid LPPD score ({val_lppd}) in fold {fold_idx}"
                        )

                    # Clear memory to prevent OOM errors
                    del train_result, val_predictions
                    jax.clear_caches()
                    jax.live_arrays().clear()
                    gc.collect()

                except Exception as e:
                    warnings.warn(f"Model fit failed in fold {fold_idx}: {str(e)}")
                    continue

            # Calculate mean validation score
            if fold_scores:
                mean_score = jnp.mean(jnp.array(fold_scores))
                std_score = jnp.std(jnp.array(fold_scores))

                # Store results
                result_entry = {
                    "prior_type": prior_type,
                    "occ_params": occ_param_dict,
                    "det_params": det_param_dict,
                    "mean_val_lppd": mean_score,
                    "std_val_lppd": std_score,
                    "fold_scores": fold_scores,
                    "n_successful_folds": len(fold_scores),
                }
                cv_results.append(result_entry)

                # Update best if this is better
                if mean_score > best_score:
                    best_score = float(mean_score)
                    best_params = {
                        "prior_type": prior_type,
                        "occ_params": occ_param_dict,
                        "det_params": det_param_dict,
                    }

                    # Refit on full data with best parameters
                    if prior_type == "normal":
                        best_prior_occ = dist.Normal(
                            occ_param_dict["loc"], occ_param_dict["scale"]
                        )
                        best_prior_det = dist.Normal(
                            det_param_dict["loc"], det_param_dict["scale"]
                        )
                    elif prior_type == "laplace":
                        best_prior_occ = dist.Laplace(
                            occ_param_dict["loc"], occ_param_dict["scale"]
                        )
                        best_prior_det = dist.Laplace(
                            det_param_dict["loc"], det_param_dict["scale"]
                        )

                    best_result = fit(
                        model_fn,
                        site_covs=site_covs,
                        obs_covs=obs_covs,
                        obs=obs,
                        regressor_occ=regressor_occ,
                        regressor_det=regressor_det,
                        prior_beta=best_prior_occ,
                        prior_alpha=best_prior_det,
                        num_samples=num_samples,
                        num_warmup=num_warmup,
                        num_chains=num_chains,
                        random_seed=random_seed,
                        timeout=timeout,
                        **kwargs,
                    )
            else:
                warnings.warn(
                    f"No successful folds for parameters: prior_type={prior_type}, occ={occ_param_dict}, det={det_param_dict}"
                )

    if best_result is None:
        raise RuntimeError(
            "Grid search failed: no successful parameter combinations found."
        )

    return GridSearchResult(best_result, best_params, best_score, cv_results)


class TestGridSearch(unittest.TestCase):

    def test_grid_search(self):

        from biolith.models import occu, simulate
        from biolith.regression import LinearRegression

        data, _ = simulate(simulate_missing=False, n_sites=8)

        gs_results = grid_search_priors(
            occu,
            **data,
            regressor_occ=LinearRegression,
            regressor_det=LinearRegression,
            prior_types=["normal", "laplace"],
            prior_params_occ={
                "normal": {"loc": [0.0], "scale": [1.0]},
                "laplace": {"loc": [0.0], "scale": [1.0]},
            },
            cv_folds=2,
            num_chains=1,
            num_warmup=30,
            num_samples=30,
            timeout=600,
        )

        # Check that both prior types were tested
        prior_types_tested = set([res["prior_type"] for res in gs_results.cv_results])
        self.assertEqual(prior_types_tested, {"normal", "laplace"})

        # Check that best params include prior type
        self.assertIn("prior_type", gs_results.best_params)
        self.assertIn(gs_results.best_params["prior_type"], ["normal", "laplace"])

    def test_grid_search_disable_occ_priors(self):
        from biolith.models import occu, simulate
        from biolith.regression import LinearRegression

        data, _ = simulate(simulate_missing=False, n_sites=8)

        # Test disabling occupancy prior tuning
        gs_results = grid_search_priors(
            occu,
            **data,
            regressor_occ=LinearRegression,
            regressor_det=LinearRegression,
            prior_types=["normal"],
            prior_params_occ=False,  # Disable occupancy tuning
            prior_params_det={"normal": {"loc": [0.0], "scale": [1.0, 2.0]}},
            cv_folds=2,
            num_chains=1,
            num_warmup=30,
            num_samples=30,
            timeout=600,
        )

        # Check that occupancy params are default
        self.assertEqual(gs_results.best_params["occ_params"]["scale"], 1.0)
        # Check that detection params were tuned
        self.assertIn(gs_results.best_params["det_params"]["scale"], [1.0, 2.0])

    def test_grid_search_disable_det_priors(self):
        from biolith.models import occu, simulate
        from biolith.regression import LinearRegression

        data, _ = simulate(simulate_missing=False, n_sites=8)

        # Test disabling detection prior tuning
        gs_results = grid_search_priors(
            occu,
            **data,
            regressor_occ=LinearRegression,
            regressor_det=LinearRegression,
            prior_types=["normal"],
            prior_params_occ={"normal": {"loc": [0.0], "scale": [1.0, 2.0]}},
            prior_params_det=False,  # Disable detection tuning
            cv_folds=2,
            num_chains=1,
            num_warmup=30,
            num_samples=30,
            timeout=600,
        )

        # Check that detection params are default
        self.assertEqual(gs_results.best_params["det_params"]["scale"], 1.0)
        # Check that occupancy params were tuned
        self.assertIn(gs_results.best_params["occ_params"]["scale"], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
