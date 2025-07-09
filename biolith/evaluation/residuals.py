from typing import Dict, Tuple

import jax.numpy as jnp


def residuals(
    posterior_samples: Dict[str, jnp.ndarray],
    obs: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculates occupancy and detection residuals from posterior samples.

    This function implements the residual definitions from Wright et al. (2019)
    to help diagnose occupancy model fit. It separates residuals for the
    occupancy process from the detection process.

    References
    ----------
        - Wright, W. J., K. M. Irvine, and M. D. Higgs. 2019. Identifying occupancy model inadequacies: can residuals separately assess detection and presence? Ecology 100(6):e02703. 10.1002/ecy.2703

    Parameters
    ----------
        posterior_samples: A dictionary containing posterior samples from a fitted model.
                          Must include 'z' (latent occupancy state), 'psi'
                          (occupancy probability), and 'prob_detection' (detection probability).
        obs: Ground truth observations of shape (n_sites, n_visits).

    Returns
    -------
        A tuple containing:
        - jnp.ndarray: Occupancy residuals of shape (n_samples, n_sites).
        - jnp.ndarray: Detection residuals of shape (n_samples, n_sites, n_visits).
                      For a given posterior draw, residuals at sites considered
                      unoccupied (z=0) are returned as np.nan.

    Examples
    --------
    >>> from biolith.models import simulate, occu
    >>> from biolith.utils import fit, predict
    >>> from biolith.evaluation import residuals
    >>> data, _ = simulate()
    >>> results = fit(occu, **data)
    >>> preds = predict(occu, results.mcmc, **data)
    >>> occ_res, det_res = residuals(preds, data["obs"])
    """

    # Get posterior samples
    z_posterior = posterior_samples["z"]
    psi_posterior = posterior_samples["psi"]
    p_posterior = posterior_samples["prob_detection"]

    # Calculate Occupancy Residuals
    # Equation (4) from the paper: o_i^[t] = z_i^[t] - psi_i^[t]
    occupancy_residuals = z_posterior - psi_posterior

    # Calculate Detection Residuals
    # Equation (5) from the paper: d_ij^[t] = y_ij - p_ij^[t], conditional on z_i^[t] = 1
    raw_detection_residuals = obs.T[None, ...] - p_posterior

    # Create a mask based on the latent state z.
    # Residuals are only defined for sites considered occupied (z=1).
    z_mask = z_posterior[:, None, :]

    # Apply the mask. Where z=0, the residual becomes NaN.
    detection_residuals_transposed = jnp.where(
        z_mask == 1, raw_detection_residuals, jnp.nan
    )
    detection_residuals = detection_residuals_transposed.transpose((0, 2, 1))

    return occupancy_residuals, detection_residuals
