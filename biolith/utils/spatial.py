from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.hsgp.approximation import hsgp_squared_exponential
from numpyro.handlers import scope


def prepare_nngp(coords: np.ndarray, n_neighbors: int = 15, c: float = 1.5):
    """Compute nearest neighbor indices and squared distance matrix."""

    n_sites = coords.shape[0]
    dists_sq = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1)
    neighbor_idx = np.zeros((n_sites, n_neighbors), dtype=int)
    for i in range(1, n_sites):
        idx = np.argsort(dists_sq[i, :i])[: min(n_neighbors, i)]
        if len(idx) > 0:
            neighbor_idx[i, : len(idx)] = idx

    ell = c * np.max(np.abs(coords)).item()

    return neighbor_idx, dists_sq, coords, ell


def sample_spatial_effects(
    coords: jnp.ndarray,
    ell: float = 1.0,
    prior_gp_sd: dist.Distribution = dist.HalfNormal(1.0),
    prior_gp_length: dist.Distribution = dist.HalfNormal(1.0),
) -> jnp.ndarray:
    """Sample NNGP spatial effects used in occupancy models."""

    gp_sd = numpyro.sample("gp_sd", prior_gp_sd)
    gp_l = numpyro.sample("gp_l", prior_gp_length)

    # Sample GP using HSGP approximation
    with scope(prefix="gp", divider="_", hide_types=["plate"]):
        w = hsgp_squared_exponential(
            x=coords,
            alpha=gp_sd,
            length=gp_l,
            m=20,  # basis functions per dimension
            ell=ell,
            non_centered=True,
        )

    return w


def simulate_spatial_effects(
    coords: np.ndarray,
    n_neighbors: int = 15,
    gp_sd: float = 1.0,
    gp_l: float = 0.2,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate spatial effects using the same NNGP mechanism."""
    if rng is None:
        rng = np.random.default_rng()
    neighbor_idx, dists_sq, coords, ell = prepare_nngp(coords, n_neighbors)
    cov = lambda d: gp_sd**2 * np.exp(-d / (gp_l**2))
    w = np.zeros(coords.shape[0])
    for i in range(coords.shape[0]):
        if i < n_neighbors:
            w[i] = rng.normal(scale=gp_sd)
        else:
            idx = neighbor_idx[i]
            C_SS = cov(dists_sq[idx[:, None], idx]) + 1e-6 * np.eye(n_neighbors)
            C_iS = cov(dists_sq[i, idx])
            A = np.linalg.solve(C_SS, C_iS)
            mu = A.dot(w[idx])
            var = gp_sd**2 - A.dot(C_iS)
            w[i] = rng.normal(mu, np.sqrt(var))
    return w, ell
