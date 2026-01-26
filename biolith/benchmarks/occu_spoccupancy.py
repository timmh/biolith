"""Benchmark script comparing biolith and spOccupancy performance on occupancy models.

This script simulates datasets of increasing size and benchmarks the fitting time for
both biolith and spOccupancy, using the same number of iterations for fair comparison.
"""

import os
import time
import warnings
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def simulate_datasets(
    base_n_sites: int = 100,
    base_time_periods: int = 20,
    n_datasets: int = 20,
    scaling_factor: float = 2,
    random_seed: int = 42,
) -> List[Tuple[Dict, Dict]]:
    """Simulate datasets of increasing size for benchmarking.

    Parameters
    ----------
    base_n_sites : int
        Base number of sites for the smallest dataset.
    base_time_periods : int
        Base number of time periods for the smallest dataset.
    n_datasets : int
        Number of datasets to generate.
    scaling_factor : float
        Factor by which to scale dataset size for each successive dataset.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    List[Tuple[Dict, Dict]]
        List of (data, true_params) tuples for each dataset size.
    """
    from biolith.models.occu import simulate

    datasets = []

    for i in range(n_datasets):
        n_sites = int(base_n_sites * (scaling_factor**i))
        time_periods = int(
            base_time_periods * (scaling_factor ** (i / 2))
        )  # Scale time periods more slowly
        deployment_days = time_periods * 7  # 7 days per session

        print(
            f"Generating dataset {i+1}/{n_datasets}: {n_sites} sites, {time_periods} time periods"
        )

        data, true_params = simulate(
            n_site_covs=2,
            n_obs_covs=1,
            n_sites=n_sites,
            deployment_days_per_site=deployment_days,
            session_duration=7,
            simulate_missing=False,  # Disable missing data for cleaner benchmarking
            random_seed=random_seed + i,
        )

        datasets.append((data, true_params))

    return datasets


def benchmark_biolith(
    data: Dict,
    num_samples: int = 500,
    num_warmup: int = 100,
    num_chains: int = 1,
    timeout: int = 3600,
) -> Tuple[float, bool]:
    """Benchmark biolith model fitting time.

    Parameters
    ----------
    data : Dict
        Data dictionary containing site_covs, obs_covs, obs, coords, ell.
    num_samples : int
        Number of MCMC samples.
    num_warmup : int
        Number of warmup samples.
    num_chains : int
        Number of MCMC chains.
    timeout : int
        Timeout in seconds.

    Returns
    -------
    Tuple[float, bool]
        Fitting time in seconds and success flag.
    """
    from biolith.models.occu import occu
    from biolith.utils import fit

    try:
        start_time = time.time()
        results = fit(
            occu,
            **data,
            num_samples=num_samples,
            num_warmup=num_warmup,
            num_chains=num_chains,
            timeout=timeout,
        )
        end_time = time.time()

        return end_time - start_time, True
    except Exception as e:
        print(f"Biolith fitting failed: {e}")
        return np.nan, False


def benchmark_spoccupancy(
    data: Dict,
    num_samples: int = 500,
) -> Tuple[float, bool]:
    """Benchmark spOccupancy model fitting time.

    Parameters
    ----------
    data : Dict
        Data dictionary containing site_covs, obs_covs, obs, coords, ell.
    num_samples : int
        Number of MCMC samples.

    Returns
    -------
    Tuple[float, bool]
        Fitting time in seconds and success flag.
    """
    try:
        import rpy2.robjects as ro
        import rpy2.robjects.numpy2ri as numpy2ri_module
        from rpy2.robjects.packages import importr

        # Import R packages
        base_r = importr("base")
        stats_r = importr("stats")
        spOccupancy_r = importr("spOccupancy")

        # Prepare data for R
        y_py = data["obs"][:, 0, :].copy()
        y_r = numpy2ri_module.py2rpy(y_py)

        # Occupancy covariates (site-level)
        occ_covs_py = np.nan_to_num(data["site_covs"].copy())
        n_sites, n_site_covs = occ_covs_py.shape

        occ_covs_r_elements = {}
        occ_formula_parts = []
        if n_site_covs > 0:
            for i in range(n_site_covs):
                cov_name = f"site_cov{i+1}"
                occ_covs_r_elements[cov_name] = numpy2ri_module.py2rpy(
                    occ_covs_py[:, i]
                )
                occ_formula_parts.append(cov_name)
        occ_covs_r_df = ro.DataFrame(occ_covs_r_elements)
        occ_formula_str = (
            "~ " + " + ".join(occ_formula_parts) if occ_formula_parts else "~ 1"
        )

        # Detection covariates (observation-level)
        det_covs_py = np.nan_to_num(data["obs_covs"][:, 0, :, :].copy())
        _, time_periods, n_obs_covs = det_covs_py.shape

        det_covs_r_elements = {}
        det_formula_parts = []
        if n_obs_covs > 0:
            for i in range(n_obs_covs):
                cov_name = f"obs_cov{i+1}"
                det_covs_r_elements[cov_name] = numpy2ri_module.py2rpy(
                    det_covs_py[:, :, i]
                )
                det_formula_parts.append(cov_name)
        det_covs_r_list = ro.ListVector(det_covs_r_elements)
        det_formula_str = (
            "~ " + " + ".join(det_formula_parts) if det_formula_parts else "~ 1"
        )

        # Consolidate data into an R list
        sp_data_r = ro.ListVector(
            {"y": y_r, "occ.covs": occ_covs_r_df, "det.covs": det_covs_r_list}
        )

        # Priors
        n_beta_params = n_site_covs + 1
        n_alpha_params = n_obs_covs + 1

        priors_list_r = ro.ListVector(
            {
                "beta.normal": ro.ListVector(
                    {
                        "mean": base_r.rep(0, n_beta_params),
                        "var": base_r.rep(1, n_beta_params),
                    }
                ),
                "alpha.normal": ro.ListVector(
                    {
                        "mean": base_r.rep(0, n_alpha_params),
                        "var": base_r.rep(1, n_alpha_params),
                    }
                ),
            }
        )

        # Fit model with timing
        start_time = time.time()
        pg_occ_results_r = spOccupancy_r.PGOcc(
            occ_formula=stats_r.as_formula(occ_formula_str),
            det_formula=stats_r.as_formula(det_formula_str),
            data=sp_data_r,
            priors=priors_list_r,
            n_samples=num_samples,
            # n.omp.threads=1  # has no influence on non-spatial models
        )
        end_time = time.time()

        return end_time - start_time, True

    except Exception as e:
        print(f"spOccupancy fitting failed: {e}")
        return np.nan, False


def run_benchmark(
    n_datasets: int = 10,
    num_samples: int = 500,
    num_warmup: int = 100,
    base_n_sites: int = 50,
    base_time_periods: int = 10,
    scaling_factor: float = 1.5,
    random_seed: int = 42,
    timeout: int = 3600,
) -> pd.DataFrame:
    """Run full benchmark comparing biolith and spOccupancy.

    Parameters
    ----------
    n_datasets : int
        Number of datasets of increasing size to test.
    num_samples : int
        Number of MCMC samples for each model.
    num_warmup : int
        Number of warmup samples for biolith.
    base_n_sites : int
        Base number of sites for smallest dataset.
    base_time_periods : int
        Base number of time periods for smallest dataset.
    scaling_factor : float
        Factor by which to scale dataset size.
    random_seed : int
        Random seed for reproducibility.
    timeout : int
        Timeout in seconds for each model fit.

    Returns
    -------
    pd.DataFrame
        Benchmark results with columns: dataset_size, n_sites, time_periods,
        biolith_time, biolith_success, spoccupancy_time, spoccupancy_success.
    """
    print(f"Generating {n_datasets} datasets of increasing size...")
    datasets = simulate_datasets(
        base_n_sites=base_n_sites,
        base_time_periods=base_time_periods,
        n_datasets=n_datasets,
        scaling_factor=scaling_factor,
        random_seed=random_seed,
    )

    results = []

    for i, (data, true_params) in enumerate(datasets):
        n_sites = data["site_covs"].shape[0]
        time_periods = data["obs_covs"].shape[1]
        dataset_size = n_sites * time_periods

        print(f"\nBenchmarking dataset {i+1}/{n_datasets} (size: {dataset_size})")
        print(f"  Sites: {n_sites}, Time periods: {time_periods}")

        # Benchmark biolith
        print("  Running biolith...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            biolith_time, biolith_success = benchmark_biolith(
                data,
                num_samples=num_samples,
                num_warmup=num_warmup,
                timeout=timeout,
            )

        print(f"    Time: {biolith_time:.2f}s, Success: {biolith_success}")

        # Benchmark spOccupancy
        print("  Running spOccupancy...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spoccupancy_time, spoccupancy_success = benchmark_spoccupancy(
                data,
                num_samples=num_samples,
            )

        print(f"    Time: {spoccupancy_time:.2f}s, Success: {spoccupancy_success}")

        results.append(
            {
                "dataset_idx": i,
                "dataset_size": dataset_size,
                "n_sites": n_sites,
                "time_periods": time_periods,
                "biolith_time": biolith_time,
                "biolith_success": biolith_success,
                "spoccupancy_time": spoccupancy_time,
                "spoccupancy_success": spoccupancy_success,
            }
        )

    return pd.DataFrame(results)


def plot_benchmark_results(
    results_df: pd.DataFrame, plot_relative: bool = False
) -> None:
    """Plot benchmark results comparing biolith and spOccupancy performance.

    Parameters
    ----------
    results_df : pd.DataFrame
        Benchmark results from run_benchmark().
    plot_relative : bool
        If True, plot the speedup ratio (spOccupancy time / biolith time).
    """
    if plot_relative:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    else:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = None

    # Filter successful runs
    biolith_success = results_df[results_df["biolith_success"]]
    spoccupancy_success = results_df[results_df["spoccupancy_success"]]

    # Plot 1: Fitting time vs dataset size
    if not biolith_success.empty:
        ax1.loglog(
            biolith_success["dataset_size"],
            biolith_success["biolith_time"],
            "o-",
            label="biolith",
            linewidth=2,
            markersize=6,
        )

    if not spoccupancy_success.empty:
        ax1.loglog(
            spoccupancy_success["dataset_size"],
            spoccupancy_success["spoccupancy_time"],
            "s-",
            label="spOccupancy",
            linewidth=2,
            markersize=6,
        )

    ax1.set_xlabel("Dataset Size (sites × time periods)")
    ax1.set_ylabel("Fitting Time (seconds)")
    ax1.set_title("Model Fitting Time vs Dataset Size")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if plot_relative:
        # Plot 2: Speedup ratio (spOccupancy time / biolith time)
        common_indices = set(biolith_success["dataset_idx"]) & set(
            spoccupancy_success["dataset_idx"]
        )
        if common_indices:
            common_results = []
            for idx in sorted(common_indices):
                biolith_row = biolith_success[
                    biolith_success["dataset_idx"] == idx
                ].iloc[0]
                spoccupancy_row = spoccupancy_success[
                    spoccupancy_success["dataset_idx"] == idx
                ].iloc[0]

                speedup = (
                    spoccupancy_row["spoccupancy_time"] / biolith_row["biolith_time"]
                )
                common_results.append(
                    {"dataset_size": biolith_row["dataset_size"], "speedup": speedup}
                )

            if common_results:
                common_df = pd.DataFrame(common_results)
                ax2.semilogx(
                    common_df["dataset_size"],
                    common_df["speedup"],
                    "o-",
                    linewidth=2,
                    markersize=6,
                    color="green",
                )
                ax2.axhline(
                    y=1,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label="Equal performance",
                )
                ax2.set_xlabel("Dataset Size (sites × time periods)")
                ax2.set_ylabel("Speedup Ratio (spOccupancy / biolith)")
                ax2.set_title("Performance Ratio\n(>1 means biolith is faster)")
                ax2.legend()
                ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/benchmark_occu_spoccupancy.png", dpi=300, bbox_inches="tight")
    plt.savefig("figures/benchmark_occu_spoccupancy.pdf", dpi=300, bbox_inches="tight")


def main():
    """Main function to run the benchmark and generate plots."""

    print("Biolith vs spOccupancy Benchmark")
    print("=" * 40)

    # Run benchmark
    results = run_benchmark(
        n_datasets=8,
        num_samples=500,
        num_warmup=100,
        base_n_sites=100,
        base_time_periods=8,
        scaling_factor=2,
        timeout=3600,
    )

    # Print summary
    print("\nBenchmark Summary:")
    print("-" * 40)
    print(f"Total datasets tested: {len(results)}")
    print(f"Biolith successful runs: {results['biolith_success'].sum()}")
    print(f"spOccupancy successful runs: {results['spoccupancy_success'].sum()}")

    if results["biolith_success"].any() and results["spoccupancy_success"].any():
        biolith_mean_time = results[results["biolith_success"]]["biolith_time"].mean()
        spoccupancy_mean_time = results[results["spoccupancy_success"]][
            "spoccupancy_time"
        ].mean()
        print(f"Mean biolith time: {biolith_mean_time:.2f}s")
        print(f"Mean spOccupancy time: {spoccupancy_mean_time:.2f}s")
        print(f"Overall speedup ratio: {spoccupancy_mean_time / biolith_mean_time:.2f}")

    # Generate plots
    plot_benchmark_results(results)

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
