#!/usr/bin/env python3
"""
Wall shear stress statistical analysis for airfoil surfaces.

This module computes robust statistical metrics of wall shear stress (τ_w)
from HDF5-stored surface data, organized by chordwise slices and flight cases.

Physics-based metrics include:
  - Mean and standard deviation
  - Skewness and excess kurtosis of fluctuations
  - Probability of flow reversal (negative wall shear stress)

Aligned with Physics of Fluids publication standards.
"""

import os
import h5py
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

# Dictionary of simulation cases to analyze
CASES = {
    "AOA5": {
        "dir": "NACA_0012_AOA5_Re50000_1716x1662x128",
        "label": r"$\alpha=5°$"
    },
    "AOA12": {
        "dir": "NACA_0012_AOA12_Re50000_1716x1662x128",
        "label": r"$\alpha=12°$"
    }
}

BASE_SIM_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations"

# ============================================================================
# Helper Functions
# ============================================================================

def compute_tau_statistics(tau_samples: np.ndarray) -> dict or None:
    """
    Compute wall shear stress statistics from a sample array.

    Computes mean, standard deviation, and statistical moments of wall shear
    stress fluctuations for direct comparison with theoretical predictions and
    experimental data.

    Parameters
    ----------
    tau_samples : np.ndarray
        Array of wall shear stress samples (any shape, will be flattened).

    Returns
    -------
    dict or None
        Dictionary with keys:
          - n_samples: number of valid samples
          - tau_mean: mean of τ_w
          - tau_std: standard deviation of τ_w (sample stddev, ddof=1)
          - tau_skewness: skewness of τ'_w = τ_w - mean
          - tau_excess_kurtosis: excess kurtosis of τ'_w
          - prob_tau_negative: fraction of samples where τ_w < 0
          - prob_tau_prime_negative: fraction of samples where τ'_w < 0

        Returns None if input is empty or all NaNs, with warning printed.

    Notes
    -----
    Skewness and excess kurtosis are computed from the fluctuations τ'_w
    to match the fluctuation formalism in the paper (τ'_w notation).
    Mathematically, these moments are invariant to centering and scaling,
    so computing from τ_w directly yields identical results. We use τ'_w
    for notational consistency with the manuscript.

    Unbiased estimators are used (bias=False in scipy.stats functions)
    to correct for small-sample bias, recommended for typical CFD sample sizes.
    """

    # Flatten array and identify valid (finite) data
    tau_flat = np.asarray(tau_samples).flatten()
    valid_mask = np.isfinite(tau_flat)
    tau_clean = tau_flat[valid_mask]

    n_clean = len(tau_clean)

    if n_clean == 0:
        print("[WARNING] No valid (finite) samples in τ_w array.")
        return None

    if n_clean < 4:
        print(f"[WARNING] Only {n_clean} valid samples; statistical moments may be unreliable.")

    # Compute mean and standard deviation
    tau_mean = np.mean(tau_clean)
    tau_std = np.std(tau_clean, ddof=1)  # Sample standard deviation

    # Compute fluctuations
    tau_prime = tau_clean - tau_mean

    # Compute skewness and excess kurtosis using unbiased estimators
    # bias=False: unbiased estimator; fisher=True: excess kurtosis (Fischer - 3)
    tau_skewness = stats.skew(tau_prime, bias=False)
    tau_excess_kurtosis = stats.kurtosis(tau_prime, fisher=True, bias=False)

    # Compute probability of negative (reversed) wall shear stress
    prob_tau_negative = np.mean(tau_clean < 0)
    prob_tau_prime_negative = np.mean(tau_prime < 0)

    return {
        'n_samples': n_clean,
        'tau_mean': tau_mean,
        'tau_std': tau_std,
        'tau_skewness': tau_skewness,
        'tau_excess_kurtosis': tau_excess_kurtosis,
        'prob_tau_negative': prob_tau_negative,
        'prob_tau_prime_negative': prob_tau_prime_negative,
    }


def save_results_csv(results: list, output_path: str) -> None:
    """
    Save statistical results to CSV file.

    Parameters
    ----------
    results : list of dict
        List of result dictionaries from compute_tau_statistics.
    output_path : str
        Path to output CSV file.
    """
    if len(results) == 0:
        print(f"[WARNING] No results to save to {output_path}")
        return

    fieldnames = list(results[0].keys())

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"  Saved: {output_path}")


def print_summary_table(case_name: str, case_label: str, results: list) -> None:
    """
    Print a formatted summary table to terminal.

    Parameters
    ----------
    case_name : str
        Case identifier (e.g., "AOA5").
    case_label : str
        Case label for display (e.g., r"$\alpha=5°$").
    results : list of dict
        List of result rows.
    """
    print(f"\n{case_name} ({case_label}):")
    print("-" * 130)
    # Build header using string concatenation to avoid backslash in f-string
    tau_prime = "τ'"
    header = (f"{'x/c':<12} {'τ_mean':<16} {'τ_std':<16} "
              f"{'Skew(' + tau_prime + ')':<14} {'Kurt(' + tau_prime + ')':<14} "
              f"{'P(τ<0)':<12} {'P(' + tau_prime + '<0)':<12}")
    print(header)
    print("-" * 130)

    for row in results:
        x_c = row['x_c']
        tau_mean = row['tau_mean']
        tau_std = row['tau_std']
        tau_skew = row['tau_skewness']
        tau_kurt = row['tau_excess_kurtosis']
        prob_neg = row['prob_tau_negative']
        prob_prime_neg = row['prob_tau_prime_negative']

        print(f"{x_c:<12.4f} {tau_mean:<16.6e} {tau_std:<16.6e} {tau_skew:<14.6f} "
              f"{tau_kurt:<14.6f} {prob_neg:<12.6f} {prob_prime_neg:<12.6f}")


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    """Execute wall shear stress statistical analysis."""

    print("=" * 80)
    print("WALL SHEAR STRESS STATISTICAL ANALYSIS")
    print("Publication-ready statistics for NACA0012 airfoil")
    print("=" * 80)

    # Dictionary to store results across all cases for cross-case comparison
    all_case_results = {}

    # ========================================================================
    # Loop over flight cases
    # ========================================================================

    for case_name, case_info in CASES.items():
        print(f"\n{'=' * 80}")
        print(f"CASE: {case_name} ({case_info['label']})")
        print(f"{'=' * 80}")

        # Construct file paths
        case_dir = os.path.join(BASE_SIM_DIR, case_info['dir'])
        pdf_analysis_dir = os.path.join(case_dir, "Mean_data", "PDF_analysis")
        surface_data_file = os.path.join(pdf_analysis_dir, "surface_data_slices.h5")

        if not os.path.exists(surface_data_file):
            print(f"[ERROR] File not found: {surface_data_file}")
            continue

        print(f"Loading surface data from: {surface_data_file}")

        # ====================================================================
        # Discover slices and load metadata
        # ====================================================================

        slice_names = []
        slice_metadata = {}

        with h5py.File(surface_data_file, "r") as f:
            slice_names = sorted(list(f.keys()))

            for s_name in slice_names:
                grp = f[s_name]
                x_c = grp.attrs.get("x_c", None)
                slice_metadata[s_name] = {'x_c': x_c}

        print(f"Found {len(slice_names)} chord locations\n")

        # ====================================================================
        # Compute statistics for each chordwise slice
        # ====================================================================

        case_results = []

        for slice_name in slice_names:
            x_c = slice_metadata[slice_name]['x_c']

            # Load wall shear stress samples from HDF5
            with h5py.File(surface_data_file, "r") as f:
                tau_samples = f[slice_name]["tau_w"][:]

            # Compute robust statistics
            stats_dict = compute_tau_statistics(tau_samples)

            if stats_dict is None:
                print(f"  {slice_name} (x/c={x_c:.4f}): [SKIPPED - invalid data]")
                continue

            # Build result row for this slice
            row = {
                'slice_name': slice_name,
                'x_c': x_c,
                'n_samples': stats_dict['n_samples'],
                'tau_mean': stats_dict['tau_mean'],
                'tau_std': stats_dict['tau_std'],
                'tau_skewness': stats_dict['tau_skewness'],
                'tau_excess_kurtosis': stats_dict['tau_excess_kurtosis'],
                'prob_tau_negative': stats_dict['prob_tau_negative'],
                'prob_tau_prime_negative': stats_dict['prob_tau_prime_negative'],
            }

            case_results.append(row)

            # Print slice-level summary
            print(f"  {slice_name} (x/c={x_c:.4f}):")
            print(f"    n_samples = {stats_dict['n_samples']}")
            print(f"    τ_mean = {stats_dict['tau_mean']:+.6e}")
            print(f"    τ_std = {stats_dict['tau_std']:.6e}")
            print(f"    Skew(τ'_w) = {stats_dict['tau_skewness']:+.6f}")
            print(f"    Kurt(τ'_w) = {stats_dict['tau_excess_kurtosis']:+.6f}")
            print(f"    P(τ_w < 0) = {stats_dict['prob_tau_negative']:.6f}")

        if len(case_results) == 0:
            print(f"[ERROR] No valid statistics computed for case {case_name}.")
            continue

        # ====================================================================
        # Save case-specific CSV
        # ====================================================================

        output_csv = os.path.join(pdf_analysis_dir, f"tau_statistics_{case_name}.csv")
        save_results_csv(case_results, output_csv)

        # Store for cross-case comparison later
        all_case_results[case_name] = {
            'results': case_results,
            'label': case_info['label'],
        }

    # ========================================================================
    # Cross-case comparison CSV
    # ========================================================================

    if len(all_case_results) > 1:
        print(f"\n{'=' * 80}")
        print("CROSS-CASE COMPARISON")
        print(f"{'=' * 80}")

        # Collect all unique x_c values across cases
        all_x_c = set()
        for case_data in all_case_results.values():
            for row in case_data['results']:
                all_x_c.add(row['x_c'])

        # Initialize comparison rows indexed by x_c
        comparison_rows = {x_c: {'x_c': x_c} for x_c in sorted(all_x_c)}

        # Fill in statistics from each case
        for case_name, case_data in all_case_results.items():
            for row in case_data['results']:
                x_c = row['x_c']

                # Add tau statistics with case suffix
                stat_keys = ['tau_mean', 'tau_std', 'tau_skewness', 'tau_excess_kurtosis',
                             'prob_tau_negative', 'prob_tau_prime_negative']
                for key in stat_keys:
                    comparison_rows[x_c][f"{case_name}_{key}"] = row[key]

        # Build fieldnames: x_c first, then statistics ordered by case
        fieldnames = ['x_c']
        stat_keys = ['tau_mean', 'tau_std', 'tau_skewness', 'tau_excess_kurtosis',
                     'prob_tau_negative', 'prob_tau_prime_negative']
        for case_name in sorted(all_case_results.keys()):
            for stat_key in stat_keys:
                fieldnames.append(f"{case_name}_{stat_key}")

        # Write comparison CSV
        output_comparison = os.path.join(BASE_SIM_DIR, "tau_statistics_comparison.csv")
        with open(output_comparison, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, restval='')
            writer.writeheader()
            for x_c in sorted(comparison_rows.keys()):
                writer.writerow(comparison_rows[x_c])

        print(f"Saved: {output_comparison}")

    # ========================================================================
    # Publication-quality plots
    # ========================================================================

    print(f"\n{'=' * 80}")
    print("GENERATING PUBLICATION PLOTS")
    print(f"{'=' * 80}")

    # Prepare plot data
    plot_data = {}
    for case_name, case_data in all_case_results.items():
        results_sorted = sorted(case_data['results'], key=lambda x: x['x_c'])
        plot_data[case_name] = {
            'label': case_data['label'],
            'x_c': [r['x_c'] for r in results_sorted],
            'tau_skewness': [r['tau_skewness'] for r in results_sorted],
            'tau_excess_kurtosis': [r['tau_excess_kurtosis'] for r in results_sorted],
            'prob_tau_negative': [r['prob_tau_negative'] for r in results_sorted],
        }

    # -------- Plot 1: Skewness vs. Chord Position --------
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for case_name, data in plot_data.items():
        ax.plot(data['x_c'], data['tau_skewness'], 'o-',
                label=data['label'], linewidth=2.5, markersize=7)

    ax.axhline(y=0, color='k', linestyle='--', alpha=0.4, linewidth=1.2)
    ax.set_xlabel(r'Chord position $x/c$ [-]', fontsize=13)
    ax.set_ylabel(r'Skewness $S(\tau_w^\prime)$ [-]', fontsize=13)
    ax.set_title(r'Wall Shear Stress Fluctuation Skewness', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle=':')
    ax.legend(fontsize=12, loc='best')
    ax.set_xlim([0, 1])

    plt.tight_layout()
    # plot_file_1 = os.path.join(BASE_SIM_DIR, "tau_skewness_vs_xc.png")
    # plt.savefig(plot_file_1, dpi=300, bbox_inches='tight')
    # print(f"Saved: {plot_file_1}")
    plt.close()

    # -------- Plot 2: Excess Kurtosis vs. Chord Position --------
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for case_name, data in plot_data.items():
        ax.plot(data['x_c'], data['tau_excess_kurtosis'], 's-',
                label=data['label'], linewidth=2.5, markersize=7)

    ax.axhline(y=0, color='k', linestyle='--', alpha=0.4, linewidth=1.2)
    ax.set_xlabel(r'Chord position $x/c$ [-]', fontsize=13)
    ax.set_ylabel(r'Excess kurtosis $\kappa(\tau_w^\prime)$ [-]', fontsize=13)
    ax.set_title(r'Wall Shear Stress Fluctuation Excess Kurtosis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle=':')
    ax.legend(fontsize=12, loc='best')
    ax.set_xlim([0, 1])

    plt.tight_layout()
    # plot_file_2 = os.path.join(BASE_SIM_DIR, "tau_excess_kurtosis_vs_xc.png")
    # plt.savefig(plot_file_2, dpi=300, bbox_inches='tight')
    # print(f"Saved: {plot_file_2}")
    plt.close()

    # -------- Plot 3: Probability of Reversed Shear Stress --------
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for case_name, data in plot_data.items():
        ax.plot(data['x_c'], data['prob_tau_negative'], '^-',
                label=data['label'], linewidth=2.5, markersize=7)

    ax.set_xlabel(r'Chord position $x/c$ [-]', fontsize=13)
    ax.set_ylabel(r'Probability $P(\tau_w < 0)$ [-]', fontsize=13)
    ax.set_title(r'Probability of Flow Reversal on Surface', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.25, linestyle=':')
    ax.legend(fontsize=12, loc='best')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, None])

    plt.tight_layout()
    # plot_file_3 = os.path.join(BASE_SIM_DIR, "tau_prob_negative_vs_xc.png")
    # plt.savefig(plot_file_3, dpi=300, bbox_inches='tight')
    # print(f"Saved: {plot_file_3}")
    plt.close()

    # ========================================================================
    # Terminal summary tables
    # ========================================================================

    print(f"\n{'=' * 80}")
    print("SUMMARY TABLES")
    print(f"{'=' * 80}")

    for case_name, case_data in all_case_results.items():
        print_summary_table(case_name, case_data['label'], case_data['results'])

    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    main()
