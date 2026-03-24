#!/usr/bin/env python3
#################################################################################
####                    SKEWNESS ANALYSIS OF SURFACE PDFs                    ####
#################################################################################

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from scipy import stats
import csv

rc('text', usetex=True)
rc('font', family='serif')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Base directory for simulations
BASE_SIM_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations"

# Cases to analyze
CASES = {
    "AOA5": {
        "dir": "NACA_0012_AOA5_Re50000_1716x1662x128",
        "label": "AOA = 5°"
    },
    "AOA12": {
        "dir": "NACA_0012_AOA12_Re50000_1716x1662x128",
        "label": "AOA = 12°"
    }
}

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("=" * 80)
print("SKEWNESS ANALYSIS OF SURFACE PRESSURE AND SHEAR STRESS FLUCTUATIONS")
print("=" * 80)

# Dictionary to store results across all cases
all_results = {}

for case_name, case_info in CASES.items():
    print(f"\n{'=' * 80}")
    print(f"PROCESSING CASE: {case_name} ({case_info['label']})")
    print(f"{'=' * 80}")

    # Construct paths
    case_dir = os.path.join(BASE_SIM_DIR, case_info['dir'])
    pdf_analysis_dir = os.path.join(case_dir, "Mean_data", "PDF_analysis")
    surface_data_file = os.path.join(pdf_analysis_dir, "surface_data_slices.h5")

    if not os.path.exists(surface_data_file):
        print(f"[WARNING] File not found: {surface_data_file}")
        continue

    # ========================================================================
    # Load slices and their metadata
    # ========================================================================
    print(f"\nLoading surface data from: {surface_data_file}")

    all_slices = []
    slice_metadata = {}
    surface_data = {}

    with h5py.File(surface_data_file, "r") as f:
        available_slices = sorted(list(f.keys()))
        print(f"Found {len(available_slices)} slice locations")

        for slice_name in available_slices:
            grp = f[slice_name]

            # Extract data
            p_samples = grp["p_w"][:]
            tau_samples = grp["tau_w"][:]

            # Extract metadata
            x_c = grp.attrs.get("x_c", None)

            all_slices.append(slice_name)
            slice_metadata[slice_name] = {'x_c': x_c}
            surface_data[slice_name] = {
                'p_w': p_samples,
                'tau_w': tau_samples
            }

    # Sort slices by x_c
    all_slices.sort(key=lambda s: slice_metadata[s]['x_c'])

    # ========================================================================
    # Compute statistics including skewness
    # ========================================================================
    print(f"\n{'=' * 80}")
    print("COMPUTING STATISTICS")
    print(f"{'=' * 80}")

    case_results = []

    for slice_name in all_slices:
        p_samples = surface_data[slice_name]['p_w']
        tau_samples = surface_data[slice_name]['tau_w']
        x_c = slice_metadata[slice_name]['x_c']

        # Compute basic statistics
        p_mean = np.mean(p_samples)
        p_std = np.std(p_samples)
        p_min = np.min(p_samples)
        p_max = np.max(p_samples)

        tau_mean = np.mean(tau_samples)
        tau_std = np.std(tau_samples)
        tau_min = np.min(tau_samples)
        tau_max = np.max(tau_samples)

        # Compute skewness (Fisher-Pearson coefficient)
        p_skewness = stats.skew(p_samples)      # skew(X) = E[(X - μ)³] / σ³
        tau_skewness = stats.skew(tau_samples)

        # Compute kurtosis (excess kurtosis)
        p_kurtosis = stats.kurtosis(p_samples)  # kurtosis(X) = E[(X - μ)⁴] / σ⁴ - 3
        tau_kurtosis = stats.kurtosis(tau_samples)

        # Compute normalized fluctuation statistics
        p_fluct = p_samples - p_mean
        tau_fluct = tau_samples - tau_mean

        p_fluct_norm = p_fluct / p_std if p_std > 0 else p_fluct
        tau_fluct_norm = tau_fluct / tau_std if tau_std > 0 else tau_fluct

        p_fluct_skewness = stats.skew(p_fluct_norm)
        tau_fluct_skewness = stats.skew(tau_fluct_norm)

        p_fluct_kurtosis = stats.kurtosis(p_fluct_norm)
        tau_fluct_kurtosis = stats.kurtosis(tau_fluct_norm)

        # Store results
        case_results.append({
            'slice_name': slice_name,
            'x_c': x_c,
            'n_samples': len(p_samples),
            # Pressure
            'p_mean': p_mean,
            'p_std': p_std,
            'p_min': p_min,
            'p_max': p_max,
            'p_skewness': p_skewness,
            'p_kurtosis': p_kurtosis,
            'p_fluct_skewness': p_fluct_skewness,
            'p_fluct_kurtosis': p_fluct_kurtosis,
            # Shear stress
            'tau_mean': tau_mean,
            'tau_std': tau_std,
            'tau_min': tau_min,
            'tau_max': tau_max,
            'tau_skewness': tau_skewness,
            'tau_kurtosis': tau_kurtosis,
            'tau_fluct_skewness': tau_fluct_skewness,
            'tau_fluct_kurtosis': tau_fluct_kurtosis,
        })

        print(f"\n  {slice_name} (x/c = {x_c:.4f}):")
        print(f"    Pressure:")
        print(f"      Mean = {p_mean:.6f}, Std = {p_std:.6f}")
        print(f"      Skewness = {p_skewness:.6f}, Kurtosis = {p_kurtosis:.6f}")
        print(f"      Normalized Skewness = {p_fluct_skewness:.6f}, Normalized Kurtosis = {p_fluct_kurtosis:.6f}")
        print(f"    Shear stress:")
        print(f"      Mean = {tau_mean:.6f}, Std = {tau_std:.6f}")
        print(f"      Skewness = {tau_skewness:.6f}, Kurtosis = {tau_kurtosis:.6f}")
        print(f"      Normalized Skewness = {tau_fluct_skewness:.6f}, Normalized Kurtosis = {tau_fluct_kurtosis:.6f}")

    # Store results
    all_results[case_name] = {
        'results': case_results,
        'label': case_info['label']
    }

    # ========================================================================
    # Save results to CSV
    # ========================================================================
    output_csv = os.path.join(pdf_analysis_dir, f"skewness_statistics_{case_name}.csv")
    if len(case_results) > 0:
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = list(case_results[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in case_results:
                writer.writerow(row)
    print(f"\nResults saved to: {output_csv}")


# ============================================================================
# COMPARISON ACROSS CASES
# ============================================================================
print(f"\n{'=' * 80}")
print("COMPARISON ACROSS CASES")
print(f"{'=' * 80}")

if len(all_results) > 1:
    # Create comparison table
    comparison_dict = {}  # Dictionary keyed by x_c values
    all_x_c = set()

    # Collect all unique x_c values
    for case_name in all_results.keys():
        results_list = all_results[case_name]['results']
        for row in results_list:
            all_x_c.add(row['x_c'])

    # Initialize comparison_dict for each x_c
    for x_c in sorted(all_x_c):
        comparison_dict[x_c] = {'x_c': x_c}

    # Fill in data from each case
    for case_name in all_results.keys():
        results_list = all_results[case_name]['results']
        for row in results_list:
            x_c = row['x_c']
            comparison_dict[x_c][f"{case_name}_p_skew"] = row['p_skewness']
            comparison_dict[x_c][f"{case_name}_tau_skew"] = row['tau_skewness']
            comparison_dict[x_c][f"{case_name}_p_fluct_skew"] = row['p_fluct_skewness']
            comparison_dict[x_c][f"{case_name}_tau_fluct_skew"] = row['tau_fluct_skewness']

    # Save comparison
    output_comparison = os.path.join(BASE_SIM_DIR, "skewness_comparison_all_cases.csv")
    if len(comparison_dict) > 0:
        # Get all fieldnames
        all_fieldnames = ['x_c']
        for case_name in sorted(all_results.keys()):
            all_fieldnames.extend([
                f"{case_name}_p_skew",
                f"{case_name}_tau_skew",
                f"{case_name}_p_fluct_skew",
                f"{case_name}_tau_fluct_skew"
            ])

        with open(output_comparison, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=all_fieldnames)
            writer.writeheader()
            for x_c in sorted(comparison_dict.keys()):
                writer.writerow(comparison_dict[x_c])
    print(f"Comparison table saved to: {output_comparison}")
    print("\nSkewness Comparison (saved to file):")


# ============================================================================
# PLOTTING
# ============================================================================
print(f"\n{'=' * 80}")
print("GENERATING PLOTS")
print(f"{'=' * 80}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for case_name, case_data in all_results.items():
    results_list = sorted(case_data['results'], key=lambda x: x['x_c'])
    label = case_data['label']

    # Extract x positions and values
    x_positions = [row['x_c'] for row in results_list]
    p_skew_vals = [row['p_skewness'] for row in results_list]
    tau_skew_vals = [row['tau_skewness'] for row in results_list]
    p_fluct_skew_vals = [row['p_fluct_skewness'] for row in results_list]
    tau_fluct_skew_vals = [row['tau_fluct_skewness'] for row in results_list]

    # Plot 1: Pressure skewness vs chord location
    axes[0, 0].plot(x_positions, p_skew_vals, 'o-', label=label, linewidth=2, markersize=6)

    # Plot 2: Shear stress skewness vs chord location
    axes[0, 1].plot(x_positions, tau_skew_vals, 's-', label=label, linewidth=2, markersize=6)

    # Plot 3: Normalized pressure fluctuation skewness vs chord location
    axes[1, 0].plot(x_positions, p_fluct_skew_vals, '^-', label=label, linewidth=2, markersize=6)

    # Plot 4: Normalized shear stress fluctuation skewness vs chord location
    axes[1, 1].plot(x_positions, tau_fluct_skew_vals, 'v-', label=label, linewidth=2, markersize=6)

# Configure plots
axes[0, 0].set_xlabel(r"Chord location (x/c)", fontsize=12)
axes[0, 0].set_ylabel(r"Pressure Skewness", fontsize=12)
axes[0, 0].set_title(r"Pressure Skewness vs Chord Location", fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(fontsize=10)

axes[0, 1].set_xlabel(r"Chord location (x/c)", fontsize=12)
axes[0, 1].set_ylabel(r"Shear Stress Skewness", fontsize=12)
axes[0, 1].set_title(r"Shear Stress Skewness vs Chord Location", fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend(fontsize=10)

axes[1, 0].set_xlabel(r"Chord location (x/c)", fontsize=12)
axes[1, 0].set_ylabel(r"Normalized Pressure Skewness", fontsize=12)
axes[1, 0].set_title(r"Normalized Pressure Fluctuation Skewness", fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend(fontsize=10)

axes[1, 1].set_xlabel(r"Chord location (x/c)", fontsize=12)
axes[1, 1].set_ylabel(r"Normalized Shear Stress Skewness", fontsize=12)
axes[1, 1].set_title(r"Normalized Shear Stress Fluctuation Skewness", fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend(fontsize=10)

plt.tight_layout()
# output_plot = os.path.join(BASE_SIM_DIR, "skewness_analysis_comparison.png")
# plt.savefig(output_plot, dpi=300, bbox_inches='tight')
# print(f"Plot saved to: {output_plot}")
plt.show()


# ============================================================================
# KURTOSIS PLOTS
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for case_name, case_data in all_results.items():
    results_list = sorted(case_data['results'], key=lambda x: x['x_c'])
    label = case_data['label']

    # Extract x positions and values
    x_positions = [row['x_c'] for row in results_list]
    p_kurt_vals = [row['p_kurtosis'] for row in results_list]
    tau_kurt_vals = [row['tau_kurtosis'] for row in results_list]
    p_fluct_kurt_vals = [row['p_fluct_kurtosis'] for row in results_list]
    tau_fluct_kurt_vals = [row['tau_fluct_kurtosis'] for row in results_list]

    # Plot 1: Pressure kurtosis vs chord location
    axes[0, 0].plot(x_positions, p_kurt_vals, 'o-', label=label, linewidth=2, markersize=6)

    # Plot 2: Shear stress kurtosis vs chord location
    axes[0, 1].plot(x_positions, tau_kurt_vals, 's-', label=label, linewidth=2, markersize=6)

    # Plot 3: Normalized pressure fluctuation kurtosis vs chord location
    axes[1, 0].plot(x_positions, p_fluct_kurt_vals, '^-', label=label, linewidth=2, markersize=6)

    # Plot 4: Normalized shear stress fluctuation kurtosis vs chord location
    axes[1, 1].plot(x_positions, tau_fluct_kurt_vals, 'v-', label=label, linewidth=2, markersize=6)

# Configure plots
axes[0, 0].set_xlabel(r"Chord location (x/c)", fontsize=12)
axes[0, 0].set_ylabel(r"Pressure Kurtosis", fontsize=12)
axes[0, 0].set_title(r"Pressure Kurtosis vs Chord Location", fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(fontsize=10)

axes[0, 1].set_xlabel(r"Chord location (x/c)", fontsize=12)
axes[0, 1].set_ylabel(r"Shear Stress Kurtosis", fontsize=12)
axes[0, 1].set_title(r"Shear Stress Kurtosis vs Chord Location", fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend(fontsize=10)

axes[1, 0].set_xlabel(r"Chord location (x/c)", fontsize=12)
axes[1, 0].set_ylabel(r"Normalized Pressure Kurtosis", fontsize=12)
axes[1, 0].set_title(r"Normalized Pressure Fluctuation Kurtosis", fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend(fontsize=10)

axes[1, 1].set_xlabel(r"Chord location (x/c)", fontsize=12)
axes[1, 1].set_ylabel(r"Normalized Shear Stress Kurtosis", fontsize=12)
axes[1, 1].set_title(r"Normalized Shear Stress Fluctuation Kurtosis", fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend(fontsize=10)

# Add horizontal line at 0 for reference (excess kurtosis = 0 for normal distribution)
for ax in axes.flat:
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)

plt.tight_layout()
# output_plot_kurtosis = os.path.join(BASE_SIM_DIR, "kurtosis_analysis_comparison.png")
# plt.savefig(output_plot_kurtosis, dpi=300, bbox_inches='tight')
# print(f"Kurtosis plot saved to: {output_plot_kurtosis}")
plt.show()

print(f"\n{'=' * 80}")
print("ANALYSIS COMPLETE")
print(f"{'=' * 80}")
