import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Set LaTeX style for plots
plt.rc('text', usetex=True)
plt.rc('font', size=16, family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')

# ============================================================================
# Configuration
# ============================================================================

# Path to the saved spectra data (contains multiple chord locations)
AOA5_DATA_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/Premultiplied_spectra"
AOA12_DATA_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Premultiplied_spectra"

# Output figure path
OUTPUT_PATH = "/home/jofre/Members/Eduard/Paper2/Figures"

# ============================================================================
# Load data
# ============================================================================

def load_spectra_data(data_dir, aoa_label):
    """Load spectra data from a directory and add AOA label."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"DATA_DIR not found: {data_dir}")

    data_files = sorted(
        f for f in os.listdir(data_dir)
        if f.endswith(".h5") and "premultiplied_spectra_data_all" in f
    )

    if len(data_files) == 0:
        raise FileNotFoundError(f"No premultiplied spectra files found in: {data_dir}")

    print(f"\nFound {len(data_files)} premultiplied spectra files for {aoa_label}:")
    for fname in data_files:
        print(f"  - {fname}")

    datasets = []
    
    for fname in data_files:
        fpath = os.path.join(data_dir, fname)
        with h5py.File(fpath, "r") as f:
            kz_plus = f["kz_plus"][...]
            y_plus = f["y_plus"][...]
            premult_psd = f["premultiplied_psd"][...]

            slice_x = float(f.attrs.get("slice_x", np.nan))
            u_tau = float(f.attrs.get("u_tau", np.nan))
            snapshot_count = int(f.attrs.get("snapshot_count", -1))
            L_z = float(f.attrs.get("spanwise_domain", np.nan))

        lambda_z_plus = (2.0 * np.pi) / kz_plus
        log10_lambda_z_plus = np.log10(lambda_z_plus)
        
        # Shift y_plus to start at 0
        y_plus_shifted = y_plus - np.nanmin(y_plus)

        datasets.append({
            "file": fname,
            "aoa": aoa_label,
            "kz_plus": kz_plus,
            "y_plus": y_plus_shifted,
            "premult_psd": premult_psd,
            "lambda_z_plus": lambda_z_plus,
            "log10_lambda_z_plus": log10_lambda_z_plus,
            "slice_x": slice_x,
            "u_tau": u_tau,
            "snapshot_count": snapshot_count,
            "L_z": L_z,
        })

    return datasets

# Load AOA5 data
datasets_aoa5 = load_spectra_data(AOA5_DATA_DIR, "AOA5")

# Load AOA12 data
datasets_aoa12 = load_spectra_data(AOA12_DATA_DIR, "AOA12")

# Combine all datasets
datasets = datasets_aoa5 + datasets_aoa12

# Print y_plus ranges for each dataset
# print("\nDataset y_plus ranges:")
# for data in datasets:
#     print(f"  - {data['aoa']} - {data['file']}: y_plus range = [{data['y_plus'].min():.1f}, {data['y_plus'].max():.1f}]")

# Global x-axis limits from all datasets (ignore non-finite values)
all_lambda = np.concatenate([d["lambda_z_plus"].ravel() for d in datasets])
finite_lambda = all_lambda[np.isfinite(all_lambda) & (all_lambda > 0)]
if finite_lambda.size == 0:
    raise ValueError("No finite positive lambda_z_plus values found for x-axis limits.")
x_min = finite_lambda.min()
x_max = finite_lambda.max()


# ============================================================================
# Create individual plots for each location (independent scales)
# ============================================================================

# print("\nCreating individual plots with independent scales...")
# for data in datasets:
#     fig, ax = plt.subplots(figsize=(8, 8))
    
#     X, Y = np.meshgrid(data["lambda_z_plus"], data["y_plus"])
#     Z = data["premult_psd"]
    
#     # Use individual min/max for each plot
#     local_min = np.nanmin(Z)
#     local_max = np.nanmax(Z)
#     local_levels = np.linspace(local_min, local_max, 30)
    
#     contour_fill = ax.contourf(X, Y, Z, levels=local_levels, cmap='RdYlBu_r', extend='both')
    
#     cbar = plt.colorbar(contour_fill, ax=ax, pad=0.02)
#     cbar.set_label(r'$k_z\,\Phi_{u_tu_t}/u_\tau^2$', fontsize=12, labelpad=15)
#     cbar.ax.tick_params(labelsize=10)
    
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.set_ylabel(r'$y^+$', fontsize=12)
#     ax.set_xlabel(r'$\lambda_z^+$', fontsize=12)
#     ax.tick_params(which='both', labelsize=11)
    
#     ax.set_title(f'Inner-scaled spanwise premultiplied power spectral density ({data["aoa"]})\n' + 
#                  f'$x_{{ss}}/c = {data["slice_x"]:.3f}$, $N_{{snapshots}} = {data["snapshot_count"]}$',
#                  fontsize=13, pad=15)
    
#     ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
#     ax.grid(True, which='major', alpha=0.5, linestyle='-', linewidth=0.7)
    
#     ax.set_ylim(0.5, 1000)
#     ax.set_box_aspect(1)
    
#     plt.tight_layout()
#     plt.show()

# ============================================================================
# Create 1x5 grid plot for AOA5 (individual scales per subplot)
# ============================================================================

# print("\nCreating 1x5 grid plot for AOA5 with independent scales...")

# # Sort AOA5 datasets by chord location and limit to 5 plots
# datasets_aoa5_sorted = sorted(datasets_aoa5, key=lambda d: d["slice_x"])
# if len(datasets_aoa5_sorted) > 5:
#     datasets_aoa5_sorted = datasets_aoa5_sorted[:5]

# n_plots = len(datasets_aoa5_sorted)
# fig, axes = plt.subplots(1, n_plots, figsize=(3.2 * n_plots, 3.2), sharey=True)

# if n_plots == 1:
#     axes = [axes]

# for ax, data in zip(axes, datasets_aoa5_sorted):
#     X, Y = np.meshgrid(data["lambda_z_plus"], data["y_plus"])
#     Z = data["premult_psd"]

#     local_min = np.nanmin(Z)
#     local_max = np.nanmax(Z)
#     local_levels = np.linspace(local_min, local_max, 30)

#     contour_fill = ax.contourf(X, Y, Z, levels=local_levels, cmap='RdYlBu_r', extend='both')

#     cbar = plt.colorbar(contour_fill, ax=ax, pad=0.02)
#     cbar.set_label(r'$k_z\,\Phi_{u_tu_t}/u_\tau^2$', fontsize=9, labelpad=10)
#     cbar.ax.tick_params(labelsize=8)

#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.set_xlabel(r'$\lambda_z^+$', fontsize=10)
#     ax.set_ylabel(r'$y^+$', fontsize=10)
#     ax.tick_params(which='both', labelsize=9)

#     ax.set_title(
#         f'$x_{{ss}}/c = {data["slice_x"]:.3f}$',
#         fontsize=10,
#         pad=6,
#     )

#     ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
#     ax.grid(True, which='major', alpha=0.5, linestyle='-', linewidth=0.7)

#     ax.set_ylim(0.5, 1000)
#     ax.set_box_aspect(1)

# fig.suptitle('Inner-scaled spanwise premultiplied power spectral density (AOA5)', fontsize=12, y=0.98)
# plt.tight_layout()
# plt.show()

# ============================================================================
# Create 1x5 grid plot for AOA12 (individual scales per subplot)
# ============================================================================

# print("\nCreating 1x5 grid plot for AOA12 with independent scales...")

# # Sort AOA12 datasets by chord location and limit to 5 plots
# datasets_aoa12_sorted = sorted(datasets_aoa12, key=lambda d: d["slice_x"])
# if len(datasets_aoa12_sorted) > 5:
#     datasets_aoa12_sorted = datasets_aoa12_sorted[:5]

# n_plots = len(datasets_aoa12_sorted)
# fig, axes = plt.subplots(1, n_plots, figsize=(3.2 * n_plots, 3.2), sharey=True)

# if n_plots == 1:
#     axes = [axes]

# for ax, data in zip(axes, datasets_aoa12_sorted):
#     X, Y = np.meshgrid(data["lambda_z_plus"], data["y_plus"])
#     Z = data["premult_psd"]

#     local_min = np.nanmin(Z)
#     local_max = np.nanmax(Z)
#     local_levels = np.linspace(local_min, local_max, 30)

#     contour_fill = ax.contourf(X, Y, Z, levels=local_levels, cmap='RdYlBu_r', extend='both')

#     cbar = plt.colorbar(contour_fill, ax=ax, pad=0.02)
#     cbar.set_label(r'$k_z\,\Phi_{u_tu_t}/u_\tau^2$', fontsize=9, labelpad=10)
#     cbar.ax.tick_params(labelsize=8)

#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.set_xlabel(r'$\lambda_z^+$', fontsize=10)
#     ax.set_ylabel(r'$y^+$', fontsize=10)
#     ax.tick_params(which='both', labelsize=9)

#     ax.set_title(
#         f'$x_{{ss}}/c = {data["slice_x"]:.3f}$',
#         fontsize=10,
#         pad=6,
#     )

#     ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
#     ax.grid(True, which='major', alpha=0.5, linestyle='-', linewidth=0.7)

#     ax.set_ylim(0.5, 1000)
#     ax.set_box_aspect(1)

# fig.suptitle('Inner-scaled spanwise premultiplied power spectral density (AOA12)', fontsize=12, y=0.98)
# plt.tight_layout()
# plt.show()

# ============================================================================
# Create 2x5 grid plot (AOA5 top row, AOA12 bottom row)
# ============================================================================

# print("\nCreating 2x5 grid plot with AOA5 (top) and AOA12 (bottom)...")

# # Sort both datasets by chord location and limit to 5 plots
# datasets_aoa5_sorted = sorted(datasets_aoa5, key=lambda d: d["slice_x"])
# datasets_aoa12_sorted = sorted(datasets_aoa12, key=lambda d: d["slice_x"])

# if len(datasets_aoa5_sorted) > 5:
#     datasets_aoa5_sorted = datasets_aoa5_sorted[:5]
# if len(datasets_aoa12_sorted) > 5:
#     datasets_aoa12_sorted = datasets_aoa12_sorted[:5]

# n_cols = max(len(datasets_aoa5_sorted), len(datasets_aoa12_sorted))
# fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 6.4), sharey=True)

# # Plot AOA5 (first row)
# for col, data in enumerate(datasets_aoa5_sorted):
#     ax = axes[0, col]
#     X, Y = np.meshgrid(data["lambda_z_plus"], data["y_plus"])
#     Z = data["premult_psd"]

#     local_min = np.nanmin(Z)
#     local_max = np.nanmax(Z)
#     local_levels = np.linspace(local_min, local_max, 30)

#     contour_fill = ax.contourf(X, Y, Z, levels=local_levels, cmap='RdYlBu_r', extend='both')

#     cbar = plt.colorbar(contour_fill, ax=ax, pad=0.02)
#     cbar.set_label(r'$k_z\,\Phi_{u_tu_t}/u_\tau^2$', fontsize=9, labelpad=10)
#     cbar.ax.tick_params(labelsize=8)

#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.set_xlim(x_min, x_max)
#     ax.set_xlabel(r'$\lambda_z^+$', fontsize=10)
#     if col == 0:
#         ax.set_ylabel(r'$y^+$ (AOA5)', fontsize=10)
#     ax.tick_params(which='both', labelsize=9)

#     ax.set_title(
#         f'$x_{{ss}}/c = {data["slice_x"]:.3f}$',
#         fontsize=10,
#         pad=6,
#     )

#     ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
#     ax.grid(True, which='major', alpha=0.5, linestyle='-', linewidth=0.7)

#     ax.set_ylim(0.5, 1000)
#     ax.set_box_aspect(1)

# # Plot AOA12 (second row)
# for col, data in enumerate(datasets_aoa12_sorted):
#     ax = axes[1, col]
#     X, Y = np.meshgrid(data["lambda_z_plus"], data["y_plus"])
#     Z = data["premult_psd"]

#     local_min = np.nanmin(Z)
#     local_max = np.nanmax(Z)
#     local_levels = np.linspace(local_min, local_max, 30)

#     contour_fill = ax.contourf(X, Y, Z, levels=local_levels, cmap='RdYlBu_r', extend='both')

#     cbar = plt.colorbar(contour_fill, ax=ax, pad=0.02)
#     cbar.set_label(r'$k_z\,\Phi_{u_tu_t}/u_\tau^2$', fontsize=9, labelpad=10)
#     cbar.ax.tick_params(labelsize=8)

#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.set_xlim(x_min, x_max)
#     ax.set_xlabel(r'$\lambda_z^+$', fontsize=10)
#     if col == 0:
#         ax.set_ylabel(r'$y^+$ (AOA12)', fontsize=10)
#     ax.tick_params(which='both', labelsize=9)

#     ax.set_title(
#         f'$x_{{ss}}/c = {data["slice_x"]:.3f}$',
#         fontsize=10,
#         pad=6,
#     )

#     ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
#     ax.grid(True, which='major', alpha=0.5, linestyle='-', linewidth=0.7)

#     ax.set_ylim(0.5, 1000)
#     ax.set_box_aspect(1)

# fig.suptitle('Inner-scaled spanwise premultiplied power spectral density (AOA5 vs AOA12)', fontsize=12, y=0.99)
# plt.tight_layout()
# plt.show()

# ============================================================================
# Create 2x3 grid plot (AOA5 row 1, AOA12 row 2 - selected chord locations)
# ============================================================================

# print("\nCreating 2x3 grid plot with x/c = 0.5, 0.7, 0.9...")

# # Filter datasets for specific chord locations
# target_locations = [0.5, 0.7, 0.9]
# tolerance = 0.01

# def filter_by_location(datasets, target_locs, tol):
#     """Filter datasets to match target chord locations."""
#     filtered = []
#     for target in sorted(target_locs):
#         for data in datasets:
#             if abs(data["slice_x"] - target) < tol:
#                 filtered.append(data)
#                 break
#     return filtered

# datasets_aoa5_selected = filter_by_location(datasets_aoa5, target_locations, tolerance)
# datasets_aoa12_selected = filter_by_location(datasets_aoa12, target_locations, tolerance)

# n_cols = 3
# fig, axes = plt.subplots(2, n_cols, figsize=(9.6, 6.4), sharey=True)

# # Plot AOA5 (first row)
# for col, data in enumerate(datasets_aoa5_selected):
#     ax = axes[0, col]
#     X, Y = np.meshgrid(data["lambda_z_plus"], data["y_plus"])
#     Z = data["premult_psd"]

#     local_min = np.nanmin(Z)
#     local_max = np.nanmax(Z)
#     local_levels = np.linspace(local_min, local_max, 30)

#     contour_fill = ax.contourf(X, Y, Z, levels=local_levels, cmap='RdYlBu_r', extend='both')

#     cbar = plt.colorbar(contour_fill, ax=ax, pad=0.02)
#     cbar.set_label(r'$k_z\,\Phi_{u_tu_t}/u_\tau^2$', fontsize=9, labelpad=10)
#     cbar.ax.tick_params(labelsize=8)

#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.set_xlim(x_min, x_max)
#     ax.set_xlabel(r'$\lambda_z^+$', fontsize=10)
#     if col == 0:
#         ax.set_ylabel(r'$y^+$ (AOA5)', fontsize=10)
#     ax.tick_params(which='both', labelsize=9)

#     ax.set_title(
#         f'$x_{{ss}}/c = {data["slice_x"]:.3f}$',
#         fontsize=10,
#         pad=6,
#     )

#     ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
#     ax.grid(True, which='major', alpha=0.5, linestyle='-', linewidth=0.7)

#     ax.set_ylim(0.5, 1000)
#     ax.set_box_aspect(1)

# # Plot AOA12 (second row)
# for col, data in enumerate(datasets_aoa12_selected):
#     ax = axes[1, col]
#     X, Y = np.meshgrid(data["lambda_z_plus"], data["y_plus"])
#     Z = data["premult_psd"]

#     local_min = np.nanmin(Z)
#     local_max = np.nanmax(Z)
#     local_levels = np.linspace(local_min, local_max, 30)

#     contour_fill = ax.contourf(X, Y, Z, levels=local_levels, cmap='RdYlBu_r', extend='both')

#     cbar = plt.colorbar(contour_fill, ax=ax, pad=0.02)
#     cbar.set_label(r'$k_z\,\Phi_{u_tu_t}/u_\tau^2$', fontsize=9, labelpad=10)
#     cbar.ax.tick_params(labelsize=8)

#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.set_xlim(x_min, x_max)
#     ax.set_xlabel(r'$\lambda_z^+$', fontsize=10)
#     if col == 0:
#         ax.set_ylabel(r'$y^+$ (AOA12)', fontsize=10)
#     ax.tick_params(which='both', labelsize=9)

#     ax.set_title(
#         f'$x_{{ss}}/c = {data["slice_x"]:.3f}$',
#         fontsize=10,
#         pad=6,
#     )

#     ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
#     ax.grid(True, which='major', alpha=0.5, linestyle='-', linewidth=0.7)

#     ax.set_ylim(0.5, 1000)
#     ax.set_box_aspect(1)

# fig.suptitle('Inner-scaled spanwise premultiplied power spectral density (x/c = 0.5, 0.7, 0.9)', fontsize=12, y=0.99)
# plt.tight_layout()
# plt.show()

# ============================================================================
# Create 2x3 grid plot with common colorbar per AOA row
# ============================================================================

# print("\nCreating 2x3 grid plot with shared colorbar for each AOA...")
# # Filter datasets for specific chord locations
# target_locations = [0.5, 0.7, 0.9]
# tolerance = 0.01

# def filter_by_location(datasets, target_locs, tol):
#     """Filter datasets to match target chord locations."""
#     filtered = []
#     for target in sorted(target_locs):
#         for data in datasets:
#             if abs(data["slice_x"] - target) < tol:
#                 filtered.append(data)
#                 break
#     return filtered

# datasets_aoa5_selected = filter_by_location(datasets_aoa5, target_locations, tolerance)
# datasets_aoa12_selected = filter_by_location(datasets_aoa12, target_locations, tolerance)

# # Compute x-axis limits from selected locations only
# all_lambda_selected = np.concatenate([d["lambda_z_plus"].ravel() for d in datasets_aoa5_selected + datasets_aoa12_selected])
# finite_lambda_selected = all_lambda_selected[np.isfinite(all_lambda_selected) & (all_lambda_selected > 0)]
# x_min_selected = finite_lambda_selected.min()
# x_max_selected = finite_lambda_selected.max()

# n_cols = 3

# # Compute shared color limits for each AOA row
# aoa5_vmin = min(np.nanmin(d["premult_psd"]) for d in datasets_aoa5_selected)
# aoa5_vmax = max(np.nanmax(d["premult_psd"]) for d in datasets_aoa5_selected)
# aoa12_vmin = min(np.nanmin(d["premult_psd"]) for d in datasets_aoa12_selected)
# aoa12_vmax = max(np.nanmax(d["premult_psd"]) for d in datasets_aoa12_selected)

# fig, axes = plt.subplots(2, n_cols, figsize=(10, 6), sharey=True)

# # Plot AOA5 (first row) with shared color scale
# for col, data in enumerate(datasets_aoa5_selected):
#     ax = axes[0, col]
#     X, Y = np.meshgrid(data["lambda_z_plus"], data["y_plus"])
#     Z = data["premult_psd"]

#     contour_fill_aoa5 = ax.contourf(
#         X, Y, Z,
#         levels=np.linspace(aoa5_vmin, aoa5_vmax, 30),
#         cmap='RdYlBu_r',
#         extend='both'
#     )

#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.set_xlim(x_min_selected, x_max_selected)
#     ax.set_xlabel(r'$\lambda_z^+$', fontsize=10)
#     if col == 0:
#         ax.set_ylabel(r'$y^+$ (AOA5)', fontsize=10)
#     ax.tick_params(which='both', labelsize=9)

#     ax.set_title(
#         f'$x_{{ss}}/c = {data["slice_x"]:.3f}$',
#         fontsize=10,
#         pad=6,
#     )

#     ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
#     ax.grid(True, which='major', alpha=0.5, linestyle='-', linewidth=0.7)

#     ax.set_ylim(0.5, 1000)
#     ax.set_box_aspect(1)

# # Plot AOA12 (second row) with shared color scale
# for col, data in enumerate(datasets_aoa12_selected):
#     ax = axes[1, col]
#     X, Y = np.meshgrid(data["lambda_z_plus"], data["y_plus"])
#     Z = data["premult_psd"]

#     contour_fill_aoa12 = ax.contourf(
#         X, Y, Z,
#         levels=np.linspace(aoa12_vmin, aoa12_vmax, 30),
#         cmap='RdYlBu_r',
#         extend='both'
#     )

#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.set_xlim(x_min_selected, x_max_selected)
#     ax.set_xlabel(r'$\lambda_z^+$', fontsize=10)
#     if col == 0:
#         ax.set_ylabel(r'$y^+$ (AOA12)', fontsize=10)
#     ax.tick_params(which='both', labelsize=9)

#     ax.set_title(
#         f'$x_{{ss}}/c = {data["slice_x"]:.3f}$',
#         fontsize=10,
#         pad=6,
#     )

#     ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
#     ax.grid(True, which='major', alpha=0.5, linestyle='-', linewidth=0.7)

#     ax.set_ylim(0.5, 1000)
#     ax.set_box_aspect(1)

# # Add shared colorbars for each row
# cbar_aoa5 = fig.colorbar(contour_fill_aoa5, ax=axes[0, :])
# cbar_aoa5.set_label(r'$k_z\,\Phi_{u_tu_t}/u_\tau^2$', fontsize=9, labelpad=10)
# cbar_aoa5.ax.tick_params(labelsize=8)

# cbar_aoa12 = fig.colorbar(contour_fill_aoa12, ax=axes[1, :])
# cbar_aoa12.set_label(r'$k_z\,\Phi_{u_tu_t}/u_\tau^2$', fontsize=9, labelpad=10)
# cbar_aoa12.ax.tick_params(labelsize=8)

# fig.suptitle('Inner-scaled spanwise premultiplied power spectral density (shared colorbars per AOA)', fontsize=12, y=0.99)
# plt.show()

# ============================================================================
# Create 2x2 grid plot with common colorbar per AOA row (x/c = 0.5, 0.7)
# ============================================================================

print("\nCreating 2x2 grid plot with shared colorbar for each AOA (x/c = 0.5, 0.7)...")

# Filter datasets for specific chord locations
target_locations_2x2 = [0.5, 0.7]

def filter_by_location(datasets, target_locs, tol=None):
    """Filter datasets to match target chord locations."""
    tol = tol if tol is not None else 0.01
    filtered = []
    for target in sorted(target_locs):
        for data in datasets:
            if abs(data["slice_x"] - target) < tol:
                filtered.append(data)
                break
    return filtered

datasets_aoa5_selected_2x2 = filter_by_location(datasets_aoa5, target_locations_2x2)
datasets_aoa12_selected_2x2 = filter_by_location(datasets_aoa12, target_locations_2x2)

# Compute x-axis limits from selected locations only
all_lambda_selected_2x2 = np.concatenate([d["lambda_z_plus"].ravel() for d in datasets_aoa5_selected_2x2 + datasets_aoa12_selected_2x2])
finite_lambda_selected_2x2 = all_lambda_selected_2x2[np.isfinite(all_lambda_selected_2x2) & (all_lambda_selected_2x2 > 0)]
x_min_selected_2x2 = finite_lambda_selected_2x2.min()
x_max_selected_2x2 = finite_lambda_selected_2x2.max()

n_cols_2x2 = 2

# Compute shared color limits for each AOA row
aoa5_vmin_2x2 = min(np.nanmin(d["premult_psd"]) for d in datasets_aoa5_selected_2x2)
aoa5_vmax_2x2 = max(np.nanmax(d["premult_psd"]) for d in datasets_aoa5_selected_2x2)
aoa12_vmin_2x2 = min(np.nanmin(d["premult_psd"]) for d in datasets_aoa12_selected_2x2)
aoa12_vmax_2x2 = max(np.nanmax(d["premult_psd"]) for d in datasets_aoa12_selected_2x2)

fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, n_cols_2x2, figure=fig, hspace=0.05, wspace=0.05)
axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(n_cols_2x2)] for i in range(2)])

# Plot AOA5 (first row) with shared color scale
for col, data in enumerate(datasets_aoa5_selected_2x2):
    ax = axes[0, col]
    X, Y = np.meshgrid(data["lambda_z_plus"], data["y_plus"])
    Z = data["premult_psd"]

    contour_fill_aoa5_2x2 = ax.contourf(
        X, Y, Z,
        levels=np.linspace(aoa5_vmin_2x2, aoa5_vmax_2x2, 30),
        cmap='RdYlBu_r',
        extend='neither'
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(x_min_selected_2x2, x_max_selected_2x2)
    ax.set_xticklabels([])
    # ax.set_xlabel(r'$\lambda_z^+$', fontsize=10)
    if col == 0:
        ax.set_ylabel(r'$y^+$', fontsize=10)    
    else:
        ax.set_yticklabels([])    
    ax.tick_params(which='both', labelsize=9)

    # ax.set_title(
    #     f'$x/c = {data["slice_x"]:.3f}$',
    #     fontsize=10,
    #     pad=6,
    # )

    ax.set_ylim(0.5, 1000)
    ax.set_box_aspect(1)

# Plot AOA12 (second row) with shared color scale
for col, data in enumerate(datasets_aoa12_selected_2x2):
    ax = axes[1, col]
    X, Y = np.meshgrid(data["lambda_z_plus"], data["y_plus"])
    Z = data["premult_psd"]

    contour_fill_aoa12_2x2 = ax.contourf(
        X, Y, Z,
        levels=np.linspace(aoa12_vmin_2x2, aoa12_vmax_2x2, 30),
        cmap='RdYlBu_r',
        extend='neither'
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(x_min_selected_2x2, x_max_selected_2x2)
    ax.set_xlabel(r'$\lambda_z^+$', fontsize=10)
    if col == 0:
        ax.set_ylabel(r'$y^+$', fontsize=10)    
    else:
        ax.set_yticklabels([])    
    ax.tick_params(which='both', labelsize=9)

    # ax.set_title(
    #     f'$x/c = {data["slice_x"]:.3f}$',
    #     fontsize=10,
    #     pad=6,
    # )

    ax.set_ylim(0.5, 1000)
    ax.set_box_aspect(1)

# Add shared colorbars for each row
cbar_aoa5_2x2 = fig.colorbar(
    contour_fill_aoa5_2x2,
    ax=axes[0, :],
    shrink=0.85
)

# Set label text
cbar_aoa5_2x2.set_label(
    r'$k_z\,\Phi_{u_tu_t}/u_\tau^2$',
    fontsize=9,
    labelpad=8
)

# Rotate label to horizontal
cbar_aoa5_2x2.ax.yaxis.label.set_rotation(0)

# Move label to the top of the colorbar
cbar_aoa5_2x2.ax.yaxis.set_label_coords(1.3, 1.05)

# Center alignment
cbar_aoa5_2x2.ax.yaxis.label.set_horizontalalignment('center')
cbar_aoa5_2x2.ax.yaxis.label.set_verticalalignment('bottom')

# Ticks
cbar_aoa5_2x2.ax.tick_params(labelsize=8)


cbar_aoa12_2x2 = fig.colorbar(
    contour_fill_aoa12_2x2, 
    ax=axes[1, :], 
    shrink=0.85,
)

# Set label text
cbar_aoa12_2x2.set_label(
    r'$k_z\,\Phi_{u_tu_t}/u_\tau^2$',
    fontsize=9,
    labelpad=8
)

# Rotate label to horizontal
cbar_aoa12_2x2.ax.yaxis.label.set_rotation(0)

# Move label to the top of the colorbar
cbar_aoa12_2x2.ax.yaxis.set_label_coords(1.3, 1.05)

# Center alignment
cbar_aoa12_2x2.ax.yaxis.label.set_horizontalalignment('center')
cbar_aoa12_2x2.ax.yaxis.label.set_verticalalignment('bottom')
# Ticks
cbar_aoa12_2x2.ax.tick_params(labelsize=8)

os.makedirs(OUTPUT_PATH, exist_ok=True)
fig.savefig(os.path.join(OUTPUT_PATH, "premultiplied_spectra_2x2.eps"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUTPUT_PATH, "premultiplied_spectra_2x2.png"), dpi=300, bbox_inches="tight")

plt.show()

# ============================================================================
# Create 2x3 grid plot with custom locations (AOA5: 0.5, 0.7, 0.9; AOA12: 0.3, 0.5, 0.7)
# ============================================================================

print("\nCreating 2x3 grid plot with custom locations (AOA5: 0.5, 0.7, 0.9; AOA12: 0.3, 0.5, 0.7)...")

# Filter datasets for specific chord locations
target_locations_aoa5_2x3 = [0.5, 0.7, 0.9]
target_locations_aoa12_2x3 = [0.3, 0.5, 0.7]

def filter_by_location(datasets, target_locs, tol=None):
    """Filter datasets to match target chord locations."""
    tol = tol if tol is not None else 0.01
    filtered = []
    for target in sorted(target_locs):
        for data in datasets:
            if abs(data["slice_x"] - target) < tol:
                filtered.append(data)
                break
    return filtered


datasets_aoa5_selected_2x3 = filter_by_location(datasets_aoa5, target_locations_aoa5_2x3)
datasets_aoa12_selected_2x3 = filter_by_location(datasets_aoa12, target_locations_aoa12_2x3)

# Compute x-axis limits from selected locations only
all_lambda_selected_2x3 = np.concatenate([d["lambda_z_plus"].ravel() for d in datasets_aoa5_selected_2x3 + datasets_aoa12_selected_2x3])
finite_lambda_selected_2x3 = all_lambda_selected_2x3[np.isfinite(all_lambda_selected_2x3) & (all_lambda_selected_2x3 > 0)]
x_min_selected_2x3 = finite_lambda_selected_2x3.min()
x_max_selected_2x3 = finite_lambda_selected_2x3.max()

n_cols_2x3 = 3

# Compute shared color limits for each AOA row
aoa5_vmin_2x3 = min(np.nanmin(d["premult_psd"]) for d in datasets_aoa5_selected_2x3)
aoa5_vmax_2x3 = max(np.nanmax(d["premult_psd"]) for d in datasets_aoa5_selected_2x3)
aoa12_vmin_2x3 = min(np.nanmin(d["premult_psd"]) for d in datasets_aoa12_selected_2x3)
aoa12_vmax_2x3 = max(np.nanmax(d["premult_psd"]) for d in datasets_aoa12_selected_2x3)

fig = plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(2, n_cols_2x3, figure=fig, hspace=0.05, wspace=0.05)
axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(n_cols_2x3)] for i in range(2)])

# Plot AOA5 (first row) with shared color scale
for col, data in enumerate(datasets_aoa5_selected_2x3):
    ax = axes[0, col]
    X, Y = np.meshgrid(data["lambda_z_plus"], data["y_plus"])
    Z = data["premult_psd"]

    contour_fill_aoa5_2x3 = ax.contourf(
        X, Y, Z,
        levels=np.linspace(aoa5_vmin_2x3, aoa5_vmax_2x3, 30),
        cmap='RdYlBu_r',
        extend='neither'
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(x_min_selected_2x3, x_max_selected_2x3)
    ax.set_xticklabels([])
    if col == 0:
        ax.set_ylabel(r'$y^+$', fontsize=10)
    else:
        ax.set_yticklabels([])
    ax.tick_params(which='both', labelsize=9)

    ax.set_ylim(0.5, 1000)
    ax.set_box_aspect(1)

# Plot AOA12 (second row) with shared color scale
for col, data in enumerate(datasets_aoa12_selected_2x3):
    ax = axes[1, col]
    X, Y = np.meshgrid(data["lambda_z_plus"], data["y_plus"])
    Z = data["premult_psd"]

    contour_fill_aoa12_2x3 = ax.contourf(
        X, Y, Z,
        levels=np.linspace(aoa12_vmin_2x3, aoa12_vmax_2x3, 30),
        cmap='RdYlBu_r',
        extend='neither'
    )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(x_min_selected_2x3, x_max_selected_2x3)
    ax.set_xlabel(r'$\lambda_z^+$', fontsize=10)
    if col == 0:
        ax.set_ylabel(r'$y^+$', fontsize=10)
    else:
        ax.set_yticklabels([])
    ax.tick_params(which='both', labelsize=9)

    ax.set_ylim(0.5, 1000)
    ax.set_box_aspect(1)

# Add shared colorbars for each row
cbar_aoa5_2x3 = fig.colorbar(
    contour_fill_aoa5_2x3,
    ax=axes[0, :],
    shrink=0.85
)

# Set label text
cbar_aoa5_2x3.set_label(
    r'$k_z\,\Phi_{u_tu_t}/u_\tau^2$',
    fontsize=9,
    labelpad=8
)

# Rotate label to horizontal
cbar_aoa5_2x3.ax.yaxis.label.set_rotation(0)

# Move label to the top of the colorbar
cbar_aoa5_2x3.ax.yaxis.set_label_coords(1.3, 1.05)

# Center alignment
cbar_aoa5_2x3.ax.yaxis.label.set_horizontalalignment('center')
cbar_aoa5_2x3.ax.yaxis.label.set_verticalalignment('bottom')
# Ticks
cbar_aoa5_2x3.ax.tick_params(labelsize=8)


cbar_aoa12_2x3 = fig.colorbar(
    contour_fill_aoa12_2x3, 
    ax=axes[1, :], 
    shrink=0.85,
)

# Set label text
cbar_aoa12_2x3.set_label(
    r'$k_z\,\Phi_{u_tu_t}/u_\tau^2$',
    fontsize=9,
    labelpad=8
)

# Rotate label to horizontal
cbar_aoa12_2x3.ax.yaxis.label.set_rotation(0)

# Move label to the top of the colorbar
cbar_aoa12_2x3.ax.yaxis.set_label_coords(1.3, 1.05)

# Center alignment
cbar_aoa12_2x3.ax.yaxis.label.set_horizontalalignment('center')
cbar_aoa12_2x3.ax.yaxis.label.set_verticalalignment('bottom')
# Ticks
cbar_aoa12_2x3.ax.tick_params(labelsize=8)

plt.show()

