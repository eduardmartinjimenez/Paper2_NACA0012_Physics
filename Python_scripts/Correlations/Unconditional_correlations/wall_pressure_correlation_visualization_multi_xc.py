"""
Wall-Pressure Correlation Visualization Script (Multi x/c Locations)

This script generates multi-location comparison visualizations of unconditional
wall-pressure stress correlations from CFD simulations. It scans a results directory
for all unconditional correlation files and produces 2 main comparison figures:

1. FIGURE 4B: Multi-location u'_rms/U_infty vs correlation field comparison
               (one row per x/c location)
2. FIGURE 5B: Multi-location spanwise correlation fields comparison
              (one row per x/c location, showing 4 spanwise separations each)

The script supports switching between two angle-of-attack (AOA) configurations:
- AOA 12 (currently active)
- AOA 5 (alternative, commented out)

Visualization windows are dynamically centered on each location's reference
coordinates and configured via offset-based parameters defined at the top.
"""

import glob
import re
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================

# # ========== ACTIVE CONFIGURATION: AOA 12 ==========
# # Results directory containing all unconditional correlation files
# RESULTS_DIR = (
#     "/home/jofre/Members/Eduard/Paper2/Simulations/"
#     "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
#     "Wall_pressure_correlations/test_4/"
# )

# # Output directory for saving figures
# OUTPUT_DIR = os.path.join(RESULTS_DIR, "Figures")

# # Visualization window offsets (relative to reference coordinates)
# # Format: OFFSET = [left/bottom_extent, right/top_extent]
# # xlim = [x_ref - left_extent, x_ref + right_extent]
# # ylim = [y_ref - bottom_extent, y_ref + top_extent]
# VIZ_XLIM_OFFSET = [0.25, 0.25]   # Symmetric x-window
# VIZ_YLIM_OFFSET = [0.02, 0.25]   # Asymmetric y-window (more extent above reference)

# ========== ALTERNATIVE CONFIGURATION: AOA 5 (COMMENTED OUT) ==========
# To switch to AOA 5, uncomment the block below and comment out the AOA 12 block above
RESULTS_DIR = (
    "/home/jofre/Members/Eduard/Paper2/Simulations/"
    "NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
    "Wall_pressure_correlations/test_2/"
)
VIZ_XLIM_OFFSET = [0.06, 0.06]
VIZ_YLIM_OFFSET = [0.01, 0.05]

# ============================================================================
# File Discovery and Sorting
# ============================================================================

print("\n" + "="*70)
print("FILE DISCOVERY")
print("="*70)

# Find all unconditional correlation files
h5_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "wall_pressure_correlation_unconditional_xc_*.h5")))

# Extract x/c values and sort numerically
def extract_xc(filepath):
    filename = os.path.basename(filepath)
    match = re.search(r'xc_([0-9.]+)\.h5', filename)
    if match:
        return float(match.group(1))
    return float('inf')

h5_files.sort(key=extract_xc)

if not h5_files:
    print(f"ERROR: No unconditional correlation files found in {RESULTS_DIR}")
    exit(1)

print(f"\nFound {len(h5_files)} unconditional correlation files:")
xc_values = []
for h5_file in h5_files:
    xc = extract_xc(h5_file)
    xc_values.append(xc)
    print(f"  x/c = {xc:.3f}: {os.path.basename(h5_file)}")

# ============================================================================
# FIGURE 4B: Multi-Location u'_rms vs Correlation Comparison
# ============================================================================
# WHAT WE'RE PLOTTING:
# - Multi-location comparison of u'_rms/U_infty and correlation at Δz=0
# - One row per chord location (x/c)
# - Left column: velocity RMS normalized by freestream
# - Right column: correlation field
# - Common color scales across all locations
#
# PURPOSE:
# - Investigates how the spatial relationship between turbulence activity
#   and wall-pressure/velocity coupling varies across different chord positions
# - Reveals if this relationship is consistent or depends on local conditions

print("\n" + "="*70)
print("FIGURE 4B: Loading data for multi-location u'_rms/R comparison")
print("="*70)

# Load all files and extract data at Δz=0
data_list = []
for h5_file in h5_files:
    with h5py.File(h5_file, 'r') as f:
        data_dict = {
            'R': f['R'][0, :, :],  # Δz=0 slice, (Ny, Nx)
            'u_rms': f['u_rms'][0, :, :],  # Δz=0 slice, (Ny, Nx)
            'x': f['x'][0, :, :],  # (Ny, Nx)
            'y': f['y'][0, :, :],  # (Ny, Nx)
            'x_c_actual': f.attrs['x_c_actual'],
            'y_actual': f.attrs['y_actual'],
            'u_infty': f.attrs.get('u_infty', 1.0),
        }
        data_list.append(data_dict)

# Compute common color scale limits across all locations
u_rms_min_all = []
u_rms_max_all = []
for d in data_list:
    u_rms_norm = d['u_rms'] / d['u_infty']
    u_rms_min_all.append(np.nanmin(u_rms_norm))
    u_rms_max_all.append(np.nanmax(u_rms_norm))

u_rms_vmin = np.min(u_rms_min_all)
u_rms_vmax = np.max(u_rms_max_all)

print(f"\nColor scale ranges:")
print(f"  u_rms/U_infty: [{u_rms_vmin:.4f}, {u_rms_vmax:.4f}]")
print(f"  Correlation:   [-1.0000,  1.0000]")

# Create figure with multiple rows (one per location)
n_locations = len(data_list)
fig, axes = plt.subplots(n_locations, 2, figsize=(14, 4.5 * n_locations), constrained_layout=True)

# Ensure axes is 2D even for single row
if n_locations == 1:
    axes = axes.reshape(1, -1)

# Prepare levels for contourf
levels_u = np.linspace(u_rms_vmin, u_rms_vmax, 16)
levels_r = np.linspace(-1.0, 1.0, 21)

# Plot each location
im1_last = None
im2_last = None

for row, data in enumerate(data_list):
    R_2d_loc = data['R']
    u_rms_2d_loc = data['u_rms']
    x_2d_loc = data['x']
    y_2d_loc = data['y']
    x_c = data['x_c_actual']
    y_c = data['y_actual']
    u_infty = data['u_infty']

    # Normalize u_rms
    u_rms_norm = u_rms_2d_loc / u_infty

    # Left panel: u_rms
    im1_last = axes[row, 0].contourf(x_2d_loc, y_2d_loc, u_rms_norm, levels=levels_u, cmap='YlOrRd')

    axes[row, 0].plot(x_c, y_c, marker='*', color='k', markersize=9, zorder=5)

    # Title only on first row
    if row == 0:
        axes[row, 0].set_title(r"$u'_{rms}/U_\infty$", fontsize=11)

    # x-label only on bottom row
    if row == n_locations - 1:
        axes[row, 0].set_xlabel('x/c', fontsize=11)
    else:
        axes[row, 0].set_xlabel('')

    # y-label only on first column (all rows)
    axes[row, 0].set_ylabel('y/c', fontsize=11)

    # x/c annotation on first column
    axes[row, 0].text(0.02, 0.98, rf'$x/c = {x_c:.3f}$',
                      transform=axes[row, 0].transAxes, fontsize=9,
                      verticalalignment='top', bbox=dict(boxstyle='round',
                      facecolor='wheat', alpha=0.5))

    axes[row, 0].set_aspect('equal', adjustable='box')

    # Right panel: correlation
    im2_last = axes[row, 1].contourf(x_2d_loc, y_2d_loc, R_2d_loc, levels=levels_r,
                                      cmap='RdBu_r', vmin=-1.0, vmax=1.0)

    # Zero contour for structural reference
    axes[row, 1].contour(x_2d_loc, y_2d_loc, R_2d_loc, levels=[0.0],
                         colors='black', linewidths=0.8, alpha=0.7)

    axes[row, 1].plot(x_c, y_c, marker='*', color='k', markersize=9, zorder=5)

    # Title only on first row
    if row == 0:
        axes[row, 1].set_title(r'$R$ at $\Delta z = 0$', fontsize=11)

    # x-label only on bottom row
    if row == n_locations - 1:
        axes[row, 1].set_xlabel('x/c', fontsize=11)
    else:
        axes[row, 1].set_xlabel('')

    # No y-label on right column
    axes[row, 1].set_ylabel('')

    axes[row, 1].set_aspect('equal', adjustable='box')

    # Apply dynamic visualization window
    xlim = [x_c - VIZ_XLIM_OFFSET[0], x_c + VIZ_XLIM_OFFSET[1]]
    ylim = [y_c - VIZ_YLIM_OFFSET[0], y_c + VIZ_YLIM_OFFSET[1]]
    axes[row, 0].set_xlim(xlim)
    axes[row, 0].set_ylim(ylim)
    axes[row, 1].set_xlim(xlim)
    axes[row, 1].set_ylim(ylim)

# Add shared colorbars for each column
cbar1 = fig.colorbar(im1_last, ax=axes[:, 0], label=r"$u'_{rms}/U_\infty$", fraction=0.046, pad=0.04)
cbar2 = fig.colorbar(im2_last, ax=axes[:, 1], label=r"Correlation coefficient, $R$", fraction=0.046, pad=0.04)
cbar2.set_ticks(np.linspace(-1, 1, 5))

fig.suptitle(
    r'Multi-location comparison of $u^\prime_{rms}/U_\infty$ and wall-pressure correlation at $\Delta z = 0$',
    fontsize=14
)


# ============================================================================
# FIGURE 5B: Multi-Location Spanwise Variation Comparison
# ============================================================================
# WHAT WE'RE PLOTTING:
# - Multi-location comparison of spanwise correlation fields at different separation
# - One row per chord location (x/c)
# - Four columns showing z_indices = [0, Nz//8, Nz//4, Nz//2] for each file
# - Common color scale across all locations and separations
#
# PURPOSE:
# - Reveals how spanwise coherence structure varies across chord positions
# - Shows if correlation pattern at different separations is consistent
# - Demonstrates spanwise extent of correlation at different airfoil sections

print("\n" + "="*70)
print("FIGURE 5B: Loading data for multi-location spanwise variation")
print("="*70)

# Load all files for FIGURE 5B
data_fig5b = []
for h5_file in h5_files:
    with h5py.File(h5_file, 'r') as f:
        Nz_loc = f['R'].shape[0]
        R_data = f['R'][:]  # Full (Nz, Ny, Nx)
        x_data = f['x'][0, :, :]  # (Ny, Nx) at Δz=0
        y_data = f['y'][0, :, :]  # (Ny, Nx) at Δz=0
        z_data = f['z'][:]  # (Nz, Ny, Nx)
        x_c_val = f.attrs['x_c_actual']
        y_c_val = f.attrs['y_actual']

        data_fig5b.append({
            'R': R_data,
            'x': x_data,
            'y': y_data,
            'z': z_data,
            'x_c': x_c_val,
            'y_c': y_c_val,
            'Nz': Nz_loc,
        })

print(f"\nLoaded data for {len(data_fig5b)} locations with fixed correlation scale [-1.0, 1.0]")

# Create figure with rows for each location, columns for each z-index
n_rows_5b = len(data_fig5b)
n_cols_5b = 4
fig, axes = plt.subplots(n_rows_5b, n_cols_5b, figsize=(16, 4 * n_rows_5b), constrained_layout=True)

# Ensure axes is 2D even for single row
if n_rows_5b == 1:
    axes = axes.reshape(1, -1)

# Color levels for all subplots (fixed physical scale)
levels_5b = np.linspace(-1.0, 1.0, 21)

# Track last image for colorbar
im_5b_last = None

# Plot each location's spanwise variation
for row_idx, data in enumerate(data_fig5b):
    R_full = data['R']  # (Nz, Ny, Nx)
    Nz_loc = data['Nz']
    x_2d_5b = data['x']
    y_2d_5b = data['y']
    z_full = data['z']
    x_c_val = data['x_c']
    y_c_val = data['y_c']

    # Compute z-indices for this location
    z_indices_5b = [0, Nz_loc//8, Nz_loc//4, Nz_loc//2]

    # Compute spanwise separations for this location
    z_ref_5b = z_full[0, 0, 0]
    z_sep_5b = [z_full[z_idx, 0, 0] - z_ref_5b for z_idx in z_indices_5b]

    # Plot the 4 z-slices for this location
    for col_idx, z_idx in enumerate(z_indices_5b):
        ax = axes[row_idx, col_idx]
        R_slice = R_full[z_idx, :, :]
        dz_sep = z_sep_5b[col_idx]

        # Plot contourf with fixed limits
        im_5b_last = ax.contourf(
            x_2d_5b, y_2d_5b, R_slice,
            levels=levels_5b,
            cmap='RdBu_r',
            vmin=-1.0,
            vmax=1.0
        )

        # Reference point
        ax.plot(x_c_val, y_c_val, marker='*', color='k', markersize=12, zorder=5)

        # Titles (only show Δz on top row)
        if row_idx == 0:
            ax.set_title(rf'$\Delta z = {dz_sep:.5f}$', fontsize=11)

        # x-label only on bottom row
        if row_idx == n_rows_5b - 1:
            ax.set_xlabel('x/c', fontsize=10)
        else:
            ax.set_xlabel('')

        # y-label only on first column
        if col_idx == 0:
            ax.set_ylabel('y/c', fontsize=10)
            # Annotation for x/c value on first column
            ax.text(0.02, 0.98, rf'$x/c = {x_c_val:.3f}$',
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='wheat', alpha=0.5))
        else:
            ax.set_ylabel('')

        ax.set_aspect('equal', adjustable='box')

        # Apply dynamic visualization window
        xlim = [x_c_val - VIZ_XLIM_OFFSET[0], x_c_val + VIZ_XLIM_OFFSET[1]]
        ylim = [y_c_val - VIZ_YLIM_OFFSET[0], y_c_val + VIZ_YLIM_OFFSET[1]]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

# Single shared colorbar
cbar_5b = fig.colorbar(im_5b_last, ax=axes, orientation='vertical',
                        fraction=0.015, pad=0.02)
cbar_5b.set_label(r'Correlation coefficient, $R$', fontsize=11)
cbar_5b.set_ticks(np.linspace(-1, 1, 5))

fig.suptitle('Multi-location comparison of wall-pressure correlation fields at different spanwise separations', fontsize=14)


# ============================================================================
# Summary Statistics
# ============================================================================

print(f"\n" + "="*70)
print("MULTI-LOCATION VISUALIZATION SUMMARY")
print("="*70)
print(f"\nVISUALIZATIONS GENERATED:")
print(f"  FIGURE 4B: Multi-location u'_rms vs correlation (Δz=0)")
print(f"  FIGURE 5B: Multi-location spanwise correlation fields (4 separations)")
print(f"\nFILES ANALYZED:")
print(f"  Directory: {RESULTS_DIR}")
print(f"  Total locations: {n_locations}")
print(f"  x/c values: {[f'{xc:.3f}' for xc in xc_values]}")
print(f"\nCOLOR SCALES USED:")
print(f"  u_rms/U_infty: [{u_rms_vmin:.4f}, {u_rms_vmax:.4f}] (common across all locations)")
print(f"  Correlation:   [-1.0000,  1.0000] (fixed physical scale)")
print(f"\nVISUALIZATION WINDOW CONFIGURATION:")
print(f"  X-window offset: [{VIZ_XLIM_OFFSET[0]:.2f}, {VIZ_XLIM_OFFSET[1]:.2f}]")
print(f"  Y-window offset: [{VIZ_YLIM_OFFSET[0]:.2f}, {VIZ_YLIM_OFFSET[1]:.2f}]")
print(f"  (Dynamically centered on each location's reference point)")
print(f"\nOUTPUT FIGURES:")


# Show all figures
plt.show()

print("\n" + "="*70)
print("MULTI-LOCATION VISUALIZATION COMPLETE")
print("="*70)
