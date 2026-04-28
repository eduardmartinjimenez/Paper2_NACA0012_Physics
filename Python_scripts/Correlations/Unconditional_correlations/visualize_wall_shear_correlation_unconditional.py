"""
Unconditional Wall-Shear Correlation Visualization Script

This script generates comprehensive visualizations of unconditional wall-shear stress
correlations from CFD simulations. It produces 5 main figures:

1. FIGURE 4:  Streamwise velocity RMS vs correlation field at a single reference location
2. FIGURE 4B: Multi-location comparison of velocity RMS and correlation across chord
3. FIGURE 5:  Spanwise correlation decay at a single reference location
4. FIGURE 5B: Multi-location spanwise correlation decay comparison
5. FIGURE 6:  Spanwise decay curves at different wall-normal heights

The script supports switching between two angle-of-attack (AOA) configurations:
- AOA 12 (currently active)
- AOA 5 (alternative, commented out)

Visualization windows are dynamically centered on reference coordinates and configured
via offset-based parameters defined at the top of the script.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================

# ========== ACTIVE CONFIGURATION: AOA 12 ==========
# Correlation data path (for visualization)
RESULT_FILE = (
    "/home/jofre/Members/Eduard/Paper2/Simulations/"
    "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
    "Wall_shear_correlations/test_5/"
    "wall_shear_correlation_unconditional_xc_0.500.h5"
)

# Probe y-coordinates for FIGURE 6 (spanwise decay curves)
# PROBE_Y_COORDS = [0.062, 0.065, 0.068, 0.075]  # x/c = 0.3
PROBE_Y_COORDS = [0.055, 0.06, 0.065, 0.07]     # x/c = 0.5 (active)
# PROBE_Y_COORDS = [0.04, 0.045, 0.055, 0.07]   # x/c = 0.7
# PROBE_Y_COORDS = [0.02, 0.025, 0.035, 0.045]  # x/c = 0.9

# Visualization window offsets (relative to reference coordinates)
# Format: OFFSET = [left/bottom_extent, right/top_extent]
# xlim = [x_ref - left_extent, x_ref + right_extent]
# ylim = [y_ref - bottom_extent, y_ref + top_extent]
VIZ_XLIM_OFFSET = [0.25, 0.25]   # Symmetric x-window
VIZ_YLIM_OFFSET = [0.02, 0.25]   # Asymmetric y-window (more extent above reference)

# ========== ALTERNATIVE CONFIGURATION: AOA 5 (COMMENTED OUT) ==========
# To switch to AOA 5, uncomment the block below and comment out the AOA 12 block above
# RESULT_FILE = (
#     "/home/jofre/Members/Eduard/Paper2/Simulations/"
#     "NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
#     "Wall_shear_correlations/test_2/"
#     "wall_shear_correlation_unconditional_xc_0.700.h5"
# )
# PROBE_Y_COORDS = [0.054, 0.056, 0.058, 0.06]   # x/c = 0.5
# # PROBE_Y_COORDS = [0.038, 0.04, 0.045, 0.05]  # x/c = 0.7
# # PROBE_Y_COORDS = [0.016, 0.02, 0.024]        # x/c = 0.9
# VIZ_XLIM_OFFSET = [0.06, 0.06]
# VIZ_YLIM_OFFSET = [0.01, 0.05]

# ============================================================================
# Load Results
# ============================================================================
# NOTE on coordinate semantics:
# - The z-dimension (Nz=128) represents different z-planes in the physical mesh
# - The FFT correlation exploits spanwise periodicity via circular convolution
# - z-index=0 corresponds to the reference z-plane at physical z = z[0,0,0]
# - z-index=k corresponds to spanwise separation Δz = z[k,0,0] - z[0,0,0]
# - This allows measuring how correlation decays away from the reference plane
# ============================================================================

print("Loading unconditional correlation results...")

with h5py.File(RESULT_FILE, 'r') as f:
    # Load correlation field
    R = f['R'][:]              # (Nz, Ny, Nx)
    u_rms = f['u_rms'][:]      # (Nz, Ny, Nx)

    # Load coordinates
    x = f['x'][:]              # (Nz, Ny, Nx)
    y = f['y'][:]              # (Nz, Ny, Nx)
    z = f['z'][:]              # (Nz, Ny, Nx)

    # Load metadata
    x_c_actual = f.attrs['x_c_actual']
    y_actual = f.attrs['y_actual']
    N_samples = f.attrs['N_samples']
    N_snapshots = f.attrs['N_snapshots']
    tau_w_mean = f.attrs['tau_w_mean']
    tau_w_rms = f.attrs['tau_w_rms']
    u_infty = f.attrs.get('u_infty', 1.0)  # Default to 1.0 if not present

Nz, Ny, Nx = R.shape

# The z-axis represents spanwise SEPARATION Δz, not physical z-coordinate.
# Index 0 = Δz=0 (same plane as reference, strongest correlation)
# Index Nz//2 = maximum separation (weakest correlation)
dz_slice = 0

print(f"Loaded unconditional correlation results:")
print(f"  Shape: (Nz={Nz}, Ny={Ny}, Nx={Nx})")
print(f"  Reference point: x/c = {x_c_actual:.4f}, y = {y_actual:.4f}")
print(f"  Wall shear: mean = {tau_w_mean:.6e}, rms = {tau_w_rms:.6e}")
print(f"  Samples: N_snapshots={N_snapshots}, N_samples={N_samples}")
print(f"  Using z-slice at Δz = {dz_slice} (in-plane correlation)")

# ============================================================================
# Extract 2D Slice at Δz=0 (In-plane, Same Spanwise Location as Reference)
# ============================================================================

R_2d = R[dz_slice, :, :]        # (Ny, Nx)
u_rms_2d = u_rms[dz_slice, :, :] # (Ny, Nx)

x_2d = x[dz_slice, :, :]  # (Ny, Nx)
y_2d = y[dz_slice, :, :]  # (Ny, Nx)

print(f"\n2D slice shape: {R_2d.shape}")

# ============================================================================
# FIGURE 4: Correlation vs Velocity RMS (Side-by-Side Comparison)
# ============================================================================
# WHAT WE'RE PLOTTING:
# - LEFT PANEL: Normalized streamwise velocity fluctuation intensity
# - RIGHT PANEL: Correlation field at Δz = 0
# - Allows direct spatial comparison between fluctuation intensity and correlation
#
# PURPOSE:
# - Investigates whether strong correlation occurs in regions of large velocity fluctuations
# - Reveals the spatial relationship between turbulence activity and wall-shear coupling

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)

# Normalize u_rms by freestream velocity
u_rms_norm = u_rms_2d / u_infty

# ---------------------------------------------------------------------------
# Left panel: streamwise fluctuation intensity
# ---------------------------------------------------------------------------
levels_u = np.linspace(np.nanmin(u_rms_norm), np.nanmax(u_rms_norm), 16)

im1 = axes[0].contourf(
    x_2d, y_2d, u_rms_norm,
    levels=levels_u,
    cmap='YlOrRd'
)

axes[0].plot(
    x_c_actual, y_actual,
    marker='*', color='k',
    markersize=9, zorder=5
)

axes[0].set_xlabel('x/c', fontsize=12)
axes[0].set_ylabel('y/c', fontsize=12)
axes[0].set_title(r'Streamwise fluctuation intensity, $u^\prime_{\mathrm{rms}}/U_\infty$', fontsize=12)
axes[0].set_aspect('equal', adjustable='box')

cbar1 = fig.colorbar(im1, ax=axes[0])
cbar1.set_label(r'$u^\prime_{\mathrm{rms}}/U_\infty$', fontsize=11)

# ---------------------------------------------------------------------------
# Right panel: correlation field at Δz = 0
# ---------------------------------------------------------------------------
levels_r = np.linspace(-1.0, 1.0, 21)

im2 = axes[1].contourf(
    x_2d, y_2d, R_2d,
    levels=levels_r,
    cmap='RdBu_r',
    vmin=-1.0,
    vmax=1.0
)

# Keep only the zero contour as a structural guide
cs_zero = axes[1].contour(
    x_2d, y_2d, R_2d,
    levels=[0.0],
    colors='black',
    linewidths=0.8,
    alpha=0.7
)

axes[1].plot(
    x_c_actual, y_actual,
    marker='*', color='k',
    markersize=9, zorder=5
)

axes[1].set_xlabel('x/c', fontsize=12)
axes[1].set_ylabel('y/c', fontsize=12)
axes[1].set_title(r'Correlation field at $\Delta z = 0$', fontsize=12)
axes[1].set_aspect('equal', adjustable='box')

cbar2 = fig.colorbar(im2, ax=axes[1])
cbar2.set_label(r'Correlation coefficient, $R$', fontsize=11)
cbar2.set_ticks(np.linspace(-1, 1, 5))

# Apply visualization window limits to both panels
# Window is relative to reference position (x_c_actual, y_actual)
xlim = [x_c_actual - VIZ_XLIM_OFFSET[0], x_c_actual + VIZ_XLIM_OFFSET[1]]
ylim = [y_actual - VIZ_YLIM_OFFSET[0], y_actual + VIZ_YLIM_OFFSET[1]]
for ax in axes:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# Overall title
fig.suptitle(
    r'Comparison between $u^\prime_{\mathrm{rms}}/U_\infty$ and correlation field at $\Delta z = 0$',
    fontsize=14
)


# ============================================================================
# FIGURE 4B: Correlation vs Velocity RMS for Multiple Chord Locations
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
#   and wall-shear coupling varies across different chord positions
# - Reveals if this relationship is consistent or depends on local conditions

import re

# Find all unconditional correlation files in the results directory
results_dir = os.path.dirname(RESULT_FILE)
all_files = os.listdir(results_dir)
h5_files = [f for f in all_files if f.startswith('wall_shear_correlation_unconditional_xc_') and f.endswith('.h5')]

# Extract x/c values and sort numerically
def extract_xc(filename):
    match = re.search(r'xc_([0-9.]+)\.h5', filename)
    if match:
        return float(match.group(1))
    return float('inf')

h5_files.sort(key=extract_xc)
print(f"\n{'='*70}")
print(f"FIGURE 4B: Found {len(h5_files)} unconditional correlation files:")
for f in h5_files:
    xc = extract_xc(f)
    print(f"  x/c = {xc:.3f}: {f}")

# Load all files and extract data at Δz=0
data_list = []
for h5_file in h5_files:
    file_path = os.path.join(results_dir, h5_file)
    with h5py.File(file_path, 'r') as f:
        data_dict = {
            'R': f['R'][0, :, :],  # Δz=0 slice, (Ny, Nx)
            'u_rms': f['u_rms'][0, :, :],  # Δz=0 slice, (Ny, Nx)
            'x': f['x'][0, :, :],  # (Ny, Nx)
            'y': f['y'][0, :, :],  # (Ny, Nx)
            'x_c_actual': f.attrs['x_c_actual'],
            'y_actual': f.attrs['y_actual'],
            'u_infty': f.attrs.get('u_infty', 1.0),
            'filename': h5_file
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

# Correlation limits (symmetric around zero)
R_all = np.concatenate([d['R'].ravel() for d in data_list])
R_min = np.min(R_all)
R_max = np.max(R_all)
R_lim = np.max(np.abs([R_min, R_max]))

print(f"Color scale ranges:")
print(f"  u_rms/U_infty: [{u_rms_vmin:.4f}, {u_rms_vmax:.4f}]")
print(f"  Correlation: [{-R_lim:.4f}, {R_lim:.4f}]")

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
    im2_last = axes[row, 1].contourf(x_2d_loc, y_2d_loc, R_2d_loc, levels=levels_r, cmap='RdBu_r', vmin=-1.0, vmax=1.0)

    # Zero contour for structural reference
    axes[row, 1].contour(x_2d_loc, y_2d_loc, R_2d_loc, levels=[0.0], colors='black', linewidths=0.8, alpha=0.7)

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

# Add shared colorbars for each column
cbar1 = fig.colorbar(im1_last, ax=axes[:, 0], label=r"$u'_{rms}/U_\infty$", fraction=0.046, pad=0.04)
cbar2 = fig.colorbar(im2_last, ax=axes[:, 1], label=r"Correlation coefficient, $R$", fraction=0.046, pad=0.04)
cbar2.set_ticks(np.linspace(-1, 1, 5))

fig.suptitle(
    r'Comparison of $u^\prime_{rms}/U_\infty$ and correlation at $\Delta z = 0$ across chord locations',
    fontsize=14
)

# Apply dynamic visualization window limits to all subplots
# Window is centered on each location's x/c and y/c using the offset values
for row in range(n_locations):
    x_c_center = data_list[row]['x_c_actual']
    y_c_center = data_list[row]['y_actual']
    xlim = [x_c_center - VIZ_XLIM_OFFSET[0], x_c_center + VIZ_XLIM_OFFSET[1]]
    ylim = [y_c_center - VIZ_YLIM_OFFSET[0], y_c_center + VIZ_YLIM_OFFSET[1]]
    axes[row, 0].set_xlim(xlim)
    axes[row, 0].set_ylim(ylim)
    axes[row, 1].set_xlim(xlim)
    axes[row, 1].set_ylim(ylim)

print(f"Analyzed {n_locations} chord locations\n")

# ============================================================================
# FIGURE 5: Spanwise Variation (Multiple z-Slices)
# ============================================================================
# WHAT WE'RE PLOTTING:
# - Correlation at different spanwise separations (Δz)
# - Shows how correlation decays as we move away from the reference z-plane
# - Demonstrates spanwise coherence of wall-shear/velocity coupling
#
# PURPOSE:
# - Quantifies spanwise extent of correlation structure
# - At Δz=0: strongest coupling (same spanwise location)
# - At Δz→max: coupling weakest (maximum spanwise separation)

# Select multiple z-slices for comparison
z_indices = [0, Nz//8, Nz//4, Nz//2]

# Compute spanwise separations (relative to first z-plane)
z_ref = z[0, 0, 0]  # Reference z-coordinate (first plane)
z_separations = [z[z_idx, 0, 0] - z_ref for z_idx in z_indices]

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)
levels = np.linspace(-1.0, 1.0, 21)

for i, z_idx in enumerate(z_indices):
    ax = axes[i]
    R_slice = R[z_idx, :, :]
    dz_separation = z_separations[i]

    im = ax.contourf(
        x_2d, y_2d, R_slice,
        levels=levels,
        cmap='RdBu_r',
        vmin=-1,
        vmax=1
    )

    ax.plot(
        x_c_actual, y_actual,
        marker='*', color='k',
        markersize=12, zorder=5
    )

    # Show both z-index and spanwise separation
    if z_idx == 0:
        title_str = rf'Index $z={z_idx}$: $\Delta z = {dz_separation:.5f}$ (reference)'
    else:
        title_str = rf'Index $z={z_idx}$: $\Delta z = {dz_separation:.5f}$'
    ax.set_title(title_str, fontsize=10)

    ax.set_xlabel('x/c', fontsize=11)
    if i == 0:
        ax.set_ylabel('y/c', fontsize=11)
    else:
        ax.set_ylabel('')

    ax.set_aspect('equal', adjustable='box')

cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.025, pad=0.02)
cbar.set_label(r'Correlation coefficient, $R$', fontsize=11)
cbar.set_ticks(np.linspace(-1, 1, 5))

# Apply visualization window limits to all 4 subplots
# Window is relative to reference position (x_c_actual, y_actual)
xlim = [x_c_actual - VIZ_XLIM_OFFSET[0], x_c_actual + VIZ_XLIM_OFFSET[1]]
ylim = [y_actual - VIZ_YLIM_OFFSET[0], y_actual + VIZ_YLIM_OFFSET[1]]
for i, ax in enumerate(axes):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

fig.suptitle('Correlation field at different spanwise separations', fontsize=14)

# ============================================================================
# FIGURE 5B: Spanwise Variation at Multiple Chord Locations
# ============================================================================
# WHAT WE'RE PLOTTING:
# - Multi-location comparison of spanwise correlation decay
# - One row per chord location (x/c)
# - Four columns showing z_indices = [0, Nz//8, Nz//4, Nz//2]
# - Common color scale across all locations and separations
#
# PURPOSE:
# - Reveals how spanwise coherence structure varies across chord positions
# - Shows if correlation decay rate is consistent or depends on location
# - Demonstrates spanwise extent of correlation at different airfoil sections

# Reuse h5_files and results_dir from FIGURE 4B section
# Load all files for FIGURE 5B
data_fig5b = []
for h5_file in h5_files:
    file_path = os.path.join(results_dir, h5_file)
    with h5py.File(file_path, 'r') as f:
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
            'filename': h5_file
        })

# Compute global correlation limits across all data
R_all_5b = np.concatenate([d['R'].ravel() for d in data_fig5b])
R_min_5b = np.min(R_all_5b)
R_max_5b = np.max(R_all_5b)

print(f"\nFIGURE 5B: Global correlation range across all locations: [{R_min_5b:.4f}, {R_max_5b:.4f}]")

# Create figure with rows for each location, columns for each z-index
n_rows_5b = len(data_fig5b)
n_cols_5b = 4
fig, axes = plt.subplots(n_rows_5b, n_cols_5b, figsize=(16, 4 * n_rows_5b), constrained_layout=True)

# Ensure axes is 2D even for single row
if n_rows_5b == 1:
    axes = axes.reshape(1, -1)

# Color levels for all subplots
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

        # Plot contourf with global limits
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

        # x/c label on bottom row only, or all if you prefer
        if row_idx == n_rows_5b - 1:
            ax.set_xlabel('x/c', fontsize=10)
        else:
            ax.set_xlabel('')

        # y/c label only on first column
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

# Single shared colorbar
cbar_5b = fig.colorbar(im_5b_last, ax=axes, orientation='vertical',
                        fraction=0.015, pad=0.02)
cbar_5b.set_label(r'Correlation coefficient, $R$', fontsize=11)
cbar_5b.set_ticks(np.linspace(-1, 1, 5))

# Apply dynamic visualization window limits to all subplots
# Window is centered on each location's x/c and y/c using the offset values
for row_idx in range(n_rows_5b):
    x_c_center = data_fig5b[row_idx]['x_c']
    y_c_center = data_fig5b[row_idx]['y_c']
    xlim = [x_c_center - VIZ_XLIM_OFFSET[0], x_c_center + VIZ_XLIM_OFFSET[1]]
    ylim = [y_c_center - VIZ_YLIM_OFFSET[0], y_c_center + VIZ_YLIM_OFFSET[1]]
    for col_idx in range(4):
        axes[row_idx, col_idx].set_xlim(xlim)
        axes[row_idx, col_idx].set_ylim(ylim)

fig.suptitle('Spanwise correlation decay at different chord locations', fontsize=14)

print(f"Analyzed {n_rows_5b} chord locations with 4 spanwise separations each")

# ============================================================================
# FIGURE 6: Spanwise Decay at Different Wall-Normal Heights
# ============================================================================
# WHAT WE'RE PLOTTING:
# - Raw correlation curves R(Δz) at different wall-normal positions (y-coordinates)
# - Shows how spanwise coherence varies with height above the wall
# - RECENTERED: Maximum correlation (Δz=0) is in the middle of the plot
#
# PURPOSE:
# - Investigates if spanwise decay is uniform across wall-normal heights
# - Identifies which heights maintain strongest spanwise coherence
# - Reveals vertical structure of spanwise correlation decay
# - More intuitive visualization for periodic correlation functions

# Compute spanwise separations for all planes
z_ref = z[0, 0, 0]  # Reference z-coordinate
dz_all = np.array([z[iz, 0, 0] - z_ref for iz in range(Nz)])

# ========== RECENTER LAG COORDINATE ==========
# For periodic correlation functions, shift so Δz=0 is centered
# Create centered lag indices: -N/2, ..., -1, 0, 1, ..., N/2-1
dz_spacing = dz_all[1] - dz_all[0] if Nz > 1 else 1.0  # Grid spacing
centered_indices = np.arange(-Nz // 2, Nz // 2)  # Centered indices
dz_centered = centered_indices * dz_spacing  # Map to actual lag values

shift = Nz // 2  # Number of positions to shift for data reordering

print(f"\nRecentering lag coordinate (periodic correlation):")
print(f"  Original: Δz[0]={dz_all[0]:.6f}, Δz[{Nz//2}]={dz_all[Nz//2]:.6f}, Δz[-1]={dz_all[-1]:.6f}")
print(f"  Centered: Δz ranges from {dz_centered[0]:.6f} to {dz_centered[-1]:.6f}")
print(f"  Grid spacing: {dz_spacing:.6f}, Nz={Nz}, shift amount={shift}")

# Find reference x-index by finding closest x-coordinate to x_c_actual
# (Using x-axis of first row to find the streamwise location)
x_at_first_row = x[0, 0, :]  # (Nx,) - x-coordinates along streamwise direction at first y
ix_ref = np.argmin(np.abs(x_at_first_row - x_c_actual))

print(f"\nFinding probe y-coordinates in mesh:")
print(f"  Reference x-index (x/c closest to {x_c_actual:.4f}): {ix_ref}, actual x/c = {x_at_first_row[ix_ref]:.4f}")

# Map probe coordinates to mesh y-indices
probe_iy_indices = []
probe_y_actual = []

for y_target in PROBE_Y_COORDS:
    # Find nearest y-index for this y-coordinate
    y_2d_at_ref_x = y[0, :, ix_ref]
    iy_nearest = np.argmin(np.abs(y_2d_at_ref_x - y_target))
    y_actual_val = y_2d_at_ref_x[iy_nearest]
    probe_iy_indices.append(iy_nearest)
    probe_y_actual.append(y_actual_val)
    print(f"  Target y={y_target:.3f} → Index {iy_nearest}, Actual y={y_actual_val:.6f}")

# Create figure with correlation curves at different y-levels
fig, ax = plt.subplots(figsize=(12, 7))

# Color map for different y-levels (from near-wall to outer)
colors = plt.cm.viridis(np.linspace(0, 1, len(PROBE_Y_COORDS)))

# Plot correlation curves for each probe height
for idx, (iy, y_val, color) in enumerate(zip(probe_iy_indices, probe_y_actual, colors)):
    # Extract correlation curve at this y-position across all Δz
    corr_at_y = R[:, iy, ix_ref]

    # Recenter the correlation curve (same shift as dz array)
    corr_at_y_centered = np.roll(corr_at_y, shift)

    # Plot the centered curve (x/c is fixed, so only label y-coordinate)
    ax.plot(dz_centered, corr_at_y_centered, 'o-', linewidth=2.0, markersize=3,
            color=color, alpha=0.8, label=f'y = {y_val:.3f}')

# Add reference lines (structural guides, not in legend)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
ax.axvline(x=0, color='green', linestyle='--', linewidth=2.0, alpha=0.5)

# Labels and formatting
ax.set_xlabel(r'Spanwise separation, $\Delta z$', fontsize=12, fontweight='bold')
ax.set_ylabel(r'Correlation coefficient, $R$', fontsize=12, fontweight='bold')
ax.set_title(r'Spanwise Decay at Different Wall-Normal Heights (x/c = {:.3f})'.format(x_c_actual),
             fontsize=13, fontweight='bold')

ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=1, framealpha=0.95)

fig.tight_layout()

# ============================================================================
# Summary Statistics
# ============================================================================
print(f"\n" + "=" * 70)
print("UNCONDITIONAL WALL-SHEAR CORRELATION ANALYSIS SUMMARY")
print("=" * 70)
print(f"\nVISUALIZATIONS GENERATED:")
print(f"  FIGURE 4:  Single-location u'_rms vs correlation comparison (Δz=0)")
print(f"  FIGURE 4B: Multi-location u'_rms vs correlation comparison across x/c")
print(f"  FIGURE 5:  Single-location spanwise correlation decay (4 z-indices)")
print(f"  FIGURE 5B: Multi-location spanwise correlation decay across x/c")
print(f"  FIGURE 6:  Spanwise decay curves at different wall-normal heights")
print(f"\nREFERENCE POINT (for Figure 4, 5, and 6):")
print(f"  x/c = {x_c_actual:.4f}, y = {y_actual:.4f}, x-index = {ix_ref}")
print(f"\nVISUALIZATION WINDOW CONFIGURATION:")
print(f"  X-window offset: [{VIZ_XLIM_OFFSET[0]:.2f}, {VIZ_XLIM_OFFSET[1]:.2f}]")
print(f"  Y-window offset: [{VIZ_YLIM_OFFSET[0]:.2f}, {VIZ_YLIM_OFFSET[1]:.2f}]")
print(f"\nSPANWISE SEPARATION RANGE (for Figure 5, 5B, 6):")
print(f"  Δz_min = {dz_all[0]:.6f} (reference plane)")
print(f"  Δz_max = {dz_all[-1]:.6f} (maximum physical separation)")
print(f"  Total extent = {dz_all[-1] - dz_all[0]:.6f}")

print(f"\nWALL-NORMAL PROBE POSITIONS (for Figure 6):")
for y_target, y_actual_val, iy in zip(PROBE_Y_COORDS, probe_y_actual, probe_iy_indices):
    corr_at_ref_z = R[0, iy, ix_ref]
    print(f"  Target y={y_target:.3f} → Actual y={y_actual_val:.6f}, R(Δz=0)={corr_at_ref_z:.4f}")

# Show all figures
plt.show()

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)
print("All plots displayed with plt.show()")
