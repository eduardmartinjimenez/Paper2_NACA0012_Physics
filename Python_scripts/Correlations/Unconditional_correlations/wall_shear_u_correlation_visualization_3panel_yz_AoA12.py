"""
Wall-Shear & Streamwise Velocity Correlation Visualization Script (3x1 Panel, Y-Z Plane)

This script generates a 3-panel visualization of unconditional correlations between
wall-shear stress (τ_w) and streamwise velocity (u') fluctuations at three chord
locations (x/c = 0.5, 0.7, 0.9), showing the Y-Z vertical cross-section at each
reference point.

Features:
- AOA-based coordinate rotation (aligns field with local flow direction)
- Y-Z plane visualization (vertical cross-section at each reference x location)
- Correlation range fixed to [0, 1]
- Dynamic visualization window centered on each location
- Consistent color scale across all panels
"""

import glob
import re
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Shared plotting defaults
plt.rc("text", usetex=True)
plt.rc("font", size=14, family="serif")
plt.rc("text.latex", preamble=r"\usepackage{amsmath} \usepackage{amssymb}")

# ============================================================================
# Configuration
# ============================================================================

# ========== AOA 12 Configuration ==========
RESULTS_DIR = (
    "/home/jofre/Members/Eduard/Paper2/Simulations/"
    "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
    "Wall_shear_correlations/test_5/"
)

# Geometrical data file (contains airfoil surface)
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

# Output directory for saving figures
OUTPUT_DIR = os.path.join(RESULTS_DIR, "Figures")

# ============================================================================
# Analysis Parameters
# ============================================================================

u_infty = 1.0
AOA = 12  # degrees
AOA_rad = np.deg2rad(AOA)
c = 1.0  # chord length

# Target chord locations for visualization
X_C_LOCATIONS = [0.5, 0.7, 0.9]

# Visualization window offsets (relative to reference coordinates)
# Format: OFFSET = [left/bottom_extent, right/top_extent]
VIZ_YLIM_OFFSET = [0.005, 0.15]  # y-window (wall-normal direction)
VIZ_ZLIM_OFFSET = [0.1, 0.1]   # z-window (spanwise direction)

# ============================================================================
# Create Output Directory
# ============================================================================

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")
else:
    print(f"Output directory: {OUTPUT_DIR}")

# ============================================================================
# File Discovery and Filtering
# ============================================================================

print("\n" + "="*70)
print("FILE DISCOVERY")
print("="*70)

# Find all unconditional correlation files
h5_files_all = sorted(glob.glob(os.path.join(RESULTS_DIR, "wall_shear_correlation_unconditional_xc_*.h5")))

# Extract x/c values
def extract_xc(filepath):
    filename = os.path.basename(filepath)
    match = re.search(r'xc_([0-9.]+)\.h5', filename)
    if match:
        return float(match.group(1))
    return float('inf')

# Filter files to match target x/c locations
h5_files = []
for xc_target in X_C_LOCATIONS:
    for h5_file in h5_files_all:
        xc_val = extract_xc(h5_file)
        if abs(xc_val - xc_target) < 0.01:  # Allow small tolerance for floating point
            h5_files.append(h5_file)
            break

if len(h5_files) != len(X_C_LOCATIONS):
    print(f"WARNING: Found {len(h5_files)} files but expected {len(X_C_LOCATIONS)} locations")
    print(f"Expected x/c values: {X_C_LOCATIONS}")
    print(f"Found x/c values: {[extract_xc(f) for f in h5_files]}")

if not h5_files:
    print(f"ERROR: No files found matching x/c locations {X_C_LOCATIONS}")
    exit(1)

print(f"\nFound {len(h5_files)} correlation files for target locations:")
xc_values = []
for h5_file in h5_files:
    xc = extract_xc(h5_file)
    xc_values.append(xc)
    print(f"  x/c = {xc:.3f}: {os.path.basename(h5_file)}")

# ============================================================================
# Load Data and Apply AOA Rotation (Y-Z Plane)
# ============================================================================

print("\n" + "="*70)
print("LOADING DATA AND APPLYING AOA ROTATION (Y-Z PLANE)")
print("="*70)

data_list = []
for h5_file in h5_files:
    with h5py.File(h5_file, 'r') as f:
        # Load correlation field - Y-Z plane at reference x location (middle of x-dimension)
        R_all = f['R'][:]  # (Nz, Ny, Nx)

        # Find the middle x-index (reference x location)
        nx = R_all.shape[2]
        x_ref_idx = nx // 2

        # Extract Y-Z plane at reference x
        R_original = R_all[:, :, x_ref_idx]  # (Nz, Ny)

        # Load all coordinates
        z_all = f['z'][:]  # (Nz, Ny, Nx)
        y_all = f['y'][:]  # (Nz, Ny, Nx)

        # Extract Y-Z plane coordinates at reference x
        z_original = z_all[:, :, x_ref_idx]  # (Nz, Ny)
        y_original = y_all[:, :, x_ref_idx]  # (Nz, Ny)

        # Get reference point
        x_c_actual = f.attrs['x_c_actual']
        y_c_actual = f.attrs['y_actual']
        z_c_actual = 0.0

        # Recenter z-dimension for periodic correlation (Δz=0 at middle)
        # This allows symmetric visualization of spanwise correlation decay
        Nz = R_original.shape[0]
        shift = Nz // 2

        # Roll correlation field along z-dimension
        R_centered = np.roll(R_original, shift, axis=0)

        # Recenter z-coordinate spacing
        z_ref = z_original[0, 0]
        dz_all = np.array([z_original[iz, 0] - z_ref for iz in range(Nz)])
        dz_spacing = dz_all[1] - dz_all[0] if Nz > 1 else 1.0

        # Create centered z-coordinate
        centered_indices = np.arange(-Nz // 2, Nz // 2)
        z_centered = centered_indices * dz_spacing

        # Create 2D grid for z-coordinates (broadcast to match y shape)
        z_centered_2d = np.tile(z_centered[:, np.newaxis], (1, y_original.shape[1]))

        # Apply AOA rotation to Y-Z coordinates
        cos_aoa = np.cos(AOA_rad)
        sin_aoa = np.sin(AOA_rad)

        # Center y-coordinates relative to reference point
        y_centered = y_original - y_c_actual
        z_centered_vals = z_centered_2d - z_c_actual

        # Apply rotation to y-coordinates (z is spanwise, unaffected by AOA)
        y_rotated = y_centered * cos_aoa
        z_rotated = z_centered_vals

        # Shift back to reference point
        y_rot_final = y_rotated + y_c_actual
        z_rot_final = z_rotated + z_c_actual

        data_dict = {
            'R': R_centered,
            'y': y_rot_final,
            'z': z_rot_final,
            'y_original': y_original,
            'z_original': z_original,
            'x_c_actual': x_c_actual,
            'y_c_actual': y_c_actual,
            'z_c_actual': z_c_actual,
            'u_infty': f.attrs.get('u_infty', 1.0),
        }
        data_list.append(data_dict)

print(f"Loaded and rotated {len(data_list)} Y-Z correlation planes by AOA = {AOA}°")

# ============================================================================
# Create 3x1 Panel Visualization (Y-Z Plane)
# ============================================================================

print("\n" + "="*70)
print("CREATING 3X1 PANEL VISUALIZATION (Y-Z PLANE)")
print("="*70)

# Compute common color scale across all locations
r_min_all = []
r_max_all = []
for d in data_list:
    R = d['R']
    r_min_all.append(np.nanmin(R))
    r_max_all.append(np.nanmax(R))

r_global_min = max(0.0, np.min(r_min_all))  # Ensure minimum is 0
r_global_max = np.max(r_max_all)

print(f"\nCorrelation statistics across all panels:")
print(f"  Global min: {r_global_min:.4f}")
print(f"  Global max: {r_global_max:.4f}")
print(f"  Visualization range: [0.0000, 1.0000]")

# ============================================================================
# Custom Colormap: monotonic fade from background to red
# ============================================================================

BACKGROUND_COLOR = "#E6F2FF"

corr_cmap = LinearSegmentedColormap.from_list(
    "correlation_fade_red",
    [
        (0.00, BACKGROUND_COLOR),  # R = 0, same as background
        (0.20, "#D7EAF7"),
        (0.40, "#F4C7B8"),
        (0.60, "#E98A72"),
        (0.80, "#D34A4A"),
        (1.00, "#8B0000"),         # R = 1
    ],
    N=256
)

levels_r = np.linspace(0.0, 1.0, 101)

# Create figure with 1 row, 3 columns
n_locations = len(data_list)
fig, axes = plt.subplots(1, n_locations, figsize=(12, 5), constrained_layout=True)

# Ensure axes is iterable even for single subplot
if n_locations == 1:
    axes = [axes]

# Set background color for each axis panel
for ax in axes:
    ax.set_facecolor(BACKGROUND_COLOR)

# Track last image for colorbar
im_last = None

# Plot each location
for col, data in enumerate(data_list):
    ax = axes[col]

    R_2d = data['R']
    y_2d = data['y']
    z_2d = data['z']
    x_c = data['x_c_actual']
    y_c = data['y_c_actual']
    z_c = data['z_c_actual']

    # Plot correlation field with range [0, 1]
    # Note: contourf expects (X, Y, Z) where X and Y are 1D or 2D grids
    im_last = ax.contourf(
        z_2d, y_2d, R_2d,
        levels=levels_r,
        cmap=corr_cmap,
        vmin=0.0,
        vmax=1.0
    )

    # Add contour line at 0.5 for structural reference
    ax.contour(
        z_2d, y_2d, R_2d,
        levels=[0.5],
        colors='black',
        linewidths=0.8,
        alpha=0.7,
        linestyles='dashdot'
    )

    # Mark reference point
    ax.plot(z_c, y_c, marker='*', color='k', markersize=12, zorder=13)

    # Title for each panel showing x/c location
    ax.set_title(
        rf'$x/c = {xc_values[col]:.2f}$',
        fontsize=12
    )

    # Z-label (spanwise) on all panels
    ax.set_xlabel(r'$z/c$', fontsize=11)

    # Y-label (wall-normal) only on first column
    if col == 0:
        ax.set_ylabel(r'$y/c$', fontsize=11)
    else:
        ax.set_ylabel('')

    ax.set_aspect('equal', adjustable='box')

    # Apply dynamic visualization window
    zlim = [z_c - VIZ_ZLIM_OFFSET[0], z_c + VIZ_ZLIM_OFFSET[1]]
    ylim = [y_c - VIZ_YLIM_OFFSET[0], y_c + VIZ_YLIM_OFFSET[1]]

    print(f"\nPanel {col+1} (x/c={xc_values[col]:.3f}):")
    print(f"  Reference point: y={y_c:.4f}, z={z_c:.4f}")
    print(f"  Visualization window: y={ylim[0]:.4f} to {ylim[1]:.4f}, z={zlim[0]:.4f} to {zlim[1]:.4f}")

    ax.set_xlim(zlim)
    ax.set_ylim(ylim)

# Add colorbar
cbar = fig.colorbar(
    im_last,
    ax=axes,
    fraction=0.046,
    pad=0.04,
    orientation='vertical',
    shrink=0.4
)

# Put colorbar label at the top, horizontally
cbar.set_ticks(np.linspace(0, 1, 6))
cbar.ax.set_title(r'$R_{\tau_w^\prime u^\prime}$', fontsize=14, pad=10)

# Save figure
output_path_png = os.path.join(OUTPUT_DIR, "wall_shear_u_correlation_1x3panel_yz_AOA12.png")
output_path_eps = os.path.join(OUTPUT_DIR, "wall_shear_u_correlation_1x3panel_yz_AOA12.eps")
fig.savefig(output_path_png, dpi=300, bbox_inches='tight')
fig.savefig(output_path_eps, dpi=300, bbox_inches='tight')
print(f"\nVisualization saved: {output_path_png}")


# ============================================================================
# Summary Statistics
# ============================================================================

print(f"\n" + "="*70)
print("VISUALIZATION SUMMARY")
print("="*70)
print(f"\nFigure Configuration:")
print(f"  Layout: 1×3 panel (one per x/c location)")
print(f"  Correlation type: τ_w vs u' fluctuations")
print(f"  Plane: Y-Z (vertical cross-section)")
print(f"  Wall-normal separation: Variable y")
print(f"  Spanwise separation: Centered Δz (Δz=0 at middle)")
print(f"  AOA rotation: {AOA}°")
print(f"\nChord Locations:")
for i, xc in enumerate(xc_values, 1):
    print(f"  Panel {i}: x/c = {xc:.3f}")

print(f"\nColor Scale:")
print(f"  Range: [0.0000, 1.0000]")
print(f"  Colormap: Custom fade (Light Blue → Dark Red)")
print(f"  Reference contour: 0.5 (black line)")
print(f"  Panel background: Light blue (fades from correlations)")

print(f"\nVisualization Window:")
print(f"  Y-offset (wall-normal): [{VIZ_YLIM_OFFSET[0]:.2f}, {VIZ_YLIM_OFFSET[1]:.2f}]")
print(f"  Z-offset (spanwise): [{VIZ_ZLIM_OFFSET[0]:.2f}, {VIZ_ZLIM_OFFSET[1]:.2f}]")

print(f"\nOutput:")
print(f"  File: {output_path_eps}")

# Show figure
plt.show()

print("\n" + "="*70)
print("VISUALIZATION COMPLETE")
print("="*70)
