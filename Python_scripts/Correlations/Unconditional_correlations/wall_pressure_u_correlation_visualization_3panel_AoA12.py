"""
Wall-Pressure & Streamwise Velocity Correlation Visualization Script (3x1 Panel)

This script generates a 3-panel visualization of unconditional correlations between
wall-pressure (p') and streamwise velocity (u') fluctuations at three chord
locations (x/c = 0.5, 0.7, 0.9).

Features:
- AOA-based coordinate rotation (aligns field with local flow direction)
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
    "Wall_pressure_correlations/test_4/"
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
VIZ_XLIM_OFFSET = [0.1, 0.35]   # Symmetric x-window
VIZ_YLIM_OFFSET = [0.07, 0.15]  # Extended y-window to show airfoil surface

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
h5_files_all = sorted(glob.glob(os.path.join(RESULTS_DIR, "wall_pressure_correlation_unconditional_xc_*.h5")))

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
# Load Airfoil Geometry
# ============================================================================

print("\n" + "="*70)
print("LOADING AIRFOIL GEOMETRY")
print("="*70)

try:
    with h5py.File(GEO_FILE, "r") as f:
        interface_points = f["interface_points"][:]

    x_interface = interface_points[:, 0]
    y_interface = interface_points[:, 1]

    # Separate upper (suction) and lower (pressure) surfaces
    y_mean = np.mean(y_interface)
    upper_mask = y_interface > y_mean
    lower_mask = ~upper_mask

    x_airfoil_upper = x_interface[upper_mask]
    y_airfoil_upper = y_interface[upper_mask]

    x_airfoil_lower = x_interface[lower_mask]
    y_airfoil_lower = y_interface[lower_mask]

    # Note: Will apply rotation per reference point when plotting
    print(f"Loaded airfoil geometry with {len(interface_points)} surface points")
    print(f"Upper surface (suction side): {len(x_airfoil_upper)} points")
    print(f"Lower surface (pressure side): {len(x_airfoil_lower)} points")
    print(f"Airfoil x range (unrotated): [{np.min(x_airfoil_upper):.4f}, {np.max(x_airfoil_upper):.4f}]")
    print(f"Airfoil y range (unrotated): [{np.min(np.concatenate([y_airfoil_upper, y_airfoil_lower])):.4f}, {np.max(np.concatenate([y_airfoil_upper, y_airfoil_lower])):.4f}]")

except FileNotFoundError:
    print(f"WARNING: Geometry file not found at {GEO_FILE}")
    print("Proceeding without airfoil surface display")
    x_airfoil_upper = None
    y_airfoil_upper = None
    x_airfoil_lower = None
    y_airfoil_lower = None

# ============================================================================
# Load Data and Apply AOA Rotation
# ============================================================================

print("\n" + "="*70)
print("LOADING DATA AND APPLYING AOA ROTATION")
print("="*70)

data_list = []
for h5_file in h5_files:
    with h5py.File(h5_file, 'r') as f:
        # Load correlation field (Δz=0 slice)
        R_original = f['R'][0, :, :]  # (Ny, Nx)

        # Load coordinates
        x_original = f['x'][0, :, :]  # (Ny, Nx)
        y_original = f['y'][0, :, :]  # (Ny, Nx)

        # Get reference point
        x_c_actual = f.attrs['x_c_actual']
        y_c_actual = f.attrs['y_actual']

        # Apply AOA rotation to coordinates and field
        # Rotation matrix: [cos(AOA)   sin(AOA) ]
        #                  [-sin(AOA)  cos(AOA) ]
        cos_aoa = np.cos(AOA_rad)
        sin_aoa = np.sin(AOA_rad)

        # Rotate coordinates relative to reference point
        x_centered = x_original - x_c_actual
        y_centered = y_original - y_c_actual

        x_rotated = x_centered * cos_aoa + y_centered * sin_aoa
        y_rotated = -x_centered * sin_aoa + y_centered * cos_aoa

        # Shift back to reference point
        x_rot_final = x_rotated + x_c_actual
        y_rot_final = y_rotated + y_c_actual

        data_dict = {
            'R': R_original,
            'x': x_rot_final,
            'y': y_rot_final,
            'x_original': x_original,
            'y_original': y_original,
            'x_c_actual': x_c_actual,
            'y_c_actual': y_c_actual,
            'u_infty': f.attrs.get('u_infty', 1.0),
        }
        data_list.append(data_dict)

print(f"Loaded and rotated {len(data_list)} correlation fields by AOA = {AOA}°")

# ============================================================================
# Create 3x1 Panel Visualization
# ============================================================================

print("\n" + "="*70)
print("CREATING 3X1 PANEL VISUALIZATION")
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
print(f"  Fixed range [0, 1]: [0.0000, 1.0000]")
print(f"  Dynamic range [0, max]: [0.0000, {r_global_max:.4f}]")

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

# ============================================================================
# PLOT 1: Create 3x1 Panel Visualization with Fixed Range [0, 1]
# ============================================================================

print("\n" + "="*70)
print("CREATING PLOT 1: FIXED RANGE [0, 1]")
print("="*70)

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
    x_2d = data['x']
    y_2d = data['y']
    x_c = data['x_c_actual']
    y_c = data['y_c_actual']

    # Plot correlation field with range [0, 1]
    im_last = ax.contourf(
        x_2d, y_2d, R_2d,
        levels=levels_r,
        cmap=corr_cmap,
        vmin=0.0,
        vmax=1.0
    )

    # Add airfoil surface overlay if available
    if x_airfoil_upper is not None and y_airfoil_upper is not None:
        # Apply AOA rotation to upper surface around the reference point
        cos_aoa = np.cos(AOA_rad)
        sin_aoa = np.sin(AOA_rad)

        # Center and rotate upper surface
        x_air_centered = x_airfoil_upper - x_c
        y_air_centered = y_airfoil_upper - y_c
        x_air_rot = x_air_centered * cos_aoa + y_air_centered * sin_aoa
        y_air_rot = -x_air_centered * sin_aoa + y_air_centered * cos_aoa
        x_air_final_upper = x_air_rot + x_c
        y_air_final_upper = y_air_rot + y_c

        # Center and rotate lower surface
        x_air_centered = x_airfoil_lower - x_c
        y_air_centered = y_airfoil_lower - y_c
        x_air_rot = x_air_centered * cos_aoa + y_air_centered * sin_aoa
        y_air_rot = -x_air_centered * sin_aoa + y_air_centered * cos_aoa
        x_air_final_lower = x_air_rot + x_c
        y_air_final_lower = y_air_rot + y_c

        # Sort upper surface by x-coordinate
        sort_idx_upper = np.argsort(x_air_final_upper)
        x_upper_sorted = x_air_final_upper[sort_idx_upper]
        y_upper_sorted = y_air_final_upper[sort_idx_upper]

        # Sort lower surface by x-coordinate (reverse for closing polygon)
        sort_idx_lower = np.argsort(x_air_final_lower)
        x_lower_sorted = x_air_final_lower[sort_idx_lower]
        y_lower_sorted = y_air_final_lower[sort_idx_lower]

        # Create closed polygon: upper surface + reversed lower surface
        x_polygon = np.concatenate([x_upper_sorted, x_lower_sorted[::-1]])
        y_polygon = np.concatenate([y_upper_sorted, y_lower_sorted[::-1]])

        # Fill the airfoil interior with light gray
        # ax.fill(x_polygon, y_polygon, color='#D3D3D3', alpha=0.6, zorder=11)
        ax.fill(x_polygon, y_polygon, color='0.85', alpha=0.35, zorder=11)

        # Plot upper surface outline
        ax.plot(x_upper_sorted, y_upper_sorted, 'k-', linewidth=1.0,
                zorder=12, label='Airfoil surface')

        # Plot lower surface outline
        ax.plot(x_lower_sorted, y_lower_sorted, 'k-', linewidth=1.0,
                zorder=12)

    # Add contour line at 0.5 for structural reference
    ax.contour(
        x_2d, y_2d, R_2d,
        levels=[0.25],
        colors='black',
        linewidths=0.8,
        alpha=0.7,
        linestyles='dashdot'
    )

    # Mark reference point
    ax.plot(x_c, y_c, marker='*', color='k', markersize=12, zorder=13)

    # Title for each panel showing x/c location
    ax.set_title(
        rf'$x/c = {xc_values[col]:.2f}$',
        fontsize=12
    )

    # x-label on all bottom panels
    ax.set_xlabel('x/c', fontsize=11)

    # y-label only on first column
    if col == 0:
        ax.set_ylabel('y/c', fontsize=11)
    else:
        ax.set_ylabel('')

    ax.set_aspect('equal', adjustable='box')

    # Apply dynamic visualization window
    xlim = [x_c - VIZ_XLIM_OFFSET[0], x_c + VIZ_XLIM_OFFSET[1]]
    ylim = [y_c - VIZ_YLIM_OFFSET[0], y_c + VIZ_YLIM_OFFSET[1]]

    print(f"\nPanel {col+1} (x/c={xc_values[col]:.3f}):")
    print(f"  Reference point: x={x_c:.4f}, y={y_c:.4f}")
    print(f"  Visualization window: x={xlim[0]:.4f} to {xlim[1]:.4f}, y={ylim[0]:.4f} to {ylim[1]:.4f}")
    if x_airfoil_upper is not None:
        airfoil_in_window = np.any((x_airfoil_upper >= xlim[0]) & (x_airfoil_upper <= xlim[1]) &
                                    (y_airfoil_upper >= ylim[0]) & (y_airfoil_upper <= ylim[1]))
        print(f"  Airfoil visible: {airfoil_in_window}")

    ax.set_xlim(xlim)
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
cbar.ax.set_title(r'$R_{p^\prime u^\prime}$', fontsize=14, pad=10)

# Save figure
output_path_png = os.path.join(OUTPUT_DIR, "wall_pressure_u_correlation_1x3panel_AOA12.png")
output_path_eps = os.path.join(OUTPUT_DIR, "wall_pressure_u_correlation_1x3panel_AOA12.eps")
fig.savefig(output_path_png, dpi=300, bbox_inches='tight')
fig.savefig(output_path_eps, dpi=300, bbox_inches='tight')
print(f"\nPlot 1 saved: {output_path_png}")

# ============================================================================
# PLOT 2: Create 3x1 Panel Visualization with Dynamic Range [0, max]
# ============================================================================

print("\n" + "="*70)
print("CREATING PLOT 2: DYNAMIC RANGE [0, {:.4f}]".format(r_global_max))
print("="*70)

# Use levels based on actual data range
levels_r_dynamic = np.linspace(r_global_min, r_global_max, 101)

# Create figure with 1 row, 3 columns
fig2, axes2 = plt.subplots(1, n_locations, figsize=(12, 5), constrained_layout=True)

# Ensure axes is iterable even for single subplot
if n_locations == 1:
    axes2 = [axes2]

# Set background color for each axis panel
for ax in axes2:
    ax.set_facecolor(BACKGROUND_COLOR)

# Track last image for colorbar
im_last2 = None

# Plot each location
for col, data in enumerate(data_list):
    ax = axes2[col]

    R_2d = data['R']
    x_2d = data['x']
    y_2d = data['y']
    x_c = data['x_c_actual']
    y_c = data['y_c_actual']

    # Plot correlation field with dynamic range [0, r_global_max]
    im_last2 = ax.contourf(
        x_2d, y_2d, R_2d,
        levels=levels_r_dynamic,
        cmap=corr_cmap,
        vmin=r_global_min,
        vmax=r_global_max
    )

    # Add airfoil surface overlay if available
    if x_airfoil_upper is not None and y_airfoil_upper is not None:
        # Apply AOA rotation to upper surface around the reference point
        cos_aoa = np.cos(AOA_rad)
        sin_aoa = np.sin(AOA_rad)

        # Center and rotate upper surface
        x_air_centered = x_airfoil_upper - x_c
        y_air_centered = y_airfoil_upper - y_c
        x_air_rot = x_air_centered * cos_aoa + y_air_centered * sin_aoa
        y_air_rot = -x_air_centered * sin_aoa + y_air_centered * cos_aoa
        x_air_final_upper = x_air_rot + x_c
        y_air_final_upper = y_air_rot + y_c

        # Center and rotate lower surface
        x_air_centered = x_airfoil_lower - x_c
        y_air_centered = y_airfoil_lower - y_c
        x_air_rot = x_air_centered * cos_aoa + y_air_centered * sin_aoa
        y_air_rot = -x_air_centered * sin_aoa + y_air_centered * cos_aoa
        x_air_final_lower = x_air_rot + x_c
        y_air_final_lower = y_air_rot + y_c

        # Sort upper surface by x-coordinate
        sort_idx_upper = np.argsort(x_air_final_upper)
        x_upper_sorted = x_air_final_upper[sort_idx_upper]
        y_upper_sorted = y_air_final_upper[sort_idx_upper]

        # Sort lower surface by x-coordinate (reverse for closing polygon)
        sort_idx_lower = np.argsort(x_air_final_lower)
        x_lower_sorted = x_air_final_lower[sort_idx_lower]
        y_lower_sorted = y_air_final_lower[sort_idx_lower]

        # Create closed polygon: upper surface + reversed lower surface
        x_polygon = np.concatenate([x_upper_sorted, x_lower_sorted[::-1]])
        y_polygon = np.concatenate([y_upper_sorted, y_lower_sorted[::-1]])

        # Fill the airfoil interior with light gray
        ax.fill(x_polygon, y_polygon, color='0.85', alpha=0.35, zorder=11)

        # Plot upper surface outline
        ax.plot(x_upper_sorted, y_upper_sorted, 'k-', linewidth=1.0,
                zorder=12, label='Airfoil surface')

        # Plot lower surface outline
        ax.plot(x_lower_sorted, y_lower_sorted, 'k-', linewidth=1.0,
                zorder=12)

    # # Add contour line at 50% of max correlation for structural reference
    # contour_level = r_global_max * 0.5
    # ax.contour(
    #     x_2d, y_2d, R_2d,
    #     levels=[contour_level],
    #     colors='black',
    #     linewidths=0.8,
    #     alpha=0.7,
    #     linestyles='dashdot'
    # )

    # Add contour line at 0.5 for structural reference
    ax.contour(
        x_2d, y_2d, R_2d,
        levels=[0.3],
        colors='black',
        linewidths=0.8,
        alpha=0.7,
        linestyles='dashdot'
    )

    # Mark reference point
    ax.plot(x_c, y_c, marker='*', color='k', markersize=12, zorder=13)

    # Title for each panel showing x/c location
    ax.set_title(
        rf'$x/c = {xc_values[col]:.2f}$',
        fontsize=12
    )

    # x-label on all bottom panels
    ax.set_xlabel('x/c', fontsize=11)

    # y-label only on first column
    if col == 0:
        ax.set_ylabel('y/c', fontsize=11)
    else:
        ax.set_ylabel('')

    ax.set_aspect('equal', adjustable='box')

    # Apply dynamic visualization window
    xlim = [x_c - VIZ_XLIM_OFFSET[0], x_c + VIZ_XLIM_OFFSET[1]]
    ylim = [y_c - VIZ_YLIM_OFFSET[0], y_c + VIZ_YLIM_OFFSET[1]]

    print(f"\nPanel {col+1} (x/c={xc_values[col]:.3f}):")
    print(f"  Reference point: x={x_c:.4f}, y={y_c:.4f}")
    print(f"  Visualization window: x={xlim[0]:.4f} to {xlim[1]:.4f}, y={ylim[0]:.4f} to {ylim[1]:.4f}")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# Add colorbar
cbar2 = fig2.colorbar(
    im_last2,
    ax=axes2,
    fraction=0.046,
    pad=0.04,
    orientation='vertical',
    shrink=0.4
)

# Put colorbar label at the top, horizontally
cbar_ticks = np.linspace(r_global_min, r_global_max, 6)
cbar2.set_ticks(cbar_ticks)
cbar2.ax.set_title(r'$R_{p^\prime u^\prime}$', fontsize=14, pad=10)

# Save figure
output_path_png2 = os.path.join(OUTPUT_DIR, "wall_pressure_u_correlation_1x3panel_AOA12_dynamic_range.png")
output_path_eps2 = os.path.join(OUTPUT_DIR, "wall_pressure_u_correlation_1x3panel_AOA12_dynamic_range.eps")
fig2.savefig(output_path_png2, dpi=300, bbox_inches='tight')
fig2.savefig(output_path_eps2, dpi=300, bbox_inches='tight')
print(f"\nPlot 2 saved: {output_path_png2}")


# ============================================================================
# Summary Statistics
# ============================================================================

print(f"\n" + "="*70)
print("VISUALIZATION SUMMARY")
print("="*70)
print(f"\nFigure Configuration:")
print(f"  Layout: 1×3 panel (one per x/c location)")
print(f"  Correlation type: p' vs u' fluctuations")
print(f"  Spanwise separation: Δz = 0")
print(f"  AOA rotation: {AOA}°")
print(f"\nChord Locations:")
for i, xc in enumerate(xc_values, 1):
    print(f"  Panel {i}: x/c = {xc:.3f}")

print(f"\nColor Scale:")
print(f"  Range: [0.0000, 1.0000]")
print(f"  Colormap: Custom fade (Light Blue → Dark Red)")
print(f"  Reference contour: 0.5 (black line)")
print(f"  Panel background: Light blue (fades from correlations)")
print(f"  Airfoil surface: Black outline (linewidth=1.0)")
print(f"  Airfoil interior: Light gray fill (alpha=0.6)")

print(f"\nVisualization Window:")
print(f"  X-offset: [{VIZ_XLIM_OFFSET[0]:.2f}, {VIZ_XLIM_OFFSET[1]:.2f}]")
print(f"  Y-offset: [{VIZ_YLIM_OFFSET[0]:.2f}, {VIZ_YLIM_OFFSET[1]:.2f}]")

print(f"\nOutput:")
print(f"  File: {output_path_eps}")

# Show figure
plt.show()

print("\n" + "="*70)
print("VISUALIZATION COMPLETE")
print("="*70)
