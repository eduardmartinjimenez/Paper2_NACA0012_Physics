import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
import glob

# ============================================================================
# Configuration
# ============================================================================
# TKE data
TKE_DATA_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
TKE_DATA_NAME = "tke_turbulent_kinetic_energy.h5"
TKE_DATA_FILE = os.path.join(TKE_DATA_PATH, TKE_DATA_NAME)

# Correlation data path
CORR_DATA_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Wall_shear_correlations/test_3"

# Output directory
OUTPUT_DIR = os.path.join(CORR_DATA_PATH, "Figures_Correlation_with_TKE_Overlay")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Figure size
FIG_SIZE = (16, 7)

# Number of contour levels for correlation
NLEVELS_CORR = 25

# ============================================================================
# Domain window relative to the reference point (x_c_actual, y_actual)
# The axis limits become: [ref + D*_MIN, ref + D*_MAX]
# Set to None to use the full data range on that axis
# ============================================================================
DX_MIN, DX_MAX = -0.5, 1.5
DY_MIN, DY_MAX = -0.05, 0.99

# ============================================================================
# TKE Overlay Configuration (as percentages of TKE range)
# ============================================================================
# Specify TKE contours as percentages of the range [tke_min, tke_max]
# These will be overlaid as line contours on the correlation maps
# Example: TKE_OVERLAY_PERCENTAGES = [25, 50, 75] for 25%, 50%, 75% of range
# Set to None to disable TKE overlay

# TKE_OVERLAY_PERCENTAGES = None  # Set to list of percentages or None to disable
TKE_OVERLAY_PERCENTAGES = [90]  # Example: 50%, 75% of TKE range

# ============================================================================
# Load TKE Data
# ============================================================================
print("=" * 70)
print("LOADING TKE DATA")
print("=" * 70)

if not os.path.exists(TKE_DATA_FILE):
    raise FileNotFoundError(f"TKE data file not found: {TKE_DATA_FILE}")

with h5py.File(TKE_DATA_FILE, "r") as f:
    tke = f["tke"][:]
    x_coords = f["x"][:]
    y_coords = f["y"][:]
    u_infty = f.attrs["u_infty"]
    AOA = f.attrs["AOA"]

# Normalize TKE
tke_norm = tke / (u_infty ** 2)
tke_norm_pos = np.where(tke_norm > 0, tke_norm, np.nan)

print(f"✓ TKE data loaded: {tke_norm.shape}")
print(f"  Range: [{np.nanmin(tke_norm_pos):.6e}, {np.nanmax(tke_norm_pos):.6e}]")

# TKE range for overlay
tke_min = np.nanmin(tke_norm_pos)
tke_max = np.nanmax(tke_norm_pos)

# ============================================================================
# Find All Correlation Files
# ============================================================================
print("\n" + "=" * 70)
print("FINDING CORRELATION FILES")
print("=" * 70)

corr_files = sorted(glob.glob(os.path.join(CORR_DATA_PATH, "wall_shear_correlation_xc_*_alpha_1.0_all_fft.h5")))

if not corr_files:
    raise FileNotFoundError(f"No correlation files found")

print(f"Found {len(corr_files)} files")

# ============================================================================
# Helper: Draw correlation panel with TKE overlay
# ============================================================================
def draw_correlation_with_tke_overlay(ax, fig, x_grid, y_grid, R, tke_norm, x_coords_tke, y_coords_tke,
                                       x_c_actual, y_actual, panel_title):
    """Draw a correlation contour panel with TKE energy overlay."""
    # Plot correlation data
    R_finite = R[np.isfinite(R)]
    vmin_local = float(np.min(R_finite))
    vmax_local = float(np.max(R_finite))
    levels_corr = np.linspace(vmin_local, vmax_local, NLEVELS_CORR)

    im = ax.contourf(x_grid, y_grid, R, levels=levels_corr, cmap='RdBu_r', extend='both')
    ax.contour(x_grid, y_grid, R, levels=10, colors='black', alpha=0.25, linewidths=0.4)

    # Overlay TKE contours if specified
    if TKE_OVERLAY_PERCENTAGES is not None:
        # Convert percentages to actual TKE values
        tke_contour_levels = []
        for pct in TKE_OVERLAY_PERCENTAGES:
            log_min = np.log10(tke_min)
            log_max = np.log10(tke_max)
            log_value = log_min + (pct / 100.0) * (log_max - log_min)
            tke_contour_levels.append(10 ** log_value)

        if tke_contour_levels:
            cs_tke = ax.contour(x_coords_tke, y_coords_tke, tke_norm, levels=tke_contour_levels,
                               colors='lime', linewidths=2.0, alpha=0.9, zorder=3, linestyles='--')
            # Add labels for TKE contours
            fmt_dict_tke = {lvl: f'{pct}%' for lvl, pct in zip(tke_contour_levels, TKE_OVERLAY_PERCENTAGES)}
            ax.clabel(cs_tke, inline=True, fontsize=8, fmt=fmt_dict_tke)

    # Mark reference point
    ax.plot(x_c_actual, y_actual, 'g*', markersize=14,
            markeredgecolor='black', markeredgewidth=1.0, zorder=5)

    ax.set_title(panel_title, fontsize=12, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('$x/c$', fontsize=11)
    ax.set_ylabel('$y/c$', fontsize=11)
    ax.grid(True, alpha=0.15)

    # Apply domain window
    if DX_MIN is not None or DX_MAX is not None:
        ax.set_xlim(
            x_c_actual + DX_MIN if DX_MIN is not None else None,
            x_c_actual + DX_MAX if DX_MAX is not None else None,
        )
    if DY_MIN is not None or DY_MAX is not None:
        ax.set_ylim(
            y_actual + DY_MIN if DY_MIN is not None else None,
            y_actual + DY_MAX if DY_MAX is not None else None,
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label('$R$', fontsize=10)
    tick_vals = sorted({vmin_local, 0.0, vmax_local})
    cbar.set_ticks(tick_vals)
    cbar.ax.tick_params(labelsize=8)

    return im

# ============================================================================
# Create Plots for Each x/c Location
# ============================================================================
print("\n" + "=" * 70)
print("CREATING VISUALIZATIONS")
print("=" * 70)

for corr_file in corr_files:
    filename = os.path.basename(corr_file)
    print(f"\nProcessing: {filename}")

    with h5py.File(corr_file, 'r') as f:
        R_PF = f['R_PF'][0, :, :]      # z=0 slice
        R_NF = f['R_NF'][0, :, :]
        x_grid = f['x'][0, :, :]
        y_grid = f['y'][0, :, :]
        x_c_actual = f.attrs['x_c_actual']
        y_actual = f.attrs['y_actual']
        N_PF = f.attrs['N_PF']
        N_NF = f.attrs['N_NF']

    # ====================================================================
    # Create 1x2 Figure (R_PF with TKE | R_NF with TKE)
    # ====================================================================
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE)

    # Panel 1: R_PF with TKE overlay
    draw_correlation_with_tke_overlay(
        axes[0], fig, x_grid, y_grid, R_PF, tke_norm_pos, x_coords, y_coords,
        x_c_actual, y_actual,
        f'PF-conditioned with TKE overlay ($N_{{PF}}={N_PF}$)'
    )

    # Panel 2: R_NF with TKE overlay
    draw_correlation_with_tke_overlay(
        axes[1], fig, x_grid, y_grid, R_NF, tke_norm_pos, x_coords, y_coords,
        x_c_actual, y_actual,
        f'NF-conditioned with TKE overlay ($N_{{NF}}={N_NF}$)'
    )

    fig.suptitle(
        f'$x/c = {float(x_c_actual):.4f}$  —  $\\alpha = {AOA}°$  —  $\\Delta z = 0$  —  TKE overlay (dashed green)',
        fontsize=14, fontweight='bold',
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    output_name = f"correlation_with_tke_overlay_xc_{x_c_actual:.4f}.png"
    output_path = os.path.join(OUTPUT_DIR, output_name)
    # plt.savefig(output_path, dpi=150, bbox_inches="tight")
    # print(f"  ✓ Saved: {output_name}")
    # plt.close()
    plt.show()

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Domain window: DX=[{DX_MIN}, {DX_MAX}], DY=[{DY_MIN}, {DY_MAX}]")
print(f"  Correlation contour levels: {NLEVELS_CORR}")
print(f"  Figure size: {FIG_SIZE}")
print(f"\nTKE overlay:")
if TKE_OVERLAY_PERCENTAGES is not None:
    print(f"  Percentages: {TKE_OVERLAY_PERCENTAGES}%")
else:
    print(f"  Disabled (set TKE_OVERLAY_PERCENTAGES to enable)")
print(f"  TKE range: [{tke_min:.6e}, {tke_max:.6e}]")
print(f"\nOutput directory:")
print(f"  {OUTPUT_DIR}")
print(f"\nTotal files generated: {len(corr_files)}")
