import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator, LogFormatterSciNotation, SymmetricalLogLocator, LogLocator
from matplotlib.colors import LogNorm

# ============================================================================
# Configuration
# ============================================================================
# # Path to TKE data
# TKE_DATA_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
# TKE_DATA_NAME = "tke_turbulent_kinetic_energy.h5"
# TKE_DATA_FILE = os.path.join(TKE_DATA_PATH, TKE_DATA_NAME)

# # Mesh/Geometry data
# GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data"
# GEO_NAME = "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
# GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

# # Output directory for figures
# FIGURES_OUTPUT_DIR = os.path.join(TKE_DATA_PATH, "Figures_TKE")
# os.makedirs(FIGURES_OUTPUT_DIR, exist_ok=True)

# Path to TKE data
TKE_DATA_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
TKE_DATA_NAME = "tke_turbulent_kinetic_energy.h5"
TKE_DATA_FILE = os.path.join(TKE_DATA_PATH, TKE_DATA_NAME)

# Mesh/Geometry data
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data"
GEO_NAME = "3d_NACA0012_Re50000_AoA5_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

# Output directory for figures
FIGURES_OUTPUT_DIR = os.path.join(TKE_DATA_PATH, "Figures_TKE")
os.makedirs(FIGURES_OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Colorbar Configuration (Log Scale)
# ============================================================================
# Set custom minimum values for colorbars (in log scale)
# Set to None to use automatic minimum (np.nanmin)
# Example: VMIN_TKE = 1e-4 or VMIN_TKE = None (automatic)

VMIN_TKE = None            # TKE minimum value (set None for auto)
VMIN_U_RMS = None            # u' RMS minimum value (set None for auto)
VMIN_V_RMS = None            # v' RMS minimum value (set None for auto)
VMIN_W_RMS = None            # w' RMS minimum value (set None for auto)

# ============================================================================
# Line Contour Configuration (as percentages of TKE range)
# ============================================================================
# Specify TKE contours as percentages of the range [tke_min, tke_max]
# Example: TKE_CONTOUR_PERCENTAGES = [25, 50, 75] for 25%, 50%, 75% of range
# Set to None to disable line contours

TKE_CONTOUR_PERCENTAGES = None  # Set to list of percentages or None to disable
# TKE_CONTOUR_PERCENTAGES = [25, 50, 75]  # Example: 25%, 50%, 75% of TKE range

# Example: To set specific values, uncomment and modify:
# VMIN_TKE = 1e-3
# VMIN_U_RMS = 1e-3
# VMIN_V_RMS = 1e-3
# VMIN_W_RMS = 1e-3
# TKE_CONTOUR_LEVELS = [1e-4, 1e-3, 1e-2, 1e-1]

# ============================================================================
# Load Data
# ============================================================================
print("=" * 70)
print("LOADING TKE DATA")
print("=" * 70)

if not os.path.exists(TKE_DATA_FILE):
    raise FileNotFoundError(f"TKE data file not found: {TKE_DATA_FILE}")

with h5py.File(TKE_DATA_FILE, "r") as f:
    # Load TKE field and RMS components
    tke = f["tke"][:]
    u_rms = f["u_prime_rms"][:]
    v_rms = f["v_prime_rms"][:]
    w_rms = f["w_prime_rms"][:]

    # Load coordinates
    x_coords = f["x"][:]
    y_coords = f["y"][:]

    # Load metadata
    u_infty = f.attrs["u_infty"]
    AOA = f.attrs["AOA"]
    n_snapshots = f.attrs["n_snapshots"]
    description = f.attrs["description"]

print(f"✓ TKE data loaded successfully")
print(f"  TKE shape: {tke.shape}")
print(f"  u'_rms shape: {u_rms.shape}")
print(f"  Coordinates shape: x={x_coords.shape}, y={y_coords.shape}")
print(f"  Metadata: u_infty={u_infty}, AOA={AOA}°, snapshots={n_snapshots}")

# Load geometrical data (airfoil surface)
geo_data = None
if os.path.exists(GEO_FILE):
    try:
        with h5py.File(GEO_FILE, "r") as f:
            geo_data = f["proj_points"][:]
        print(f"✓ Geometrical data loaded")
    except Exception as e:
        print(f"[WARNING] Could not load geometrical data: {e}")

# ============================================================================
# Statistics
# ============================================================================
print("\n" + "=" * 70)
print("TKE STATISTICS")
print("=" * 70)
print(f"TKE min: {np.nanmin(tke):.6e}")
print(f"TKE max: {np.nanmax(tke):.6e}")
print(f"TKE mean: {np.nanmean(tke):.6e}")
print(f"TKE median: {np.nanmedian(tke):.6e}")

print(f"\nu'_rms min: {np.nanmin(u_rms):.6e}, max: {np.nanmax(u_rms):.6e}")
print(f"v'_rms min: {np.nanmin(v_rms):.6e}, max: {np.nanmax(v_rms):.6e}")
print(f"w'_rms min: {np.nanmin(w_rms):.6e}, max: {np.nanmax(w_rms):.6e}")

# ============================================================================
# Normalize by u_infty
# ============================================================================
# Normalize TKE by u_infty^2
tke_norm = tke / (u_infty ** 2)
u_rms_norm = u_rms / u_infty
v_rms_norm = v_rms / u_infty
w_rms_norm = w_rms / u_infty

print(f"\nNormalized values (by u_infty={u_infty}):")
print(f"TKE/u_∞² min: {np.nanmin(tke_norm):.6e}, max: {np.nanmax(tke_norm):.6e}")
print(f"u'_rms/u_∞ min: {np.nanmin(u_rms_norm):.6e}, max: {np.nanmax(u_rms_norm):.6e}")
print(f"v'_rms/u_∞ min: {np.nanmin(v_rms_norm):.6e}, max: {np.nanmax(v_rms_norm):.6e}")
print(f"w'_rms/u_∞ min: {np.nanmin(w_rms_norm):.6e}, max: {np.nanmax(w_rms_norm):.6e}")

# ============================================================================
# Colorbar Range Information
# ============================================================================
print("\n" + "=" * 70)
print("COLORBAR RANGES (Log Scale)")
print("=" * 70)

# Compute positive values for log scale
tke_norm_pos = np.where(tke_norm > 0, tke_norm, np.nan)
u_rms_norm_pos = np.where(u_rms_norm > 0, u_rms_norm, np.nan)
v_rms_norm_pos = np.where(v_rms_norm > 0, v_rms_norm, np.nan)
w_rms_norm_pos = np.where(w_rms_norm > 0, w_rms_norm, np.nan)

tke_range_min = VMIN_TKE if VMIN_TKE is not None else np.nanmin(tke_norm_pos)
tke_range_max = np.nanmax(tke_norm_pos)

u_rms_range_min = VMIN_U_RMS if VMIN_U_RMS is not None else np.nanmin(u_rms_norm_pos)
u_rms_range_max = np.nanmax(u_rms_norm_pos)

v_rms_range_min = VMIN_V_RMS if VMIN_V_RMS is not None else np.nanmin(v_rms_norm_pos)
v_rms_range_max = np.nanmax(v_rms_norm_pos)

w_rms_range_min = VMIN_W_RMS if VMIN_W_RMS is not None else np.nanmin(w_rms_norm_pos)
w_rms_range_max = np.nanmax(w_rms_norm_pos)

print(f"TKE/u_∞²        : [{tke_range_min:.6e}, {tke_range_max:.6e}]")
print(f"u'_rms/u_∞      : [{u_rms_range_min:.6e}, {u_rms_range_max:.6e}]")
print(f"v'_rms/u_∞      : [{v_rms_range_min:.6e}, {v_rms_range_max:.6e}]")
print(f"w'_rms/u_∞      : [{w_rms_range_min:.6e}, {w_rms_range_max:.6e}]")


# ============================================================================
# Plot 1: TKE Contour Map
# ============================================================================
print("\nGenerating TKE contour map...")

fig, ax = plt.subplots(figsize=(12, 8))

# Create contour plot with normalized TKE using log scale
# Filter out zero/negative values for log scale (already computed above)
tke_min = tke_range_min
tke_max = tke_range_max

levels = np.logspace(np.log10(tke_min), np.log10(tke_max), 20)
cf = ax.contourf(x_coords, y_coords, tke_norm_pos, levels=levels, cmap="viridis",
                 norm=LogNorm(vmin=tke_min, vmax=tke_max))
cs = ax.contour(x_coords, y_coords, tke_norm_pos, levels=levels[::2], colors="black",
                alpha=0.3, linewidths=0.5, norm=LogNorm(vmin=tke_min, vmax=tke_max))


ax.set_xlabel("x/c", fontsize=12)
ax.set_xlim(-0.15, 1.5)
ax.set_ylabel("y/c", fontsize=12)
ax.set_ylim(-0.1, 0.5)
ax.set_title(f"Turbulent Kinetic Energy (TKE) - AoA={AOA}°", fontsize=14, fontweight="bold")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)

cbar = plt.colorbar(cf, ax=ax, label=r"TKE / $u_\infty^2$", fraction=0.02)

# Set up logarithmic locator for ticks at each order of magnitude
cbar.ax.yaxis.set_major_locator(LogLocator(base=10, numticks=15))
cbar.ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))
cbar.ax.tick_params(labelsize=8)

# Add minor ticks
cbar.ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)*0.1, numticks=100))

# Add min/max values on the colorbar
cbar.ax.text(1.3, tke_min, f'{tke_min:.2e}', transform=cbar.ax.transData,
             fontsize=9, verticalalignment='center', fontweight='bold', color='red')
cbar.ax.text(1.3, tke_max, f'{tke_max:.2e}', transform=cbar.ax.transData,
             fontsize=9, verticalalignment='center', fontweight='bold', color='red')

# Add line contours at specific TKE values if specified
if TKE_CONTOUR_PERCENTAGES is not None:
    print(f"\nAdding line contours at TKE percentages: {TKE_CONTOUR_PERCENTAGES}%")

    # Convert percentages to actual TKE values (in log scale)
    contour_levels = []
    for pct in TKE_CONTOUR_PERCENTAGES:
        # Map percentage (0-100) to the log-scale range
        # In log scale: value = 10^(log10(min) + (pct/100) * (log10(max) - log10(min)))
        log_min = np.log10(tke_min)
        log_max = np.log10(tke_max)
        log_value = log_min + (pct / 100.0) * (log_max - log_min)
        contour_levels.append(10 ** log_value)

    if contour_levels:
        cs = ax.contour(x_coords, y_coords, tke_norm_pos, levels=contour_levels,
                       colors='red', linewidths=1.5, alpha=0.8)
        # Add labels to contour lines with percentage and absolute value
        fmt_dict = {lvl: f'{pct}%' for lvl, pct in zip(contour_levels, TKE_CONTOUR_PERCENTAGES)}
        ax.clabel(cs, inline=True, fontsize=8, fmt=fmt_dict)
        print(f"✓ {len(contour_levels)} contour lines plotted")
plt.tight_layout()
# plt.savefig(os.path.join(FIGURES_OUTPUT_DIR, "01_TKE_contour_map.png"), dpi=150, bbox_inches="tight")
# print("✓ Saved: 01_TKE_contour_map.png")
# plt.close()
plt.show()


# ============================================================================
# Plot 2: RMS Components Contour Maps (4-panel)
# ============================================================================
# print("Generating RMS components contour maps...")

# fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# fields = [u_rms_norm_pos, v_rms_norm_pos, w_rms_norm_pos, tke_norm_pos]
# field_ranges = [
#     (u_rms_range_min, u_rms_range_max),
#     (v_rms_range_min, v_rms_range_max),
#     (w_rms_range_min, w_rms_range_max),
#     (tke_range_min, tke_range_max)
# ]
# titles = ["u' RMS", "v' RMS", "w' RMS", "TKE"]
# cmaps = ["Blues", "Greens", "Purples", "viridis"]
# labels = [r"$u'_{rms} / u_\infty$", r"$v'_{rms} / u_\infty$", r"$w'_{rms} / u_\infty$", r"TKE / $u_\infty^2$"]

# for idx, (ax, field, (field_min, field_max), title, cmap, label) in enumerate(zip(axes.flat, fields, field_ranges, titles, cmaps, labels)):
#     # Log-spaced contour levels
#     levels = np.logspace(np.log10(field_min), np.log10(field_max), 15)
#     cf = ax.contourf(x_coords, y_coords, field, levels=levels, cmap=cmap,
#                      norm=LogNorm(vmin=field_min, vmax=field_max))

#     ax.set_xlabel("x/c", fontsize=10)
#     ax.set_xlim(-0.15, 1.5)
#     ax.set_ylabel("y/c", fontsize=10)
#     ax.set_ylim(-0.1, 0.5)
#     ax.set_title(title, fontsize=11, fontweight="bold")
#     ax.set_aspect("equal")
#     ax.grid(True, alpha=0.2)

#     cbar = plt.colorbar(cf, ax=ax, label=label, fraction=0.02)

#     # Set up logarithmic locator for ticks at each order of magnitude
#     cbar.ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
#     cbar.ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))
#     cbar.ax.tick_params(labelsize=7)

#     # Add minor ticks
#     cbar.ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)*0.1, numticks=100))

#     # Add min/max values on the colorbar
#     cbar.ax.text(1.2, field_min, f'{field_min:.1e}', transform=cbar.ax.transData,
#                  fontsize=7, verticalalignment='center', fontweight='bold', color='red')
#     cbar.ax.text(1.2, field_max, f'{field_max:.1e}', transform=cbar.ax.transData,
#                  fontsize=7, verticalalignment='center', fontweight='bold', color='red')

#     # Add line contours at specific TKE values only for the TKE panel (idx=3)
#     if idx == 3 and TKE_CONTOUR_LEVELS is not None:
#         # Filter contour levels to be within the data range
#         valid_levels = [level for level in TKE_CONTOUR_LEVELS if field_min <= level <= field_max]

#         if valid_levels:
#             cs = ax.contour(x_coords, y_coords, field, levels=valid_levels,
#                            colors='red', linewidths=1.5, alpha=0.8)
#             # Add labels to contour lines
#             ax.clabel(cs, inline=True, fontsize=7, fmt='%.1e')

# plt.tight_layout()
# # plt.savefig(os.path.join(FIGURES_OUTPUT_DIR, "02_RMS_components_contours.png"), dpi=150, bbox_inches="tight")
# # print("✓ Saved: 02_RMS_components_contours.png")
# # plt.close()
# plt.show()
print("\n" + "=" * 70)
print("PLOTTING COMPLETE")
print("=" * 70)
print(f"All figures saved to: {FIGURES_OUTPUT_DIR}")
print(f"\nGenerated plots:")
print("  1. 01_TKE_contour_map.png - Main TKE field visualization")
print("  2. 02_RMS_components_contours.png - Individual RMS components")

