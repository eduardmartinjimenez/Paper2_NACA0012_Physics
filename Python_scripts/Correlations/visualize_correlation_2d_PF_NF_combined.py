import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================
# BASE_RESULTS_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Wall_shear_correlations/test_3"
BASE_RESULTS_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/Wall_shear_correlations/test_1"
# BASE_RESULTS_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Wall_pressure_correlations/"
OUTPUT_DIR = os.path.join(BASE_RESULTS_DIR, "Figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

X_C_LOCATIONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# X_C_LOCATIONS = [0.5]

ALPHA = 1.0

RESULT_FILES = {
    x_c: os.path.join(BASE_RESULTS_DIR, f"wall_shear_correlation_xc_{x_c:.3f}_alpha_{ALPHA:.1f}_all_fft.h5")
    for x_c in X_C_LOCATIONS
}
# RESULT_FILES = {
#     x_c: os.path.join(BASE_RESULTS_DIR, f"wall_pressure_correlation_xc_{x_c:.3f}_alpha_{ALPHA:.1f}_all_fft.h5")
#     for x_c in X_C_LOCATIONS
# }

NLEVELS = 25

# Figure size (width, height) in inches — increase to make plots bigger
FIG_SIZE = (20, 10)

# Domain window relative to the reference point (x_c_actual, y_actual).
# The axis limits become: [ref + D*_MIN, ref + D*_MAX]
# Set to None to use the full data range on that axis.
DX_MIN, DX_MAX = -0.5, 1.5
DY_MIN, DY_MAX = -0.05, 0.99

# ============================================================================
# Load data from all chord locations
# ============================================================================
print("Loading correlation results for all chord locations...")

data_all = {}

for x_c, filepath in RESULT_FILES.items():
    if not os.path.exists(filepath):
        print(f"  [WARNING] File not found for x/c = {x_c}: {filepath}")
        continue

    with h5py.File(filepath, 'r') as f:
        R_PF = f['R_PF'][0, :, :]      # Dz=0 slice -> (Ny, Nx)
        R_NF = f['R_NF'][0, :, :]
        x_grid = f['x'][0, :, :]
        y_grid = f['y'][0, :, :]
        x_c_actual = f.attrs['x_c_actual']
        y_actual = f.attrs['y_actual']
        N_PF = f.attrs['N_PF']
        N_NF = f.attrs['N_NF']

    data_all[x_c] = {
        'R_PF': R_PF,
        'R_NF': R_NF,
        'x_grid': x_grid,
        'y_grid': y_grid,
        'x_c_actual': x_c_actual,
        'y_actual': y_actual,
        'N_PF': N_PF,
        'N_NF': N_NF,
    }

    print(f"  x/c = {x_c:.1f}: loaded ({R_PF.shape[0]}x{R_PF.shape[1]}), "
          f"N_PF={N_PF}, N_NF={N_NF}")

if len(data_all) == 0:
    raise RuntimeError("No result files found!")

x_c_sorted = sorted(data_all.keys())

# ============================================================================
# Helper: draw one contour panel
# ============================================================================
def draw_panel(ax, fig, x_grid, y_grid, R, x_c_actual, y_actual, panel_title):
    R_finite = R[np.isfinite(R)]
    vmin_local = float(np.min(R_finite))
    vmax_local = float(np.max(R_finite))
    levels = np.linspace(vmin_local, vmax_local, NLEVELS)

    im = ax.contourf(x_grid, y_grid, R,
                     levels=levels, cmap='RdBu_r', extend='both')
    ax.contour(x_grid, y_grid, R,
               levels=10, colors='black', alpha=0.25, linewidths=0.4)

    ax.plot(x_c_actual, y_actual, 'g*', markersize=14,
            markeredgecolor='black', markeredgewidth=1.0, zorder=5)

    ax.set_title(panel_title, fontsize=12, fontweight='bold')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('$x/c$', fontsize=11)
    ax.set_ylabel('$y/c$', fontsize=11)
    ax.grid(True, alpha=0.15)

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

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('$R$', fontsize=10)
    tick_vals = sorted({vmin_local, 0.0, vmax_local})
    cbar.set_ticks(tick_vals)
    cbar.ax.tick_params(labelsize=8)


# ============================================================================
# One combined 1x2 figure (PF | NF) per x/c location
# ============================================================================
print("\nGenerating combined PF/NF figures per x/c location...")

for x_c in x_c_sorted:
    d = data_all[x_c]

    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE)

    draw_panel(
        axes[0], fig,
        d['x_grid'], d['y_grid'], d['R_PF'],
        d['x_c_actual'], d['y_actual'],
        f'PF-conditioned  ($N_{{PF}}={d["N_PF"]}$)',
    )

    draw_panel(
        axes[1], fig,
        d['x_grid'], d['y_grid'], d['R_NF'],
        d['x_c_actual'], d['y_actual'],
        f'NF-conditioned  ($N_{{NF}}={d["N_NF"]}$)',
    )

    fig.suptitle(
        f'$x/c = {x_c:.1f}$  —  $\\alpha = {ALPHA}$  —  $\\Delta z = 0$',
        fontsize=14, fontweight='bold',
    )

    plt.tight_layout()

    plt.show()

print("\nDone.")
