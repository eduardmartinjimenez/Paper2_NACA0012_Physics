import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================
BASE_RESULTS_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Wall_shear_correlations/"
OUTPUT_DIR = os.path.join(BASE_RESULTS_DIR, "Figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# X_C_LOCATIONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
X_C_LOCATIONS = [0.3, 0.5, 0.7, 0.9]

ALPHA = 1.0

RESULT_FILES = {
    x_c: os.path.join(BASE_RESULTS_DIR, f"wall_shear_correlation_xc_{x_c:.3f}_alpha_{ALPHA:.1f}_all_fft.h5")
    for x_c in X_C_LOCATIONS
}

# Common colormap limits.
# Options:
#   - Set manually:              CLIM_MODE = 'manual',    VMIN = -0.5, VMAX = 0.5
#   - Symmetric global max:      CLIM_MODE = 'auto_sym'   (+-max|R| across all files)
#   - Absolute global range:     CLIM_MODE = 'auto'       (min/max across all files)
#   - Fixed percentile clipping: CLIM_MODE = 'percentile', PMIN = 2, PMAX = 98
CLIM_MODE = 'auto'
VMIN = -0.5
VMAX = 0.5
PMIN, PMAX = 2, 98   # only used when CLIM_MODE = 'percentile'
NLEVELS = 25

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
        R_all = f['R_all'][0, :, :]
        x_grid = f['x'][0, :, :]
        y_grid = f['y'][0, :, :]
        x_c_actual = f.attrs['x_c_actual']
        y_actual = f.attrs['y_actual']
        N_PF = f.attrs['N_PF']
        N_NF = f.attrs['N_NF']
        N_all = f.attrs['N_all']

    data_all[x_c] = {
        'R_PF': R_PF,
        'R_NF': R_NF,
        'R_all': R_all,
        'x_grid': x_grid,
        'y_grid': y_grid,
        'x_c_actual': x_c_actual,
        'y_actual': y_actual,
        'N_PF': N_PF,
        'N_NF': N_NF,
        'N_all': N_all,
    }

    print(f"  x/c = {x_c:.1f}: loaded ({R_PF.shape[0]}x{R_PF.shape[1]}), "
          f"N_PF={N_PF}, N_NF={N_NF}")

if len(data_all) == 0:
    raise RuntimeError("No result files found!")

# ============================================================================
# Compute colormap limits based on CLIM_MODE
# ============================================================================
all_R = np.concatenate([
    np.concatenate([d['R_PF'].ravel(), d['R_NF'].ravel(), d['R_all'].ravel()])
    for d in data_all.values()
])
all_R = all_R[np.isfinite(all_R)]

if CLIM_MODE == 'auto_sym':
    vmax_abs = np.max(np.abs(all_R))
    VMIN, VMAX = -vmax_abs, vmax_abs
    print(f"  Auto symmetric limits: [{VMIN:.3f}, {VMAX:.3f}]")
elif CLIM_MODE == 'auto':
    VMIN, VMAX = float(np.min(all_R)), float(np.max(all_R))
    print(f"  Auto limits: [{VMIN:.3f}, {VMAX:.3f}]")
elif CLIM_MODE == 'percentile':
    VMIN = float(np.percentile(all_R, PMIN))
    VMAX = float(np.percentile(all_R, PMAX))
    print(f"  Percentile [{PMIN},{PMAX}]% limits: [{VMIN:.3f}, {VMAX:.3f}]")
else:  # 'manual'
    print(f"  Manual limits: [{VMIN:.3f}, {VMAX:.3f}]")

x_c_sorted = sorted(data_all.keys())
n_panels = len(x_c_sorted)
ncols = 4
nrows = int(np.ceil(n_panels / ncols))

# ============================================================================
# Helper: create one multi-panel figure for a given correlation type
# ============================================================================
def plot_multipanel(corr_key, corr_title, fig_label):
    """Plot 2D contour maps at Dz=0 for all chord locations."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes)

    for idx, x_c in enumerate(x_c_sorted):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        d = data_all[x_c]

        # Local colormap limits for this panel
        R = d[corr_key]
        R_finite = R[np.isfinite(R)]
        vmin_local = float(np.min(R_finite))
        vmax_local = float(np.max(R_finite))
        levels = np.linspace(vmin_local, vmax_local, NLEVELS)

        im = ax.contourf(d['x_grid'], d['y_grid'], R,
                         levels=levels, cmap='RdBu_r', extend='both')
        ax.contour(d['x_grid'], d['y_grid'], R,
                   levels=10, colors='black', alpha=0.25, linewidths=0.4)

        # Mark reference point
        ax.plot(d['x_c_actual'], d['y_actual'], 'g*', markersize=14,
                markeredgecolor='black', markeredgewidth=1.0, zorder=5)

        ax.set_title(f'$x/c = {x_c:.1f}$', fontsize=13, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.15)

        if col == 0:
            ax.set_ylabel('$y/c$', fontsize=12)
        if row == nrows - 1:
            ax.set_xlabel('$x/c$', fontsize=12)

        # Individual colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('$R$', fontsize=10)
        cbar.set_ticks([vmin_local, 0, vmax_local])
        cbar.ax.tick_params(labelsize=8)

    # Hide unused panels
    for idx in range(n_panels, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        f'{corr_title} at $\\Delta z = 0$  —  $\\alpha = {ALPHA}$',
        fontsize=15, fontweight='bold', y=1.01,
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR, f"correlation_2d_{fig_label}_all_xc_alpha_{ALPHA:.1f}.png"
    )
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.show()


# ============================================================================
# Create the three figures
# ============================================================================
plot_multipanel(
    'R_all',
    'Unconditional correlation $R_{all}(\\tau\'_w, u\'_s)$',
    'Rall',
)

plot_multipanel(
    'R_PF',
    'PF-conditioned correlation $R_{PF}(\\tau\'_w, u\'_s)$',
    'RPF',
)

plot_multipanel(
    'R_NF',
    'NF-conditioned correlation $R_{NF}(\\tau\'_w, u\'_s)$',
    'RNF',
)

print("\nDone.")
