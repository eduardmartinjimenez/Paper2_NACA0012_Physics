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

X_C_LOCATIONS = [0.3, 0.5, 0.7, 0.9]
ALPHA = 1.0

# Angle of attack: simulation frame has chord along x-axis and inflow rotated
# by AOA. Rotating coordinates by +AOA puts inflow horizontal.
AOA_DEG = 12.0
AOA_RAD = np.deg2rad(AOA_DEG)

RESULT_FILES = {
    x_c: os.path.join(BASE_RESULTS_DIR, f"wall_shear_correlation_xc_{x_c:.3f}_alpha_{ALPHA:.1f}_all_fft.h5")
    for x_c in X_C_LOCATIONS
}

NLEVELS = 25

# ============================================================================
# Rotation helper
# ============================================================================
def rotate(x, y, angle_rad):
    """Rotate coordinates by angle_rad (counterclockwise).
    Transforms simulation frame (chord along x) to physical frame
    (inflow along x)."""
    x_rot =  x * np.cos(angle_rad) + y * np.sin(angle_rad)
    y_rot = -x * np.sin(angle_rad) + y * np.cos(angle_rad)
    return x_rot, y_rot

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
        R_PF  = f['R_PF'][0, :, :]
        R_NF  = f['R_NF'][0, :, :]
        R_all = f['R_all'][0, :, :]
        x_grid = f['x'][0, :, :]
        y_grid = f['y'][0, :, :]
        x_c_actual = f.attrs['x_c_actual']
        y_actual   = f.attrs['y_actual']
        N_PF  = f.attrs['N_PF']
        N_NF  = f.attrs['N_NF']
        N_all = f.attrs['N_all']

    # Rotate coordinate grids into physical frame
    x_rot, y_rot = rotate(x_grid, y_grid, AOA_RAD)

    # Rotate reference point
    x_ref_rot, y_ref_rot = rotate(x_c_actual, y_actual, AOA_RAD)

    data_all[x_c] = {
        'R_PF':      R_PF,
        'R_NF':      R_NF,
        'R_all':     R_all,
        'x_grid':    x_rot,
        'y_grid':    y_rot,
        'x_ref':     x_ref_rot,
        'y_ref':     y_ref_rot,
        'N_PF':  N_PF,
        'N_NF':  N_NF,
        'N_all': N_all,
    }

    print(f"  x/c = {x_c:.1f}: loaded, ref rotated to "
          f"({x_ref_rot:.4f}, {y_ref_rot:.4f})")

if len(data_all) == 0:
    raise RuntimeError("No result files found!")

x_c_sorted = sorted(data_all.keys())
n_panels = len(x_c_sorted)
ncols = min(4, n_panels)
nrows = int(np.ceil(n_panels / ncols))

# ============================================================================
# Helper: create one multi-panel figure for a given correlation type
# ============================================================================
def plot_multipanel(corr_key, corr_title, fig_label):
    """Plot 2D contour maps at Dz=0 for all chord locations (rotated frame)."""
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes)

    for idx, x_c in enumerate(x_c_sorted):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        d = data_all[x_c]

        R = d[corr_key]
        R_finite = R[np.isfinite(R)]
        vmin_local = float(np.min(R_finite))
        vmax_local = float(np.max(R_finite))
        levels = np.linspace(vmin_local, vmax_local, NLEVELS)

        im = ax.contourf(d['x_grid'], d['y_grid'], R,
                         levels=levels, cmap='RdBu_r', extend='both')
        ax.contour(d['x_grid'], d['y_grid'], R,
                   levels=10, colors='black', alpha=0.25, linewidths=0.4)

        # Reference point (rotated)
        ax.plot(d['x_ref'], d['y_ref'], 'g*', markersize=14,
                markeredgecolor='black', markeredgewidth=1.0, zorder=5)

        ax.set_title(f'$x/c = {x_c:.1f}$', fontsize=13, fontweight='bold')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.15)

        if col == 0:
            ax.set_ylabel('$y\'/c$  (normal to inflow)', fontsize=12)
        if row == nrows - 1:
            ax.set_xlabel('$x\'/c$  (streamwise)', fontsize=12)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('$R$', fontsize=10)
        cbar.set_ticks([vmin_local, 0, vmax_local])
        cbar.ax.tick_params(labelsize=8)

    for idx in range(n_panels, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        f'{corr_title} at $\\Delta z = 0$  —  $\\alpha = {ALPHA}$,  '
        f'AoA = {AOA_DEG}° rotation applied',
        fontsize=14, fontweight='bold', y=1.01,
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR, f"correlation_2d_{fig_label}_all_xc_alpha_{ALPHA:.1f}_rotated.png"
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
