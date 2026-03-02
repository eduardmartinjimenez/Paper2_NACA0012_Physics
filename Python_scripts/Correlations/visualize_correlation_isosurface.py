import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

# ============================================================================
# Configuration
# ============================================================================
# Result file from correlation analysis
RESULT_FILE = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Wall_shear_correlations/wall_shear_correlation_xc_0.700_alpha_0.5_all_fft_2.h5"

# Output directory
OUTPUT_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Wall_shear_correlations/Figures/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output base path
OUTPUT_BASE = os.path.join(OUTPUT_DIR, "correlation_isosurface_fft2_07_alpha05")

# Isosurface levels to plot (positive values; negative will be added automatically)
R_LEVELS = [0.1, 0.2]

# Colors for positive / negative isosurfaces
COLOR_POS = (0.8, 0.2, 0.2, 0.35)   # red, semi-transparent
COLOR_NEG = (0.2, 0.2, 0.8, 0.35)   # blue, semi-transparent

# ============================================================================
# Load results
# ============================================================================
print("Loading correlation results...")

with h5py.File(RESULT_FILE, 'r') as f:
    R_PF  = f['R_PF'][:]      # (Nz, Ny, Nx)
    R_NF  = f['R_NF'][:]
    x     = f['x'][:]         # (Nz, Ny, Nx)
    y     = f['y'][:]
    z     = f['z'][:]

    x_c_actual = f.attrs['x_c_actual']
    y_actual   = f.attrs['y_actual']
    N_PF       = f.attrs['N_PF']
    N_NF       = f.attrs['N_NF']

Nz, Ny, Nx = R_PF.shape

# ============================================================================
# Build coordinate axes
# ============================================================================
# x and y are rectilinear: extract 1D arrays from a single slice
x_1d = x[0, 0, :]       # (Nx,)
y_1d = y[0, :, 0]       # (Ny,)

# z-axis represents spanwise SEPARATION Dz.
# Physical z-coordinates are stored; compute Dz = z - zpip install scikit-image[0]
dz_1d = z[:, 0, 0] - z[0, 0, 0]   # (Nz,)

# Due to periodicity, Dz > Lz/2 wraps around to negative separation.
# Keep only the first half (0 to ~Lz/2) for a clearer visualization.
Nz_half = Nz // 2 + 1
dz_1d   = dz_1d[:Nz_half]

R_PF_half = R_PF[:Nz_half, :, :]
R_NF_half = R_NF[:Nz_half, :, :]

print(f"Loaded correlation results:")
print(f"  Full shape : (Nz={Nz}, Ny={Ny}, Nx={Nx})")
print(f"  Half shape : (Nz_half={Nz_half}, Ny={Ny}, Nx={Nx})")
print(f"  Dz range   : [{dz_1d[0]:.4f}, {dz_1d[-1]:.4f}]")
print(f"  x  range   : [{x_1d[0]:.4f}, {x_1d[-1]:.4f}]")
print(f"  y  range   : [{y_1d[0]:.4f}, {y_1d[-1]:.4f}]")
print(f"  Reference  : x/c = {x_c_actual:.4f}, y = {y_actual:.4f}")
print(f"  Samples    : N_PF={N_PF}, N_NF={N_NF}")


def _extract_isosurface(field, level, dz_1d, y_1d, x_1d):
    """Extract an isosurface and map vertices from index to physical coords."""
    try:
        verts_idx, faces, _, _ = marching_cubes(field, level)
    except (ValueError, RuntimeError):
        # No isosurface at this level
        return None, None

    # marching_cubes returns vertices as (iz, iy, ix) in index space.
    # Map to physical coordinates by linear interpolation.
    verts_phys = np.empty_like(verts_idx)
    verts_phys[:, 0] = np.interp(verts_idx[:, 0],
                                  np.arange(len(dz_1d)), dz_1d)   # Dz
    verts_phys[:, 1] = np.interp(verts_idx[:, 1],
                                  np.arange(len(y_1d)),  y_1d)    # y
    verts_phys[:, 2] = np.interp(verts_idx[:, 2],
                                  np.arange(len(x_1d)),  x_1d)    # x

    return verts_phys, faces


def _add_isosurface(ax, verts, faces, color):
    """Add a triangulated isosurface to a 3D axis."""
    if verts is None:
        return
    # Build collection of triangles: each face references 3 vertex indices
    # Poly3DCollection expects coordinates as (x, y, z) = (streamwise, wall-normal, Dz)
    triangles = verts[faces]
    # Reorder columns: stored as (Dz, y, x) -> plot as (x, y, Dz)
    triangles = triangles[:, :, [2, 1, 0]]

    mesh = Poly3DCollection(triangles, alpha=color[3])
    mesh.set_facecolor(color[:3])
    mesh.set_edgecolor((0, 0, 0, 0.05))
    ax.add_collection3d(mesh)


# ============================================================================
# FIGURE: Isosurfaces for each R level — PF vs NF side by side
# ============================================================================
for R_level in R_LEVELS:
    print(f"\nPlotting isosurfaces at |R| = {R_level} ...")

    fig = plt.figure(figsize=(16, 7))

    for col, (label, R_data) in enumerate([('PF', R_PF_half),
                                            ('NF', R_NF_half)]):
        ax = fig.add_subplot(1, 2, col + 1, projection='3d')

        # Positive isosurface (+R_level)
        v_pos, f_pos = _extract_isosurface(R_data, R_level, dz_1d, y_1d, x_1d)
        _add_isosurface(ax, v_pos, f_pos, COLOR_POS)

        # Negative isosurface (-R_level)
        v_neg, f_neg = _extract_isosurface(R_data, -R_level, dz_1d, y_1d, x_1d)
        _add_isosurface(ax, v_neg, f_neg, COLOR_NEG)

        # Mark reference point at Dz=0
        ax.scatter([x_c_actual], [y_actual], [0],
                   c='green', s=120, marker='*', edgecolors='black',
                   linewidths=1, zorder=10, label='Reference point')

        # Axis labels
        ax.set_xlabel('$x/c$', fontsize=12, labelpad=8)
        ax.set_ylabel('$y/c$', fontsize=12, labelpad=8)
        ax.set_zlabel('$\\Delta z / c$', fontsize=12, labelpad=8)

        ax.set_xlim(x_1d[0], x_1d[-1])
        ax.set_ylim(y_1d[0], y_1d[-1])
        ax.set_zlim(dz_1d[0], dz_1d[-1])

        n_label = N_PF if label == 'PF' else N_NF
        ax.set_title(f'{label} Correlation  ($N = {n_label}$)',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)

    # Build a legend for the isosurface colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_POS[:3], alpha=COLOR_POS[3],
              label=f'$R = +{R_level}$'),
        Patch(facecolor=COLOR_NEG[:3], alpha=COLOR_NEG[3],
              label=f'$R = -{R_level}$'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=2, fontsize=12, frameon=True)

    fig.suptitle(
        f'Isosurfaces of $R(\\tau\'_w, u\'_s)$ at $|R| = {R_level}$\n'
        f'Reference: $x/c = {x_c_actual:.3f}$, $y = {y_actual:.4f}$',
        fontsize=14, fontweight='bold', y=0.98,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])

    output_path = f"{OUTPUT_BASE}_R{R_level:.2f}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved to: {output_path}")

plt.show()

print("\nDone.")
