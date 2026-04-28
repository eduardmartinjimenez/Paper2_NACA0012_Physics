import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# ============================================================================
# Configuration
# ============================================================================
BASE_RESULTS_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Wall_shear_correlations/test_3"
OUTPUT_DIR = os.path.join(BASE_RESULTS_DIR, "Figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# X_C_LOCATIONS = [0.5]
# X_C_LOCATIONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
X_C_LOCATIONS = [0.3, 0.5, 0.7, 0.9]


ALPHA = 1.0

# Angle of attack
AOA_DEG = 12.0
AOA_RAD = np.deg2rad(AOA_DEG)

# Fluid / simulation reference parameters
Re_c    = 50000
rho_ref = 1.0
u_infty = 1.0
c_ref   = 1.0
nu_ref  = u_infty * c_ref / Re_c

# Wall-normal sampling parameters (same style as Mean_velocity_profiles.py)
WALL_NORMAL_LENGTH = 0.4   # length in chord units above the surface
N_SAMPLE_POINTS    = 500   # query points along the vertical line

RESULT_FILES = {
    x_c: os.path.join(BASE_RESULTS_DIR, f"wall_shear_correlation_xc_{x_c:.3f}_alpha_{ALPHA:.1f}_all_fft.h5")
    for x_c in X_C_LOCATIONS
}

# ============================================================================
# Rotation helper
# ============================================================================
def rotate(x, y, angle_rad):
    """Rotate (x, y) into flow-aligned frame by angle_rad (counterclockwise).
    After rotation: x' is streamwise (inflow direction), y' is wall-normal."""
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    x_rot =  x * ca + y * sa
    y_rot = -x * sa + y * ca
    return x_rot, y_rot

# ============================================================================
# Load data and extract wall-normal profiles via KDTree
# ============================================================================
print("Loading correlation results and extracting wall-normal profiles via KDTree...")

profiles = {}

for x_c, filepath in RESULT_FILES.items():
    if not os.path.exists(filepath):
        print(f"  [WARNING] File not found for x/c = {x_c}: {filepath}")
        continue

    with h5py.File(filepath, 'r') as f:
        R_PF  = f['R_PF'][:]     # (Nz, Ny, Nx)
        R_NF  = f['R_NF'][:]
        R_all = f['R_all'][:]
        x_grid = f['x'][:]       # (Nz, Ny, Nx)
        y_grid = f['y'][:]
        z_grid = f['z'][:]
        x_c_actual = f.attrs['x_c_actual']
        y_actual   = f.attrs['y_actual']
        N_PF  = f.attrs['N_PF']
        N_NF  = f.attrs['N_NF']
        N_all = f.attrs['N_all']
        tau_w_mean = float(f.attrs['tau_w_mean'])

    # -----------------------------------------------------------------------
    # Work on the Dz=0 (in-plane) slice
    # -----------------------------------------------------------------------
    x_2d = x_grid[0, :, :]   # (Ny, Nx)
    y_2d = y_grid[0, :, :]

    R_PF_2d  = R_PF[0, :, :]
    R_NF_2d  = R_NF[0, :, :]
    R_all_2d = R_all[0, :, :]

    # Rotate mesh coordinates into flow-aligned frame
    x_2d_rot, y_2d_rot = rotate(x_2d, y_2d, AOA_RAD)

    # Rotate the reference wall point
    x_ref_rot, y_ref_rot = rotate(x_c_actual, y_actual, AOA_RAD)

    # -----------------------------------------------------------------------
    # Build KDTree on the flattened 2D rotated mesh (same as Mean_velocity_profiles.py)
    # -----------------------------------------------------------------------
    Ny, Nx = x_2d_rot.shape
    x_flat = x_2d_rot.ravel()
    y_flat = y_2d_rot.ravel()
    tree = cKDTree(np.column_stack((x_flat, y_flat)))

    # -----------------------------------------------------------------------
    # Sample along a true vertical line at x' = x'_ref, from y'_wall upward
    # -----------------------------------------------------------------------
    y_query = np.linspace(y_ref_rot, y_ref_rot + WALL_NORMAL_LENGTH, N_SAMPLE_POINTS)
    x_query = np.full_like(y_query, x_ref_rot)

    distances, flat_indices = tree.query(np.column_stack((x_query, y_query)))

    # Convert flat indices to (j, i) grid indices
    j_indices = flat_indices // Nx
    i_indices = flat_indices % Nx

    # Remove duplicate grid cells (multiple query points mapped to the same cell)
    ij_pairs = np.column_stack((i_indices, j_indices))
    _, unique_idx = np.unique(ij_pairs, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)  # preserve order along the line

    j_u = j_indices[unique_idx]
    i_u = i_indices[unique_idx]

    x_prime_profile = x_2d_rot[j_u, i_u]
    y_prime_profile = y_2d_rot[j_u, i_u]
    R_PF_profile    = R_PF_2d[j_u, i_u]
    R_NF_profile    = R_NF_2d[j_u, i_u]
    R_all_profile   = R_all_2d[j_u, i_u]

    # Wall-normal distance from the surface
    eta = y_prime_profile - y_ref_rot

    print(f"  x/c = {x_c:.1f}: x'_ref={x_ref_rot:.4f}, y'_ref={y_ref_rot:.4f}, "
          f"{len(unique_idx)} unique mesh points, max dist={distances[unique_idx].max():.6f}")

    # -----------------------------------------------------------------------
    # Spanwise (Dz) profiles at (y = y_ref, x = x_ref) — same as before
    # -----------------------------------------------------------------------
    # Use the in-plane ix_ref to locate the spanwise slice
    x_1d_rot = x_2d_rot[x_2d_rot.shape[0] // 2, :]
    ix_ref = int(np.argmin(np.abs(x_1d_rot - x_ref_rot)))
    iy_ref = int(np.argmin(np.abs(y_2d_rot[:, ix_ref] - y_ref_rot)))

    Nz  = R_PF.shape[0]
    z_1d = z_grid[:, iy_ref, ix_ref]
    dz   = float(z_1d[1] - z_1d[0])
    Dz   = np.fft.fftshift(np.fft.fftfreq(Nz)) * (Nz * dz)

    profiles[x_c] = {
        # KDTree-sampled wall-normal profile
        'x_prime':    x_prime_profile,
        'y_prime':    y_prime_profile,
        'eta':        eta,
        'y_ref':      y_ref_rot,
        'x_ref_rot':  x_ref_rot,
        'x_c_actual': x_c_actual,
        'R_PF':       R_PF_profile,
        'R_NF':       R_NF_profile,
        'R_all':      R_all_profile,
        # Metadata
        'N_PF':  N_PF,
        'N_NF':  N_NF,
        'N_all': N_all,
        'tau_w_mean': tau_w_mean,
        # 2D mesh for the extraction visualization
        'x_2d_rot': x_2d_rot,
        'y_2d_rot': y_2d_rot,
        # Spanwise profiles
        'Dz':      Dz,
        'dz':      dz,
        'Nz':      Nz,
        'R_PF_z':  np.fft.fftshift(R_PF[:, iy_ref, ix_ref]),
        'R_NF_z':  np.fft.fftshift(R_NF[:, iy_ref, ix_ref]),
        'R_all_z': np.fft.fftshift(R_all[:, iy_ref, ix_ref]),
    }

if len(profiles) == 0:
    raise RuntimeError("No result files found!")

x_c_sorted = sorted(profiles.keys())
colors = plt.cm.viridis(np.linspace(0, 1, len(profiles)))

# ============================================================================
# Plot 0: Extraction line visualization in the rotated mesh
# ============================================================================
print("\nCreating extraction line visualization...")

fig0, ax = plt.subplots(1, 1, figsize=(10, 8))

for (x_c, data), color in zip([(xc, profiles[xc]) for xc in x_c_sorted], colors):
    x_flat = data['x_2d_rot'].ravel()
    y_flat = data['y_2d_rot'].ravel()
    ax.scatter(x_flat, y_flat, s=0.1, c='lightgray', alpha=0.3, rasterized=True)
    break  # mesh is the same for all x/c

for (x_c, data), color in zip([(xc, profiles[xc]) for xc in x_c_sorted], colors):
    ax.scatter(data['x_prime'], data['y_prime'],
               s=20, color=color, alpha=0.8, zorder=3,
               label=f'$x/c = {x_c:.1f}$ (KDTree points)', edgecolors='black', linewidth=0.3)
    # Draw the vertical sampling line
    ax.plot([data['x_ref_rot'], data['x_ref_rot']],
            [data['y_ref'], data['y_ref'] + WALL_NORMAL_LENGTH],
            '--', color=color, linewidth=1.5, alpha=0.7)
    ax.plot(data['x_ref_rot'], data['y_ref'], 'o', color=color,
            markersize=8, zorder=4, label=f'Wall point ($x/c={x_c:.1f}$)')

ax.axhline(0, color='red', linewidth=1.5, linestyle='--', alpha=0.6, label='$y\'=0$')
ax.set_xlabel("$x'$ (streamwise, chord units)", fontsize=13)
ax.set_ylabel("$y'$ (wall-normal, chord units)", fontsize=13)
ax.set_title('Wall-normal extraction lines via KDTree — rotated frame ($\\Delta z=0$)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, f"kdtree_extraction_visualization_alpha_{ALPHA:.1f}.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {output_path}")
plt.show()

# ============================================================================
# Plot 1: R vs (y' - y'_wall)  [chord units]
# ============================================================================
# print("\nCreating wall-normal profiles (chord units)...")

# fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
# corr_types = [('R_PF', 'PF', 0), ('R_NF', 'NF', 1), ('R_all', 'ALL', 2)]

# for corr_key, corr_label, ax_idx in corr_types:
#     ax = axes1[ax_idx]
#     for (x_c, data), color in zip([(xc, profiles[xc]) for xc in x_c_sorted], colors):
#         mask = data['eta'] >= 0
#         ax.plot(data[corr_key][mask], data['eta'][mask],
#                 linewidth=2.0, color=color, label=f'$x/c = {x_c:.1f}$', alpha=0.85)

#     ax.axhline(0, color='grey', linewidth=0.5, linestyle='-', alpha=0.5)
#     ax.axvline(0, color='grey', linewidth=0.5, linestyle='-', alpha=0.5)
#     ax.set_xlabel('Correlation $R$', fontsize=13)
#     ax.set_title(f'$R_{{{corr_label}}}$', fontsize=14, fontweight='bold')
#     ax.grid(True, alpha=0.3)
#     if ax_idx == 0:
#         ax.set_ylabel("$y' - y'_{\\mathrm{wall}}$  (chord units)", fontsize=13)
#         ax.legend(fontsize=10, loc='best', ncol=2)

# fig1.suptitle(
#     f'Wall-normal profiles of $R(\\tau\'_w,\\, u\'_s)$ — KDTree sampling, '
#     f'$\\Delta z=0$, AoA={AOA_DEG}°, $\\alpha={ALPHA}$',
#     fontsize=13, fontweight='bold', y=1.02,
# )
# plt.tight_layout()
# output_path = os.path.join(OUTPUT_DIR, f"kdtree_correlation_wall_normal_eta_alpha_{ALPHA:.1f}.png")
# plt.savefig(output_path, dpi=150, bbox_inches='tight')
# print(f"  Saved: {output_path}")
# plt.show()

# ============================================================================
# Plot 2: R vs y+  [wall units, log scale]
# ============================================================================
print("\nCreating wall-normal profiles (wall units)...")

corr_types = [('R_PF', 'PF', 0), ('R_NF', 'NF', 1), ('R_all', 'ALL', 2)]
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for corr_key, corr_label, ax_idx in corr_types:
    ax = axes2[ax_idx]
    for (x_c, data), color in zip([(xc, profiles[xc]) for xc in x_c_sorted], colors):
        u_tau  = np.sqrt(np.abs(data['tau_w_mean']) / rho_ref)
        mask   = data['eta'] >= 0
        y_plus = data['eta'][mask] * u_tau / nu_ref

        ax.plot(y_plus, data[corr_key][mask],
                linewidth=2.0, color=color, label=f'$x/c = {x_c:.1f}$', alpha=0.85)

    ax.axhline(0, color='grey', linewidth=0.5, linestyle='-', alpha=0.5)
    ax.set_xlabel('$y^+$', fontsize=13)
    ax.set_title(f'$R_{{{corr_label}}}$', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_xlim(left=1e-1)
    ax.grid(True, alpha=0.3, which='both')
    if ax_idx == 0:
        ax.set_ylabel('Correlation $R$', fontsize=13)
        ax.legend(fontsize=10, loc='best', ncol=2)

fig2.suptitle(
    f'Wall-normal profiles of $R(\\tau\'_w,\\, u\'_s)$ vs $y^+$ — KDTree sampling, '
    f'$\\Delta z=0$, AoA={AOA_DEG}°, $\\alpha={ALPHA}$',
    fontsize=13, fontweight='bold', y=1.02,
)
plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, f"kdtree_correlation_wall_normal_yplus_alpha_{ALPHA:.1f}.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {output_path}")
plt.show()

