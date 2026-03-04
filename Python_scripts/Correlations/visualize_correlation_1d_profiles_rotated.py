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

X_C_LOCATIONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ALPHA = 1.0

# Angle of attack: simulation frame has chord along x-axis and inflow rotated
# by AOA. Rotating coordinates by +AOA puts inflow horizontal and gives the
# true wall-normal direction (y') for profile extraction.
AOA_DEG = 12.0
AOA_RAD = np.deg2rad(AOA_DEG)

# Fluid / simulation reference parameters
Re_c    = 50000
rho_ref = 1.0
u_infty = 1.0
c_ref   = 1.0
nu_ref  = u_infty * c_ref / Re_c

RESULT_FILES = {
    x_c: os.path.join(BASE_RESULTS_DIR, f"wall_shear_correlation_xc_{x_c:.3f}_alpha_{ALPHA:.1f}_all_fft.h5")
    for x_c in X_C_LOCATIONS
}

# ============================================================================
# Rotation helper (same as Mean_velocity_profiles_dense_AoA12.py)
# ============================================================================
def rotate(x, y, angle_rad):
    """Rotate (x, y) into flow-aligned frame by angle_rad (counterclockwise).
    After rotation: x' is streamwise (inflow direction), y' is wall-normal."""
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    x_rot =  x * ca + y * sa
    y_rot = -x * sa + y * ca
    return x_rot, y_rot

# ============================================================================
# Load data from all chord locations
# ============================================================================
print("Loading correlation results for all chord locations...")

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
    # Rotate coordinates into flow-aligned frame (Dz=0 slice for 2D work)
    # -----------------------------------------------------------------------
    x_2d = x_grid[0, :, :]   # (Ny, Nx)
    y_2d = y_grid[0, :, :]

    x_2d_rot, y_2d_rot = rotate(x_2d, y_2d, AOA_RAD)

    # Rotate the reference wall point
    x_ref_rot, y_ref_rot = rotate(x_c_actual, y_actual, AOA_RAD)

    # -----------------------------------------------------------------------
    # In-plane (Dz=0) correlations
    # -----------------------------------------------------------------------
    R_PF_2d  = R_PF[0, :, :]
    R_NF_2d  = R_NF[0, :, :]
    R_all_2d = R_all[0, :, :]

    # -----------------------------------------------------------------------
    # Find the column in the rotated frame whose x' is closest to x'_ref.
    # x' varies mainly along axis=1 (columns), so we use the middle row.
    # This gives the wall-normal profile at the correct streamwise location.
    # -----------------------------------------------------------------------
    x_1d_rot = x_2d_rot[x_2d_rot.shape[0] // 2, :]   # representative row
    ix_ref = int(np.argmin(np.abs(x_1d_rot - x_ref_rot)))

    # Extract wall-normal profiles along this column in rotated y' coordinates
    y_prime_profile = y_2d_rot[:, ix_ref]   # y' values along the column
    R_PF_profile    = R_PF_2d[:, ix_ref]
    R_NF_profile    = R_NF_2d[:, ix_ref]
    R_all_profile   = R_all_2d[:, ix_ref]

    # Wall-normal distance from the surface (y' - y'_wall)
    eta = y_prime_profile - y_ref_rot        # distance from surface in rot. frame

    # -----------------------------------------------------------------------
    # Spanwise (Dz) profiles at (y = y_ref, x = x_ref)
    # -----------------------------------------------------------------------
    iy_ref = int(np.argmin(np.abs(y_prime_profile - y_ref_rot)))
    Nz = R_PF.shape[0]
    z_1d = z_grid[:, iy_ref, ix_ref]
    dz   = float(z_1d[1] - z_1d[0])
    Dz   = np.fft.fftshift(np.fft.fftfreq(Nz)) * (Nz * dz)

    profiles[x_c] = {
        'y_prime':      y_prime_profile,
        'eta':          eta,
        'y_ref':        y_ref_rot,
        'x_ref_rot':    x_ref_rot,
        'x_c_actual':   x_c_actual,
        'R_PF':         R_PF_profile,
        'R_NF':         R_NF_profile,
        'R_all':        R_all_profile,
        'N_PF':  N_PF,
        'N_NF':  N_NF,
        'N_all': N_all,
        'tau_w_mean':   tau_w_mean,
        'Dz':           Dz,
        'dz':           dz,
        'Nz':           Nz,
        'R_PF_z':  np.fft.fftshift(R_PF[:, iy_ref, ix_ref]),
        'R_NF_z':  np.fft.fftshift(R_NF[:, iy_ref, ix_ref]),
        'R_all_z': np.fft.fftshift(R_all[:, iy_ref, ix_ref]),
    }

    print(f"  x/c = {x_c:.1f}: x'_ref={x_ref_rot:.4f}, y'_ref={y_ref_rot:.4f}, "
          f"ix_ref={ix_ref}, Nz={Nz}")

if len(profiles) == 0:
    raise RuntimeError("No result files found!")

x_c_sorted = sorted(profiles.keys())
colors = plt.cm.viridis(np.linspace(0, 1, len(profiles)))

# ============================================================================
# Plot 1: Wall-normal profiles R vs (y' - y'_wall)  [chord units]
# ============================================================================
print("\nCreating wall-normal profiles plot (chord units)...")

fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

corr_types = [('R_PF', 'PF', 0), ('R_NF', 'NF', 1), ('R_all', 'ALL', 2)]

for corr_key, corr_label, ax_idx in corr_types:
    ax = axes1[ax_idx]

    for (x_c, data), color in zip([(xc, profiles[xc]) for xc in x_c_sorted], colors):
        # Only plot points above the surface (eta >= 0)
        mask = data['eta'] >= 0
        ax.plot(data[corr_key][mask], data['eta'][mask],
                linewidth=2.0, color=color, label=f'$x/c = {x_c:.1f}$', alpha=0.85)

    ax.axhline(0, color='grey', linewidth=0.5, linestyle='-', alpha=0.5)
    ax.axvline(0, color='grey', linewidth=0.5, linestyle='-', alpha=0.5)
    ax.set_xlabel('Correlation coefficient $R$', fontsize=13)
    ax.set_title(f'$R_{{{corr_label}}}$', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if ax_idx == 0:
        ax.set_ylabel("$y' - y'_{wall}$  (wall-normal, chord units)", fontsize=13)
        ax.legend(fontsize=10, loc='best', ncol=2)

fig1.suptitle(
    f'Wall-normal profiles of $R(\\tau\'_w, u\'_s)$ at $\\Delta z = 0$'
    f' — rotated frame (AoA = {AOA_DEG}°), $\\alpha = {ALPHA}$',
    fontsize=14, fontweight='bold', y=1.02,
)
plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, f"correlation_1d_profiles_rotated_eta_alpha_{ALPHA:.1f}.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {output_path}")
plt.show()

# ============================================================================
# Plot 2: Wall-normal profiles R vs y+  (wall units)
# ============================================================================
print("\nCreating wall-normal profiles plot (wall units)...")

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

for corr_key, corr_label, ax_idx in corr_types:
    ax = axes2[ax_idx]

    for (x_c, data), color in zip([(xc, profiles[xc]) for xc in x_c_sorted], colors):
        u_tau  = np.sqrt(np.abs(data['tau_w_mean']) / rho_ref)
        mask   = data['eta'] >= 0
        y_plus = data['eta'][mask] * u_tau / nu_ref

        ax.plot(data[corr_key][mask], y_plus,
                linewidth=2.0, color=color, label=f'$x/c = {x_c:.1f}$', alpha=0.85)

    ax.axhline(0, color='grey', linewidth=0.5, linestyle='-', alpha=0.5)
    ax.axvline(0, color='grey', linewidth=0.5, linestyle='-', alpha=0.5)
    ax.set_xlabel('Correlation coefficient $R$', fontsize=13)
    ax.set_title(f'$R_{{{corr_label}}}$', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    if ax_idx == 0:
        ax.set_ylabel("$y^+$", fontsize=13)
        ax.legend(fontsize=10, loc='best', ncol=2)

fig2.suptitle(
    f'Wall-normal profiles of $R(\\tau\'_w, u\'_s)$ vs $y^+$ at $\\Delta z = 0$'
    f' — rotated frame (AoA = {AOA_DEG}°), $\\alpha = {ALPHA}$',
    fontsize=14, fontweight='bold', y=1.02,
)
plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, f"correlation_1d_profiles_rotated_yplus_alpha_{ALPHA:.1f}.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {output_path}")
plt.show()

# ============================================================================
# Plot 3: Spanwise profiles R vs Dz
# ============================================================================
print("\nCreating spanwise profiles plot...")

fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

corr_types_z = [('R_PF_z', 'PF', 0), ('R_NF_z', 'NF', 1), ('R_all_z', 'ALL', 2)]

for corr_key, corr_label, ax_idx in corr_types_z:
    ax = axes3[ax_idx]

    for (x_c, data), color in zip([(xc, profiles[xc]) for xc in x_c_sorted], colors):
        ax.plot(data['Dz'], data[corr_key],
                linewidth=2.0, color=color, label=f'$x/c = {x_c:.1f}$', alpha=0.85)

    ax.axvline(0, color='grey', linewidth=0.5, linestyle='-', alpha=0.5)
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='-', alpha=0.5)
    ax.set_xlabel(r'$\Delta z / c$', fontsize=13)
    ax.set_title(f'$R_{{{corr_label}}}(\\Delta z)$', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if ax_idx == 0:
        ax.set_ylabel('Correlation coefficient $R$', fontsize=13)
        ax.legend(fontsize=10, loc='best', ncol=2)

fig3.suptitle(
    f'Spanwise profiles of $R(\\tau\'_w, u\'_s)$ at $y = y_{{ref}}$'
    f' — rotated frame (AoA = {AOA_DEG}°), $\\alpha = {ALPHA}$',
    fontsize=14, fontweight='bold', y=1.02,
)
plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, f"correlation_1d_profiles_rotated_z_alpha_{ALPHA:.1f}.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {output_path}")
plt.show()

print("\nDone.")
