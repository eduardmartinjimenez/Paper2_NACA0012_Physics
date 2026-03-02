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

# Chord locations and corresponding result files
X_C_LOCATIONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # x/c locations to plot
ALPHA = 2.0

# Fluid / simulation reference parameters (dimensionless simulation units)
Re_c    = 50000   # chord Reynolds number
rho_ref = 1.0     # reference density
u_infty = 1.0     # free-stream velocity
c_ref   = 1.0     # chord length
nu_ref  = u_infty * c_ref / Re_c   # kinematic viscosity = 1/Re_c

RESULT_FILES = {
    x_c: os.path.join(BASE_RESULTS_DIR, f"wall_shear_correlation_xc_{x_c:.3f}_alpha_{ALPHA:.1f}_all_fft.h5")
    for x_c in X_C_LOCATIONS
}

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
        R_PF = f['R_PF'][:]      # (Nz, Ny, Nx)
        R_NF = f['R_NF'][:]
        R_all = f['R_all'][:]
        x_grid = f['x'][:]       # (Nz, Ny, Nx)
        y_grid = f['y'][:]
        x_c_actual = f.attrs['x_c_actual']
        y_actual = f.attrs['y_actual']
        N_PF = f.attrs['N_PF']
        N_NF = f.attrs['N_NF']
        N_all = f.attrs['N_all']
        tau_w_mean = float(f.attrs['tau_w_mean'])

    # Extract 2D slice at Dz=0 (in-plane correlation)
    R_PF_2d = R_PF[0, :, :]     # (Ny_crop, Nx_crop)
    R_NF_2d = R_NF[0, :, :]
    R_all_2d = R_all[0, :, :]
    x_2d = x_grid[0, :, :]      # (Ny_crop, Nx_crop)
    y_2d = y_grid[0, :, :]

    # Find x-index closest to the reference point within the crop window
    x_1d = x_2d[0, :]           # x varies along axis=1
    ix_ref = np.argmin(np.abs(x_1d - x_c_actual))

    # Extract wall-normal profiles at x = x_ref, Dz = 0
    y_profile = y_2d[:, ix_ref]
    R_PF_profile = R_PF_2d[:, ix_ref]
    R_NF_profile = R_NF_2d[:, ix_ref]
    R_all_profile = R_all_2d[:, ix_ref]

    profiles[x_c] = {
        'y': y_profile,
        'y_ref': y_actual,
        'x_c_actual': x_c_actual,
        'R_PF': R_PF_profile,
        'R_NF': R_NF_profile,
        'R_all': R_all_profile,
        'N_PF': N_PF,
        'N_NF': N_NF,
        'N_all': N_all,
        'tau_w_mean': tau_w_mean,
    }

    print(f"  x/c = {x_c:.1f}: loaded (Ny={len(y_profile)}, x_actual={x_c_actual:.4f}, y_ref={y_actual:.4f})")

if len(profiles) == 0:
    raise RuntimeError("No result files found!")

# ============================================================================
# Plot 1: Wall-normal profiles of R_PF, R_NF, R_all for each chord location
#         (physical coordinates: y - y_ref in chord units)
# ============================================================================
fig, axes = plt.subplots(1, len(profiles), figsize=(6 * len(profiles), 6), sharey=False)

if len(profiles) == 1:
    axes = [axes]

for ax, (x_c, data) in zip(axes, sorted(profiles.items())):
    y_rel = data['y'] - data['y_ref']  # wall-normal distance from surface

    ax.plot(y_rel, data['R_all'], 'k-',  linewidth=2.0, label='$R_{all}$')
    ax.plot(y_rel, data['R_PF'],  'r--', linewidth=1.5, label='$R_{PF}$')
    ax.plot(y_rel, data['R_NF'],  'b-.', linewidth=1.5, label='$R_{NF}$')

    ax.axvline(0, color='grey', linewidth=0.5, linestyle='-')
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='-')

    ax.set_xlabel('$(y - y_{ref})/c$', fontsize=13)
    ax.set_ylabel('Correlation coefficient $R$', fontsize=13)
    ax.set_title(f'$x/c = {x_c:.1f}$', fontsize=14, fontweight='bold')
    ax.set_xlim(left=0)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)

fig.suptitle(
    f'Wall-normal profiles of $R(\\tau\'_w, u\'_s)$ at $\\Delta z = 0$\n'
    f'$\\alpha = {ALPHA}$',
    fontsize=15, fontweight='bold', y=1.02,
)

plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, f"correlation_1d_profiles_alpha_{ALPHA:.1f}_Dz0.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved figure to: {output_path}")

plt.show()

# ============================================================================
# Plot 2: Wall-normal profiles of R_PF, R_NF, R_all vs y+ (wall units)
# ============================================================================
fig2, axes2 = plt.subplots(1, len(profiles), figsize=(6 * len(profiles), 6), sharey=False)

if len(profiles) == 1:
    axes2 = [axes2]

for ax, (x_c, data) in zip(axes2, sorted(profiles.items())):
    # Friction velocity from mean wall shear stress
    u_tau = np.sqrt(np.abs(data['tau_w_mean']) / rho_ref)

    # Wall-normal distance in wall units: y+ = (y - y_wall) * u_tau / nu
    y_rel = data['y'] - data['y_ref']
    y_plus = y_rel * u_tau / nu_ref

    ax.plot(y_plus, data['R_all'], 'k-',  linewidth=2.0, label='$R_{all}$')
    ax.plot(y_plus, data['R_PF'],  'r--', linewidth=1.5, label='$R_{PF}$')
    ax.plot(y_plus, data['R_NF'],  'b-.', linewidth=1.5, label='$R_{NF}$')

    ax.axvline(0, color='grey', linewidth=0.5, linestyle='-')
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='-')

    ax.set_xlabel('$y^+$', fontsize=13)
    ax.set_ylabel('Correlation coefficient $R$', fontsize=13)
    ax.set_title(f'$x/c = {x_c:.1f}$  ($u_{{\\tau}} = {u_tau:.4f}$)', fontsize=14, fontweight='bold')
    ax.set_xlim(left=0)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)

fig2.suptitle(
    f'Wall-normal profiles of $R(\\tau\'_w, u\'_s)$ vs $y^+$ at $\\Delta z = 0$\n'
    f'$\\alpha = {ALPHA}$',
    fontsize=15, fontweight='bold', y=1.02,
)

plt.tight_layout()

output_path_yplus = os.path.join(OUTPUT_DIR, f"correlation_1d_profiles_yplus_alpha_{ALPHA:.1f}_Dz0.png")
plt.savefig(output_path_yplus, dpi=150, bbox_inches='tight')
print(f"Saved y+ figure to: {output_path_yplus}")

plt.show()
