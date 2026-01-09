import os
import sys
import h5py
import numpy as np
import glob
import gc
import matplotlib.pyplot as plt

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"

if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader

# Paths
#DATA_FILE_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/Test/data/3d_NACA0012_Re50000_AoA5_strain_rate_tensor_test_mpi.h5"
#MESH_FILE_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/Test/3d_NACA0012_Re10000_AoA5-CROP-MESH.h5"
DATA_FILE_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/3d_NACA0012_Re50000_AoA12_strain_rate_tensor_batch_33056839.h5"
MESH_FILE_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/3d_NACA0012_Re50000_AoA12-CROP-MESH.h5"

# Check if the data file exists
if os.path.exists(DATA_FILE_PATH):
    print(f"Data file exists! {DATA_FILE_PATH}")
else:
    print(f"Data file does not exist: {DATA_FILE_PATH}")

# Check if the mesh file exists
if os.path.exists(MESH_FILE_PATH):
    print(f"Mesh file exists! {MESH_FILE_PATH}")
else:
    print(f"Mesh file does not exist: {MESH_FILE_PATH}")

# Reference parameters
rho_ref = 1.0  # Reference density [kg/m3]
u_infty = 1.0  # Free-stream velocity [m/s]
c = 1.0  # Airfoil chord length [m]
Re_D = 50000  # Reynolds number [-]
mu_ref = rho_ref * u_infty * c / Re_D  # Dynamic viscosity [Pa s]
nu_ref = mu_ref / rho_ref  # Kinetic viscosity [m2/s]

# Load Sij_Sij_temporal_avg from saved file
with h5py.File(DATA_FILE_PATH, "r") as f:
    Sij_Sij_temporal_avg = f["Sij_Sij_temporal_avg"][:]
    print(f"Loaded Sij_Sij_temporal_avg from {DATA_FILE_PATH}.")

# Average strain-rate tensor in the z-direction (after temporal mean)
Sij_Sij_avg = np.mean(Sij_Sij_temporal_avg, axis=0)

# Compute epsilon prime
epsilon_prime = 2 * nu_ref * Sij_Sij_avg
print(f"Mean dissipation rate (epsilon_prime) shape: {epsilon_prime.shape}")

# Kolmogorov length scale
eta = (nu_ref**3 / epsilon_prime) ** (1 / 4)
print(f"Kolmogorov length scale (eta) shape: {eta.shape}")

# Kolmogorov time scale
tau_eta = (nu_ref / epsilon_prime) ** (1 / 2)
print(f"Kolmogorov time scale (tau_eta) shape: {tau_eta.shape}")

# minimum Kolmogorov time scale
tau_eta_min = np.min(tau_eta[np.isfinite(tau_eta)])
print(f"Minimum Kolmogorov time scale in domain: {tau_eta_min:.3e} s")

# Compute Kolmogorov length scale
# Only compute where epsilon is positive and finite
eta = np.full_like(epsilon_prime, np.nan, dtype=float)
mask_eps = np.isfinite(epsilon_prime) & (epsilon_prime > 0.0)
eta[mask_eps] = (nu_ref**3 / epsilon_prime[mask_eps]) ** (1 / 4)
print(f"Kolmogorov length scale computed: shape = {eta.shape}")

# Mesh load
loader = CompressedSnapshotLoader(MESH_FILE_PATH)
x_data, y_data, z_data = loader.x, loader.y, loader.z
tag_ibm_data = loader.tag_ibm
x = x_data[1, 1:-1, 1:-1]
y = y_data[1, 1:-1, 1:-1]
z = z_data[0:3, 1, 1]

print(f"Mesh loaded: x shape = {x.shape}")

# Plot Kolmogorov length scale
# 2D colormap of eta
# plt.figure(figsize=(10, 6))
# pc = plt.pcolormesh(x, y, eta, shading="auto", cmap="plasma", vmin=0.0, vmax=0.015)
# plt.colorbar(pc, label=r"Kolmogorov length scale $\eta$")
# plt.xlabel("x [m]"); plt.ylabel("y [m]")
# plt.title("Kolmogorov Length Scale")
# plt.tight_layout()
# plt.show()

# Compute grid spacings
dx = (x[1:-1, 2:] - x[1:-1, :-2]) / 2
dy = (y[2:, 1:-1] - y[:-2, 1:-1]) / 2
dz = z[2] - z[1]

# Compute equivalent cube size
d = (dx * dy * dz) ** (1 / 3)
#d = (dx * dy ) ** (1 / 2)

print(f"Grid spacing computed: shape = {d.shape}")

# plt.figure(figsize=(10, 6))
# c = plt.pcolormesh(
#    x[1:-1, 1:-1],
#    y[1:-1, 1:-1],
#    d,
#    shading="auto",
#    cmap="plasma",
# #    vmin=0.0,
# #    vmax=0.02,
# )
# plt.colorbar(c, label=r"Cubic cell size $\Delta$")
# plt.xlabel("x [m]")
# plt.ylabel("y [m]")
# plt.title("Equivalent Cube Cell Size Field")
# plt.tight_layout()
# plt.show()

# Compute delta/eta ratio
eta_mid = eta[1:-1, 1:-1]
delta_eta_ratio = np.where(eta_mid > 0, d / eta_mid, np.nan)
print(f"delta/eta ratio computed: shape = {delta_eta_ratio.shape}")


plt.figure(figsize=(10, 6))
c = plt.pcolormesh(
   x[1:-1, 1:-1],
   y[1:-1, 1:-1],
   delta_eta_ratio,
   shading="auto",
   cmap="plasma",
#    vmin=0.0,
#    vmax=5,
)
plt.colorbar(c, label=r"$\Delta / \eta$")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Mesh resolution relative to Kolmogorov scale")
plt.tight_layout()
plt.show()



# # ---------- Directional Δ/η ratios ----------
# # Interior coordinates to match dx, dy shapes
# x_mid = x[1:-1, 1:-1]
# y_mid = y[1:-1, 1:-1]

# # Ensure positive spacings (curvilinear grids can flip local sign)
# dx_mid = np.abs(dx)                     # (Ny-2, Nx-2)
# dy_mid = np.abs(dy)                     # (Ny-2, Nx-2)

# # dz: use mean spacing along z (adjust if you need local dz)
# z_line = z_data[:, 1, 1]
# dz_val = float(np.mean(np.diff(z_line))) if z_line.size > 1 else float(z[2]-z[1])
# dz_mid = np.full_like(dx_mid, np.abs(dz_val))   # broadcast to 2D

# # Match eta to interior
# eta_mid = eta[1:-1, 1:-1]

# # Valid mask: finite, positive eta; also mask solids if tag_ibm is available
# valid = np.isfinite(eta_mid) & (eta_mid > 0)
# if tag_ibm_data is not None:
#     # Fluid = 0 (adjust if your convention differs)
#     fluid_2d = np.any(tag_ibm_data == 0, axis=0)      # (Ny, Nx)
#     valid &= fluid_2d[1:-1, 1:-1]

# # Compute ratios
# dx_over_eta = np.full_like(eta_mid, np.nan)
# dy_over_eta = np.full_like(eta_mid, np.nan)
# dz_over_eta = np.full_like(eta_mid, np.nan)

# dx_over_eta[valid] = dx_mid[valid] / eta_mid[valid]
# dy_over_eta[valid] = dy_mid[valid] / eta_mid[valid]
# dz_over_eta[valid] = dz_mid[valid] / eta_mid[valid]

# # Helper for nice color limits (clip outliers)
# def vmax99(a):
#     v = a[np.isfinite(a)]
#     return float(np.percentile(v, 99)) if v.size else 1.0

# # --- Plot Δx/η ---
# plt.figure(figsize=(10, 6))
# vm = vmax99(dx_over_eta)
# pc = plt.pcolormesh(x_mid, y_mid, dx_over_eta, shading="auto", cmap="plasma", vmin=0.0, vmax=vm)
# plt.colorbar(pc, label=r"$\Delta x / \eta$")
# CS = plt.contour(x_mid, y_mid, dx_over_eta, levels=[1, 2], colors="k", linewidths=0.8)
# plt.clabel(CS, inline=True, fontsize=8, fmt="%.0f")
# plt.xlabel("x [m]"); plt.ylabel("y [m]")
# plt.title(r"Directional resolution: $\Delta x / \eta$")
# plt.tight_layout(); plt.show()

# # --- Plot Δy/η ---
# plt.figure(figsize=(10, 6))
# vm = vmax99(dy_over_eta)
# pc = plt.pcolormesh(x_mid, y_mid, dy_over_eta, shading="auto", cmap="plasma", vmin=0.0, vmax=vm)
# plt.colorbar(pc, label=r"$\Delta y / \eta$")
# CS = plt.contour(x_mid, y_mid, dy_over_eta, levels=[1, 2], colors="k", linewidths=0.8)
# plt.clabel(CS, inline=True, fontsize=8, fmt="%.0f")
# plt.xlabel("x [m]"); plt.ylabel("y [m]")
# plt.title(r"Directional resolution: $\Delta y / \eta$")
# plt.tight_layout(); plt.show()

# # --- Plot Δz/η ---
# plt.figure(figsize=(10, 6))
# vm = vmax99(dz_over_eta)
# pc = plt.pcolormesh(x_mid, y_mid, dz_over_eta, shading="auto", cmap="plasma", vmin=0.0, vmax=vm)
# plt.colorbar(pc, label=r"$\Delta z / \eta$")
# CS = plt.contour(x_mid, y_mid, dz_over_eta, levels=[1, 2], colors="k", linewidths=0.8)
# plt.clabel(CS, inline=True, fontsize=8, fmt="%.0f")
# plt.xlabel("x [m]"); plt.ylabel("y [m]")
# plt.title(r"Directional resolution: $\Delta z / \eta$")
# plt.tight_layout(); plt.show()

# # Quick percentiles to see what's limiting globally
# def pct(a, p):
#     v = a[np.isfinite(a)]
#     return np.nan if v.size == 0 else np.percentile(v, p)

# for name, arr in [("Δx/η", dx_over_eta), ("Δy/η", dy_over_eta), ("Δz/η", dz_over_eta)]:
#     print(f"{name}: P50={pct(arr,50):.2f}, P90={pct(arr,90):.2f}, P99={pct(arr,99):.2f}")



# Define x/c positions to extract profiles
x_targets = [0.3, 0.6, 0.8, 0.95, 1.5, 2]
x_mid = x[1:-1, 1:-1] 
y_mid = y[1:-1, 1:-1]

fig, axes = plt.subplots(1, len(x_targets), figsize=(12, 5), sharey=True)
for i, xt in enumerate(x_targets):
    idx_x = int(np.argmin(np.abs(x_mid[0, :] - xt)))   # column index in interior grid
    y_vals = y_mid[:, idx_x]
    h_over_eta = delta_eta_ratio[:, idx_x]

    ax = axes[i]
    ax.plot(h_over_eta, y_vals, linewidth=1)
    ax.set_title(f"$x/c={xt}$")
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel(r"$h/\eta$")
    ax.grid(True, linestyle="--", alpha=0.5)

axes[0].set_ylabel(r"$y/c$")
plt.suptitle(r"Resolution profile: $h/\eta$ vs $y/c$")
plt.tight_layout(); plt.subplots_adjust(top=0.85)
plt.show()
