import os
import sys
import h5py
import numpy as np
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

# Surface data directory (output from compute_surface_variables.py)
SURFACE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Snapshots/batch_30658504/Surface_data/"

# Geometrical data (for surface coordinates, normals, etc.)
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

# Mesh data (for z-coordinates)
MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
MESH_NAME = "3d_NACA0012_Re50000_AoA12-CROP-MESH.h5"
MESH_FILE = os.path.join(MESH_PATH, MESH_NAME)

# Reference parameters
rho_ref = 1.0   # Reference density [kg/m3]
u_infty = 1.0   # Free-stream velocity [m/s]
q_inf   = 0.5 * rho_ref * u_infty**2  # Dynamic pressure [Pa]

# ============================================================================
# Load geometrical data
# ============================================================================
print("=" * 70)
print("LOADING GEOMETRICAL DATA")
print("=" * 70)

with h5py.File(GEO_FILE, "r") as f:
    interface_points    = f["interface_points"][:]      # (N_surf, 3)
    proj_points         = f["proj_points"][:]           # (N_surf, 3)
    proj_normals        = f["proj_normals"][:]          # (N_surf, 3)
    proj_distances      = f["proj_distances"][:]        # (N_surf,)
    interface_indices_i = f["interface_indices_i"][:]   # (N_surf,)
    interface_indices_j = f["interface_indices_j"][:]   # (N_surf,)

N_surf = len(interface_indices_i)

# Surface (x, y) coordinates from projected wall points
x_surface = proj_points[:, 0]  # (N_surf,)
y_surface = proj_points[:, 1]  # (N_surf,)

# z-coordinates from mesh — strip ghost cells (first and last z-planes)
with h5py.File(MESH_FILE, "r") as f:
    z_full = f["z"][:, 0, 0]       # (Nz_full,) = (130,)
z_phys = z_full[1:-1]              # (Nz_phys,) = (128,) — physical planes only
Nz_phys = len(z_phys)

print(f"  N_surf  = {N_surf}")
print(f"  Nz_full = {len(z_full)}  →  Nz_phys = {Nz_phys}  (ghost cells removed)")
print(f"  z range: [{z_phys[0]:.6f}, {z_phys[-1]:.6f}]")

# ============================================================================
# Discover and load all surface data snapshots
# ============================================================================
print("\n" + "=" * 70)
print("LOADING SURFACE DATA SNAPSHOTS")
print("=" * 70)

surface_files = sorted(Path(SURFACE_DIR).glob("surface_*.h5"))
N_snapshots = len(surface_files)
print(f"  Found {N_snapshots} surface data files")

if N_snapshots == 0:
    raise RuntimeError(f"No surface data files found in {SURFACE_DIR}")

# Peek at the first file to confirm dimensions
with h5py.File(surface_files[0], "r") as f:
    sample_shape = f["p_w"].shape
    print(f"  Sample file shape: p_w = {sample_shape}")
    assert sample_shape == (Nz_phys, N_surf), (
        f"Shape mismatch: file has {sample_shape}, expected ({Nz_phys}, {N_surf})"
    )

# Extract time-step labels from snapshot attribute (fallback to filename)
timestep_labels = []
for fpath in surface_files:
    with h5py.File(fpath, "r") as f:
        if "snapshot" in f.attrs:
            snap_name = f.attrs["snapshot"]
            # Extract number from e.g. "3d_NACA0012_Re50000_AoA12_6350000-COMP-DATA"
            parts = snap_name.replace("-COMP-DATA", "").split("_")
            timestep_labels.append(int(parts[-1]))
        else:
            # Fallback: parse from filename
            parts = fpath.stem.replace("surface_", "").replace("-COMP-DATA", "").split("_")
            timestep_labels.append(int(parts[-1]))
timestep_labels = np.array(timestep_labels)

# Pre-allocate arrays: (N_snapshots, Nz_phys, N_surf)
p_w_all   = np.empty((N_snapshots, Nz_phys, N_surf), dtype=np.float32)
tau_w_all = np.empty((N_snapshots, Nz_phys, N_surf), dtype=np.float32)
p_bulk_all = np.empty(N_snapshots, dtype=np.float64)

mem_gb = 2 * p_w_all.nbytes / 1e9  # p_w + tau_w
print(f"  Allocating {mem_gb:.2f} GB for {N_snapshots} snapshots")

for i, fpath in enumerate(surface_files):
    with h5py.File(fpath, "r") as f:
        p_w_all[i]   = f["p_w"][:]
        tau_w_all[i]  = f["tau_w"][:]
        p_bulk_all[i] = f.attrs["p_bulk"]
    if (i + 1) % 50 == 0 or i == 0 or i == N_snapshots - 1:
        print(f"  [{i+1:4d}/{N_snapshots}] Loaded: {fpath.name}")

print(f"\n  p_w   shape: {p_w_all.shape}   (N_snapshots, Nz_phys, N_surf)")
print(f"  tau_w shape: {tau_w_all.shape}   (N_snapshots, Nz_phys, N_surf)")

# ============================================================================
# Compute statistics
# ============================================================================
print("\n" + "=" * 70)
print("SURFACE DATA STATISTICS")
print("=" * 70)

# --- Time-averaged fields (mean over snapshots) ---
p_w_mean   = np.mean(p_w_all,   axis=0)  # (Nz_phys, N_surf)
tau_w_mean = np.mean(tau_w_all, axis=0)   # (Nz_phys, N_surf)

# --- Spanwise-averaged (mean over z) ---
p_w_mean_z   = np.mean(p_w_mean,   axis=0)  # (N_surf,)
tau_w_mean_z = np.mean(tau_w_mean, axis=0)   # (N_surf,)

# --- Cp and Cf from the time-and-spanwise-averaged fields ---
Cp_values = p_w_mean_z / q_inf    # (N_surf,)
Cf_values = tau_w_mean_z / q_inf  # (N_surf,)

print(f"  p_w   — min: {np.nanmin(p_w_mean):.6e},  max: {np.nanmax(p_w_mean):.6e},  mean: {np.nanmean(p_w_mean):.6e}")
print(f"  tau_w — min: {np.nanmin(tau_w_mean):.6e},  max: {np.nanmax(tau_w_mean):.6e},  mean: {np.nanmean(tau_w_mean):.6e}")
print(f"  Cp    — min: {np.min(Cp_values):.6e},  max: {np.max(Cp_values):.6e}")
print(f"  Cf    — min: {np.min(Cf_values):.6e},  max: {np.max(Cf_values):.6e}")
print(f"  p_bulk (mean over snapshots): {np.mean(p_bulk_all):.6e}")

# ============================================================================
# Summary of available variables
# ============================================================================
print("\n" + "=" * 70)
print("AVAILABLE VARIABLES")
print("=" * 70)
print(f"""
  Geometry:
    x_surface          ({N_surf},)                        — wall x-coordinates
    y_surface          ({N_surf},)                        — wall y-coordinates
    z_phys             ({Nz_phys},)                       — physical z-coordinates (no ghosts)
    proj_normals       ({N_surf}, 3)                      — wall-normal vectors
    proj_distances     ({N_surf},)                        — wall-normal distances

  Per-snapshot 3D surface data:
    p_w_all            ({N_snapshots}, {Nz_phys}, {N_surf})  — wall pressure (p_surf - p_bulk)
    tau_w_all          ({N_snapshots}, {Nz_phys}, {N_surf})  — wall shear stress

  Time-averaged:
    p_w_mean           ({Nz_phys}, {N_surf})              — time-averaged wall pressure
    tau_w_mean         ({Nz_phys}, {N_surf})              — time-averaged wall shear stress

  Time- and spanwise-averaged:
    p_w_mean_z         ({N_surf},)                        — <p_w>(t,z)
    tau_w_mean_z       ({N_surf},)                        — <tau_w>(t,z)
    Cp_values          ({N_surf},)                        — pressure coefficient
    Cf_values          ({N_surf},)                        — skin friction coefficient

  Metadata:
    timestep_labels    ({N_snapshots},)                   — time-step indices
    p_bulk_all         ({N_snapshots},)                   — bulk pressure per snapshot
""")
print("=" * 70)