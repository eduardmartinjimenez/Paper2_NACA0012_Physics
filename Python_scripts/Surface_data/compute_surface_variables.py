import os
import sys
import h5py
import numpy as np
from pathlib import Path
import time

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader

# ============================================================================
# Configuration
# ============================================================================

# Save results directory (one .h5 file per snapshot)
SAVE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Snapshots/batch_35801606/Surface_data/"
os.makedirs(SAVE_DIR, exist_ok=True)

# Geometrical data file
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA5_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

# Mesh data file
MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
MESH_NAME = "3d_NACA0012_Re50000_AoA5-CROP-MESH.h5"
MESH_FILE = os.path.join(MESH_PATH, MESH_NAME)

# Snapshot data directory
SNAPSHOTS_DIR = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA5_Re50000_1716x1662x128/Steady_state/batch_35801606/"

# Reference parameters
rho_ref = 1.0   # Reference density [kg/m3]
u_infty = 1.0   # Free-stream velocity [m/s]
c = 1.0         # Airfoil chord length [m]
Re_c = 50000    # Reynolds number [-]
mu_ref = rho_ref * u_infty * c / Re_c   # Dynamic viscosity [Pa s]
q_inf = 0.5 * rho_ref * u_infty**2      # Dynamic pressure [Pa]

# ============================================================================
# Utilities
# ============================================================================

def assert_exists(path: str, kind: str = "File") -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{kind} does not exist: {path}")
    print(f"  [OK] {kind}: {path}")

# ============================================================================
# Load geometry and mesh (done once)
# ============================================================================
print("=" * 70)
print("LOADING GEOMETRY AND MESH")
print("=" * 70)

assert_exists(GEO_FILE, "Geometrical data")
assert_exists(MESH_FILE, "Mesh data")

# Load geometrical data (2D interface information, same for every z-plane)
with h5py.File(GEO_FILE, "r") as f:
    interface_points = f["interface_points"][:]    # (N_surf, 3)
    proj_points      = f["proj_points"][:]         # (N_surf, 3)
    proj_normals     = f["proj_normals"][:]        # (N_surf, 3)
    proj_distances   = f["proj_distances"][:]      # (N_surf,)
    interface_indices_i = f["interface_indices_i"][:]  # (N_surf,) x-indices
    interface_indices_j = f["interface_indices_j"][:]  # (N_surf,) y-indices

N_surf = len(interface_indices_i)
print(f"  Number of 2D interface points: {N_surf}")

# Initialize loader (loads mesh + topology once)
loader = CompressedSnapshotLoader(MESH_FILE)
x_data = loader.x          # (Nz_full, Ny, Nx) = (130, 1586, 1620)
y_data = loader.y
z_data = loader.z
tag_ibm_data = loader.tag_ibm
Nz_full, Ny, Nx = x_data.shape
# Physical z-planes (exclude ghost cells at z=0 and z=Nz_full-1)
Nz_phys = Nz_full - 2   # 128 physical z-planes
print(f"  Mesh shape full (Nz, Ny, Nx): ({Nz_full}, {Ny}, {Nx})")
print(f"  Physical z-planes (excluding ghosts): {Nz_phys}")

# ============================================================================
# Pre-compute quantities that are identical for every snapshot
# ============================================================================

# --- Tangent vectors (2D, same for every z-plane) ---
# For a wall normal n = (nx, ny, nz), the in-plane tangent is t = (ny, -nx, 0)
tangent_vectors = np.stack(
    [proj_normals[:, 1], -proj_normals[:, 0], np.zeros(N_surf, dtype=proj_normals.dtype)],
    axis=1,
)  # (N_surf, 3)
tangent_vectors /= np.linalg.norm(tangent_vectors, axis=1, keepdims=True)

# --- Cell volumes for bulk-pressure computation ---
# Only physical z-planes (1:-1), all x/y (no ghost cells in x,y for cropped domain)
# Central differences for cell sizes
dx = 0.5 * (x_data[1:-1, :, 2:] - x_data[1:-1, :, :-2])    # (Nz_phys, Ny, Nx-2)
dy = 0.5 * (y_data[1:-1, 2:, :] - y_data[1:-1, :-2, :])    # (Nz_phys, Ny-2, Nx)
dz = 0.5 * (z_data[2:, :, :] - z_data[:-2, :, :])           # (Nz_phys, Ny, Nx)
# Use interior points where all three differences are defined
# Trim to common shape: (Nz_phys, Ny-2, Nx-2)
cell_volume = dx[:, 1:-1, :] * dy[:, :, 1:-1] * dz[:, 1:-1, 1:-1]

# Fluid mask on physical interior (same shape as cell_volume)
fluid_mask = tag_ibm_data[1:-1, 1:-1, 1:-1] == 0

# Pre-compute total fluid volume (constant across snapshots)
volume_sum = np.sum(cell_volume[fluid_mask])

# --- 3D surface indices: replicate 2D indices for every physical z-plane ---
# Physical z-planes: indices 1, 2, ..., Nz_full-2 (skip ghost planes 0 and Nz_full-1)
z_phys = np.arange(1, Nz_full - 1)  # (Nz_phys,)
# Resulting shapes: (Nz_phys, N_surf) — one row per physical z-plane
j_3d = np.broadcast_to(interface_indices_j[np.newaxis, :], (Nz_phys, N_surf))
i_3d = np.broadcast_to(interface_indices_i[np.newaxis, :], (Nz_phys, N_surf))
k_3d = np.broadcast_to(z_phys[:, np.newaxis], (Nz_phys, N_surf))

# Collect snapshot files
snapshot_files = sorted(Path(SNAPSHOTS_DIR).glob("*.h5"))
N_snapshots = len(snapshot_files)
print(f"  Number of snapshots found: {N_snapshots}")

if N_snapshots == 0:
    raise RuntimeError(f"No .h5 snapshots found in {SNAPSHOTS_DIR}")

# ============================================================================
# Process each snapshot
# ============================================================================
print("\n" + "=" * 70)
print("PROCESSING SNAPSHOTS")
print("=" * 70)

for snap_idx, snap_path in enumerate(snapshot_files):
    t_start = time.time()
    snap_name = snap_path.stem  # e.g. "3d_NACA0012_Re50000_AoA12_6350000-COMP-DATA"
    save_name = f"surface_{snap_name}.h5"
    save_path = os.path.join(SAVE_DIR, save_name)

    # Skip if already computed
    if os.path.exists(save_path):
        print(f"  [{snap_idx+1}/{N_snapshots}] SKIP (exists): {save_name}")
        continue

    print(f"  [{snap_idx+1}/{N_snapshots}] Processing: {snap_path.name} ...", end=" ", flush=True)

    # Load instantaneous compressed fields
    fields = loader.load_snapshot(str(snap_path))

    # Reconstruct full 3D fields (NaN outside fluid)
    p_full = loader.reconstruct_field(fields["p"])   # (Nz, Ny, Nx)
    u_full = loader.reconstruct_field(fields["u"])
    v_full = loader.reconstruct_field(fields["v"])
    w_full = loader.reconstruct_field(fields["w"])

    # ------------------------------------------------------------------
    # 1. Bulk pressure (volume-weighted average over all fluid cells)
    # ------------------------------------------------------------------
    p_interior = p_full[1:-1, 1:-1, 1:-1]  # match cell_volume shape
    p_bulk = np.nansum(p_interior[fluid_mask] * cell_volume[fluid_mask]) / volume_sum

    # ------------------------------------------------------------------
    # 2. Surface pressure: p_w = p_surface - p_bulk
    #    Extract pressure at every (z, j_interface, i_interface)
    # ------------------------------------------------------------------
    p_surface = p_full[k_3d, j_3d, i_3d]  # (Nz_phys, N_surf)
    p_w = p_surface - p_bulk               # (Nz_phys, N_surf)

    # ------------------------------------------------------------------
    # 3. Wall shear stress (tau_w) via scalar projection
    #    For each z-plane, extract (u, v, w) at the interface points,
    #    project onto the tangent vector, and use tau_w = mu * u_t / d_n
    # ------------------------------------------------------------------
    u_surf = u_full[k_3d, j_3d, i_3d]  # (Nz_phys, N_surf)
    v_surf = v_full[k_3d, j_3d, i_3d]
    w_surf = w_full[k_3d, j_3d, i_3d]

    # Tangential velocity: dot product with tangent vector for each point
    # tangent_vectors is (N_surf, 3), broadcast over z
    u_t = (u_surf * tangent_vectors[:, 0][np.newaxis, :]
         + v_surf * tangent_vectors[:, 1][np.newaxis, :]
         + w_surf * tangent_vectors[:, 2][np.newaxis, :])  # (Nz_phys, N_surf)

    # Wall shear stress: tau_w = mu * u_t / delta_n
    # proj_distances is (N_surf,), broadcast over z
    tau_w = mu_ref * u_t / proj_distances[np.newaxis, :]  # (Nz_phys, N_surf)

    # ------------------------------------------------------------------
    # 4. Save results for this snapshot
    # ------------------------------------------------------------------
    with h5py.File(save_path, "w") as f:
        f.create_dataset("p_w",   data=p_w,   dtype=np.float32)
        f.create_dataset("tau_w", data=tau_w,  dtype=np.float32)
        # Store metadata
        f.attrs["p_bulk"]   = float(p_bulk)

    elapsed = time.time() - t_start
    print(f"done  ({elapsed:.1f}s)  |  p_bulk={p_bulk:.6e}  "
          f"tau_w=[{np.nanmin(tau_w):.4e}, {np.nanmax(tau_w):.4e}]  "
          f"p_w=[{np.nanmin(p_w):.4e}, {np.nanmax(p_w):.4e}]")

    # Free memory
    del p_full, u_full, v_full, w_full, fields
    del p_surface, p_w, u_surf, v_surf, w_surf, u_t, tau_w

print("\n" + "=" * 70)
print(f"ALL DONE — {N_snapshots} snapshots processed.")
print(f"Output directory: {SAVE_DIR}")
print("=" * 70)

