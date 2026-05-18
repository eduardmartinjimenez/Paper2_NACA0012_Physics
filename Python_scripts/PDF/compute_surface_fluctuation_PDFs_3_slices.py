import os
import sys
import h5py
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from glob import glob

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader


# ============================================================================
# Configuration
# ============================================================================

# # Base directory containing all slice folders
# SLICES_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA5_Re50000_1716x1662x128/Slice_data/"

# # Pattern to match all slice location directories
# SLICE_PATTERN = "slice_*"

# # Pattern to match slice snapshots
# SNAPSHOT_PATTERN = "slice_*-COMP-DATA.h5"

# # Geometrical data file
# GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
# GEO_NAME = "3d_NACA0012_Re50000_AoA5_Geometrical_Data.h5"
# GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

# # Output directory
# OUTPUT_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/PDF_analysis/"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# Base directory containing all slice folders
SLICES_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA5_Re50000_1716x1662x128/Slice_data/"

# Pattern to match all slice location directories
SLICE_PATTERN = "slice_*"

# Pattern to match slice snapshots
SNAPSHOT_PATTERN = "slice_*-COMP-DATA.h5"

# Geometrical data file
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA5_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

# Output directory
OUTPUT_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/PDF_analysis/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Reference parameters
rho_ref = 1.0
u_infty = 1.0
c = 1.0
Re_c = 50000
q_inf = 0.5 * rho_ref * u_infty**2
AOA = 5
mu_ref = rho_ref * u_infty * c / Re_c

# ============================================================================
# ANALYSIS PARAMETERS
# ============================================================================

# Number of bins for histogram (PDF)
N_BINS = 250

# ============================================================================
# Load geometrical data
# ============================================================================
print("\n" + "=" * 70)
print("LOADING GEOMETRICAL DATA")
print("=" * 70)

with h5py.File(GEO_FILE, "r") as f:
    interface_points = f["interface_points"][:]
    interface_indices_i = f["interface_indices_i"][:]
    interface_indices_j = f["interface_indices_j"][:]
    proj_normals = f["proj_normals"][...].astype(np.float64)
    proj_distances = f["proj_distances"][...].astype(np.float64)

N_surf = len(interface_points)
print(f"  Number of 2D interface points: {N_surf}")

# ============================================================================
# Load slice mesh data
# ============================================================================
print("\n" + "=" * 70)
print("LOADING SLICE MESH DATA")
print("=" * 70)

slice_dirs = sorted(glob(os.path.join(SLICES_PATH, SLICE_PATTERN)))
print(f"  Found {len(slice_dirs)} slice directories")

mesh_data = {}
for slice_dir in slice_dirs:
    mesh_matches = glob(os.path.join(slice_dir, "slice_*-CROP-MESH.h5"))
    if not mesh_matches:
        print(f"  [WARNING] No mesh file found in {slice_dir}")
        continue

    slice_name = os.path.basename(slice_dir)
    mesh_file_path = mesh_matches[0]

    loader = CompressedSnapshotLoader(mesh_file_path)

    # Exclude ghost cells at spanwise boundaries
    x_data = loader.x[1:-1, :, :]
    y_data = loader.y[1:-1, :, :]
    z_data = loader.z[1:-1, :, :]

    x_coord = x_data[0, 0, 0]

    mesh_data[slice_name] = {
        'mesh_file': mesh_file_path,
        'x': x_data,
        'y': y_data,
        'z': z_data,
        'x_chord': x_coord,
    }

    print(f"  {slice_name}: Mesh shape {x_data.shape}, x-chord location: {x_coord:.4f}")

if len(mesh_data) == 0:
    raise RuntimeError("No mesh files found for any slice.")

# ============================================================================
# Find closest interface point on suction side for each slice
# ============================================================================
print("\n" + "=" * 70)
print("FINDING CLOSEST INTERFACE POINTS")
print("=" * 70)

closest_interface_points = {}

suction_side_indices = np.where(interface_points[:, 1] >= 0)[0]
if len(suction_side_indices) == 0:
    raise RuntimeError("No suction-side interface points found (y >= 0).")

suction_side_points = interface_points[suction_side_indices]

for slice_name, slice_info in mesh_data.items():
    x_slice = slice_info['x_chord']

    x_distances = np.abs(suction_side_points[:, 0] - x_slice)
    closest_idx = np.argmin(x_distances)
    closest_point = suction_side_points[closest_idx]
    closest_global_idx = suction_side_indices[closest_idx]

    closest_interface_points[slice_name] = {
        'point': closest_point,
        'index': closest_global_idx,
        'x_c': closest_point[0] / c,
        'distance': x_distances[closest_idx],
    }

    print(f"  {slice_name}: Closest interface point at x={closest_point[0]:.4f}, "
          f"distance={x_distances[closest_idx]:.6f}")

print(f"\nFound closest interface points for {len(closest_interface_points)} slices")

# ============================================================================
# Find closest y-grid index for each slice
# ============================================================================
print("\n" + "=" * 70)
print("FINDING CLOSEST SLICE Y-GRID INDICES")
print("=" * 70)

for slice_name, slice_info in mesh_data.items():
    y_data = slice_info['y']
    interface_y = closest_interface_points[slice_name]['point'][1]

    y_slice = y_data[0, :, 0]

    y_distances = np.abs(y_slice - interface_y)
    closest_y_idx = np.argmin(y_distances)

    closest_interface_points[slice_name]['y_index'] = closest_y_idx
    closest_interface_points[slice_name]['y_value'] = y_slice[closest_y_idx]

    print(f"  {slice_name}: Closest y-index={closest_y_idx}, "
          f"y_grid={y_slice[closest_y_idx]:.6f}, "
          f"interface_y={interface_y:.6f}, "
          f"distance={y_distances[closest_y_idx]:.6f}")

print(f"\nFound closest y-indices for {len(closest_interface_points)} slices")

def get_slice_snapshot_files(slice_dir: str) -> list:
    files = sorted(Path(slice_dir).glob(SNAPSHOT_PATTERN))
    return [str(f) for f in files if "avg" not in f.name]

# ============================================================================
# Visualize interface points, slices, and closest points
# ============================================================================
print("\n" + "=" * 70)
print("VISUALIZING INTERFACE POINTS AND SLICE LOCATIONS")
print("=" * 70)

fig, ax = plt.subplots(figsize=(12, 8))

# Plot all interface points
ax.scatter(interface_points[:, 0], interface_points[:, 1], 
           c='lightgray', s=20, alpha=0.5, label='All interface points')

# Plot suction-side interface points
ax.scatter(suction_side_points[:, 0], suction_side_points[:, 1], 
           c='blue', s=30, alpha=0.7, label='Suction-side interface points')

# Plot each slice location and its closest interface point
colors = plt.cm.tab10(np.linspace(0, 1, len(mesh_data)))

for color_idx, (slice_name, interface_info) in enumerate(closest_interface_points.items()):
    x_slice = mesh_data[slice_name]['x_chord']
    closest_point = interface_info['point']
    y_idx = interface_info['y_index']
    slice_y = mesh_data[slice_name]['y'][0, y_idx, 0]
    
    # Plot slice location (vertical line at x_slice)
    ax.axvline(x=x_slice, color=colors[color_idx], linestyle='--', alpha=0.6, linewidth=1.5)
    
    # Plot closest interface point
    ax.scatter(closest_point[0], closest_point[1], 
              color=colors[color_idx], s=150, marker='*', 
              edgecolors='black', linewidth=1.5,
              label=f'{slice_name} (x/c={interface_info["x_c"]:.3f})')
    
    # Plot the y-grid index point on slice
    ax.scatter(x_slice, slice_y, 
              color=colors[color_idx], s=100, marker='x', linewidth=2,
              alpha=0.8)

ax.set_xlabel('x (chord-normalized)', fontsize=12)
ax.set_ylabel('y (distance from chord)', fontsize=12)
ax.set_title('Interface Points, Slice Locations, and Closest Surface Points', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=10)
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'interface_points_visualization.png'), dpi=150)
print(f"Visualization saved to {os.path.join(OUTPUT_DIR, 'interface_points_visualization.png')}")
plt.show()

# ============================================================================
# Collect surface data from slice snapshots
# ============================================================================
print("\n" + "=" * 70)
print("COLLECTING SURFACE DATA FROM SLICE SNAPSHOTS")
print("=" * 70)

surface_data = {}
slice_dir_map = {os.path.basename(d): d for d in slice_dirs}

for slice_name in mesh_data.keys():
    surface_data[slice_name] = {
        'p_w': [],
        'tau_w': [],
    }

total_snapshots = 0

for slice_name, slice_info in mesh_data.items():
    slice_dir = slice_dir_map.get(slice_name)
    if not slice_dir:
        print(f"  [WARNING] No directory found for {slice_name}")
        continue

    snapshot_files = get_slice_snapshot_files(slice_dir)
    if len(snapshot_files) == 0:
        print(f"  [WARNING] No slice snapshots found for {slice_name} in {slice_dir}")
        continue

    total_snapshots += len(snapshot_files)

    idx_point = closest_interface_points[slice_name]['index']
    y_idx = closest_interface_points[slice_name]['y_index']
    x_c_slice = closest_interface_points[slice_name]['x_c']
    print(f"Processing {len(snapshot_files)} snapshots for {slice_name} (x/c={x_c_slice:.3f})...")

    loader = CompressedSnapshotLoader(slice_info['mesh_file'])

    normal_at_point = proj_normals[idx_point]
    tangent_at_point = np.array([normal_at_point[1], -normal_at_point[0], 0.0])
    tangent_norm = np.linalg.norm(tangent_at_point)
    if tangent_norm == 0.0:
        print(f"  [WARNING] Zero tangent norm for {slice_name}; skipping")
        continue
    tangent_at_point /= tangent_norm

    distance_at_point = proj_distances[idx_point]
    if distance_at_point <= 0.0:
        print(f"  [WARNING] Non-positive wall distance for {slice_name}; skipping")
        continue

    for idx, snapshot_file in enumerate(snapshot_files):
        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  Processing snapshot {idx+1}/{len(snapshot_files)}...", flush=True)

        try:
            fields = loader.load_snapshot(snapshot_file)

            u_data = loader.reconstruct_field(fields["u"])[1:-1, :, :]
            v_data = loader.reconstruct_field(fields["v"])[1:-1, :, :]
            w_data = loader.reconstruct_field(fields["w"])[1:-1, :, :]
            p_data = loader.reconstruct_field(fields["p"])[1:-1, :, :]

            u_line = u_data[:, y_idx, 0]
            v_line = v_data[:, y_idx, 0]
            w_line = w_data[:, y_idx, 0]
            p_line = p_data[:, y_idx, 0]

            u_t_line = (u_line * tangent_at_point[0] +
                        v_line * tangent_at_point[1] +
                        w_line * tangent_at_point[2])

            tau_line = mu_ref * u_t_line / distance_at_point

            valid_mask = np.isfinite(p_line) & np.isfinite(tau_line)
            surface_data[slice_name]['p_w'].extend(p_line[valid_mask])
            surface_data[slice_name]['tau_w'].extend(tau_line[valid_mask])

        except Exception as e:
            print(f"  [WARNING] Error processing {snapshot_file}: {e}")
            continue

if total_snapshots == 0:
    raise RuntimeError("No slice snapshots were processed.")

# =========================================================================
# Save surface data to HDF5 file
# =========================================================================
print("\n" + "=" * 70)
print("SAVING SURFACE DATA TO HDF5")
print("=" * 70)

# surface_data_file = os.path.join(OUTPUT_DIR, "surface_data_slices.h5")
surface_data_file = os.path.join(OUTPUT_DIR, "surface_data_slices_AoA5_Re50000.h5")

with h5py.File(surface_data_file, "w") as f:
    for slice_name, data in surface_data.items():
        grp = f.create_group(slice_name)
        
        p_samples = np.asarray(data['p_w'])
        tau_samples = np.asarray(data['tau_w'])
        
        grp.create_dataset("p_w", data=p_samples)
        grp.create_dataset("tau_w", data=tau_samples)
        
        # Store metadata
        grp.attrs["x_c"] = closest_interface_points[slice_name]["x_c"]
        grp.attrs["interface_x"] = closest_interface_points[slice_name]["point"][0]
        grp.attrs["interface_y"] = closest_interface_points[slice_name]["point"][1]
        grp.attrs["interface_index"] = closest_interface_points[slice_name]["index"]
        grp.attrs["y_grid_index"] = closest_interface_points[slice_name]["y_index"]
        grp.attrs["num_samples_p"] = len(p_samples)
        grp.attrs["num_samples_tau"] = len(tau_samples)
        
        print(f"  {slice_name}:")
        print(f"    - p_w: {len(p_samples)} samples, range [{np.min(p_samples):.4f}, {np.max(p_samples):.4f}]")
        print(f"    - tau_w: {len(tau_samples)} samples, range [{np.min(tau_samples):.4f}, {np.max(tau_samples):.4f}]")

print(f"\nSurface data saved to: {surface_data_file}")

# =========================================================================
# Compute histogram (PDF)
# =========================================================================
print("\n" + "=" * 70)
print("COMPUTING PDFs")
print("=" * 70)

pdf_data = {}

for slice_name, data in surface_data.items():
    if len(data['p_w']) == 0:
        print(f"  [WARNING] No samples collected for {slice_name}")
        continue

    p_samples = np.asarray(data['p_w'])
    tau_samples = np.asarray(data['tau_w'])

    p_hist, p_bin_edges = np.histogram(p_samples, bins=N_BINS, density=True)
    tau_hist, tau_bin_edges = np.histogram(tau_samples, bins=N_BINS, density=True)

    p_bin_centers = 0.5 * (p_bin_edges[:-1] + p_bin_edges[1:])
    tau_bin_centers = 0.5 * (tau_bin_edges[:-1] + tau_bin_edges[1:])

    pdf_data[slice_name] = {
        'p_bins': p_bin_centers,
        'p_pdf': p_hist,
        'tau_bins': tau_bin_centers,
        'tau_pdf': tau_hist,
    }

    print(f"\n  {slice_name}:")
    print(f"    Pressure PDF:")
    print(f"      Number of samples: {len(p_samples)}")
    print(f"      Data range:     [{np.min(p_samples):.4f}, {np.max(p_samples):.4f}]")
    print(f"      Max PDF value:  {np.max(p_hist):.4f}")
    print(f"    Shear stress PDF:")
    print(f"      Number of samples: {len(tau_samples)}")
    print(f"      Data range:     [{np.min(tau_samples):.4f}, {np.max(tau_samples):.4f}]")
    print(f"      Max PDF value:  {np.max(tau_hist):.4f}")

# =========================================================================
# Plot PDFs for each slice location
# =========================================================================
print("\n" + "=" * 70)
print("PLOTTING PDFs")
print("=" * 70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

for slice_name, pdf_info in pdf_data.items():
    x_c_slice = closest_interface_points[slice_name]['x_c']
    label = f"{slice_name} (x/c = {x_c_slice:.2f})"

    ax1.plot(pdf_info['p_bins'], pdf_info['p_pdf'], label=label)
    ax2.plot(pdf_info['tau_bins'], pdf_info['tau_pdf'], label=label)

ax1.set_xlabel("Pressure $p$")
ax1.set_ylabel("PDF")
ax1.grid(True)
ax1.legend()

ax2.set_xlabel("Shear stress $\\tau_w$")
ax2.set_ylabel("PDF")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

# =========================================================================
# Compute and Plot Normalized Fluctuation PDFs
# =========================================================================
print("\n" + "=" * 70)
print("COMPUTING NORMALIZED FLUCTUATION PDFs")
print("=" * 70)

normalized_pdf_data = {}

for slice_name, data in surface_data.items():
    if len(data['p_w']) == 0:
        continue

    p_samples = np.asarray(data['p_w'])
    tau_samples = np.asarray(data['tau_w'])

    p_mean_at_xc = np.mean(p_samples)
    tau_mean_at_xc = np.mean(tau_samples)
    p_std = np.std(p_samples)
    tau_std = np.std(tau_samples)

    if p_std == 0.0 or tau_std == 0.0:
        print(f"  [WARNING] Zero std for {slice_name}; skipping normalized PDFs")
        continue

    p_fluct_norm = (p_samples - p_mean_at_xc) / p_std
    tau_fluct_norm = (tau_samples - tau_mean_at_xc) / tau_std

    p_hist, p_bin_edges = np.histogram(p_fluct_norm, bins=N_BINS, density=True)
    tau_hist, tau_bin_edges = np.histogram(tau_fluct_norm, bins=N_BINS, density=True)

    p_bin_centers = 0.5 * (p_bin_edges[:-1] + p_bin_edges[1:])
    tau_bin_centers = 0.5 * (tau_bin_edges[:-1] + tau_bin_edges[1:])

    normalized_pdf_data[slice_name] = {
        'p_bins': p_bin_centers,
        'p_pdf': p_hist,
        'tau_bins': tau_bin_centers,
        'tau_pdf': tau_hist,
    }

    p_norm_mean = np.mean(p_fluct_norm)
    p_norm_var = np.var(p_fluct_norm)
    tau_norm_mean = np.mean(tau_fluct_norm)
    tau_norm_var = np.var(tau_fluct_norm)

    print(f"\n  {slice_name}:")
    print(f"    <p> = {p_mean_at_xc:.6f}, std(p) = {p_std:.6f}")
    print(f"    <tau_w> = {tau_mean_at_xc:.6f}, std(tau_w) = {tau_std:.6f}")
    print(f"    Normalized p' range: [{np.min(p_fluct_norm):.2f}, {np.max(p_fluct_norm):.2f}]")
    print(f"    Normalized tau_w' range: [{np.min(tau_fluct_norm):.2f}, {np.max(tau_fluct_norm):.2f}]")
    print(f"    --- Verification ---")
    print(f"    Normalized p':    mean = {p_norm_mean:.6f}, var = {p_norm_var:.6f}")
    print(f"    Normalized tau_w': mean = {tau_norm_mean:.6f}, var = {tau_norm_var:.6f}")

print("\n" + "=" * 70)
print("PLOTTING NORMALIZED FLUCTUATION PDFs")
print("=" * 70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

for slice_name, pdf_info in normalized_pdf_data.items():
    x_c_slice = closest_interface_points[slice_name]['x_c']
    label = f"{slice_name} (x/c = {x_c_slice:.2f})"

    ax1.plot(pdf_info['p_bins'], pdf_info['p_pdf'], label=label)
    ax2.plot(pdf_info['tau_bins'], pdf_info['tau_pdf'], label=label)

ax1.set_xlabel("Normalized Pressure Fluctuation $p'$")
ax1.set_ylabel("PDF")
ax1.grid(True)
ax1.legend()

ax2.set_xlabel("Normalized Shear Stress Fluctuation $\\tau_w'$")
ax2.set_ylabel("PDF")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()

