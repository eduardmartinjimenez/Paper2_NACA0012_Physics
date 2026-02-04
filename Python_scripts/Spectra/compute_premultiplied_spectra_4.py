import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader
from scipy.spatial import cKDTree

# ============================================================================
# Configuration
# ============================================================================

# Save results
SAVE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/Premultiplied_spectra/"
SAVE_NAME = "premultiplied_spectra_data_9.h5"
SAVE_PATH = os.path.join(SAVE_DIR, SAVE_NAME)

# Geometrical data file
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA5_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

# Mesh slice path
MESH_SLICE_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Slices_data/slices_batch_1/slice_9_compr/"
# MESH_SLICE_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Slices_data/slices_test/last_slice/"
MESH_SLICE_NAME = "slice_9-CROP-MESH.h5"
MESH_SLICE_FILE = os.path.join(MESH_SLICE_PATH, MESH_SLICE_NAME)

# Slices data path
SLICES_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Slices_data/slices_batch_1/slice_9_compr/"
# SLICES_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Slices_data/slices_test/"

# Average data path
AVG_SLICE_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Slices_data/slices_batch_1/slice_9_compr/last_slice"
AVG_SLICE_NAME = "slice_9_14302400-COMP-DATA.h5"
# AVG_SLICE_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Slices_data/slices_test/last_slice/"
# AVG_SLICE_NAME =  "slice_1_14302400-COMP-DATA.h5"
AVG_SLICE_FILE = os.path.join(AVG_SLICE_PATH, AVG_SLICE_NAME)

# Reference parameters
rho_ref = 1.0  # Reference density [kg/m3]
u_infty = 1.0  # Free-stream velocity [m/s]
c = 1.0  # Airfoil chord length [m]
Re_c = 50000  # Reynolds number [-]
mu_ref = rho_ref * u_infty * c / Re_c  # Dynamic viscosity [Pa s]
nu_ref = mu_ref / rho_ref  # Kinetic viscosity [m2/s]

# ============================================================================
# Utilities
# ============================================================================

def assert_exists(path: str, kind: str = "File") -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{kind} does not exist: {path}")
    print(f"{kind} exists: {path}")

def get_slice_files(slices_path: str) -> list:
    """Get all snapshot slice files in directory, sorted by timestamp."""
    slice_files = []
    for file in sorted(Path(slices_path).glob("slice_*-COMP-DATA.h5")):
        if "avg" not in file.name:  # Exclude average files
            slice_files.append(str(file))
    return slice_files

# ============================================================================
# Load geometrical data and mesh
# ============================================================================

assert_exists(GEO_FILE, "Geometrical data file")
assert_exists(MESH_SLICE_FILE, "Mesh slice file")

# Load geometrical data
with h5py.File(GEO_FILE, "r") as f:
    interface_points = f["interface_points"][...].astype(np.float64)
    proj_normals = f["proj_normals"][...].astype(np.float64)
    proj_distances = f["proj_distances"][...].astype(np.float64)

# Filter suction side interface points (y >= 0)
suction_side_points = interface_points[interface_points[:, 1] >= 0]

# Load compressed slice data (mesh)
loader = CompressedSnapshotLoader(MESH_SLICE_FILE)

# Load mesh slice data (exclude ghost cells at spanwise boundaries)
x_data = loader.x[1:-1, :, :]
y_data = loader.y[1:-1, :, :]
z_data = loader.z[1:-1, :, :]
print(f"Mesh shape: {x_data.shape}")

# Spanwise domain parameters
z_line = z_data[:, 0, 0]
nz = z_line.size
dz = abs(z_line[1] - z_line[0])
L_z = dz * nz  # Spanwise domain length

print(f"Spanwise domain: nz={nz}, dz={dz:.6e} m, Lz={L_z:.6e} m")

# Get the slice's x coordinate (all same, take any)
slice_x = x_data[0, 0, 0]
print(f"Slice x coordinate: {slice_x:.6f}")

# Find the interface point on suction side closest to this x
x_distances = np.abs(suction_side_points[:, 0] - slice_x)
closest_idx = np.argmin(x_distances)
closest_interface_point = suction_side_points[closest_idx]
interface_y = closest_interface_point[1]

print(f"Closest interface point: ({closest_interface_point[0]:.6f}, {interface_y:.6f})")

# Find the slice y grid index closest to the interface y coordinate
slice_y_unique = np.unique(y_data[:, :, 0][0, :])  # Extract y values along wall-normal direction
y_distances = np.abs(slice_y_unique - interface_y)
j_closest = np.argmin(y_distances)
slice_y_at_interface = slice_y_unique[j_closest]

print(f"Slice y grid index closest to interface: {j_closest}")
print(f"Slice y value at interface: {slice_y_at_interface:.6f}")

# Crop y coordinates to only include fluid points (from interface outwards)
j_start_fluid = j_closest
slice_y_unique_fluid = slice_y_unique[j_start_fluid:]
ny_fluid = len(slice_y_unique_fluid)
print(f"Fluid region: {ny_fluid} points from j={j_start_fluid} onwards")

# Extract the normal direction at the closest interface point
tree_full = cKDTree(interface_points[:, :2])
_, idx_full = tree_full.query(closest_interface_point[:2])
normal_at_closest_point = proj_normals[idx_full]
distance_at_closest_point = proj_distances[idx_full]

# Compute tangential direction at the closest interface point
tangent_at_closest_point = np.array([normal_at_closest_point[1], -normal_at_closest_point[0], 0.0])
tangent_at_closest_point /= np.linalg.norm(tangent_at_closest_point)

print(f"Normal at closest interface point: {normal_at_closest_point}")
print(f"Tangent at closest interface point: {tangent_at_closest_point}")
print(f"Wall distance at closest interface point: {distance_at_closest_point:.6e}")

# ============================================================================
# Load average slice data to compute friction velocity
# ============================================================================

if AVG_SLICE_FILE and os.path.exists(AVG_SLICE_FILE):
    print(f"\nLoading average file: {AVG_SLICE_FILE}")
    avg_fields = loader.load_snapshot_avg(AVG_SLICE_FILE)
    
    # Reconstruct average fields
    avg_u_data = loader.reconstruct_field(avg_fields["avg_u"])
    avg_v_data = loader.reconstruct_field(avg_fields["avg_v"])
    avg_w_data = loader.reconstruct_field(avg_fields["avg_w"])
    
    # Exclude ghost cells from average data
    avg_u_data = avg_u_data[1:-1, :, :]
    avg_v_data = avg_v_data[1:-1, :, :]
    avg_w_data = avg_w_data[1:-1, :, :]
    
    # Decompose average velocity into tangential component
    avg_u_t = (avg_u_data * tangent_at_closest_point[0] +
               avg_v_data * tangent_at_closest_point[1] +
               avg_w_data * tangent_at_closest_point[2])
    
    # Spanwise average of average tangential velocity (average over first axis = z)
    span_avg_avg_u_t = np.nanmean(avg_u_t, axis=0)
    u_t_closest_avg = span_avg_avg_u_t[j_closest, 0]
    
    # Crop to fluid region only
    span_avg_avg_u_t_fluid = span_avg_avg_u_t[j_start_fluid:, :]
    
    # Compute wall shear stress and friction velocity
    tau_w_closest = mu_ref * u_t_closest_avg / distance_at_closest_point
    u_tau = np.sqrt(np.abs(tau_w_closest) / rho_ref)
    y_plus_interface = u_tau * distance_at_closest_point / nu_ref
    
    print(f"Wall shear stress: {tau_w_closest:.6e} Pa")
    print(f"Friction velocity: {u_tau:.6e} m/s")
    print(f"y+ at interface: {y_plus_interface:.6e}")
else:
    print("Warning: Average file not found")
    sys.exit(1)

# ============================================================================
# Load and process snapshot slices
# ============================================================================

slice_files = get_slice_files(SLICES_PATH)
print(f"\nFound {len(slice_files)} snapshot files")

if len(slice_files) == 0:
    print("No snapshot files found!")
    sys.exit(1)

# Spanwise wavenumber
kz_full = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz)   # (nz,) rad/m two-sided
pos_mask = kz_full > 0.0    # keep only kz>0 (drop kz=0, negative)
kz_pos = kz_full[pos_mask]  # (nk_pos,)
kz_plus_pos = kz_pos * nu_ref / u_tau   # inner-scaled positive side
nk_pos = kz_pos.size

print(f"nz={nz}, dz={dz:.3e}, Lz={L_z:.3e}")
print(f"Positive kz bins: {nk_pos}, kz range: {kz_pos.min():.3e} to {kz_pos.max():.3e} rad/m")

# Use only fluid region for spectral analysis
ny = ny_fluid
y_unique = slice_y_unique_fluid
print(f"Fluid region y locations: {ny}")

# PSD accumulator: one-sided (positive kz only)
E_kz = np.zeros((ny, nk_pos), dtype=np.float64)  # Energy spectral density
snapshot_count = 0

print(f"\nProcessing {len(slice_files)} snapshots...")
print(f"Spanwise resolution: {nz}")
print(f"Wall-normal fluid grid points: {ny}")

# ============================================================================
# Main loop through snapshots
# ============================================================================

# Variance check settings (print once for debugging)
j_test = min(10, ny - 1)     # choose a y-index not too close to the wall (in fluid region)
do_variance_check = True

for i, slice_file in enumerate(slice_files):
    if not os.path.exists(slice_file):
        print(f"File not found: {slice_file}")
        continue
    
    try:
        # Load snapshot data
        fields = loader.load_snapshot(slice_file)
        
        # Reconstruct fields
        u_data = loader.reconstruct_field(fields["u"])
        v_data = loader.reconstruct_field(fields["v"])
        w_data = loader.reconstruct_field(fields["w"])
        
        # Exclude ghost cells from snapshot data
        u_data = u_data[1:-1, :, :]
        v_data = v_data[1:-1, :, :]
        w_data = w_data[1:-1, :, :]
        
        # Decompose velocity into tangential component
        u_t = (u_data * tangent_at_closest_point[0] +
               v_data * tangent_at_closest_point[1] +
               w_data * tangent_at_closest_point[2])
        
        # Extract only fluid region
        u_t_fluid = u_t[:, j_start_fluid:, :]
        
        # Compute fluctuations by subtracting the mean field
        u_t_fluct = u_t_fluid - span_avg_avg_u_t_fluid
        
        # Spanwise FFT for each y location in fluid region
        for j in range(ny):
            u_line = u_t_fluct[:, j, 0] # real signal u'(z), length nz
           
            U_full = np.fft.fft(u_line) # two-sided FFT, length nz
            Phi2_full = (dz / nz) * (np.abs(U_full) ** 2)   # two-sided variance density
            
            # Keep only kz>0 and apply one-sided correction (x2)
            Phi1_pos = 2.0 * Phi2_full[pos_mask]    # length nk_pos
            E_kz[j, :] += Phi1_pos

            # Variance consistency check (only on first snapshot)
            if do_variance_check and i == 0 and j == j_test:
                var_phys = float(np.mean(u_line**2))
                var_spec = float(np.sum(Phi1_pos) / L_z)

                print(f"\n[Variance check at j={j_test}]")
                print(f"  Physical space: {var_phys:.6e}")
                print(f"  Spectral space: {var_spec:.6e}")
                print(f"  Ratio: {var_spec/var_phys:.6f}\n")

                do_variance_check = False

        snapshot_count += 1
        
        if (i + 1) % max(1, len(slice_files) // 10) == 0:
            print(f"  Processed {i + 1}/{len(slice_files)} snapshots")
    
    except Exception as e:
        print(f"Error processing {slice_file}: {e}")
        continue

if snapshot_count == 0:
    print("No snapshots were successfully processed!")
    sys.exit(1)

# Average over snapshots
E_kz /= snapshot_count

print(f"\nProcessed {snapshot_count} snapshots successfully")

# ============================================================================
# Compute inner-scaled premultiplied spectra
# ============================================================================

# Premultiplied spectrum: kz * Phi / u_tau^2  (more standard than kz_plus * Phi)
premult_pos = (kz_pos[np.newaxis, :] * E_kz) / (u_tau ** 2)   # (ny, nk_pos)

print(f"Wavenumber range: {kz_pos.min():.2e} to {kz_pos.max():.2e}")
print(f"Spectrum range: {np.nanmin(premult_pos):.2e} to {np.nanmax(premult_pos):.2e}")

# ============================================================================
# Create visualization
# ============================================================================

# Compute y+ for all wall-normal locations
y_plus_all = u_tau * y_unique / nu_ref

# Create contour plot with publication-quality formatting
fig, ax = plt.subplots(figsize=(12, 8))

# Use logarithmic scale for wavelength (lambda_z^+ = 2*pi / k_z^+)
lambda_z_plus = (2.0 * np.pi) / kz_plus_pos
log10_lambda_z_plus = np.log10(lambda_z_plus)
X, Y = np.meshgrid(log10_lambda_z_plus, y_plus_all)
Z = premult_pos

# Create filled contour plot with smooth levels
n_levels = 30
levels = np.linspace(np.nanmin(Z), np.nanmax(Z), n_levels)
contour_fill = ax.contourf(X, Y, Z, levels=levels, cmap='RdYlBu_r', extend='both')

# Add contour lines for better visibility
contour_lines = ax.contour(X, Y, Z, levels=[1.0, 3.8], colors='black', linewidths=2.0, alpha=0.7)
ax.clabel(contour_lines, inline=True, fontsize=10, fmt='%.1f')

# Add colorbar with proper formatting
cbar = plt.colorbar(contour_fill, ax=ax, pad=0.02)
cbar.set_label(r'$k_z\,\Phi_{u_tu_t}/u_\tau^2$', fontsize=12, labelpad=15)
cbar.ax.tick_params(labelsize=10)

# Set axis labels and scales
ax.set_yscale('log')
ax.set_ylabel(r'$y^+$', fontsize=12)
ax.set_xlabel(r'$\log_{10}(\lambda_z^+)$', fontsize=12)
ax.tick_params(which='both', labelsize=11)

# Professional title
ax.set_title(f'Inner-scaled spanwise premultiplied power spectral density\n' + 
             f'$x_{{ss}}/c = {slice_x:.3f}$, $N_{{snapshots}} = {snapshot_count}$',
             fontsize=13, pad=15)

# Improve grid visibility
ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
ax.grid(True, which='major', alpha=0.5, linestyle='-', linewidth=0.7)

# Set reasonable axis limits
ax.set_xlim(log10_lambda_z_plus.min(), log10_lambda_z_plus.max())
ax.set_ylim(y_plus_all.min(), y_plus_all.max())

plt.tight_layout()
# plt.savefig(os.path.join(SLICES_PATH, "premultiplied_spectra.png"), dpi=300, bbox_inches='tight')
# print(f"\nFigure saved to: {os.path.join(SLICES_PATH, 'premultiplied_spectra.png')}")
plt.show()

# ============================================================================
# Save results
# ============================================================================

output_file = os.path.join(SAVE_DIR, SAVE_NAME)
with h5py.File(output_file, "w") as f:
    f.create_dataset("kz_plus", data=kz_plus_pos)
    f.create_dataset("y_plus", data=y_plus_all)
    f.create_dataset("premultiplied_psd", data=premult_pos)
    f.attrs["slice_x"] = slice_x
    f.attrs["u_tau"] = u_tau
    f.attrs["snapshot_count"] = snapshot_count
    f.attrs["spanwise_domain"] = L_z

print(f"Data saved to: {output_file}")
