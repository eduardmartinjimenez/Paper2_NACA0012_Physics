"""
Test script to validate premultiplied spectra using Parseval's theorem.

Parseval's theorem (energy conservation):
  ∫|u'(z)|² dz = ∫|U(k)|² dk

For a one-sided spectrum with proper normalization:
  var_phys = <u'^2> (physical space variance)
  var_spec = ∫Φ(k) dk / (2π)  (spectral integration)
"""

import os
import sys
import h5py
import numpy as np
from pathlib import Path

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader
from scipy.spatial import cKDTree

# ============================================================================
# Configuration
# ============================================================================

GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA5_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

MESH_SLICE_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA5_Re50000_1716x1662x128/Slice_data/slice_5/"
MESH_SLICE_NAME = "slice_5-CROP-MESH.h5"
MESH_SLICE_FILE = os.path.join(MESH_SLICE_PATH, MESH_SLICE_NAME)

SLICES_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA5_Re50000_1716x1662x128/Slice_data/slice_5/"

AVG_SLICE_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA5_Re50000_1716x1662x128/Slice_data/slice_5/last_slice/"
AVG_SLICE_NAME =  "slice_5_19420800-COMP-DATA.h5"
AVG_SLICE_FILE = os.path.join(AVG_SLICE_PATH, AVG_SLICE_NAME)

# Reference parameters
rho_ref = 1.0
u_infty = 1.0
c = 1.0
Re_c = 50000
mu_ref = rho_ref * u_infty * c / Re_c
nu_ref = mu_ref / rho_ref

# ============================================================================
# Load geometrical data and mesh
# ============================================================================

print("=" * 80)
print("PARSEVAL THEOREM VALIDATION TEST")
print("=" * 80)

with h5py.File(GEO_FILE, "r") as f:
    interface_points = f["interface_points"][...].astype(np.float64)
    proj_normals = f["proj_normals"][...].astype(np.float64)
    proj_distances = f["proj_distances"][...].astype(np.float64)

suction_side_points = interface_points[interface_points[:, 1] >= 0]

loader = CompressedSnapshotLoader(MESH_SLICE_FILE)

x_data = loader.x[1:-1, :, :]
y_data = loader.y[1:-1, :, :]
z_data = loader.z[1:-1, :, :]

z_line = z_data[:, 0, 0]
nz = z_line.size
dz = z_line[1] - z_line[0]
L_z = dz * nz

print(f"\nMesh configuration:")
print(f"  nz={nz}, dz={dz:.6e} m, Lz={L_z:.6e} m")

slice_x = x_data[0, 0, 0]
x_distances = np.abs(suction_side_points[:, 0] - slice_x)
closest_idx = np.argmin(x_distances)
closest_interface_point = suction_side_points[closest_idx]
interface_y = closest_interface_point[1]

slice_y_unique = np.unique(y_data[:, :, 0][0, :])
y_distances = np.abs(slice_y_unique - interface_y)
j_closest = np.argmin(y_distances)

j_start_fluid = j_closest
slice_y_unique_fluid = slice_y_unique[j_start_fluid:]
ny_fluid = len(slice_y_unique_fluid)

tree_full = cKDTree(interface_points[:, :2])
_, idx_full = tree_full.query(closest_interface_point[:2])
normal_at_closest_point = proj_normals[idx_full]
distance_at_closest_point = proj_distances[idx_full]

tangent_at_closest_point = np.array([normal_at_closest_point[1], -normal_at_closest_point[0], 0.0])
tangent_at_closest_point /= np.linalg.norm(tangent_at_closest_point)

# Load average file for friction velocity
if AVG_SLICE_FILE and os.path.exists(AVG_SLICE_FILE):
    avg_fields = loader.load_snapshot_avg(AVG_SLICE_FILE)
    
    avg_u_data = loader.reconstruct_field(avg_fields["avg_u"])
    avg_v_data = loader.reconstruct_field(avg_fields["avg_v"])
    avg_w_data = loader.reconstruct_field(avg_fields["avg_w"])
    
    avg_u_data = avg_u_data[1:-1, :, :]
    avg_v_data = avg_v_data[1:-1, :, :]
    avg_w_data = avg_w_data[1:-1, :, :]
    
    avg_u_t = (avg_u_data * tangent_at_closest_point[0] +
               avg_v_data * tangent_at_closest_point[1] +
               avg_w_data * tangent_at_closest_point[2])
    
    span_avg_avg_u_t = np.nanmean(avg_u_t, axis=0)
    u_t_closest_avg = span_avg_avg_u_t[j_closest, 0]
    span_avg_avg_u_t_fluid = span_avg_avg_u_t[j_start_fluid:, :]
    
    tau_w_closest = mu_ref * u_t_closest_avg / distance_at_closest_point
    u_tau = np.sqrt(np.abs(tau_w_closest) / rho_ref)
    
    print(f"  Friction velocity: u_tau={u_tau:.6e} m/s")
else:
    print("ERROR: Average file not found")
    sys.exit(1)

# ============================================================================
# Get first snapshot file
# ============================================================================

slice_files = sorted(Path(SLICES_PATH).glob("slice_*-COMP-DATA.h5"))
slice_files = [f for f in slice_files if "avg" not in f.name]

if len(slice_files) == 0:
    print("No snapshot files found!")
    sys.exit(1)

snapshot_file = str(slice_files[0])
print(f"\nLoading first snapshot: {Path(snapshot_file).name}")

# ============================================================================
# Load single snapshot and compute FFT
# ============================================================================

fields = loader.load_snapshot(snapshot_file)

u_data = loader.reconstruct_field(fields["u"])
v_data = loader.reconstruct_field(fields["v"])
w_data = loader.reconstruct_field(fields["w"])

u_data = u_data[1:-1, :, :]
v_data = v_data[1:-1, :, :]
w_data = w_data[1:-1, :, :]

u_t = (u_data * tangent_at_closest_point[0] +
       v_data * tangent_at_closest_point[1] +
       w_data * tangent_at_closest_point[2])

u_t_fluid = u_t[:, j_start_fluid:, :]
# u_t_fluct = u_t_fluid - span_avg_avg_u_t_fluid
u_t_fluct = u_t_fluid - np.mean(u_t_fluid, axis=0)  # Remove spanwise mean at each y location


# ============================================================================
# Test Parseval at multiple y+ locations
# ============================================================================

print("\n" + "=" * 80)
print("PARSEVAL THEOREM VALIDATION")
print("=" * 80)

# Wavenumber
kz_full = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz)
pos_mask = kz_full > 0.0
kz_pos = kz_full[pos_mask]
dk = kz_pos[1] - kz_pos[0] if len(kz_pos) > 1 else 1.0

print(f"\nWavenumber parameters:")
print(f"  Positive kz bins: {len(kz_pos)}")
print(f"  dk (spacing): {dk:.6e} rad/m")

# Test at several y locations
test_y_indices = [5, 10, 20, 50, 100]
test_y_indices = [j for j in test_y_indices if j < len(slice_y_unique_fluid)]

print(f"\nTesting {len(test_y_indices)} wall-normal locations:\n")

for j_test in test_y_indices:
    u_line = u_t_fluct[:, j_test, 0]  # length nz
    y_plus = u_tau * slice_y_unique_fluid[j_test] / nu_ref
    
    # Physical space variance
    # var_phys = 0.5*np.mean((u_line-np.mean(u_line)) ** 2)
    # var_phys = np.mean((u_line-np.mean(u_line)) ** 2)
    var_phys = np.mean(u_line ** 2)
    
    # Spectral computation (same as in main script)
    # U_full = np.fft.fft(u_line) / nz
    U_full = np.fft.fft(u_line)
    Phi2_full = (dz / nz) * (np.abs(U_full) ** 2)  # two-sided variance density
    # Phi2_full = (1.0 / nz) * (np.abs(U_full) ** 2) 
    # Phi2_full = 0.5 * (np.abs(U_full) ** 2)  # one-sided variance density with normalization
    Phi1_pos = 2.0 * Phi2_full[pos_mask]  # one-sided
    
    # Spectral variance via integration
    # var_spec = np.sum(Phi1_pos)*dk
    var_spec = np.sum(Phi1_pos) * dk / (2.0 * np.pi)

    # Error
    error = abs(var_spec - var_phys) / var_phys * 100
    
    print(f"  y+ = {y_plus:8.2f} (j={j_test:3d})")
    print(f"    Physical space variance:  {var_phys:.6e}")
    print(f"    Spectral space variance:  {var_spec:.6e}")
    print(f"    Relative error:           {error:.4f} %")
    print()

# ============================================================================
# Detailed analysis at one location
# ============================================================================

j_detailed = test_y_indices[len(test_y_indices) // 2]  # Middle location
u_line = u_t_fluct[:, j_detailed, 0]
y_plus_detail = u_tau * slice_y_unique_fluid[j_detailed] / nu_ref

print("=" * 80)
print(f"DETAILED ANALYSIS AT y+ = {y_plus_detail:.2f} (j={j_detailed})")
print("=" * 80)

# var_phys = 0.5 * np.mean((u_line-np.mean(u_line)) ** 2)
# var_phys = np.mean((u_line-np.mean(u_line)) ** 2)
var_phys = np.mean(u_line ** 2)


# U_full = np.fft.fft(u_line) / nz
# Phi2_full = 0.5 * (np.abs(U_full) ** 2)  # one-sided variance density with normalization
U_full = np.fft.fft(u_line)
Phi2_full = (dz / nz) * (np.abs(U_full) ** 2)  # two-sided variance density

Phi1_pos = 2.0 * Phi2_full[pos_mask]  # one-sided
# var_spec = np.sum(Phi1_pos) * dk
var_spec = np.sum(Phi1_pos) * dk / (2.0 * np.pi)

print(f"\nPhysical space:")
print(f"  <u'^2> = {var_phys:.6e}")
print(f"  min(u') = {u_line.min():.6e}")
print(f"  max(u') = {u_line.max():.6e}")
print(f"  std(u') = {np.std(u_line):.6e}")

print(f"\nSpectral space (one-sided):")
print(f"  Φ1(k) range: [{Phi1_pos.min():.6e}, {Phi1_pos.max():.6e}]")
print(f"  ∑Φ1(k)*dk/(2π) = {var_spec:.6e}")

print(f"\nParseval verification:")
print(f"  var_phys  = {var_phys:.6e}")
print(f"  var_spec  = {var_spec:.6e}")
print(f"  Ratio (spec/phys) = {var_spec/var_phys:.6f}")
print(f"  Relative error    = {abs(var_spec - var_phys) / var_phys * 100:.4f} %")

if abs(var_spec - var_phys) / var_phys < 0.01:
    print(f"\n✓ PARSEVAL THEOREM VALIDATED (error < 1%)")
else:
    print(f"\n✗ WARNING: Large error detected")

print("\n" + "=" * 80)
