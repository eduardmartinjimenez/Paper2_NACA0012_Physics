"""
Timing Test Script - Comparing Region Filtering Performance

This script tests the optimized CompressedSnapshotLoader with timing for:
1. Full domain loading (with ghost cell exclusion)
2. Region-specific loading
3. Performance comparisons
"""

import numpy as np
import h5py
import os
import time
from data_loader_functions_cropp import CompressedSnapshotLoader


print("\n" + "="*80)
print("TIMING TEST: OPTIMIZED LOADER (WITH REGION FILTERING)")
print("="*80 + "\n")

# Define paths
MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
MESH_NAME = "3d_NACA0012_Re50000_AoA12-CROP-MESH.h5"
MESH_FILE = os.path.join(MESH_PATH, MESH_NAME)

SNAPSHOT_PATH_AVG = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/last_snapshot/"
SNAPSHOT_NAME_AVG = "3d_NACA0012_Re50000_AoA12_avg_26500000-COMP-DATA.h5"
SNAPSHOT_FILE_AVG = os.path.join(SNAPSHOT_PATH_AVG, SNAPSHOT_NAME_AVG)

SNAPSHOT_PATH_PRI = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/Steady_state/batch_30658504/"
SNAPSHOT_NAME_PRI = "3d_NACA0012_Re50000_AoA12_6350000-COMP-DATA.h5"
SNAPSHOT_FILE_PRI = os.path.join(SNAPSHOT_PATH_PRI, SNAPSHOT_NAME_PRI)

# Check files
if os.path.exists(MESH_FILE):
    print(f"Mesh file exists! {MESH_FILE}")
else:
    print(f"Mesh file does not exist: {MESH_FILE}")
    exit(1)

if os.path.exists(SNAPSHOT_FILE_AVG):
    print(f"Average data file exists!")
else:
    print(f"Average Data file does not exist!")

if os.path.exists(SNAPSHOT_FILE_PRI):
    print(f"Primitive data file exists!")
else:
    print(f"Primitive Data file does not exist!")


# ============================================================================
# PART 1: TEST FULL DOMAIN LOADING
# ============================================================================
print("\n" + "="*80)
print("PART 1: FULL DOMAIN LOADING (default: exclude_z_ghosts=True)")
print("="*80)

print("\n" + "-"*80)
print("TIMER 1A: Initialize loader (load whole mesh)")
print("-"*80)
t_start = time.time()
loader_full = CompressedSnapshotLoader(MESH_FILE)
t_mesh_full = time.time() - t_start
print(f"✓ Time to load mesh: {t_mesh_full:.4f} seconds")

# Get coordinates
x_full = loader_full.x
y_full = loader_full.y
z_full = loader_full.z
tag_ibm_full = loader_full.tag_ibm

print(f"  Mesh shape: {x_full.shape}, Fluid points: {loader_full.N_points:,}")

print("\n" + "-"*80)
print("TIMER 2A: Load primitive snapshot (full domain)")
print("-"*80)
t_start = time.time()
fields_full = loader_full.load_snapshot(SNAPSHOT_FILE_PRI)
t_snapshot_pri_full = time.time() - t_start
print(f"✓ Time to load primitive snapshot: {t_snapshot_pri_full:.4f} seconds")

print("\n" + "-"*80)
print("TIMER 3A: Load averaged snapshot (full domain)")
print("-"*80)
t_start = time.time()
fields_avg_full = loader_full.load_snapshot_avg(SNAPSHOT_FILE_AVG)
t_snapshot_avg_full = time.time() - t_start
print(f"✓ Time to load averaged snapshot: {t_snapshot_avg_full:.4f} seconds")

print("\n" + "-"*80)
print("TIMER 4A: Reconstruct full 3D velocity field (u)")
print("-"*80)
t_start = time.time()
u_pri_full = loader_full.reconstruct_field(fields_full["u"])
t_reconstruct_full = time.time() - t_start
print(f"✓ Time to reconstruct u field: {t_reconstruct_full:.4f} seconds")

# Reconstruct other fields (not timed)
u_inst_full = loader_full.reconstruct_field(fields_avg_full["u"])
u_avg_full = loader_full.reconstruct_field(fields_avg_full["avg_u"])

# Summary
print("\n" + "="*80)
print("TIMING SUMMARY - FULL DOMAIN")
print("="*80)
total_full = t_mesh_full + t_snapshot_pri_full + t_snapshot_avg_full + t_reconstruct_full
print(f"  1. Mesh loading:              {t_mesh_full:.4f} seconds")
print(f"  2. Primitive snapshot:        {t_snapshot_pri_full:.4f} seconds")
print(f"  3. Averaged snapshot:         {t_snapshot_avg_full:.4f} seconds")
print(f"  4. Field reconstruction (u):  {t_reconstruct_full:.4f} seconds")
print(f"  ---")
print(f"  TOTAL TIME:                   {total_full:.4f} seconds")
print("="*80 + "\n")


# ============================================================================
# PART 2: TEST REGION-SPECIFIC LOADING
# ============================================================================
print("\n" + "="*80)
print("PART 2: REGION-SPECIFIC LOADING")
print("="*80)

# Define region
REGION = (-0.2, 1.2, -0.3, 0.3, 0.0, 0.1)
x_min, x_max, y_min, y_max, z_min, z_max = REGION

print(f"\nRequested region (physical coordinates):")
print(f"  X: [{x_min:.4f}, {x_max:.4f}]")
print(f"  Y: [{y_min:.4f}, {y_max:.4f}]")
print(f"  Z: [{z_min:.4f}, {z_max:.4f}]")

print("\n" + "-"*80)
print("TIMER 1B: Initialize loader with region (load + filter mesh)")
print("-"*80)
t_start = time.time()
loader_region = CompressedSnapshotLoader(MESH_FILE, region=REGION)
t_mesh_region = time.time() - t_start
print(f"✓ Time to load and filter mesh: {t_mesh_region:.4f} seconds")

# Get coordinates
x_region = loader_region.x
y_region = loader_region.y
z_region = loader_region.z
tag_ibm_region = loader_region.tag_ibm

print(f"\n  Memory savings:")
print(f"    Grid points: {x_region.size:,} vs {x_full.size:,} (reduction: {100*(1 - x_region.size/x_full.size):.1f}%)")
print(f"    Fluid points: {loader_region.N_points:,} vs {loader_full.N_points:,} (reduction: {100*(1 - loader_region.N_points/loader_full.N_points):.1f}%)")

print("\n" + "-"*80)
print("TIMER 2B: Load primitive snapshot (region)")
print("-"*80)
t_start = time.time()
fields_region = loader_region.load_snapshot(SNAPSHOT_FILE_PRI)
t_snapshot_pri_region = time.time() - t_start
print(f"✓ Time to load primitive snapshot: {t_snapshot_pri_region:.4f} seconds")
print(f"  Speedup vs full domain: {t_snapshot_pri_full/t_snapshot_pri_region:.2f}x")

print("\n" + "-"*80)
print("TIMER 3B: Load averaged snapshot (region)")
print("-"*80)
t_start = time.time()
fields_avg_region = loader_region.load_snapshot_avg(SNAPSHOT_FILE_AVG)
t_snapshot_avg_region = time.time() - t_start
print(f"✓ Time to load averaged snapshot: {t_snapshot_avg_region:.4f} seconds")
print(f"  Speedup vs full domain: {t_snapshot_avg_full/t_snapshot_avg_region:.2f}x")

print("\n" + "-"*80)
print("TIMER 4B: Reconstruct 3D velocity field (u) for region")
print("-"*80)
t_start = time.time()
u_pri_region = loader_region.reconstruct_field(fields_region["u"])
t_reconstruct_region = time.time() - t_start
print(f"✓ Time to reconstruct u field: {t_reconstruct_region:.4f} seconds")
print(f"  Speedup vs full domain: {t_reconstruct_full/t_reconstruct_region:.2f}x")

# Reconstruct other fields (not timed)
u_inst_region = loader_region.reconstruct_field(fields_avg_region["u"])
u_avg_region = loader_region.reconstruct_field(fields_avg_region["avg_u"])

# Summary
print("\n" + "="*80)
print("TIMING SUMMARY - REGION LOADING")
print("="*80)
total_region = t_mesh_region + t_snapshot_pri_region + t_snapshot_avg_region + t_reconstruct_region
print(f"  1. Mesh loading + filtering:  {t_mesh_region:.4f} seconds")
print(f"  2. Primitive snapshot:        {t_snapshot_pri_region:.4f} seconds")
print(f"  3. Averaged snapshot:         {t_snapshot_avg_region:.4f} seconds")
print(f"  4. Field reconstruction (u):  {t_reconstruct_region:.4f} seconds")
print(f"  ---")
print(f"  TOTAL TIME:                   {total_region:.4f} seconds")
print("="*80 + "\n")


# ============================================================================
# PART 3: OVERALL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("PERFORMANCE COMPARISON")
print("="*80)

speedup = total_full / total_region

print(f"\nOverall Performance:")
print(f"  Full domain time:   {total_full:.4f} seconds")
print(f"  Region time:        {total_region:.4f} seconds")
print(f"  Overall speedup:    {speedup:.2f}x")
print(f"  Time saved:         {total_full - total_region:.4f} seconds ({100*(total_full-total_region)/total_full:.1f}%)")

print(f"\nDetailed breakdown:")
print(f"  {'Operation':<25} {'Full (s)':<12} {'Region (s)':<12} {'Speedup':<10}")
print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
print(f"  {'Mesh loading':<25} {t_mesh_full:<12.4f} {t_mesh_region:<12.4f} {t_mesh_full/t_mesh_region:<10.2f}x")
print(f"  {'Primitive snapshot':<25} {t_snapshot_pri_full:<12.4f} {t_snapshot_pri_region:<12.4f} {t_snapshot_pri_full/t_snapshot_pri_region:<10.2f}x")
print(f"  {'Averaged snapshot':<25} {t_snapshot_avg_full:<12.4f} {t_snapshot_avg_region:<12.4f} {t_snapshot_avg_full/t_snapshot_avg_region:<10.2f}x")
print(f"  {'Field reconstruction':<25} {t_reconstruct_full:<12.4f} {t_reconstruct_region:<12.4f} {t_reconstruct_full/t_reconstruct_region:<10.2f}x")

print("\n" + "="*80 + "\n")


# ============================================================================
# VERIFICATION
# ============================================================================
print("--- Data Verification ---")
print(f"\nFull Domain:")
print(f"  Mesh: {x_full.shape}, u_avg: {u_avg_full.shape}")
print(f"  u_avg range: [{np.nanmin(u_avg_full):.6f}, {np.nanmax(u_avg_full):.6f}], NaN: {np.isnan(u_avg_full).sum():,}")

print(f"\nRegion:")
print(f"  Mesh: {x_region.shape}, u_avg: {u_avg_region.shape}")
print(f"  u_avg range: [{np.nanmin(u_avg_region):.6f}, {np.nanmax(u_avg_region):.6f}], NaN: {np.isnan(u_avg_region).sum():,}")

# Verify coordinates within bounds
x_pts, y_pts, z_pts = loader_region.get_coordinates()
x_ok = np.all((x_pts >= x_min) & (x_pts <= x_max))
y_ok = np.all((y_pts >= y_min) & (y_pts <= y_max))
z_ok = np.all((z_pts >= z_min) & (z_pts <= z_max))
print(f"\nRegion bounds check: X={'✓' if x_ok else '✗'}, Y={'✓' if y_ok else '✗'}, Z={'✓' if z_ok else '✗'}")

print("\n--- Verification Complete ---\n")
