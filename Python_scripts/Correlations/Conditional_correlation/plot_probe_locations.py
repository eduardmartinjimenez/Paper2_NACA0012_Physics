"""
Quick visualization of the surface reference point and the velocity probe.
No snapshot data is loaded - only geometry and mesh files are needed.
Keep X_C_REF, PROBE_X, PROBE_Y in sync with wall_shear_probe_validation.py.
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader

# ============================================================================
# Parameters  (keep in sync with wall_shear_probe_validation.py)
# ============================================================================

X_C_REF = 0.5    # x/c of the surface reference point (suction side)
PROBE_X = -0.2   # target probe x coordinate
PROBE_Y =  0.5   # target probe y coordinate

MESH_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
MESH_FILE = os.path.join(MESH_PATH, "3d_NACA0012_Re50000_AoA12-CROP-MESH.h5")

GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
GEO_FILE = os.path.join(GEO_PATH, "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5")

OUTPUT_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Wall_shear_correlations/Probe_validation/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# Load geometry
# ============================================================================
with h5py.File(GEO_FILE, "r") as f:
    interface_points = f["interface_points"][:]

x_iface = interface_points[:, 0]
y_iface = interface_points[:, 1]
upper_mask = y_iface > np.mean(y_iface)

# Find nearest surface point on suction side
upper_idx   = np.where(upper_mask)[0]
surf_idx    = upper_idx[np.argmin(np.abs(x_iface[upper_idx] - X_C_REF))]
surf_x      = float(x_iface[surf_idx])
surf_y      = float(y_iface[surf_idx])

print(f"Surface ref  : target x/c={X_C_REF:.2f}  ->  actual ({surf_x:.4f}, {surf_y:.4f})")

# ============================================================================
# Load mesh and find nearest probe node (midplane)
# ============================================================================
loader   = CompressedSnapshotLoader(MESH_FILE)
x_data   = loader.x[1:-1, :, :]   # (Nz_phys, Ny, Nx)
y_data   = loader.y[1:-1, :, :]

Nz_phys      = x_data.shape[0]
midplane_idx = Nz_phys // 2

x_mid = x_data[midplane_idx, :, :]   # (Ny, Nx)
y_mid = y_data[midplane_idx, :, :]

dist_2d  = np.sqrt((x_mid - PROBE_X) ** 2 + (y_mid - PROBE_Y) ** 2)
flat_idx = np.argmin(dist_2d)
iy_p, ix_p = np.unravel_index(flat_idx, dist_2d.shape)

probe_x = float(x_mid[iy_p, ix_p])
probe_y = float(y_mid[iy_p, ix_p])
probe_d = float(dist_2d[iy_p, ix_p])

print(f"Velocity probe: target ({PROBE_X:.3f}, {PROBE_Y:.3f})  "
      f"->  actual ({probe_x:.4f}, {probe_y:.4f})  dist={probe_d:.5f}")
if probe_d > 0.05:
    print("  [WARNING] Probe is more than 0.05 away from the target - adjust PROBE_X/Y")

# ============================================================================
# Plot
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                          gridspec_kw={'width_ratios': [2, 1]})

# ---------- left panel: full domain view ----------
ax = axes[0]

ax.scatter(x_iface[upper_mask],  y_iface[upper_mask],
           c='steelblue', s=3, alpha=0.6, label='Suction side')
ax.scatter(x_iface[~upper_mask], y_iface[~upper_mask],
           c='tomato',    s=3, alpha=0.6, label='Pressure side')

# Dot grid in midplane (every 20th node for context, skip NaN-masked interior)
step = 20
xg = x_mid[::step, ::step].ravel()
yg = y_mid[::step, ::step].ravel()
ax.scatter(xg, yg, c='lightgrey', s=1, alpha=0.4, zorder=1)

# Surface reference
ax.scatter(surf_x, surf_y, c='limegreen', s=250, marker='*',
           edgecolors='black', lw=1.5, zorder=6,
           label=f'Surface ref  x/c={surf_x:.3f}')

# Velocity probe
ax.scatter(probe_x, probe_y, c='orange', s=200, marker='^',
           edgecolors='black', lw=1.5, zorder=6,
           label=f'Probe  ({probe_x:.3f}, {probe_y:.3f})')

# Dashed connector
ax.plot([surf_x, probe_x], [surf_y, probe_y],
        'k--', lw=1.0, alpha=0.45, zorder=5)

ax.set_xlabel('x/c', fontsize=13)
ax.set_ylabel('y/c', fontsize=13)
ax.set_title('Full domain view  (midplane grid subsampled)', fontsize=13)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.25)
ax.set_aspect('equal', adjustable='box')

# ---------- right panel: zoom around the two points ----------
ax = axes[1]

pad = 0.15
x_lo = min(surf_x, probe_x) - pad
x_hi = max(surf_x, probe_x) + pad
y_lo = min(surf_y, probe_y) - pad
y_hi = max(surf_y, probe_y) + pad

# All grid nodes in the bounding box
in_box = ((x_mid >= x_lo) & (x_mid <= x_hi) &
          (y_mid >= y_lo) & (y_mid <= y_hi))
ax.scatter(x_mid[in_box], y_mid[in_box],
           c='lightgrey', s=4, alpha=0.6, zorder=1, label='Grid nodes')

# Airfoil surface in box
in_box_surf = ((x_iface >= x_lo) & (x_iface <= x_hi) &
               (y_iface >= y_lo) & (y_iface <= y_hi))
ax.scatter(x_iface[in_box_surf & upper_mask],  y_iface[in_box_surf & upper_mask],
           c='steelblue', s=8, zorder=3)
ax.scatter(x_iface[in_box_surf & ~upper_mask], y_iface[in_box_surf & ~upper_mask],
           c='tomato',    s=8, zorder=3)

ax.scatter(surf_x, surf_y, c='limegreen', s=280, marker='*',
           edgecolors='black', lw=1.5, zorder=6,
           label=f'Surface ref\n({surf_x:.4f}, {surf_y:.4f})')
ax.scatter(probe_x, probe_y, c='orange', s=230, marker='^',
           edgecolors='black', lw=1.5, zorder=6,
           label=f'Probe\n({probe_x:.4f}, {probe_y:.4f})')

ax.annotate(f'target: ({PROBE_X:.3f}, {PROBE_Y:.3f})\nsnapped: ({probe_x:.4f}, {probe_y:.4f})',
            xy=(probe_x, probe_y),
            xytext=(probe_x + (x_hi - x_lo) * 0.08, probe_y - (y_hi - y_lo) * 0.12),
            fontsize=8.5, color='darkorange',
            arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.0))

ax.plot([surf_x, probe_x], [surf_y, probe_y],
        'k--', lw=1.0, alpha=0.45, zorder=5)

ax.set_xlim(x_lo, x_hi)
ax.set_ylim(y_lo, y_hi)
ax.set_xlabel('x/c', fontsize=13)
ax.set_ylabel('y/c', fontsize=13)
ax.set_title('Zoom view', fontsize=13)
ax.legend(fontsize=9, loc='best')
ax.grid(True, alpha=0.25)
ax.set_aspect('equal', adjustable='box')

plt.suptitle(
    f"Probe validation setup  |  "
    f"midplane z={midplane_idx}/{Nz_phys-1}",
    fontsize=12, fontweight='bold', y=1.01
)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, f"probe_locations_xc{X_C_REF:.2f}_px{PROBE_X:.3f}_py{PROBE_Y:.3f}.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved to: {out_path}")
