"""
Visualization Script for Signal Correlation Analysis
======================================================

Loads saved velocity fluctuation and TKE data, then creates:
1. Geometric visualization: Surface point + probe point + airfoil surface
2. Time series comparison: τ' (surface) vs u', v', w' (fixed probe)
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"
if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader

# ============================================================================
# CONFIGURATION
# ============================================================================

### AOA 12º

# Mesh slice for geometry
MESH_SLICE_PATH = "/home/jofre/disc2/Members/Eduard/NACA_0012_AOA12_Re50000_1716x1662x128/Slice_data/slice_9/"
MESH_SLICE_NAME = "slice_9-CROP-MESH.h5"

# Geometric data (airfoil surface)
GEO_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Geometrical_data/"
GEO_NAME = "3d_NACA0012_Re50000_AoA12_Geometrical_Data.h5"
GEO_FILE = os.path.join(GEO_PATH, GEO_NAME)

# Signal data (output from signal_correlation.py)
SIGNAL_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Signal_correlation/"
SIGNAL_FILE = os.path.join(SIGNAL_DIR, "velocity_timeseries_slice_9_test.h5")

# Physical parameters
AOA_deg = 12.0
AOA_rad = np.radians(AOA_deg)

# Sampling frequency for plotting
# frequency = 1: plot all points
# frequency = N: plot 1 every N points
PLOT_FREQUENCY = 1  # Change this to downsample for faster plotting

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def assert_exists(path: str, kind: str = "File") -> None:
    """Check path exists and print confirmation."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{kind} does not exist: {path}")
    print(f"✓ {kind} exists: {path}")


# ============================================================================
# LOAD DATA
# ============================================================================

print("="*70)
print("LOADING DATA")
print("="*70)

assert_exists(GEO_FILE, "Geometrical data file")
assert_exists(SIGNAL_FILE, "Signal data file")

# Load geometrical data
print("\nLoading airfoil surface geometry...")
with h5py.File(GEO_FILE, "r") as f:
    interface_points = f["interface_points"][...].astype(np.float64)
    proj_normals = f["proj_normals"][...].astype(np.float64)
    proj_distances = f["proj_distances"][...].astype(np.float64)

# Extract suction and pressure side surfaces
suction_side_points = interface_points[interface_points[:, 1] >= 0]
pressure_side_points = interface_points[interface_points[:, 1] < 0]
print(f"  Suction side points: {suction_side_points.shape[0]}")
print(f"  Pressure side points: {pressure_side_points.shape[0]}")

# Load mesh
print("Loading mesh geometry...")
mesh_file = os.path.join(MESH_SLICE_PATH, MESH_SLICE_NAME)
loader = CompressedSnapshotLoader(mesh_file)
x_data = loader.x[1:-1, :, :]
y_data = loader.y[1:-1, :, :]
z_data = loader.z[1:-1, :, :]
slice_x = x_data[0, 0, 0]
print(f"  Slice x-coordinate: {slice_x:.6f}")
print(f"  Mesh shape (nz, ny, nx): {x_data.shape}")

# Extract grid coordinates
y_unique = np.unique(y_data[:, :, 0][0, :])
z_unique = np.unique(z_data[:, 0, 0])

# Load signal data
print("\nLoading signal data...")
with h5py.File(SIGNAL_FILE, "r") as f:
    # Read metadata
    num_probes = f.attrs['num_probes']
    probe_labels = [label.decode('utf-8') if isinstance(label, bytes) else label
                    for label in f.attrs['probe_labels']]
    probe_locations_actual = f.attrs['probe_locations_actual']

    print(f"  Number of probes: {num_probes}")
    print(f"  Probe labels: {probe_labels}")

    # Load time series data
    signal_data = {}
    for probe_id in range(num_probes):
        probe_group = f[f'probes/probe_{probe_id}']
        probe_label = probe_group.attrs['label']

        # Initialize base data
        signal_data[probe_id] = {
            'label': probe_label,
            'y_actual': probe_group.attrs['y_actual'],
            'iterations': probe_group['iterations'][...],
            'time': probe_group['time'][...],
        }

        # Load data based on probe type
        if probe_label == 'surface':
            # Surface probe: only has tau_prime
            signal_data[probe_id]['tau_prime'] = probe_group['tau_prime'][...]
            # Initialize velocity fields as None for surface probe
            signal_data[probe_id]['u_prime'] = None
            signal_data[probe_id]['v_prime'] = None
            signal_data[probe_id]['w_prime'] = None
            signal_data[probe_id]['tke'] = None
        else:
            # Velocity probe: has velocity fluctuations and TKE
            signal_data[probe_id]['u_prime'] = probe_group['u_prime'][...]
            signal_data[probe_id]['v_prime'] = probe_group['v_prime'][...]
            signal_data[probe_id]['w_prime'] = probe_group['w_prime'][...]
            signal_data[probe_id]['tke'] = probe_group['tke'][...]
            # Initialize tau_prime as None for velocity probe
            signal_data[probe_id]['tau_prime'] = None

        print(f"  Probe {probe_id} ({signal_data[probe_id]['label']}): "
              f"y={signal_data[probe_id]['y_actual']:.6e}, "
              f"timesteps={len(signal_data[probe_id]['iterations'])}")

# ============================================================================
# PLOT 1: GEOMETRIC VISUALIZATION
# ============================================================================

print("\n" + "="*70)
print("PLOT 1: GEOMETRIC VISUALIZATION")
print("="*70)

# Find closest surface point at this slice x-location
x_distances = np.abs(suction_side_points[:, 0] - slice_x)
closest_idx = np.argmin(x_distances)
surface_point = suction_side_points[closest_idx]

print(f"Surface point found at x={surface_point[0]:.6f}, y={surface_point[1]:.6f}")

# Get probe y-coordinates
probe_y_coords = [signal_data[i]['y_actual'] for i in sorted(signal_data.keys())]

fig, ax = plt.subplots(figsize=(14, 9))

# Plot airfoil surfaces
ax.scatter(suction_side_points[:, 0], suction_side_points[:, 1],
          s=15, c='blue', label='Suction side', zorder=3, alpha=0.5)
ax.scatter(pressure_side_points[:, 0], pressure_side_points[:, 1],
          s=15, c='red', label='Pressure side', zorder=3, alpha=0.5)

# Plot slice plane
ax.axvline(x=slice_x, color='green', linewidth=2.5, linestyle='--',
           label=f'Slice plane (x={slice_x:.4f})', zorder=2, alpha=0.8)

# Plot surface point
ax.scatter(surface_point[0], surface_point[1], s=300, c='orange', marker='*',
          label=f'Surface point (y={surface_point[1]:.4e})',
          zorder=5, edgecolors='black', linewidths=2)

# Plot probe points on slice line
for probe_id in sorted(signal_data.keys()):
    probe = signal_data[probe_id]
    label = probe['label']
    y_actual = probe['y_actual']

    if label == 'surface':
        marker, color = '^', 'purple'
        marker_size = 12
        edge_width = 1.5
    else:
        marker, color = 's', 'cyan'
        marker_size = 12
        edge_width = 1.5

    # Plot on the slice line
    ax.plot(slice_x, y_actual, marker, markersize=marker_size, color=color,
            markeredgecolor='black', markeredgewidth=edge_width, zorder=5,
            label=f"Probe {probe_id}: {label} (y={y_actual:.4e})")

ax.set_xlabel('x (chord)', fontsize=12, fontweight='bold')
ax.set_ylabel('y (chord)', fontsize=12, fontweight='bold')
ax.set_title(f'Airfoil Surface, Slice Plane, and Probe Locations\nAOA={AOA_deg}°, slice_x={slice_x:.4f}',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(loc='upper left', fontsize=10, framealpha=0.9, ncol=2)
ax.set_aspect('equal')
ax.margins(0.05)

plt.tight_layout()

print(f"\n{'='*70}")
print("DISPLAYING GEOMETRIC VISUALIZATION")
print(f"{'='*70}")
# plt.show()

# ============================================================================
# PLOT 2: TIME SERIES COMPARISON
# ============================================================================

print("\n" + "="*70)
print("PLOT: TIME SERIES COMPARISON - ALL SIGNALS")
print("="*70)

# Separate probes
surface_probe = None
velocity_probe = None

for probe_id in sorted(signal_data.keys()):
    if signal_data[probe_id]['label'] == 'surface':
        surface_probe = (probe_id, signal_data[probe_id])
    else:
        velocity_probe = (probe_id, signal_data[probe_id])

if surface_probe is None or velocity_probe is None:
    raise ValueError("Expected one surface probe and one velocity probe")

surf_id, surf_data = surface_probe
vel_id, vel_data = velocity_probe

# Apply frequency downsampling for plotting
time_surf = surf_data['time'][::PLOT_FREQUENCY]
time_vel = vel_data['time'][::PLOT_FREQUENCY]

# Create figure with 5 subplots (tau', u', v', w', TKE)
fig, axes = plt.subplots(5, 1, figsize=(14, 14))

colors = {'surface': '#d62728', 'velocity': '#1f77b4'}

print(f"\nPlotting with frequency = {PLOT_FREQUENCY}")
print(f"  Original points: {len(surf_data['time'])}")
print(f"  Plotted points: {len(time_surf)}")

# Plot 1: Wall shear stress fluctuation (τ')
ax = axes[0]
tau_prime = surf_data['tau_prime'][::PLOT_FREQUENCY]
ax.plot(time_surf, tau_prime, linewidth=0.8, color=colors['surface'], alpha=0.85)
ax.set_ylabel("τ' (Pa)", fontsize=10, fontweight='bold')
ax.set_title(f"Probe {surf_id}: Surface - Wall Shear Stress Fluctuation",
             fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

# Statistics box
stats_text = (f"Mean: {np.mean(tau_prime):.4e} Pa\n"
              f"Std: {np.std(tau_prime):.4e} Pa\n"
              f"Min: {np.min(tau_prime):.4e} Pa\n"
              f"Max: {np.max(tau_prime):.4e} Pa\n"
              f"RMS: {np.sqrt(np.mean(tau_prime**2)):.4e} Pa")
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
        fontsize=8, family='monospace')

# Plot 2: Streamwise velocity fluctuation (u')
ax = axes[1]
u_prime = vel_data['u_prime'][::PLOT_FREQUENCY]
ax.plot(time_vel, u_prime, linewidth=0.8, color=colors['velocity'], alpha=0.85)
ax.set_ylabel("u' (m/s)", fontsize=10, fontweight='bold')
ax.set_title(f"Probe {vel_id}: Fixed Height - Streamwise Velocity Fluctuation",
             fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

stats_text = (f"Mean: {np.mean(u_prime):.4e} m/s\n"
              f"Std: {np.std(u_prime):.4e} m/s\n"
              f"Min: {np.min(u_prime):.4e} m/s\n"
              f"Max: {np.max(u_prime):.4e} m/s\n"
              f"RMS: {np.sqrt(np.mean(u_prime**2)):.4e} m/s")
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.85),
        fontsize=8, family='monospace')

# Plot 3: Cross-streamwise velocity fluctuation (v')
ax = axes[2]
v_prime = vel_data['v_prime'][::PLOT_FREQUENCY]
ax.plot(time_vel, v_prime, linewidth=0.8, color='#2ca02c', alpha=0.85)
ax.set_ylabel("v' (m/s)", fontsize=10, fontweight='bold')
ax.set_title(f"Probe {vel_id}: Fixed Height - Cross-streamwise Velocity Fluctuation",
             fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

stats_text = (f"Mean: {np.mean(v_prime):.4e} m/s\n"
              f"Std: {np.std(v_prime):.4e} m/s\n"
              f"Min: {np.min(v_prime):.4e} m/s\n"
              f"Max: {np.max(v_prime):.4e} m/s\n"
              f"RMS: {np.sqrt(np.mean(v_prime**2)):.4e} m/s")
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.85),
        fontsize=8, family='monospace')

# Plot 4: Spanwise velocity fluctuation (w')
ax = axes[3]
w_prime = vel_data['w_prime'][::PLOT_FREQUENCY]
ax.plot(time_vel, w_prime, linewidth=0.8, color='#ff7f0e', alpha=0.85)
ax.set_ylabel("w' (m/s)", fontsize=10, fontweight='bold')
ax.set_title(f"Probe {vel_id}: Fixed Height - Spanwise Velocity Fluctuation",
             fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

stats_text = (f"Mean: {np.mean(w_prime):.4e} m/s\n"
              f"Std: {np.std(w_prime):.4e} m/s\n"
              f"Min: {np.min(w_prime):.4e} m/s\n"
              f"Max: {np.max(w_prime):.4e} m/s\n"
              f"RMS: {np.sqrt(np.mean(w_prime**2)):.4e} m/s")
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85),
        fontsize=8, family='monospace')

# Plot 5: Turbulent Kinetic Energy (TKE)
ax = axes[4]
tke = vel_data['tke'][::PLOT_FREQUENCY]
ax.plot(time_vel, tke, linewidth=0.8, color='#9467bd', alpha=0.85)
ax.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
ax.set_ylabel('TKE (m/s)²', fontsize=10, fontweight='bold')
ax.set_title(f"Probe {vel_id}: Fixed Height - Turbulent Kinetic Energy",
             fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')

stats_text = (f"Mean: {np.mean(tke):.4e} (m/s)²\n"
              f"Min: {np.min(tke):.4e} (m/s)²\n"
              f"Max: {np.max(tke):.4e} (m/s)²")
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
        fontsize=8, family='monospace')

plt.tight_layout()

print(f"\n{'='*70}")
print("DISPLAYING COMBINED PLOT")
print(f"{'='*70}")
plt.show()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)
print(f"\nSurface Probe (Probe {surf_id}):")
print(f"  Location: y = {surf_data['y_actual']:.6e} chord")
print(f"  Quantity: Wall Shear Stress Fluctuation τ'(t)")
print(f"  Mean τ': {np.mean(tau_prime):.6e} Pa")
print(f"  Std τ': {np.std(tau_prime):.6e} Pa")
print(f"  RMS τ': {np.sqrt(np.mean(tau_prime**2)):.6e} Pa")

print(f"\nVelocity Probe (Probe {vel_id}):")
print(f"  Location: y = {vel_data['y_actual']:.6e} chord")
print(f"  Quantities: u'(t), v'(t), w'(t)")
print(f"\n  Streamwise (u'):")
print(f"    Mean: {np.mean(u_prime):.6e} m/s")
print(f"    Std: {np.std(u_prime):.6e} m/s")
print(f"    RMS: {np.sqrt(np.mean(u_prime**2)):.6e} m/s")
print(f"\n  Cross-streamwise (v'):")
print(f"    Mean: {np.mean(v_prime):.6e} m/s")
print(f"    Std: {np.std(v_prime):.6e} m/s")
print(f"    RMS: {np.sqrt(np.mean(v_prime**2)):.6e} m/s")
print(f"\n  Spanwise (w'):")
print(f"    Mean: {np.mean(w_prime):.6e} m/s")
print(f"    Std: {np.std(w_prime):.6e} m/s")
print(f"    RMS: {np.sqrt(np.mean(w_prime**2)):.6e} m/s")

tke = vel_data['tke']
print(f"\n  Turbulent Kinetic Energy (TKE):")
print(f"    Mean: {np.mean(tke):.6e} (m/s)²")
print(f"    Min: {np.min(tke):.6e} (m/s)²")
print(f"    Max: {np.max(tke):.6e} (m/s)²")

print(f"\nTime Series Info:")
print(f"  Time range: {vel_data['time'][0]:.6e} to {vel_data['time'][-1]:.6e} s")
print(f"  Number of timesteps: {len(vel_data['time'])}")
print(f"  Total duration: {vel_data['time'][-1] - vel_data['time'][0]:.6e} s")

print("="*70)
print("\nVisualization complete!")
