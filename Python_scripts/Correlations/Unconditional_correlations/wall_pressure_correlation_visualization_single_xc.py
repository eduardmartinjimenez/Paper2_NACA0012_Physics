"""
Wall-Pressure Correlation Visualization Script (Single x/c Location)

This script generates visualizations of unconditional wall-pressure stress correlations
from CFD simulations at a single reference chord location. It produces 3 main figures:

1. FIGURE 4:  Streamwise velocity RMS vs correlation field at reference location
2. FIGURE 5:  Correlation fields at different spanwise separations
3. FIGURE 6:  Spanwise correlation profiles at different wall-normal heights

The script supports switching between two angle-of-attack (AOA) configurations:
- AOA 12 (currently active)
- AOA 5 (alternative, commented out)

Visualization windows are dynamically centered on the reference coordinates and
configured via offset-based parameters defined at the top of the script.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================

# # ========== ACTIVE CONFIGURATION: AOA 12 ==========
# Correlation data path (for visualization)
RESULT_FILE = (
    "/home/jofre/Members/Eduard/Paper2/Simulations/"
    "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
    "Wall_pressure_correlations/test_4/"
    "wall_pressure_correlation_unconditional_xc_0.900.h5"
)

# Probe y-coordinates for FIGURE 6 (spanwise correlation profiles)
# PROBE_Y_COORDS = [0.062, 0.065, 0.068, 0.075]  # x/c = 0.3
# PROBE_Y_COORDS = [0.055, 0.06, 0.065, 0.07]   # x/c = 0.5 
# PROBE_Y_COORDS = [0.04, 0.045, 0.055, 0.07]   # x/c = 0.7
PROBE_Y_COORDS = [0.02, 0.025, 0.035, 0.045]  # x/c = 0.9

# # Visualization window offsets (relative to reference coordinates)
# # Format: OFFSET = [left/bottom_extent, right/top_extent]
# # xlim = [x_ref - left_extent, x_ref + right_extent]
# # ylim = [y_ref - bottom_extent, y_ref + top_extent]
VIZ_XLIM_OFFSET = [0.5, 0.5]  
VIZ_YLIM_OFFSET = [0.02, 0.5]   

# ========== ALTERNATIVE CONFIGURATION: AOA 5 ==========
# To switch to AOA 5, uncomment the block below and comment out the AOA 12 block above
# RESULT_FILE = (
#     "/home/jofre/Members/Eduard/Paper2/Simulations/"
#     "NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/"
#     "Wall_pressure_correlations/test_1/"
#     "wall_pressure_correlation_unconditional_xc_0.900.h5"
# )


# # PROBE_Y_COORDS = [0.054, 0.056, 0.058, 0.06]   # x/c = 0.5
# # PROBE_Y_COORDS = [0.038, 0.04, 0.045, 0.05]  # x/c = 0.7
# PROBE_Y_COORDS = [0.016, 0.02, 0.024]        # x/c = 0.9
# VIZ_XLIM_OFFSET = [0.5, 0.5]
# VIZ_YLIM_OFFSET = [0.01, 0.5]

# ============================================================================
# Load Results
# ============================================================================
# NOTE on coordinate semantics:
# - The z-dimension (Nz=128) represents different z-planes in the physical mesh
# - The FFT correlation exploits spanwise periodicity via circular convolution
# - z-index=0 corresponds to the reference z-plane at physical z = z[0,0,0]
# - z-index=k corresponds to spanwise separation Δz = z[k,0,0] - z[0,0,0]
# - This allows measuring how correlation decays away from the reference plane
# ============================================================================

print("Loading unconditional wall-pressure correlation results...")

with h5py.File(RESULT_FILE, 'r') as f:
    # Load correlation field
    R = f['R'][:]              # (Nz, Ny, Nx)
    u_rms = f['u_rms'][:]      # (Nz, Ny, Nx)

    # Load coordinates
    x = f['x'][:]              # (Nz, Ny, Nx)
    y = f['y'][:]              # (Nz, Ny, Nx)
    z = f['z'][:]              # (Nz, Ny, Nx)

    # Load metadata
    x_c_actual = f.attrs['x_c_actual']
    y_actual = f.attrs['y_actual']
    N_samples = f.attrs['N_samples']
    N_snapshots = f.attrs['N_snapshots']
    p_total_mean = f.attrs['p_total_mean']
    p_total_rms = f.attrs['p_total_rms']
    u_infty = f.attrs.get('u_infty', 1.0)  # Default to 1.0 if not present

Nz, Ny, Nx = R.shape

# The z-axis represents spanwise SEPARATION Δz, not physical z-coordinate.
# Index 0 = Δz=0 (same plane as reference, strongest correlation)
# Index Nz//2 = maximum separation (weakest correlation)
dz_slice = 0

print(f"Loaded unconditional wall-pressure correlation results:")
print(f"  Shape: (Nz={Nz}, Ny={Ny}, Nx={Nx})")
print(f"  Reference point: x/c = {x_c_actual:.4f}, y = {y_actual:.4f}")
print(f"  Total pressure: mean = {p_total_mean:.6e}, rms = {p_total_rms:.6e}")
print(f"  Samples: N_snapshots={N_snapshots}, N_samples={N_samples}")
print(f"  Using z-slice at Δz = {dz_slice} (in-plane correlation)")

# ============================================================================
# Extract 2D Slice at Δz=0 (In-plane, Same Spanwise Location as Reference)
# ============================================================================

R_2d = R[dz_slice, :, :]        # (Ny, Nx)
u_rms_2d = u_rms[dz_slice, :, :] # (Ny, Nx)

x_2d = x[dz_slice, :, :]  # (Ny, Nx)
y_2d = y[dz_slice, :, :]  # (Ny, Nx)

print(f"\n2D slice shape: {R_2d.shape}")

# ============================================================================
# Correlation Scale Selection
# ============================================================================
# Compute real min/max of correlation field across all z-indices
R_min_real = np.nanmin(R)
R_max_real = np.nanmax(R)

print(f"\n{'='*70}")
print(f"CORRELATION SCALE STATISTICS:")
print(f"{'='*70}")
print(f"  Real min/max range:     [{R_min_real:.4f}, {R_max_real:.4f}]")
print(f"  Standard range:         [-1.0000,  1.0000]")
print(f"\nChoose correlation colorbar scale:")
print(f"  Option 1: Use real min/max  [{R_min_real:.4f}, {R_max_real:.4f}]")
print(f"  Option 2: Use standard scale [-1.0000,  1.0000] (Recommended for comparison)")
print(f"{'='*70}")

# Ask user for choice
valid_choice = False
while not valid_choice:
    user_input = input("\nEnter choice (1 or 2) [default: 2]: ").strip()
    if user_input == '':
        user_input = '2'
    if user_input in ['1', '2']:
        valid_choice = True
        choice = int(user_input)
    else:
        print("Invalid choice. Please enter 1 or 2.")

# Set colorbar limits based on choice
if choice == 1:
    R_vmin = R_min_real
    R_vmax = R_max_real
    use_real_scale = True
    print(f"\n✓ Using real min/max scale: [{R_vmin:.4f}, {R_vmax:.4f}]")
else:
    R_vmin = -1.0
    R_vmax = 1.0
    use_real_scale = False
    print(f"\n✓ Using standard scale: [-1.0000,  1.0000]")

# Define color levels based on choice
if use_real_scale:
    levels_r = np.linspace(R_vmin, R_vmax, 21)
else:
    levels_r = np.linspace(-1.0, 1.0, 21)

# ============================================================================
# FIGURE 4: Correlation vs Velocity RMS (Side-by-Side Comparison)
# ============================================================================
# WHAT WE'RE PLOTTING:
# - LEFT PANEL: Normalized streamwise velocity fluctuation intensity
# - RIGHT PANEL: Correlation field at Δz = 0
# - Allows direct spatial comparison between fluctuation intensity and correlation
#
# PURPOSE:
# - Investigates whether strong pressure-velocity correlation occurs in regions
#   of large velocity fluctuations
# - Reveals the spatial relationship between turbulence activity and
#   wall-pressure/velocity coupling

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)

# Normalize u_rms by freestream velocity
u_rms_norm = u_rms_2d / u_infty

# ---------------------------------------------------------------------------
# Left panel: streamwise fluctuation intensity
# ---------------------------------------------------------------------------
levels_u = np.linspace(np.nanmin(u_rms_norm), np.nanmax(u_rms_norm), 16)

im1 = axes[0].contourf(
    x_2d, y_2d, u_rms_norm,
    levels=levels_u,
    cmap='YlOrRd'
)

axes[0].plot(
    x_c_actual, y_actual,
    marker='*', color='k',
    markersize=9, zorder=5
)

axes[0].set_xlabel('x/c', fontsize=12)
axes[0].set_ylabel('y/c', fontsize=12)
axes[0].set_title(r'Streamwise fluctuation intensity, $u^\prime_{\mathrm{rms}}/U_\infty$', fontsize=12)
axes[0].set_aspect('equal', adjustable='box')

cbar1 = fig.colorbar(im1, ax=axes[0])
cbar1.set_label(r'$u^\prime_{\mathrm{rms}}/U_\infty$', fontsize=11)

# ---------------------------------------------------------------------------
# Right panel: correlation field at Δz = 0
# ---------------------------------------------------------------------------
im2 = axes[1].contourf(
    x_2d, y_2d, R_2d,
    levels=levels_r,
    cmap='RdBu_r',
    vmin=R_vmin,
    vmax=R_vmax
)

# Keep only the zero contour as a structural guide
cs_zero = axes[1].contour(
    x_2d, y_2d, R_2d,
    levels=[0.0],
    colors='black',
    linewidths=0.8,
    alpha=0.7
)

axes[1].plot(
    x_c_actual, y_actual,
    marker='*', color='k',
    markersize=9, zorder=5
)

axes[1].set_xlabel('x/c', fontsize=12)
axes[1].set_ylabel('y/c', fontsize=12)
axes[1].set_title(r'Correlation field at $\Delta z = 0$', fontsize=12)
axes[1].set_aspect('equal', adjustable='box')

cbar2 = fig.colorbar(im2, ax=axes[1])
cbar2.set_label(r'Correlation coefficient, $R$', fontsize=11)
cbar2.set_ticks(np.linspace(R_vmin, R_vmax, 5))

# Apply visualization window limits to both panels
# Window is relative to reference position (x_c_actual, y_actual)
xlim = [x_c_actual - VIZ_XLIM_OFFSET[0], x_c_actual + VIZ_XLIM_OFFSET[1]]
ylim = [y_actual - VIZ_YLIM_OFFSET[0], y_actual + VIZ_YLIM_OFFSET[1]]
for ax in axes:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

# Overall title
fig.suptitle(
    r'Comparison between $u^\prime_{\mathrm{rms}}/U_\infty$ and wall-pressure correlation at $\Delta z = 0$',
    fontsize=14
)

# ============================================================================
# FIGURE 5: Correlation Fields at Different Spanwise Separations
# ============================================================================
# WHAT WE'RE PLOTTING:
# - Correlation at different spanwise separations (Δz)
# - Shows how correlation varies as we move away from the reference z-plane
# - Demonstrates spanwise coherence of wall-pressure/velocity coupling
#
# PURPOSE:
# - Quantifies spanwise extent of correlation structure
# - At Δz=0: strongest coupling (same spanwise location)
# - At Δz→max: coupling weakest (maximum spanwise separation)

# Select multiple z-slices for comparison
z_indices = [0, Nz//8, Nz//4, Nz//2]

# Compute spanwise separations (relative to first z-plane)
z_ref = z[0, 0, 0]  # Reference z-coordinate (first plane)
z_separations = [z[z_idx, 0, 0] - z_ref for z_idx in z_indices]

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)

for i, z_idx in enumerate(z_indices):
    ax = axes[i]
    R_slice = R[z_idx, :, :]
    dz_separation = z_separations[i]

    im = ax.contourf(
        x_2d, y_2d, R_slice,
        levels=levels_r,
        cmap='RdBu_r',
        vmin=R_vmin,
        vmax=R_vmax
    )

    ax.plot(
        x_c_actual, y_actual,
        marker='*', color='k',
        markersize=12, zorder=5
    )

    # Show both z-index and spanwise separation
    if z_idx == 0:
        title_str = rf'Index $z={z_idx}$: $\Delta z = {dz_separation:.5f}$ (reference)'
    else:
        title_str = rf'Index $z={z_idx}$: $\Delta z = {dz_separation:.5f}$'
    ax.set_title(title_str, fontsize=10)

    ax.set_xlabel('x/c', fontsize=11)
    if i == 0:
        ax.set_ylabel('y/c', fontsize=11)
    else:
        ax.set_ylabel('')

    ax.set_aspect('equal', adjustable='box')

cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.025, pad=0.02)
cbar.set_label(r'Correlation coefficient, $R$', fontsize=11)
cbar.set_ticks(np.linspace(R_vmin, R_vmax, 5))

# Apply visualization window limits to all 4 subplots
# Window is relative to reference position (x_c_actual, y_actual)
xlim = [x_c_actual - VIZ_XLIM_OFFSET[0], x_c_actual + VIZ_XLIM_OFFSET[1]]
ylim = [y_actual - VIZ_YLIM_OFFSET[0], y_actual + VIZ_YLIM_OFFSET[1]]
for i, ax in enumerate(axes):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

fig.suptitle('Wall-pressure correlation field at different spanwise separations', fontsize=14)

# ============================================================================
# FIGURE 6: Spanwise Correlation Profiles at Different Wall-Normal Heights
# ============================================================================
# WHAT WE'RE PLOTTING:
# - Correlation curves R(Δz) at different wall-normal positions (y-coordinates)
# - Shows how spanwise coherence varies with height above the wall
# - RECENTERED: Maximum correlation (Δz=0) is in the middle of the plot
#
# PURPOSE:
# - Investigates if spanwise correlation profile varies across wall-normal heights
# - Identifies which heights maintain strongest spanwise coherence
# - Reveals vertical structure of spanwise correlation

# Compute spanwise separations for all planes (using centered lag coordinate)
z_ref = z[0, 0, 0]  # Reference z-coordinate
dz_all = np.array([z[iz, 0, 0] - z_ref for iz in range(Nz)])

# ========== RECENTER LAG COORDINATE ==========
# For periodic correlation functions, shift so Δz=0 is centered
# Create centered lag indices: -N/2, ..., -1, 0, 1, ..., N/2-1
dz_spacing = dz_all[1] - dz_all[0] if Nz > 1 else 1.0  # Grid spacing
centered_indices = np.arange(-Nz // 2, Nz // 2)  # Centered indices
dz_centered = centered_indices * dz_spacing  # Map to actual lag values

shift = Nz // 2  # Number of positions to shift for data reordering

print(f"\nRecentering lag coordinate (periodic correlation):")
print(f"  Original: Δz[0]={dz_all[0]:.6f}, Δz[{Nz//2}]={dz_all[Nz//2]:.6f}, Δz[-1]={dz_all[-1]:.6f}")
print(f"  Centered: Δz ranges from {dz_centered[0]:.6f} to {dz_centered[-1]:.6f}")
print(f"  Grid spacing: {dz_spacing:.6f}, Nz={Nz}, shift amount={shift}")

# Find reference x-index by finding closest x-coordinate to x_c_actual
# (Using x-axis of first row to find the streamwise location)
x_at_first_row = x[0, 0, :]  # (Nx,) - x-coordinates along streamwise direction at first y
ix_ref = np.argmin(np.abs(x_at_first_row - x_c_actual))

print(f"\nFinding probe y-coordinates in mesh:")
print(f"  Reference x-index (x/c closest to {x_c_actual:.4f}): {ix_ref}, actual x/c = {x_at_first_row[ix_ref]:.4f}")

# Map probe coordinates to mesh y-indices
probe_iy_indices = []
probe_y_actual = []

for y_target in PROBE_Y_COORDS:
    # Find nearest y-index for this y-coordinate at the reference x-index
    y_2d_at_ref_x = y[0, :, ix_ref]
    iy_nearest = np.argmin(np.abs(y_2d_at_ref_x - y_target))
    y_actual_val = y_2d_at_ref_x[iy_nearest]
    probe_iy_indices.append(iy_nearest)
    probe_y_actual.append(y_actual_val)
    print(f"  Target y={y_target:.3f} → Index {iy_nearest}, Actual y={y_actual_val:.6f}")

# Create figure with correlation curves at different y-levels
fig, ax = plt.subplots(figsize=(12, 7))

# Color map for different y-levels (from near-wall to outer)
colors = plt.cm.viridis(np.linspace(0, 1, len(PROBE_Y_COORDS)))

# Plot correlation curves for each probe height
for idx, (iy, y_val, color) in enumerate(zip(probe_iy_indices, probe_y_actual, colors)):
    # Extract correlation curve at this y-position across all Δz
    corr_at_y = R[:, iy, ix_ref]

    # Recenter the correlation curve (same shift as dz array)
    corr_at_y_centered = np.roll(corr_at_y, shift)

    # Plot the centered curve (x/c is fixed, so only label y-coordinate)
    ax.plot(dz_centered, corr_at_y_centered, 'o-', linewidth=2.0, markersize=3,
            color=color, alpha=0.8, label=f'y = {y_val:.3f}')

# Add reference lines (structural guides, not in legend)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
ax.axvline(x=0, color='green', linestyle='--', linewidth=2.0, alpha=0.5)

# Labels and formatting
ax.set_xlabel(r'Spanwise separation, $\Delta z$', fontsize=12, fontweight='bold')
ax.set_ylabel(r'Correlation coefficient, $R$', fontsize=12, fontweight='bold')
ax.set_title(r'Spanwise Correlation Profiles at Different Wall-Normal Heights (x/c = {:.3f})'.format(x_c_actual),
             fontsize=13, fontweight='bold')

ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=1, framealpha=0.95)

fig.tight_layout()

# ============================================================================
# Summary Statistics
# ============================================================================
print(f"\n" + "=" * 70)
print("UNCONDITIONAL WALL-PRESSURE CORRELATION ANALYSIS SUMMARY (Single x/c)")
print("=" * 70)
print(f"\nVISUALIZATIONS GENERATED:")
print(f"  FIGURE 4:  Velocity RMS vs correlation field comparison (Δz=0)")
print(f"  FIGURE 5:  Correlation fields at different spanwise separations")
print(f"  FIGURE 6:  Spanwise correlation profiles at selected wall-normal heights")
print(f"\nCORRELATION COLORBAR SCALE:")
if use_real_scale:
    print(f"  Scale Type: Real min/max (adaptive)")
    print(f"  Range: [{R_vmin:.4f}, {R_vmax:.4f}]")
else:
    print(f"  Scale Type: Standard (-1 to 1)")
    print(f"  Range: [-1.0000,  1.0000]")
print(f"\nREFERENCE POINT:")
print(f"  x/c = {x_c_actual:.4f}, y = {y_actual:.4f}, x-index = {ix_ref}")
print(f"\nVISUALIZATION WINDOW CONFIGURATION:")
print(f"  X-window offset: [{VIZ_XLIM_OFFSET[0]:.2f}, {VIZ_XLIM_OFFSET[1]:.2f}]")
print(f"  Y-window offset: [{VIZ_YLIM_OFFSET[0]:.2f}, {VIZ_YLIM_OFFSET[1]:.2f}]")
print(f"\nSPANWISE SEPARATION RANGE:")
print(f"  Δz_min = {dz_all[0]:.6f} (reference plane)")
print(f"  Δz_max = {dz_all[-1]:.6f} (maximum physical separation)")
print(f"  Total extent = {dz_all[-1] - dz_all[0]:.6f}")
print(f"\nFIGURE 5 Z-INDICES:")
print(f"  Selected indices: {z_indices}")
print(f"  Corresponding separations: {[f'{dz:.6f}' for dz in z_separations]}")
print(f"\nWALL-NORMAL PROBE POSITIONS (Figure 6):")
for y_target, y_actual_val, iy in zip(PROBE_Y_COORDS, probe_y_actual, probe_iy_indices):
    corr_at_ref_z = R[0, iy, ix_ref]
    print(f"  Target y={y_target:.3f} → Actual y={y_actual_val:.6f}, R(Δz=0)={corr_at_ref_z:.4f}")

# Show all figures
plt.show()

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)
print("All plots displayed with plt.show()")
