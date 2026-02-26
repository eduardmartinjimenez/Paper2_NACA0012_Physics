import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ============================================================================
# Configuration
# ============================================================================
# Result file from correlation analysis
RESULT_FILE = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Wall_shear_correlations/wall_shear_correlation_xc_0.500_alpha_0.5_all_fft_2.h5"

# Output directory
OUTPUT_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Wall_shear_correlations/Figures/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Output base path (directory + base filename without suffix/extension)
OUTPUT_BASE = os.path.join(OUTPUT_DIR, "correlation_2d_all_fft_05_alpha05_2")

# ============================================================================
# Load results
# ============================================================================
print("Loading correlation results...")

with h5py.File(RESULT_FILE, 'r') as f:
    # Load correlation fields
    R_PF = f['R_PF'][:]      # (Nz, Ny, Nx)
    R_NF = f['R_NF'][:]      # (Nz, Ny, Nx)
    R_all = f['R_all'][:]    # (Nz, Ny, Nx)
    u_rms = f['u_rms'][:]    # (Nz, Ny, Nx)
    
    # Load coordinates
    x = f['x'][:]            # (Nz, Ny, Nx)
    y = f['y'][:]            # (Nz, Ny, Nx)
    z = f['z'][:]            # (Nz, Ny, Nx)
    
    # Load metadata
    x_c_actual = f.attrs['x_c_actual']
    y_actual = f.attrs['y_actual']
    N_PF = f.attrs['N_PF']
    N_NF = f.attrs['N_NF']
    N_all = f.attrs['N_all']

Nz, Ny, Nx = R_all.shape
z_mid = Nz // 2

print(f"Loaded correlation results:")
print(f"  Shape: (Nz={Nz}, Ny={Ny}, Nx={Nx})")
print(f"  Reference point: x/c = {x_c_actual:.4f}, y = {y_actual:.4f}")
print(f"  Samples: N_all={N_all}, N_PF={N_PF}, N_NF={N_NF}")
print(f"  Using z-slice at z_mid = {z_mid}")

# ============================================================================
# Extract 2D slices at z_mid
# ============================================================================
R_PF_2d = R_PF[z_mid, :, :]   # (Ny, Nx)
R_NF_2d = R_NF[z_mid, :, :]   # (Ny, Nx)
R_all_2d = R_all[z_mid, :, :] # (Ny, Nx)
u_rms_2d = u_rms[z_mid, :, :] # (Ny, Nx)

x_2d = x[z_mid, :, :]  # (Ny, Nx)
y_2d = y[z_mid, :, :]  # (Ny, Nx)
z_2d = z[z_mid, :, :]  # (Ny, Nx)

print(f"\n2D slice shapes: {R_PF_2d.shape}")

# ============================================================================
# FIGURE 1: Three-panel comparison (PF, NF, All)
# ============================================================================
# WHAT WE'RE PLOTTING:
# - R_PF: Correlation coefficient during POSITIVE WALL SHEAR STRESS (ejection) events
#         Shows which regions have velocity correlated with upward wall shear
# - R_NF: Correlation coefficient during NEGATIVE WALL SHEAR STRESS (sweep) events
#         Shows which regions have velocity correlated with downward wall shear
# - R_all: Overall correlation coefficient across all events
#          Shows the general relationship without event conditioning
# 
# MAP INTERPRETATION:
# - RED regions: positive correlation (high velocity when high wall shear)
# - BLUE regions: negative correlation (low velocity when high wall shear)
# - WHITE regions: no correlation (velocity independent of wall shear)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Common colormap limits
vmin = -0.5
vmax = 0.5

titles = ['PF Correlation\n(Ejection Events)', 'NF Correlation\n(Sweep Events)', 'All Correlation\n(Unconditioned)']
data_list = [R_PF_2d, R_NF_2d, R_all_2d]

for ax, title, data in zip(axes, titles, data_list):
    # Create contour plot
    im = ax.contourf(x_2d, y_2d, data, levels=20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    
    # Add contour lines
    cs = ax.contour(x_2d, y_2d, data, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    
    # Mark reference point (where tau'_w is measured)
    ax.plot(x_c_actual, y_actual, 'g*', markersize=20, markeredgecolor='black', 
            markeredgewidth=1.5, label='Reference point', zorder=5)
    
    # Labels and title
    ax.set_xlabel('x/c (streamwise)', fontsize=12)
    ax.set_ylabel('y/c (wall-normal)', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal', adjustable='box')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Coefficient', fontsize=11)

# Add legend
axes[0].legend(loc='upper left', fontsize=11)

# Overall title
fig.suptitle(f'Wall Shear - Streamwise Velocity Correlation at z_mid={z_mid}\n' + 
             f'Reference point tau\'_w at: x/c={x_c_actual:.3f}, y={y_actual:.4f}',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()

# Save figure
output_path = f"{OUTPUT_BASE}_slice_z{z_mid:03d}.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved FIGURE 1 (three-panel comparison) to: {output_path}")

# ============================================================================
# FIGURE 2: Detailed R_all with contour labels
# ============================================================================
# WHAT WE'RE PLOTTING:
# - Overall correlation coefficient R_all in detail
# - Shows spatial structure: regions upstream/downstream and above/below reference point
# - Contour lines help identify local maxima and minima
# - Useful for understanding where wall shear fluctuations influence velocity most strongly

fig, ax = plt.subplots(figsize=(10, 8))

# Create contour plot
levels = np.linspace(vmin, vmax, 25)
im = ax.contourf(x_2d, y_2d, R_all_2d, levels=levels, cmap='RdBu_r')

# Add contour lines with labels
cs = ax.contour(x_2d, y_2d, R_all_2d, levels=10, colors='black', alpha=0.4, linewidths=0.8)
ax.clabel(cs, inline=True, fontsize=8, fmt='%g')

# Mark reference point
ax.plot(x_c_actual, y_actual, 'g*', markersize=25, markeredgecolor='white', 
        markeredgewidth=2, label='Reference point (tau\'_w measurement)', zorder=5)

# Labels and title
ax.set_xlabel('x/c (streamwise distance from reference)', fontsize=13, fontweight='bold')
ax.set_ylabel('y/c (wall-normal distance from reference)', fontsize=13, fontweight='bold')
ax.set_title(f'Overall Correlation Coefficient - z_mid={z_mid}\n' +
             'Contours show local extrema of correlation strength',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.2)
ax.set_aspect('equal', adjustable='box')
ax.legend(loc='upper left', fontsize=12)

# Colorbar
cbar = plt.colorbar(im, ax=ax, label='Correlation Coefficient R')
cbar.ax.tick_params(labelsize=11)

# Add metadata text
textstr = f'Total samples: {N_all}\nPF (ejection): {N_PF}\nNF (sweep): {N_NF}'
ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()

# Save figure
output_path = f"{OUTPUT_BASE}_all_detailed_z{z_mid:03d}.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved FIGURE 2 (detailed R_all) to: {output_path}")

# ============================================================================
# FIGURE 3: Velocity RMS vs Correlation
# ============================================================================
# WHAT WE'RE PLOTTING:
# LEFT PANEL (u'_rms):
#   - Magnitude of velocity fluctuations at each point
#   - Shows where flow is most turbulent/energetic
#   - High RMS = strong velocity variations
#
# RIGHT PANEL (R_all):
#   - Correlation coefficient (same as Figure 2)
#   - Allows visual comparison: does high velocity fluctuation → high correlation?
#   - Often correlation is stronger where velocity RMS is high

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: u_rms
im1 = axes[0].contourf(x_2d, y_2d, u_rms_2d, levels=20, cmap='viridis')
axes[0].plot(x_c_actual, y_actual, 'r*', markersize=20, markeredgecolor='white', 
            markeredgewidth=1.5, zorder=5, label='Reference point')
axes[0].set_xlabel('x/c', fontsize=12)
axes[0].set_ylabel('y/c', fontsize=12)
axes[0].set_title('Streamwise Velocity RMS (u\'_rms)', fontsize=13, fontweight='bold')
axes[0].set_aspect('equal', adjustable='box')
axes[0].grid(True, alpha=0.2)
axes[0].legend(loc='upper left', fontsize=11)
cbar1 = plt.colorbar(im1, ax=axes[0], label='u\'_rms magnitude')

# Right: Overall correlation
im2 = axes[1].contourf(x_2d, y_2d, R_all_2d, levels=20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
axes[1].plot(x_c_actual, y_actual, 'g*', markersize=20, markeredgecolor='white', 
            markeredgewidth=1.5, zorder=5, label='Reference point')
axes[1].set_xlabel('x/c', fontsize=12)
axes[1].set_ylabel('y/c', fontsize=12)
axes[1].set_title('Correlation Coefficient (R_all)', fontsize=13, fontweight='bold')
axes[1].set_aspect('equal', adjustable='box')
axes[1].grid(True, alpha=0.2)
axes[1].legend(loc='upper left', fontsize=11)
cbar2 = plt.colorbar(im2, ax=axes[1], label='Correlation R')

fig.suptitle(f'Velocity Field Context vs Correlation at z_mid={z_mid}\n' +
             'Left: Where is flow most turbulent? Right: Where is strongest correlation with wall shear?',
             fontsize=14, fontweight='bold')
plt.tight_layout()

# Save figure
output_path = f"{OUTPUT_BASE}_vs_urms_z{z_mid:03d}.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved FIGURE 3 (velocity RMS vs correlation) to: {output_path}")

plt.show()

print("\n" + "="*70)
print("VISUALIZATION COMPLETE")
print("="*70)
print(f"All figures saved to: {OUTPUT_DIR}")
print("\n" + "="*70)
print("EXPLANATION OF PLOTS")
print("="*70)
print("""
FIGURE 1: Three-panel comparison
  - LEFT (PF): Shows correlation during EJECTION events (high positive wall shear)
    → Red regions = velocity increases when wall shear increases (coherent structure)
    → Blue regions = velocity decreases when wall shear increases (opposing flow)
  
  - MIDDLE (NF): Shows correlation during SWEEP events (high negative wall shear)
    → Red regions = velocity increases when wall shear decreases (coherent structure)
    → Blue regions = velocity decreases when wall shear decreases
  
  - RIGHT (R_all): Shows overall correlation across ALL events
    → Unconditioned by event type
    → Reveals general relationship between wall shear and velocity

FIGURE 2: Detailed correlation with contours
  - Enhanced view of R_all with contour lines showing local extrema
  - Helps identify gradients and feature structures
  - Contour labels show correlation strength at specific locations

FIGURE 3: Velocity RMS vs Correlation
  - LEFT: Magnitude of velocity fluctuations (u'_rms)
    → High values = turbulent regions with large velocity swings
  - RIGHT: Same correlation as before for direct comparison
  - Visual inspection shows if correlation is stronger in energetic regions

KEY OBSERVATIONS:
- Positive correlation means wall shear and velocity fluctuations are in-phase
- Regions near reference point show strongest correlation
- Correlation typically decays with distance from reference point
- Event-conditioned (PF/NF) may show different patterns than unconditioned (All)
""")
