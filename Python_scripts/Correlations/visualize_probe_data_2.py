"""
visualize_probe_data_2.py
=========================
Visualize probe time signals (from probe_time_signals_2.py) together with the
spatial correlation field (from wall_shear_correlations_2.py).

PLOTS
-----
  1. Probe locations on the 2D spatial correlation map R_all(Dz=0)
  2. Phase relationship: tau'_w overlaid with strongest +R and -R probes
  3. Normalised time series of tau'_w and all u' signals
  4. Convergence of R(0) for all probe pairs
  5. Scatter plots of tau'_w vs u' for the three strongest-correlated pairs
  6. Power spectral density (PSD) of all signals
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================================================================
# CONFIGURATION
# ============================================================================

# Probe data file (output of probe_time_signals_2.py)
PROBE_FILE = (
    "/home/jofre/Members/Eduard/Paper2/Simulations/"
    "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
    "Probe_time_signals/probe_time_signals_2.h5"
)

# Spatial correlation data file (output of wall_shear_correlations_2.py)
CORR_FILE = (
    "/home/jofre/Members/Eduard/Paper2/Simulations/"
    "NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/"
    "Wall_shear_correlations/test_3/"
    "wall_shear_correlation_xc_0.500_alpha_1.0_all_fft.h5"
)

# Output directory (set to None to only display interactively)
OUTPUT_DIR = None

# ============================================================================
# LOAD PROBE DATA
# ============================================================================
print("=" * 70)
print("PROBE DATA VISUALIZATION")
print("=" * 70)

print(f"\nLoading probe data from:\n  {PROBE_FILE}")

with h5py.File(PROBE_FILE, "r") as f:
    n_snapshots     = int(f.attrs['n_snapshots'])
    n_surf_probes   = int(f.attrs['n_surf_probes'])
    n_domain_probes = int(f.attrs['n_domain_probes'])
    AOA_deg         = int(f.attrs['AOA_deg'])

    # Surface probes
    surf_probes = []
    for k in range(n_surf_probes):
        g = f[f"time_series/surf_{k}"]
        surf_probes.append({
            'xc_actual' : float(g.attrs['xc_actual']),
            'y_actual'  : float(g.attrs['y_actual']),
            'tau_w_mean': float(g.attrs['tau_w_mean']),
            'tau_w_rms' : float(g.attrs['tau_w_rms']),
            'tau_prime' : g['tau_prime'][:],
        })

    # Domain probes
    domain_probes = []
    for j in range(n_domain_probes):
        g = f[f"time_series/domain_{j}"]
        domain_probes.append({
            'x_actual' : float(g.attrs['x_actual']),
            'y_actual' : float(g.attrs['y_actual']),
            'u_mean'   : float(g.attrs['u_mean']),
            'u_prime'  : g['u_prime'][:],
        })

    # Zero-lag correlations and convergence
    correlations = {}
    for k in range(n_surf_probes):
        for j in range(n_domain_probes):
            g = f[f"correlations/surf{k}_dom{j}"]
            correlations[(k, j)] = {
                'R0'        : float(g.attrs['R0']),
                'running_R0': g['running_R0'][:],
            }

print(f"  {n_surf_probes} surface probe(s), {n_domain_probes} domain probes, "
      f"{n_snapshots} snapshots")

# Print summary table
print("\n  Domain probe summary:")
print(f"  {'Probe':>6s}  {'x':>7s}  {'y':>7s}  {'u_mean':>8s}  {'R(0)':>8s}")
for j, dp in enumerate(domain_probes):
    R0 = correlations[(0, j)]['R0']
    print(f"  D{j:>4d}  {dp['x_actual']:7.3f}  {dp['y_actual']:7.3f}"
          f"  {dp['u_mean']:8.4f}  {R0:+8.4f}")


# ============================================================================
# LOAD SPATIAL CORRELATION DATA
# ============================================================================
print(f"\nLoading spatial correlation from:\n  {CORR_FILE}")

with h5py.File(CORR_FILE, "r") as f:
    R_all  = f['R_all'][:]     # (Nz, Ny_crop, Nx_crop)
    x_corr = f['x'][:]        # (Nz, Ny_crop, Nx_crop)
    y_corr = f['y'][:]
    xc_ref = float(f.attrs['x_c_actual'])
    yc_ref = float(f.attrs['y_actual'])

# Take the Dz=0 slice (first z-index, since z-axis = relative separation)
R_all_2d = R_all[0, :, :]       # (Ny_crop, Nx_crop)
x_2d     = x_corr[0, :, :]
y_2d     = y_corr[0, :, :]

print(f"  R_all shape: {R_all.shape}")
print(f"  Dz=0 slice : {R_all_2d.shape}")
print(f"  Reference pt: x/c={xc_ref:.3f}, y={yc_ref:.4f}")


# ============================================================================
# Derived quantities
# ============================================================================
snap_idx = np.arange(n_snapshots)

# R(0) values sorted by magnitude for later use
R0_list = [(j, correlations[(0, j)]['R0']) for j in range(n_domain_probes)]
R0_sorted = sorted(R0_list, key=lambda x: abs(x[1]), reverse=True)


# ============================================================================
# PLOT 1: Probe locations on the 2D spatial correlation map
# ============================================================================
print("\n1. Probe locations on spatial correlation map...")

fig1, ax1 = plt.subplots(figsize=(12, 7))

# Clip R_all to [-1, 1] for robust plotting
R_plot = np.clip(R_all_2d, -1.0, 1.0)

# Hide NaN / zero regions inside the airfoil
R_plot = np.ma.masked_where(np.abs(R_plot) < 1e-12, R_plot)

# Symmetric colorbar range based on data (exclude self-correlation peak at ref point)
R_abs_max = np.max(np.abs(R_plot))   # use full data range
levels = np.linspace(-R_abs_max, R_abs_max, 41)
cf = ax1.contourf(x_2d, y_2d, R_plot, levels=levels,
                  cmap='RdBu_r', extend='both')
cbar = plt.colorbar(cf, ax=ax1, shrink=0.85, pad=0.02)
cbar.set_label(r'$R_{\tau_w^\prime u^\prime}$  (all, $\Delta z = 0$)', fontsize=12)

# Contour lines at key levels
ax1.contour(x_2d, y_2d, R_plot,
            levels=[-0.3, -0.2, -0.1, 0.1, 0.2, 0.3],
            colors='k', linewidths=0.5, alpha=0.4)

# Surface reference point
ax1.scatter(xc_ref, yc_ref, c='lime', s=350, marker='*',
            edgecolors='black', linewidths=2, zorder=20,
            label=f'Surface ref: x/c={xc_ref:.2f}')

# Domain probes — color by R(0) sign
for j, dp in enumerate(domain_probes):
    R0 = correlations[(0, j)]['R0']
    color = 'orangered' if R0 < 0 else 'dodgerblue'
    ax1.scatter(dp['x_actual'], dp['y_actual'],
                c=color, s=160, marker='^', edgecolors='black', linewidths=1.5,
                zorder=20)
    ax1.annotate(f"D{j}\nR={R0:+.2f}",
                 xy=(dp['x_actual'], dp['y_actual']),
                 xytext=(8, 8), textcoords='offset points',
                 fontsize=8, fontweight='bold',
                 color=color,
                 bbox=dict(boxstyle='round,pad=0.2', fc='white',
                           ec=color, alpha=0.85))

# Domain limits (crop to interesting region)
x_dom_min = min(dp['x_actual'] for dp in domain_probes) - 0.15
x_dom_max = max(dp['x_actual'] for dp in domain_probes) + 0.15
y_dom_min = yc_ref - 0.05
y_dom_max = max(dp['y_actual'] for dp in domain_probes) + 0.1

ax1.set_xlim(x_dom_min, x_dom_max)
ax1.set_ylim(y_dom_min, y_dom_max)
ax1.set_xlabel('x/c', fontsize=13)
ax1.set_ylabel('y/c', fontsize=13)
ax1.set_title(r'Domain probe locations on spatial $R_{\tau_w^\prime u^\prime}$'
              r' ($\Delta z = 0$)',
              fontsize=13, fontweight='bold')
ax1.set_aspect('equal', adjustable='box')
ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)

plt.tight_layout()


# ============================================================================
# PLOT 2: Phase relationship — overlay tau'_w with strongest +R and -R probes
# ============================================================================
print("2. Phase relationship (in-phase vs anti-phase)...")

# Find strongest positive and negative R(0) probes
j_pos = max(range(n_domain_probes), key=lambda j: correlations[(0, j)]['R0'])
j_neg = min(range(n_domain_probes), key=lambda j: correlations[(0, j)]['R0'])
R0_pos = correlations[(0, j_pos)]['R0']
R0_neg = correlations[(0, j_neg)]['R0']

fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

tau_norm_full = surf_probes[0]['tau_prime'] / surf_probes[0]['tau_w_rms']

# Zoom window (show ~20% of the signal so individual events are visible)
n_show = n_snapshots // 5
idx_show = slice(0, n_show)

# --- Top panel: tau'_w vs positive-R probe (in-phase) ---
dp_pos = domain_probes[j_pos]
u_rms_pos = np.std(dp_pos['u_prime'])
u_norm_pos = dp_pos['u_prime'] / u_rms_pos

ax2a.plot(snap_idx[idx_show], tau_norm_full[idx_show],
          color='steelblue', lw=1.0, label=r"$\tau'_w / \tau_{rms}$ (surface)")
ax2a.plot(snap_idx[idx_show], u_norm_pos[idx_show],
          color='orangered', lw=1.0, alpha=0.8,
          label=f"$u'/\\sigma_u$ at D{j_pos} ({dp_pos['x_actual']:.2f}, "
                f"{dp_pos['y_actual']:.2f})")
ax2a.axhline(0, color='k', lw=0.5, ls=':')
ax2a.set_ylabel('Normalised fluctuation', fontsize=11)
ax2a.set_title(f"In-phase (positive correlation): D{j_pos}, R(0) = {R0_pos:+.3f}",
               fontsize=12, fontweight='bold', loc='left')
ax2a.legend(fontsize=9, loc='upper right')
ax2a.grid(True, alpha=0.3)

# --- Bottom panel: tau'_w vs negative-R probe (anti-phase) ---
dp_neg = domain_probes[j_neg]
u_rms_neg = np.std(dp_neg['u_prime'])
u_norm_neg = dp_neg['u_prime'] / u_rms_neg

ax2b.plot(snap_idx[idx_show], tau_norm_full[idx_show],
          color='steelblue', lw=1.0, label=r"$\tau'_w / \tau_{rms}$ (surface)")
ax2b.plot(snap_idx[idx_show], u_norm_neg[idx_show],
          color='orangered', lw=1.0, alpha=0.8,
          label=f"$u'/\\sigma_u$ at D{j_neg} ({dp_neg['x_actual']:.2f}, "
                f"{dp_neg['y_actual']:.2f})")
ax2b.axhline(0, color='k', lw=0.5, ls=':')
ax2b.set_ylabel('Normalised fluctuation', fontsize=11)
ax2b.set_xlabel('Snapshot index', fontsize=11)
ax2b.set_title(f"Anti-phase (negative correlation): D{j_neg}, R(0) = {R0_neg:+.3f}",
               fontsize=12, fontweight='bold', loc='left')
ax2b.legend(fontsize=9, loc='upper right')
ax2b.grid(True, alpha=0.3)

fig2.suptitle(r'Phase relationship: $\tau^\prime_w$ vs $u^\prime$ at correlated probes',
              fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])


# ============================================================================
# PLOT 3: Normalised time series — tau'_w and all u' on shared x-axis
# ============================================================================
print("3. Time series overview...")

n_panels = 1 + n_domain_probes
fig3 = plt.figure(figsize=(14, 2.5 * n_panels))
gs3 = GridSpec(n_panels, 1, figure=fig3, hspace=0.15)

# Surface probe
ax = fig3.add_subplot(gs3[0, 0])
tau_norm = surf_probes[0]['tau_prime'] / surf_probes[0]['tau_w_rms']
ax.plot(snap_idx, tau_norm, color='steelblue', lw=0.6)
ax.axhline(0, color='k', lw=0.5, ls=':')
ax.set_ylabel(r"$\tau'_w / \tau_{rms}$", fontsize=10)
ax.set_title(f"Surface probe S0: x/c = {surf_probes[0]['xc_actual']:.3f}",
             fontsize=11, fontweight='bold', loc='left')
ax.grid(True, alpha=0.3)
ax.set_xticklabels([])

# Domain probes
for j, dp in enumerate(domain_probes):
    ax = fig3.add_subplot(gs3[1 + j, 0], sharex=fig3.axes[0])
    u_rms = np.std(dp['u_prime'])
    u_norm = dp['u_prime'] / u_rms if u_rms > 1e-14 else dp['u_prime']
    R0 = correlations[(0, j)]['R0']
    ax.plot(snap_idx, u_norm, color='tomato', lw=0.6)
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.set_ylabel(r"$u'/\sigma_u$", fontsize=10)
    ax.set_title(f"D{j}: ({dp['x_actual']:.3f}, {dp['y_actual']:.3f})  |  "
                 f"R(0) = {R0:+.3f}",
                 fontsize=10, fontweight='bold', loc='left')
    ax.grid(True, alpha=0.3)
    if j < n_domain_probes - 1:
        ax.set_xticklabels([])

fig3.axes[-1].set_xlabel('Snapshot index', fontsize=11)
fig3.suptitle('Normalised fluctuation time series',
              fontsize=13, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])


# ============================================================================
# PLOT 4: Convergence of R(0)
# ============================================================================
print("4. Convergence of R(0)...")

fig4, ax4 = plt.subplots(figsize=(12, 6))

cmap = plt.cm.tab10
for j in range(n_domain_probes):
    corr = correlations[(0, j)]
    ax4.plot(snap_idx, corr['running_R0'],
             color=cmap(j), lw=1.2,
             label=f"D{j} ({domain_probes[j]['x_actual']:.2f}, "
                   f"{domain_probes[j]['y_actual']:.2f})  R={corr['R0']:+.3f}")

ax4.axhline(0, color='k', lw=0.5, ls=':')
ax4.set_xlabel('Number of snapshots', fontsize=12)
ax4.set_ylabel(r'Running $R(0)$', fontsize=12)
ax4.set_title('Convergence of zero-lag correlation', fontsize=13, fontweight='bold')
ax4.legend(fontsize=9, loc='best', ncol=2, framealpha=0.9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()


# ============================================================================
# PLOT 5: Scatter plots — tau'_w vs u' for the 3 strongest correlated pairs
# ============================================================================
print("5. Scatter plots for strongest pairs...")

n_scatter = min(3, n_domain_probes)
top_pairs = R0_sorted[:n_scatter]

fig5, axes5 = plt.subplots(1, n_scatter, figsize=(5 * n_scatter, 5))
if n_scatter == 1:
    axes5 = [axes5]

tau_norm = surf_probes[0]['tau_prime'] / surf_probes[0]['tau_w_rms']

for idx, (j, R0) in enumerate(top_pairs):
    ax = axes5[idx]
    dp = domain_probes[j]
    u_rms = np.std(dp['u_prime'])
    u_norm = dp['u_prime'] / u_rms if u_rms > 1e-14 else dp['u_prime']

    ax.scatter(tau_norm, u_norm, s=6, alpha=0.4,
               c='steelblue', edgecolors='none')

    # Least-squares fit line
    coeffs = np.polyfit(tau_norm, u_norm, 1)
    x_fit = np.linspace(tau_norm.min(), tau_norm.max(), 100)
    ax.plot(x_fit, np.polyval(coeffs, x_fit), 'r-', lw=2,
            label=f'slope = {coeffs[0]:.2f}')

    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.axvline(0, color='k', lw=0.5, ls=':')
    ax.set_xlabel(r"$\tau'_w / \tau_{rms}$", fontsize=11)
    ax.set_ylabel(r"$u' / \sigma_u$", fontsize=11)
    ax.set_title(f"D{j}: ({dp['x_actual']:.2f}, {dp['y_actual']:.2f})\n"
                 f"R(0) = {R0:+.3f}",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

fig5.suptitle(r'Joint distributions: $\tau^\prime_w$ vs $u^\prime$'
              ' (strongest pairs)',
              fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()


# ============================================================================
# PLOT 6: Power spectral density (PSD)
# ============================================================================
print("6. Power spectral density...")

fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(14, 5))

# --- Surface probe PSD ---
tau_raw = surf_probes[0]['tau_prime']
freqs_tau = np.fft.rfftfreq(n_snapshots)
psd_tau = np.abs(np.fft.rfft(tau_raw)) ** 2 / n_snapshots

ax6a.semilogy(freqs_tau[1:], psd_tau[1:], color='steelblue', lw=0.8)
ax6a.set_xlabel('Normalised frequency (1/snapshot)', fontsize=11)
ax6a.set_ylabel('PSD', fontsize=11)
ax6a.set_title(f"Surface probe S0: x/c = {surf_probes[0]['xc_actual']:.3f}",
               fontsize=11, fontweight='bold')
ax6a.grid(True, alpha=0.3, which='both')

# --- Domain probes PSD ---
for j, dp in enumerate(domain_probes):
    freqs = np.fft.rfftfreq(n_snapshots)
    psd_u = np.abs(np.fft.rfft(dp['u_prime'])) ** 2 / n_snapshots
    ax6b.semilogy(freqs[1:], psd_u[1:], lw=0.8, color=cmap(j),
                  label=f"D{j} ({dp['x_actual']:.2f}, {dp['y_actual']:.2f})")

ax6b.set_xlabel('Normalised frequency (1/snapshot)', fontsize=11)
ax6b.set_ylabel('PSD', fontsize=11)
ax6b.set_title('Domain probes', fontsize=11, fontweight='bold')
ax6b.legend(fontsize=8, loc='best', framealpha=0.9)
ax6b.grid(True, alpha=0.3, which='both')

fig6.suptitle('Power Spectral Density', fontsize=13, fontweight='bold', y=1.0)
plt.tight_layout()


# ============================================================================
# SAVE OR SHOW
# ============================================================================
print("\n" + "=" * 70)

if OUTPUT_DIR is not None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i, fig in enumerate([fig1, fig2, fig3, fig4, fig5, fig6], start=1):
        name = f"probe_vis_{i:02d}.png"
        fig.savefig(os.path.join(OUTPUT_DIR, name), dpi=150, bbox_inches='tight')
        print(f"  Saved: {name}")

plt.show()

print("DONE")
print("=" * 70)
