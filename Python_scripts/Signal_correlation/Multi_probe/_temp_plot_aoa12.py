"""
Multi-Location Coherence Maps
==============================

Visualizes wall-signal/streamwise-velocity coherence maps at three chordwise
locations (x/c = 0.5, 0.7, 0.9) for a selected angle of attack.

Generates two figures:
- Figure 1: γ²_{τ_w u_s}(St_c, y+) at three x/c locations
- Figure 2: γ²_{p_w u_s}(St_c, y+) at three x/c locations

Each figure has 3 rows (one per location) and uses a shared colorbar.
Data is loaded from pre-computed HDF5 timeseries files.
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# LaTeX style
plt.rc('text', usetex=True)
plt.rc('font', size=14, family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')

# ============================================================================
# CONFIGURATION
# ============================================================================

AOA_deg = 12.0  # Change to 12.0 for AoA 12º

if AOA_deg == 5.0:
    BASE_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/"
elif AOA_deg == 12.0:
    BASE_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/"
else:
    raise ValueError(f"AoA {AOA_deg}° not supported")

CACHE_PATH = os.path.join(BASE_PATH, "Mean_data/Coherence/")
OUTPUT_DIR = CACHE_PATH

# Chordwise locations
XC_LOCATIONS = [0.5, 0.7, 0.9]

# Physical parameters
rho_ref = 1.0
u_infty = 1.0
c = 1.0
Re_c = 50000
mu_ref = rho_ref * u_infty * c / Re_c
nu_ref = mu_ref / rho_ref

# Spectral analysis parameters (Welch's method)
NPERSEG = 4096
NOVERLAP = NPERSEG // 2
WINDOW = 'hann'
DETREND_TYPE = 'linear'

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def preprocess_signal_for_welch(x, detrend_type='linear'):
    """Preprocess signal for Welch/CSD computations."""
    x_arr = np.asarray(x, dtype=float)
    valid_idx = ~np.isnan(x_arr)
    x_clean = x_arr[valid_idx]

    if detrend_type == 'linear':
        x_preprocessed = signal.detrend(x_clean, type='linear')
    elif detrend_type == 'constant':
        x_preprocessed = x_clean - np.mean(x_clean)
    elif detrend_type is None:
        x_preprocessed = x_clean
    else:
        raise ValueError(f"Unsupported detrend_type: {detrend_type}")

    return x_preprocessed


def compute_coherence_at_location(h5_file, xc_value):
    """
    Compute coherence between wall signals and streamwise velocity at all probe heights.

    Returns:
        y_plus: wall-normal positions in wall units
        f: frequency array
        coherence_tau_w: 2D coherence array (probe_height x frequency)
        coherence_p_w: 2D coherence array (probe_height x frequency)
    """
    with h5py.File(h5_file, 'r') as f:
        # Load metadata
        probe_y_actual = f['_metadata/probe_y_actual'][:]

        # Extract time series
        tau_w_prime = f['tau_w_prime'][:]  # (Nt, nz)
        pressure_prime = f['pressure_prime'][:]  # (Nt, nz)
        time_data = f['time'][:]

        nz = tau_w_prime.shape[1]
        n_probes = len(probe_y_actual)

        # Compute sampling frequency from time data
        dt = np.mean(np.diff(time_data))
        fs = 1.0 / dt

        print(f"  x/c = {xc_value:.1f}: Nt={tau_w_prime.shape[0]}, nz={nz}, n_probes={n_probes}, fs={fs:.1f} Hz")

        # Initialize output arrays
        coherence_tau_w = np.zeros((n_probes, NPERSEG // 2 + 1))
        coherence_p_w = np.zeros((n_probes, NPERSEG // 2 + 1))

        # For each probe height
        for j in range(n_probes):
            u_s_prime_j = f[f'u_s_prime_{j}'][:]  # (Nt, nz)

            # Average over spanwise direction
            tau_w_avg_z = np.mean(tau_w_prime, axis=1)  # (Nt,)
            p_w_avg_z = np.mean(pressure_prime, axis=1)  # (Nt,)
            u_s_avg_z = np.mean(u_s_prime_j, axis=1)  # (Nt,)

            # Preprocess signals
            tau_w_clean = preprocess_signal_for_welch(tau_w_avg_z, detrend_type=DETREND_TYPE)
            p_w_clean = preprocess_signal_for_welch(p_w_avg_z, detrend_type=DETREND_TYPE)
            u_s_clean = preprocess_signal_for_welch(u_s_avg_z, detrend_type=DETREND_TYPE)

            # Compute Welch PSD and CSD
            f_welch, Suu = signal.welch(u_s_clean, fs=fs, window=WINDOW, nperseg=NPERSEG,
                                        noverlap=NOVERLAP, detrend=False, scaling='spectrum')
            _, Stau_w_u = signal.csd(tau_w_clean, u_s_clean, fs=fs, window=WINDOW, nperseg=NPERSEG,
                                     noverlap=NOVERLAP, detrend=False)
            _, Sp_w_u = signal.csd(p_w_clean, u_s_clean, fs=fs, window=WINDOW, nperseg=NPERSEG,
                                   noverlap=NOVERLAP, detrend=False)

            # Compute auto-spectra for wall signals
            _, Stau_w = signal.welch(tau_w_clean, fs=fs, window=WINDOW, nperseg=NPERSEG,
                                     noverlap=NOVERLAP, detrend=False, scaling='spectrum')
            _, Sp_w = signal.welch(p_w_clean, fs=fs, window=WINDOW, nperseg=NPERSEG,
                                   noverlap=NOVERLAP, detrend=False, scaling='spectrum')

            # Compute squared coherence γ² = |CSD|² / (Sxx * Syy)
            with np.errstate(divide='ignore', invalid='ignore'):
                coh_tau_w = np.abs(Stau_w_u) ** 2 / (Stau_w * Suu + 1e-16)
                coh_p_w = np.abs(Sp_w_u) ** 2 / (Sp_w * Suu + 1e-16)

            coherence_tau_w[j, :] = np.clip(coh_tau_w, 0, 1)
            coherence_p_w[j, :] = np.clip(coh_p_w, 0, 1)

        # Convert frequency to Strouhal number
        St_c = f_welch * c / u_infty

        # Convert wall-normal position to wall units
        # y+ = y * u_tau / nu, where u_tau is friction velocity
        # Use a representative friction velocity (could be refined)
        # For now, estimate from the probe coordinates
        Re_tau = np.sqrt(Re_c / 2)  # Rough estimate
        u_tau = Re_tau * nu_ref / c
        y_plus = probe_y_actual * u_tau / nu_ref

        return y_plus, St_c, coherence_tau_w, coherence_p_w


# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("="*70)
print(f"MULTI-LOCATION COHERENCE MAPS - AoA {AOA_deg}°")
print("="*70)

# Load data for all three locations
all_data = {}
for xc in XC_LOCATIONS:
    h5_file = os.path.join(CACHE_PATH, f"timeseries_both_xc_{xc:.3f}.h5")
    if not os.path.exists(h5_file):
        print(f"✗ File not found: {h5_file}")
        sys.exit(1)

    print(f"\nProcessing x/c = {xc:.1f}...")
    y_plus, St_c, coh_tau, coh_pw = compute_coherence_at_location(h5_file, xc)
    all_data[xc] = {
        'y_plus': y_plus,
        'St_c': St_c,
        'coh_tau': coh_tau,
        'coh_pw': coh_pw
    }

# Find frequency range to skip (low frequencies with low coherence)
freq_start = 1  # Start from index 1 to skip zero frequency

# ============================================================================
# FIGURE 1: TAU_W COHERENCE (3 rows, 1 column)
# ============================================================================

print("\nGenerating Figure 1: τ_w coherence...")

fig1, axes1 = plt.subplots(3, 1, figsize=(10, 12))

# Collect all data for shared colorbar
all_coh_tau = np.concatenate([all_data[xc]['coh_tau'][:, freq_start:] for xc in XC_LOCATIONS])
vmin_tau = np.min(all_coh_tau)
vmax_tau = np.max(all_coh_tau)
levels_tau = np.linspace(vmin_tau, vmax_tau, 21)

for idx, xc in enumerate(XC_LOCATIONS):
    ax = axes1[idx]
    data = all_data[xc]
    y_plus = data['y_plus']
    St_c = data['St_c']
    coh_tau = data['coh_tau']

    # Only plot positive y+
    positive_y_mask = y_plus > 0
    y_plus_plot = y_plus[positive_y_mask]
    Z = coh_tau[positive_y_mask, freq_start:]

    X, Y = np.meshgrid(St_c[freq_start:], y_plus_plot)

    cf = ax.contourf(X, Y, Z, levels=levels_tau, cmap='YlOrRd', vmin=vmin_tau, vmax=vmax_tau)

    ax.set_xlabel(r'$St_c = f c/U_\infty$', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$y^+$', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Secondary y-axis for y/c
    u_tau = np.sqrt(Re_c / 2) * nu_ref / c
    y_plus_to_y_over_c = lambda y_plus_val: y_plus_val * nu_ref / (u_tau * c)
    y_over_c_to_y_plus = lambda y_over_c_val: y_over_c_val * u_tau * c / nu_ref
    secax = ax.secondary_yaxis('right', functions=(y_plus_to_y_over_c, y_over_c_to_y_plus))
    secax.set_ylabel(r'$(y - y_w)/c$', fontsize=12, fontweight='bold')
    secax.set_yscale('log')

    ax.set_title(rf'$x/c = {xc:.1f}$', fontsize=13, fontweight='bold')
    ax.set_ylim(min(y_plus_plot), max(y_plus_plot))
    ax.grid(True, alpha=0.2, which='both')

# Add shared colorbar
cbar1 = fig1.colorbar(cf, ax=axes1, label=r'$\gamma^2_{\tau_w u_s}$', pad=0.02)
fig1.suptitle(rf'Wall Shear Stress Coherence with Streamwise Velocity, AoA = {AOA_deg:.0f}$^\circ$',
              fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

png_file1 = os.path.join(OUTPUT_DIR, f"multi_location_coherence_tau_w_AOA{AOA_deg:.0f}.png")
eps_file1 = os.path.join(OUTPUT_DIR, f"multi_location_coherence_tau_w_AOA{AOA_deg:.0f}.eps")
plt.savefig(png_file1, dpi=300, bbox_inches='tight')
plt.savefig(eps_file1, bbox_inches='tight')
print(f"✓ Figure 1 saved:")
print(f"  PNG: {png_file1}")
print(f"  EPS: {eps_file1}")
plt.close()

# ============================================================================
# FIGURE 2: PRESSURE COHERENCE (3 rows, 1 column)
# ============================================================================

print("\nGenerating Figure 2: p_w coherence...")

fig2, axes2 = plt.subplots(3, 1, figsize=(10, 12))

# Collect all data for shared colorbar
all_coh_pw = np.concatenate([all_data[xc]['coh_pw'][:, freq_start:] for xc in XC_LOCATIONS])
vmin_pw = np.min(all_coh_pw)
vmax_pw = np.max(all_coh_pw)
levels_pw = np.linspace(vmin_pw, vmax_pw, 21)

for idx, xc in enumerate(XC_LOCATIONS):
    ax = axes2[idx]
    data = all_data[xc]
    y_plus = data['y_plus']
    St_c = data['St_c']
    coh_pw = data['coh_pw']

    # Only plot positive y+
    positive_y_mask = y_plus > 0
    y_plus_plot = y_plus[positive_y_mask]
    Z = coh_pw[positive_y_mask, freq_start:]

    X, Y = np.meshgrid(St_c[freq_start:], y_plus_plot)

    cf = ax.contourf(X, Y, Z, levels=levels_pw, cmap='YlOrRd', vmin=vmin_pw, vmax=vmax_pw)

    ax.set_xlabel(r'$St_c = f c/U_\infty$', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$y^+$', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Secondary y-axis for y/c
    u_tau = np.sqrt(Re_c / 2) * nu_ref / c
    y_plus_to_y_over_c = lambda y_plus_val: y_plus_val * nu_ref / (u_tau * c)
    y_over_c_to_y_plus = lambda y_over_c_val: y_over_c_val * u_tau * c / nu_ref
    secax = ax.secondary_yaxis('right', functions=(y_plus_to_y_over_c, y_over_c_to_y_plus))
    secax.set_ylabel(r'$(y - y_w)/c$', fontsize=12, fontweight='bold')
    secax.set_yscale('log')

    ax.set_title(rf'$x/c = {xc:.1f}$', fontsize=13, fontweight='bold')
    ax.set_ylim(min(y_plus_plot), max(y_plus_plot))
    ax.grid(True, alpha=0.2, which='both')

# Add shared colorbar
cbar2 = fig2.colorbar(cf, ax=axes2, label=r'$\gamma^2_{p_w u_s}$', pad=0.02)
fig2.suptitle(rf'Wall Pressure Coherence with Streamwise Velocity, AoA = {AOA_deg:.0f}$^\circ$',
              fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

png_file2 = os.path.join(OUTPUT_DIR, f"multi_location_coherence_p_w_AOA{AOA_deg:.0f}.png")
eps_file2 = os.path.join(OUTPUT_DIR, f"multi_location_coherence_p_w_AOA{AOA_deg:.0f}.eps")
plt.savefig(png_file2, dpi=300, bbox_inches='tight')
plt.savefig(eps_file2, bbox_inches='tight')
print(f"✓ Figure 2 saved:")
print(f"  PNG: {png_file2}")
print(f"  EPS: {eps_file2}")
plt.show()

print("\n" + "="*70)
print("COMPLETE")
print("="*70)
