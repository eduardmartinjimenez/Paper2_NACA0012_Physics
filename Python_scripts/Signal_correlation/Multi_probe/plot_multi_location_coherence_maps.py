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

AOA_deg = 5.0  

# Snapshot frequencies
#VLINE_FREQUENCIES_2 = [12.5]
VLINE_FREQUENCIES_2 = None
VLINE_FREQUENCIES = None

# Vertical line frequencies (Strouhal numbers) for AoA 5º
VLINE_FREQUENCIES = [2, 12.5]

# # Vertical line frequencies (Strouhal numbers) for AoA 12º
# VLINE_FREQUENCIES = [2.0, 100.0]

if AOA_deg == 5.0:
    BASE_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/"
elif AOA_deg == 12.0:
    BASE_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/"
else:
    raise ValueError(f"AoA {AOA_deg}° not supported")

CACHE_PATH = os.path.join(BASE_PATH, "Mean_data/Coherence/")
OUTPUT_DIR = CACHE_PATH
COHERENCE_CACHE_DIR = os.path.join(CACHE_PATH, "coherence_cache")

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


def compute_spectra_welch(signal1, signal2, fs, window='hann', nperseg=None,
                         noverlap=None, detrend='linear'):
    """Compute autospectra and complex cross-spectrum using Welch's method."""
    signal1_arr = np.asarray(signal1, dtype=float)
    signal2_arr = np.asarray(signal2, dtype=float)

    valid_mask = ~(np.isnan(signal1_arr) | np.isnan(signal2_arr))
    signal1_masked = signal1_arr[valid_mask]
    signal2_masked = signal2_arr[valid_mask]

    if detrend is None:
        signal1_preprocessed = signal1_masked
        signal2_preprocessed = signal2_masked
    else:
        signal1_preprocessed = preprocess_signal_for_welch(signal1_masked, detrend_type=detrend)
        signal2_preprocessed = preprocess_signal_for_welch(signal2_masked, detrend_type=detrend)

    f, S_11 = signal.welch(signal1_preprocessed, fs=fs, window=window,
                           nperseg=nperseg, noverlap=noverlap, scaling='density')

    _, S_22 = signal.welch(signal2_preprocessed, fs=fs, window=window,
                           nperseg=nperseg, noverlap=noverlap, scaling='density')

    _, S_12 = signal.csd(signal1_preprocessed, signal2_preprocessed, fs=fs,
                        window=window, nperseg=nperseg, noverlap=noverlap,
                        scaling='density')

    return f, S_11, S_22, S_12


def compute_coherence_at_location(h5_file, xc_value, cache_dir=None):
    """
    Compute coherence between wall signals and streamwise velocity at all probe heights.
    Computes spectra for each z location separately, then averages spectra before coherence.
    Results are cached for fast reuse.

    Returns:
        n_over_c: wall-normal distances in chord units
        y_plus: wall-normal positions in wall units
        St_c: frequency array (Strouhal number)
        coherence_tau_w: 2D coherence array (probe_height x frequency)
        coherence_p_w: 2D coherence array (probe_height x frequency)
    """
    # Check cache first
    cache_file = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"coherence_xc_{xc_value:.3f}.h5")
        if os.path.exists(cache_file):
            print(f"  Loading from cache: {cache_file}")
            with h5py.File(cache_file, 'r') as cf:
                n_over_c = cf['n_over_c'][:]
                y_plus = cf['y_plus'][:]
                St_c = cf['St_c'][:]
                coherence_tau_w = cf['coherence_tau_w'][:]
                coherence_p_w = cf['coherence_p_w'][:]
            return n_over_c, y_plus, St_c, coherence_tau_w, coherence_p_w

    with h5py.File(h5_file, 'r') as f:
        # Load metadata
        probe_n_actual = f['_metadata/probe_n_actual'][:]
        probe_y_actual = f['_metadata/probe_y_actual'][:]

        # Extract time series
        tau_w_prime = f['tau_w_prime'][:]  # (Nt, nz)
        tau_w_z = f['tau_w_z'][:]  # (Nt, nz) - full signal for u_tau calculation
        pressure_prime = f['pressure_prime'][:]  # (Nt, nz)
        time_data = f['time'][:]

        nz = tau_w_prime.shape[1]
        n_probes = len(probe_n_actual)

        # Compute sampling frequency from time data
        dt = np.mean(np.diff(time_data))
        fs = 1.0 / dt

        print(f"  x/c = {xc_value:.1f}: Nt={tau_w_prime.shape[0]}, nz={nz}, n_probes={n_probes}, fs={fs:.1f} Hz")

        # Compute u_tau from mean of full wall shear signal (NOT fluctuating part)
        tau_w_mean_ref = float(np.mean(tau_w_z))
        u_tau_ref = np.sqrt(np.abs(tau_w_mean_ref) / rho_ref)
        print(f"    tau_w_mean: {tau_w_mean_ref:.6e}, u_tau: {u_tau_ref:.6e}")

        # Initialize output arrays
        n_freq = NPERSEG // 2 + 1
        coherence_tau_w = np.zeros((n_probes, n_freq))
        coherence_p_w = np.zeros((n_probes, n_freq))

        # For each probe height
        for j in range(n_probes):
            if (j + 1) % max(1, n_probes // 5) == 0 or j == 0:
                print(f"    Probe {j + 1}/{n_probes}")

            u_s_prime_j = f[f'u_s_prime_{j}'][:]  # (Nt, nz)

            # Collect spectra for all z-planes (NOT averaged first)
            S_tau_w_all_z = []
            S_p_w_all_z = []
            S_u_s_all_z = []
            S_tau_w_u_s_all_z = []
            S_p_w_u_s_all_z = []

            for iz in range(nz):
                tau_w_z_signal = tau_w_prime[:, iz]
                p_w_z_signal = pressure_prime[:, iz]
                u_s_z_signal = u_s_prime_j[:, iz]

                # Preprocess all signals
                tau_w_clean = preprocess_signal_for_welch(tau_w_z_signal, detrend_type=DETREND_TYPE)
                p_w_clean = preprocess_signal_for_welch(p_w_z_signal, detrend_type=DETREND_TYPE)
                u_s_clean = preprocess_signal_for_welch(u_s_z_signal, detrend_type=DETREND_TYPE)

                # Compute Welch spectra (single computation per signal pair)
                f_welch, S_tau_w, S_u_s_tau, S_tau_w_u_s = compute_spectra_welch(
                    tau_w_clean, u_s_clean, fs, window=WINDOW, nperseg=NPERSEG,
                    noverlap=NOVERLAP, detrend=None  # Already detrended
                )
                _, S_p_w, S_u_s_p, S_p_w_u_s = compute_spectra_welch(
                    p_w_clean, u_s_clean, fs, window=WINDOW, nperseg=NPERSEG,
                    noverlap=NOVERLAP, detrend=None  # Already detrended
                )
                _, S_u_s = signal.welch(u_s_clean, fs=fs, window=WINDOW, nperseg=NPERSEG,
                                       noverlap=NOVERLAP, detrend=False, scaling='density')

                S_tau_w_all_z.append(S_tau_w)
                S_p_w_all_z.append(S_p_w)
                S_u_s_all_z.append(S_u_s)
                S_tau_w_u_s_all_z.append(S_tau_w_u_s)
                S_p_w_u_s_all_z.append(S_p_w_u_s)

            # Average spectra over z
            S_tau_w_mean = np.mean(S_tau_w_all_z, axis=0)
            S_p_w_mean = np.mean(S_p_w_all_z, axis=0)
            S_u_s_mean = np.mean(S_u_s_all_z, axis=0)
            S_tau_w_u_s_mean = np.mean(S_tau_w_u_s_all_z, axis=0)
            S_p_w_u_s_mean = np.mean(S_p_w_u_s_all_z, axis=0)

            # Compute coherence from averaged spectra
            coherence_tau_w[j, :] = np.abs(S_tau_w_u_s_mean)**2 / (S_tau_w_mean * S_u_s_mean + 1e-30)
            coherence_p_w[j, :] = np.abs(S_p_w_u_s_mean)**2 / (S_p_w_mean * S_u_s_mean + 1e-30)
            coherence_tau_w[j, :] = np.clip(coherence_tau_w[j, :], 0, 1)
            coherence_p_w[j, :] = np.clip(coherence_p_w[j, :], 0, 1)

        # Convert frequency to Strouhal number
        St_c = f_welch * c / u_infty

        # Compute y+ from u_tau calculated from full wall-shear signal
        y_plus = probe_n_actual * u_tau_ref / nu_ref

    # Save to cache for future use
    if cache_file is not None:
        print(f"  Saving to cache: {cache_file}")
        with h5py.File(cache_file, 'w') as cf:
            cf.create_dataset('n_over_c', data=probe_n_actual)
            cf.create_dataset('y_plus', data=y_plus)
            cf.create_dataset('St_c', data=St_c)
            cf.create_dataset('coherence_tau_w', data=coherence_tau_w)
            cf.create_dataset('coherence_p_w', data=coherence_p_w)

    return probe_n_actual, y_plus, St_c, coherence_tau_w, coherence_p_w


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
    n_over_c, y_plus, St_c, coh_tau, coh_pw = compute_coherence_at_location(h5_file, xc, cache_dir=COHERENCE_CACHE_DIR)
    all_data[xc] = {
        'n_over_c': n_over_c,
        'y_plus': y_plus,
        'St_c': St_c,
        'coh_tau': coh_tau,
        'coh_pw': coh_pw
    }

# ============================================================================
# DIAGNOSTIC CHECKS
# ============================================================================

print("\n" + "="*70)
print("DIAGNOSTIC CHECKS")
print("="*70)

for xc in XC_LOCATIONS:
    data = all_data[xc]
    print(f"\nx/c = {xc}")
    print(f"  y_plus shape: {data['y_plus'].shape}")
    print(f"  St_c shape: {data['St_c'].shape}")
    print(f"  coh_tau shape: {data['coh_tau'].shape}")
    print(f"  coh_pw shape: {data['coh_pw'].shape}")
    print(f"  coh_tau range: {np.nanmin(data['coh_tau']):.4e}, {np.nanmax(data['coh_tau']):.4e}")
    print(f"  coh_pw range: {np.nanmin(data['coh_pw']):.4e}, {np.nanmax(data['coh_pw']):.4e}")
    print(f"  y_plus range: {np.nanmin(data['y_plus']):.4e}, {np.nanmax(data['y_plus']):.4e}")

# Find frequency range to skip (low frequencies with low coherence)
freq_start = 1  # Start from index 1 to skip zero frequency

# ============================================================================
# FIGURE 1: TAU_W COHERENCE (3 rows, 1 column)
# ============================================================================

print("\nGenerating Figure 1: τ_w coherence...")

fig1, axes1 = plt.subplots(3, 1, figsize=(6.5, 9.0))

fig1.subplots_adjust(
    left=0.13,
    right=0.88,
    bottom=0.08,
    top=0.86,
    hspace=0.35
)

# Collect all data for shared colorbar
all_coh_tau = np.concatenate([all_data[xc]['coh_tau'][:, freq_start:] for xc in XC_LOCATIONS])
# vmin_tau = np.min(all_coh_tau)
# vmax_tau = np.max(all_coh_tau)
vmin_tau = 0.0
vmax_tau = 1.0
levels_tau = np.linspace(vmin_tau, vmax_tau, 21)


for idx, xc in enumerate(XC_LOCATIONS):
    ax = axes1[idx]
    data = all_data[xc]
    y_plus = data['y_plus']
    St_c = data['St_c']
    coh_tau = data['coh_tau']
    n_over_c = data['n_over_c']

    # Only plot positive y+
    positive_y_mask = y_plus > 0
    y_plus_plot = y_plus[positive_y_mask]
    Z = coh_tau[positive_y_mask, freq_start:]
    n_over_c_plot = n_over_c[positive_y_mask]

    X, Y = np.meshgrid(St_c[freq_start:], y_plus_plot)

    cf = ax.contourf(X, Y, Z, levels=levels_tau, cmap='YlOrRd', vmin=vmin_tau, vmax=vmax_tau)

    ax.set_xlabel(r'$St_c = f c/U_\infty$', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$y^+$', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Secondary y-axis for (y-y_w)/c = n/c using actual data points
    def y_plus_to_n_over_c(yp):
        u_tau_ref = y_plus_plot[0] * nu_ref / n_over_c_plot[0]
        return yp * nu_ref / u_tau_ref

    def n_over_c_to_y_plus(n_c):
        u_tau_ref = y_plus_plot[0] * nu_ref / n_over_c_plot[0]
        return n_c * u_tau_ref / nu_ref

    secax = ax.secondary_yaxis(
        'right',
        functions=(y_plus_to_n_over_c, n_over_c_to_y_plus)
    )

    secax.set_ylabel(r'$y/c$', fontsize=12, fontweight='bold')
    secax.set_yscale('log')

    if idx == len(XC_LOCATIONS) - 1:
        ax.set_xlabel(r'$St_c = f c/U_\infty$', fontsize=12, fontweight='bold')
    else:
        ax.set_xlabel('')
        ax.tick_params(labelbottom=False)

    # ax.set_title(rf'$x/c = {xc:.1f}$', fontsize=13, fontweight='bold')
    ax.set_ylim(min(y_plus_plot), max(y_plus_plot))
    #ax.grid(True, alpha=0.2, which='both')

    # Add vertical lines at specific frequencies
    if VLINE_FREQUENCIES is not None:
        for freq in VLINE_FREQUENCIES:
            ax.axvline(freq, color='black', linestyle=':', linewidth=1.5, alpha=0.6)

    # Add vertical lines at specific frequencies
    if VLINE_FREQUENCIES_2 is not None:
        for freq in VLINE_FREQUENCIES_2:
            ax.axvline(freq, color='red', linestyle=':', linewidth=1.5, alpha=0.6)

# Add shared horizontal colorbar at the top
cbar_ax1 = fig1.add_axes([0.28, 0.905, 0.50, 0.012])
cbar1 = fig1.colorbar(cf, cax=cbar_ax1, orientation='horizontal')

# Put colorbar ticks on top
cbar1.ax.xaxis.set_ticks_position('top')

# Manual label on the left of the colorbar
cbar_ax1.text(
    -0.08, 0.5,
    r'$\gamma^2_{\tau_w u_s}$',
    transform=cbar_ax1.transAxes,
    ha='right',
    va='center',
    fontsize=12
)

# fig1.suptitle(
#     rf'Wall Shear Stress Coherence with Streamwise Velocity, AoA = {AOA_deg:.0f}$^\circ$',
#     fontsize=14,
#     fontweight='bold',
#     y=0.985
# )

png_file1 = os.path.join(OUTPUT_DIR, f"multi_location_coherence_tau_w_AOA{AOA_deg:.0f}.png")
eps_file1 = os.path.join(OUTPUT_DIR, f"multi_location_coherence_tau_w_AOA{AOA_deg:.0f}.eps")
plt.savefig(png_file1, dpi=300, bbox_inches='tight')
plt.savefig(eps_file1, bbox_inches='tight')
print(f"✓ Figure 1 saved:")
print(f"  PNG: {png_file1}")
print(f"  EPS: {eps_file1}")

# ============================================================================
# FIGURE 2: PRESSURE COHERENCE (3 rows, 1 column)
# ============================================================================

print("\nGenerating Figure 2: p_w coherence...")

fig2, axes2 = plt.subplots(3, 1, figsize=(6.5, 9.0))

fig2.subplots_adjust(
    left=0.13,
    right=0.88,
    bottom=0.08,
    top=0.86,
    hspace=0.35
)

# Collect all data for shared colorbar
all_coh_pw = np.concatenate([all_data[xc]['coh_pw'][:, freq_start:] for xc in XC_LOCATIONS])
# vmin_pw = np.min(all_coh_pw)
# vmax_pw = np.max(all_coh_pw)
vmin_pw = 0.0
vmax_pw = 1.0
levels_pw = np.linspace(vmin_pw, vmax_pw, 21)

for idx, xc in enumerate(XC_LOCATIONS):
    ax = axes2[idx]
    data = all_data[xc]
    y_plus = data['y_plus']
    St_c = data['St_c']
    coh_pw = data['coh_pw']
    n_over_c = data['n_over_c']

    # Only plot positive y+
    positive_y_mask = y_plus > 0
    y_plus_plot = y_plus[positive_y_mask]
    Z = coh_pw[positive_y_mask, freq_start:]
    n_over_c_plot = n_over_c[positive_y_mask]

    X, Y = np.meshgrid(St_c[freq_start:], y_plus_plot)

    cf = ax.contourf(X, Y, Z, levels=levels_pw, cmap='YlOrRd', vmin=vmin_pw, vmax=vmax_pw)

    ax.set_xlabel(r'$St_c = f c/U_\infty$', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$y^+$', fontsize=12, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Secondary y-axis for (y-y_w)/c = n/c using actual data points
    def y_plus_to_n_over_c(yp):
        u_tau_ref = y_plus_plot[0] * nu_ref / n_over_c_plot[0]
        return yp * nu_ref / u_tau_ref

    def n_over_c_to_y_plus(n_c):
        u_tau_ref = y_plus_plot[0] * nu_ref / n_over_c_plot[0]
        return n_c * u_tau_ref / nu_ref

    secax = ax.secondary_yaxis('right', functions=(y_plus_to_n_over_c, n_over_c_to_y_plus))
    secax.set_ylabel(r'$y/c$', fontsize=12, fontweight='bold')
    secax.set_yscale('log')

    if idx == len(XC_LOCATIONS) - 1:
        ax.set_xlabel(r'$St_c = f c/U_\infty$', fontsize=12, fontweight='bold')
    else:
        ax.set_xlabel('')
        ax.tick_params(labelbottom=False)

    ax.set_ylim(min(y_plus_plot), max(y_plus_plot))
    #ax.grid(True, alpha=0.2, which='both')

    # Add vertical lines at specific frequencies
    if VLINE_FREQUENCIES is not None:
        for freq in VLINE_FREQUENCIES:
            ax.axvline(freq, color='black', linestyle=':', linewidth=1.5, alpha=0.6)

    # Add vertical lines at specific frequencies
    if VLINE_FREQUENCIES_2 is not None:
        for freq in VLINE_FREQUENCIES_2:
            ax.axvline(freq, color='red', linestyle=':', linewidth=1.5, alpha=0.6)

# Add shared horizontal colorbar at the top
cbar_ax2 = fig2.add_axes([0.28, 0.905, 0.50, 0.012])
cbar2 = fig2.colorbar(cf, cax=cbar_ax2, orientation='horizontal')

# Put colorbar ticks on top
cbar2.ax.xaxis.set_ticks_position('top')

# Manual label on the left of the colorbar
cbar_ax2.text(
    -0.08, 0.5,
    r'$\gamma^2_{p_w u_s}$',
    transform=cbar_ax2.transAxes,
    ha='right',
    va='center',
    fontsize=12
)

# fig2.suptitle(
#     rf'Wall Pressure Coherence with Streamwise Velocity, AoA = {AOA_deg:.0f}$^\circ$',
#     fontsize=14,
#     fontweight='bold',
#     y=0.985
# )

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
