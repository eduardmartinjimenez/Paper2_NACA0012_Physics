"""
Band-Split Spectral Analysis - Two-Band Decomposition (Improved)
================================================================

Loads processed signals (tau_prime, u_prime) from full-spanwise spectral analysis
and performs a frequency band-split analysis:
- Keeps tau_prime broadband
- Splits u_prime into low-pass (f* <= cutoff) and high-pass (f* > cutoff)
- Analyzes cross-spectrum and coherence for each band separately
- Averages results over z for final statistics

Improvements:
- Variance-based energy fractions (correct and sum to ~100%)
- Welch frequency array tracked explicitly (f_welch, f_star_welch)
- Coherence masking to avoid plotting where filtered signal is negligible
- Orthogonality checks for FFT-based split
- Negative coherence values displayed correctly
- More rigorous frequency consistency checks
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Physical parameters
U_inf = 1.0
c = 1.0
F_STAR_CUTOFF = 10.0  # Band split cutoff in nondimensional frequency

# Paths
INPUT_FILE = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Signal_correlation/full_spanwise_spectral_slice_9.h5"
OUTPUT_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Signal_correlation/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Welch parameters
NPERSEG_BASE = 4096
NOVERLAP_BASE = NPERSEG_BASE // 2
WINDOW = 'hann'

# Coherence masking threshold (coherence plotted only where PSD > eps * max(PSD))
EPS_COHERENCE_MASK = 1e-6

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_input_data(input_file):
    """Load tau_prime, u_prime, and metadata from HDF5."""
    print(f"Loading from: {input_file}")

    with h5py.File(input_file, 'r') as f:
        tau_prime = f['timeseries/tau_prime'][...]
        u_prime = f['timeseries/u_prime'][...]
        iterations = f['timeseries/iterations'][...]
        time = f['timeseries/time'][...]
        f_array = f['frequency/f'][...]
        f_star = f['frequency/f_star'][...]

        # Metadata
        fs = f['timeseries'].attrs['sampling_frequency']
        Nt = f['timeseries'].attrs['Nt']
        Nz = f['timeseries'].attrs['Nz']

    return {
        'tau_prime': tau_prime,
        'u_prime': u_prime,
        'iterations': iterations,
        'time': time,
        'f': f_array,
        'f_star': f_star,
        'fs': fs,
        'Nt': Nt,
        'Nz': Nz
    }


def validate_signals(data):
    """Check data integrity and basic properties."""
    tau_prime = data['tau_prime']
    u_prime = data['u_prime']
    time = data['time']

    assert tau_prime.shape == u_prime.shape, "Shape mismatch"
    assert time.shape[0] == tau_prime.shape[0], "Time dimension mismatch"
    assert not np.any(np.isnan(tau_prime)), "NaN in tau_prime"
    assert not np.any(np.isnan(u_prime)), "NaN in u_prime"

    # Check zero mean over time for each z
    tau_mean_over_z = np.abs(np.mean(tau_prime, axis=0))
    u_mean_over_z = np.abs(np.mean(u_prime, axis=0))

    return {
        'tau_max_abs_mean': np.max(tau_mean_over_z),
        'u_max_abs_mean': np.max(u_mean_over_z)
    }


def get_welch_params(Nt):
    """Determine Welch parameters based on data length."""
    nperseg = NPERSEG_BASE
    noverlap = NOVERLAP_BASE

    if nperseg > Nt:
        nperseg = Nt // 2
        noverlap = nperseg // 2

    return nperseg, noverlap


def split_signal_fft(signal_z, f, f_cutoff):
    """
    Split a 1D signal into low-pass and high-pass using FFT.

    Args:
        signal_z: 1D array of time series data
        f: 1D array of positive frequencies (from rfft)
        f_cutoff: Cutoff frequency in Hz

    Returns:
        lp_part: Low-pass filtered signal (f <= f_cutoff)
        hp_part: High-pass filtered signal (f > f_cutoff)
    """
    # Compute FFT (assuming real signal)
    fft_vals = np.fft.rfft(signal_z)

    # Create mask
    mask_lp = f <= f_cutoff
    mask_hp = f > f_cutoff

    # Create filtered FFT
    fft_lp = fft_vals.copy()
    fft_lp[mask_hp] = 0

    fft_hp = fft_vals.copy()
    fft_hp[mask_lp] = 0

    # Inverse FFT to time domain
    lp_part = np.fft.irfft(fft_lp, n=len(signal_z))
    hp_part = np.fft.irfft(fft_hp, n=len(signal_z))

    return lp_part, hp_part


def compute_psd_welch(signal_1d, fs, nperseg, noverlap, window):
    """Compute PSD using Welch's method. Returns (f, psd) where f is the frequency array."""
    signal_centered = signal_1d - np.mean(signal_1d)
    f, psd = signal.welch(
        signal_centered,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'
    )
    return f, psd


def compute_cross_spectrum_welch(signal1, signal2, fs, nperseg, noverlap, window):
    """Compute cross-spectrum magnitude using Welch's method. Returns (f, magnitude)."""
    signal1_centered = signal1 - np.mean(signal1)
    signal2_centered = signal2 - np.mean(signal2)
    f, csd = signal.csd(
        signal1_centered,
        signal2_centered,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'
    )
    return f, np.abs(csd)


def compute_coherence_welch(signal1, signal2, fs, nperseg, noverlap, window):
    """Compute magnitude-squared coherence. Returns (f, coherence)."""
    signal1_centered = signal1 - np.mean(signal1)
    signal2_centered = signal2 - np.mean(signal2)
    f, coh = signal.coherence(
        signal1_centered,
        signal2_centered,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap
    )
    return f, coh


def verify_frequency_consistency(f_ref, f_new, label):
    """Verify that two frequency arrays match."""
    if not np.allclose(f_ref, f_new):
        raise ValueError(
            f"Frequency array mismatch for {label}. "
            f"Reference length {len(f_ref)}, new length {len(f_new)}"
        )


def plot_sample_timeseries(data, u_prime_lp, u_prime_hp, z_indices, output_dir):
    """Plot time series at selected z locations for visual validation."""
    time = data['time']
    u_prime = data['u_prime']

    for iz in z_indices:
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))

        # Original
        ax = axes[0]
        ax.plot(time, u_prime[:, iz], linewidth=0.5, color='#1f77b4', alpha=0.8)
        ax.set_ylabel("u'", fontsize=10, fontweight='bold')
        ax.set_title(f"Original u_prime at z-index {iz}", fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Low-pass
        ax = axes[1]
        ax.plot(time, u_prime_lp[:, iz], linewidth=0.5, color='#2ca02c', alpha=0.8)
        ax.set_ylabel("u'_lp", fontsize=10, fontweight='bold')
        ax.set_title(f"Low-pass u_prime (f* <= {F_STAR_CUTOFF}) at z-index {iz}",
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # High-pass
        ax = axes[2]
        ax.plot(time, u_prime_hp[:, iz], linewidth=0.5, color='#ff7f0e', alpha=0.8)
        ax.set_ylabel("u'_hp", fontsize=10, fontweight='bold')
        ax.set_title(f"High-pass u_prime (f* > {F_STAR_CUTOFF}) at z-index {iz}",
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Reconstruction check
        ax = axes[3]
        reconstructed = u_prime_lp[:, iz] + u_prime_hp[:, iz]
        ax.plot(time, u_prime[:, iz], linewidth=1.0, color='#1f77b4', alpha=0.8,
                label='Original')
        ax.plot(time, reconstructed, linewidth=1.0, color='#d62728', alpha=0.6,
                label='Reconstructed (lp + hp)', linestyle='--')
        ax.set_xlabel("Time (s)", fontsize=10, fontweight='bold')
        ax.set_ylabel("u'", fontsize=10, fontweight='bold')
        ax.set_title(f"Reconstruction Verification at z-index {iz}",
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        plt.tight_layout()
        plt.show()


def plot_psd_validation(data, u_prime_lp, u_prime_hp, z_indices, fs, nperseg,
                        noverlap, f_star_welch, output_dir):
    """Plot PSDs of original and filtered signals at selected z locations."""
    u_prime = data['u_prime']

    freq_idx_start = 1  # Skip zero frequency

    for iz in z_indices:
        # Compute PSDs
        _, psd_original = compute_psd_welch(u_prime[:, iz], fs, nperseg, noverlap, WINDOW)
        _, psd_lp = compute_psd_welch(u_prime_lp[:, iz], fs, nperseg, noverlap, WINDOW)
        _, psd_hp = compute_psd_welch(u_prime_hp[:, iz], fs, nperseg, noverlap, WINDOW)
        _, psd_reconstructed = compute_psd_welch(
            u_prime_lp[:, iz] + u_prime_hp[:, iz], fs, nperseg, noverlap, WINDOW
        )

        fig, ax = plt.subplots(figsize=(12, 8))

        ax.loglog(f_star_welch[freq_idx_start:], psd_original[freq_idx_start:],
                 'o-', linewidth=1.5, markersize=2, color='#1f77b4', alpha=0.8,
                 label='Original u_prime')
        ax.loglog(f_star_welch[freq_idx_start:], psd_lp[freq_idx_start:],
                 's-', linewidth=1.5, markersize=2, color='#2ca02c', alpha=0.8,
                 label=f'Low-pass (f* <= {F_STAR_CUTOFF})')
        ax.loglog(f_star_welch[freq_idx_start:], psd_hp[freq_idx_start:],
                 '^-', linewidth=1.5, markersize=2, color='#ff7f0e', alpha=0.8,
                 label=f'High-pass (f* > {F_STAR_CUTOFF})')
        ax.loglog(f_star_welch[freq_idx_start:], psd_reconstructed[freq_idx_start:],
                 'v--', linewidth=1.5, markersize=2, color='#d62728', alpha=0.8,
                 label='Reconstructed (lp + hp)')

        ax.axvline(F_STAR_CUTOFF, color='red', linestyle='--', linewidth=2,
                  alpha=0.7, label=f'Cutoff f* = {F_STAR_CUTOFF}')

        ax.set_xlabel("Nondimensional Frequency f* = f·c/U∞ [-]", fontsize=11, fontweight='bold')
        ax.set_ylabel("PSD [signal²/Hz]", fontsize=11, fontweight='bold')
        ax.set_title(f"PSD Validation: Original vs Filtered at z-index {iz}",
                    fontsize=12, fontweight='bold')
        ax.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)
        ax.legend(fontsize=10, loc='upper right')

        plt.tight_layout()
        plt.show()


def plot_final_results(f_star_welch, psd_tau_mean, psd_u_mean, psd_u_lp_mean, psd_u_hp_mean,
                       csd_mag_broadband_mean, csd_mag_lp_mean, csd_mag_hp_mean,
                       coherence_broadband_mean, coherence_lp_mean, coherence_hp_mean,
                       mask_lp_valid, mask_hp_valid,
                       covariance_stats, correlation_stats, output_dir):
    """Create final summary plots."""
    freq_idx_start = 1

    # Figure 1: PSD comparison
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.loglog(f_star_welch[freq_idx_start:], psd_tau_mean[freq_idx_start:],
             'o-', linewidth=1.5, markersize=3, color='#d62728', alpha=0.8,
             label="τ' broadband")
    ax.loglog(f_star_welch[freq_idx_start:], psd_u_mean[freq_idx_start:],
             's-', linewidth=1.5, markersize=3, color='#1f77b4', alpha=0.8,
             label="u' broadband")
    ax.loglog(f_star_welch[freq_idx_start:], psd_u_lp_mean[freq_idx_start:],
             '^-', linewidth=1.5, markersize=3, color='#2ca02c', alpha=0.8,
             label="u' low-pass")
    ax.loglog(f_star_welch[freq_idx_start:], psd_u_hp_mean[freq_idx_start:],
             'v-', linewidth=1.5, markersize=3, color='#ff7f0e', alpha=0.8,
             label="u' high-pass")

    ax.axvline(F_STAR_CUTOFF, color='red', linestyle='--', linewidth=2, alpha=0.7)

    ax.set_xlabel("Nondimensional Frequency f* = f·c/U∞ [-]", fontsize=11, fontweight='bold')
    ax.set_ylabel("PSD [signal²/Hz]", fontsize=11, fontweight='bold')
    ax.set_title("PSD Comparison: Broadband vs Band-Split", fontsize=12, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)
    ax.legend(fontsize=10, loc='upper right')

    plt.tight_layout()
    plt.show()

    # Figure 2: Cross-spectrum magnitude comparison
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.loglog(f_star_welch[freq_idx_start:], csd_mag_broadband_mean[freq_idx_start:],
             'o-', linewidth=1.5, markersize=3, color='#1f77b4', alpha=0.8,
             label='τ vs u broadband')
    ax.loglog(f_star_welch[freq_idx_start:], csd_mag_lp_mean[freq_idx_start:],
             's-', linewidth=1.5, markersize=3, color='#2ca02c', alpha=0.8,
             label='τ vs u low-pass')
    ax.loglog(f_star_welch[freq_idx_start:], csd_mag_hp_mean[freq_idx_start:],
             '^-', linewidth=1.5, markersize=3, color='#ff7f0e', alpha=0.8,
             label='τ vs u high-pass')

    ax.axvline(F_STAR_CUTOFF, color='red', linestyle='--', linewidth=2, alpha=0.7)

    ax.set_xlabel("Nondimensional Frequency f* = f·c/U∞ [-]", fontsize=11, fontweight='bold')
    ax.set_ylabel("|S_τu| [signal product / Hz]", fontsize=11, fontweight='bold')
    ax.set_title("Cross-Spectrum Magnitude: Band Comparison", fontsize=12, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)
    ax.legend(fontsize=10, loc='upper right')

    plt.tight_layout()
    plt.show()

    # Figure 3: Coherence comparison with masking
    # Only plot coherence where the filtered signal carries meaningful energy
    fig, ax = plt.subplots(figsize=(12, 8))

    # Broadband coherence (always plotted)
    ax.semilogx(f_star_welch[freq_idx_start:], coherence_broadband_mean[freq_idx_start:],
               'o-', linewidth=1.5, markersize=3, color='#1f77b4', alpha=0.8,
               label='γ²(τ, u) broadband')

    # Low-pass coherence (only where mask is true)
    coh_lp_masked = coherence_lp_mean.copy()
    coh_lp_masked[~mask_lp_valid] = np.nan
    ax.semilogx(f_star_welch[freq_idx_start:], coh_lp_masked[freq_idx_start:],
               's-', linewidth=1.5, markersize=3, color='#2ca02c', alpha=0.8,
               label='γ²(τ, u_lp) low-pass (masked)')

    # High-pass coherence (only where mask is true)
    coh_hp_masked = coherence_hp_mean.copy()
    coh_hp_masked[~mask_hp_valid] = np.nan
    ax.semilogx(f_star_welch[freq_idx_start:], coh_hp_masked[freq_idx_start:],
               '^-', linewidth=1.5, markersize=3, color='#ff7f0e', alpha=0.8,
               label='γ²(τ, u_hp) high-pass (masked)')

    ax.axvline(F_STAR_CUTOFF, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_ylim([0, 1.05])

    ax.set_xlabel("Nondimensional Frequency f* = f·c/U∞ [-]", fontsize=11, fontweight='bold')
    ax.set_ylabel("γ²(f) [-]", fontsize=11, fontweight='bold')
    ax.set_title("Coherence: Band Comparison (masked where PSD is small)",
                fontsize=12, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
    ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)
    ax.legend(fontsize=10, loc='upper right')

    plt.tight_layout()
    plt.show()

    # Figure 4: Zero-lag statistics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bands = ['Broadband', 'Low-pass', 'High-pass']
    covariances = [covariance_stats['broadband'], covariance_stats['lp'], covariance_stats['hp']]
    correlations = [correlation_stats['broadband'], correlation_stats['lp'], correlation_stats['hp']]

    x_pos = np.arange(len(bands))
    width = 0.35

    # Covariance plot
    ax1.bar(x_pos, covariances, width, color=['#1f77b4', '#2ca02c', '#ff7f0e'], alpha=0.8)
    ax1.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    ax1.set_ylabel('Covariance(τ, u) [Pa·(m/s)]', fontsize=11, fontweight='bold')
    ax1.set_title('Zero-Lag Covariance by Band', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(bands)
    ax1.grid(True, alpha=0.3, axis='y')

    # Correlation plot with automatic y-limits to show negative values
    y_min = min(0, np.min(correlations) - 0.1)
    y_max = max(1.0, np.max(correlations) + 0.1)

    ax2.bar(x_pos, correlations, width, color=['#1f77b4', '#2ca02c', '#ff7f0e'], alpha=0.8)
    ax2.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
    ax2.set_ylabel('Correlation(τ, u) [-]', fontsize=11, fontweight='bold')
    ax2.set_title('Zero-Lag Correlation by Band', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bands)
    ax2.set_ylim([y_min, y_max])
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def save_results_hdf5(output_file, data, u_prime_lp, u_prime_hp, f_cutoff,
                      psd_tau_mean, psd_u_mean, psd_u_lp_mean, psd_u_hp_mean,
                      csd_mag_broadband_mean, csd_mag_lp_mean, csd_mag_hp_mean,
                      coherence_broadband_mean, coherence_lp_mean, coherence_hp_mean,
                      mask_lp_valid, mask_hp_valid,
                      covariance_stats, correlation_stats, nperseg, noverlap,
                      f_welch, f_star_welch):
    """Save results to HDF5."""
    print(f"\nSaving to: {output_file}")

    with h5py.File(output_file, 'w') as hf:
        # Timeseries group
        ts_group = hf.create_group('timeseries')
        ts_group.create_dataset('u_prime_lp', data=u_prime_lp, compression='gzip')
        ts_group.attrs['u_prime_lp_description'] = f'Low-pass u_prime (f* <= {F_STAR_CUTOFF})'

        ts_group.create_dataset('u_prime_hp', data=u_prime_hp, compression='gzip')
        ts_group.attrs['u_prime_hp_description'] = f'High-pass u_prime (f* > {F_STAR_CUTOFF})'

        ts_group.create_dataset('time', data=data['time'])
        ts_group.attrs['Nt'] = data['Nt']
        ts_group.attrs['Nz'] = data['Nz']

        # Frequency group
        freq_group = hf.create_group('frequency')
        freq_group.create_dataset('f_welch', data=f_welch)
        freq_group.create_dataset('f_star_welch', data=f_star_welch)
        freq_group.attrs['frequency_computed_from_welch'] = True
        freq_group.attrs['nperseg'] = nperseg

        # Spectral group
        spec_group = hf.create_group('spectral')

        spec_group.create_dataset('psd_tau_mean', data=psd_tau_mean, compression='gzip')
        spec_group.attrs['psd_tau_mean_description'] = 'Mean PSD of tau_prime (broadband)'

        spec_group.create_dataset('psd_u_mean', data=psd_u_mean, compression='gzip')
        spec_group.attrs['psd_u_mean_description'] = 'Mean PSD of u_prime (broadband)'

        spec_group.create_dataset('psd_u_lp_mean', data=psd_u_lp_mean, compression='gzip')
        spec_group.attrs['psd_u_lp_mean_description'] = f'Mean PSD of u_prime_lp (f* <= {F_STAR_CUTOFF})'

        spec_group.create_dataset('psd_u_hp_mean', data=psd_u_hp_mean, compression='gzip')
        spec_group.attrs['psd_u_hp_mean_description'] = f'Mean PSD of u_prime_hp (f* > {F_STAR_CUTOFF})'

        spec_group.create_dataset('csd_mag_broadband_mean', data=csd_mag_broadband_mean,
                                 compression='gzip')
        spec_group.attrs['csd_mag_broadband_description'] = 'Mean |S_tau_u| broadband'

        spec_group.create_dataset('csd_mag_lp_mean', data=csd_mag_lp_mean, compression='gzip')
        spec_group.attrs['csd_mag_lp_description'] = f'Mean |S_tau_u_lp| for f* <= {F_STAR_CUTOFF}'

        spec_group.create_dataset('csd_mag_hp_mean', data=csd_mag_hp_mean, compression='gzip')
        spec_group.attrs['csd_mag_hp_description'] = f'Mean |S_tau_u_hp| for f* > {F_STAR_CUTOFF}'

        spec_group.create_dataset('coherence_broadband_mean', data=coherence_broadband_mean,
                                 compression='gzip')
        spec_group.attrs['coherence_broadband_description'] = 'Mean γ²(τ, u) broadband'

        spec_group.create_dataset('coherence_lp_mean', data=coherence_lp_mean, compression='gzip')
        spec_group.attrs['coherence_lp_description'] = f'Mean γ²(τ, u_lp) for f* <= {F_STAR_CUTOFF}'

        spec_group.create_dataset('coherence_hp_mean', data=coherence_hp_mean, compression='gzip')
        spec_group.attrs['coherence_hp_description'] = f'Mean γ²(τ, u_hp) for f* > {F_STAR_CUTOFF}'

        # Coherence masking information
        spec_group.create_dataset('mask_lp_valid', data=mask_lp_valid.astype(bool))
        spec_group.attrs['mask_lp_valid_description'] = (
            f'Boolean mask: coherence_lp plotted only where psd_u_lp_mean > {EPS_COHERENCE_MASK} * max'
        )

        spec_group.create_dataset('mask_hp_valid', data=mask_hp_valid.astype(bool))
        spec_group.attrs['mask_hp_valid_description'] = (
            f'Boolean mask: coherence_hp plotted only where psd_u_hp_mean > {EPS_COHERENCE_MASK} * max'
        )

        spec_group.attrs['eps_coherence_mask'] = EPS_COHERENCE_MASK
        spec_group.attrs['nperseg'] = nperseg
        spec_group.attrs['noverlap'] = noverlap
        spec_group.attrs['window'] = WINDOW
        spec_group.attrs['f_cutoff_hz'] = f_cutoff
        spec_group.attrs['f_cutoff_star'] = F_STAR_CUTOFF

        # Summary group
        summary_group = hf.create_group('summary')

        summary_group.attrs['covariance_broadband'] = covariance_stats['broadband']
        summary_group.attrs['covariance_lp'] = covariance_stats['lp']
        summary_group.attrs['covariance_hp'] = covariance_stats['hp']

        summary_group.attrs['correlation_broadband'] = correlation_stats['broadband']
        summary_group.attrs['correlation_lp'] = correlation_stats['lp']
        summary_group.attrs['correlation_hp'] = correlation_stats['hp']

        # Root metadata
        hf.attrs['description'] = 'Band-split spectral analysis (improved version)'
        hf.attrs['filter_type'] = 'FFT-based sharp cutoff'
        hf.attrs['Nt'] = data['Nt']
        hf.attrs['Nz'] = data['Nz']
        hf.attrs['fs'] = data['fs']
        hf.attrs['f_cutoff_hz'] = f_cutoff
        hf.attrs['f_cutoff_star'] = F_STAR_CUTOFF

    print(f"✓ Results saved")


# ============================================================================
# MAIN SCRIPT
# ============================================================================

print("="*70)
print("PHASE 1: LOAD DATA")
print("="*70)

data = load_input_data(INPUT_FILE)

tau_prime = data['tau_prime']
u_prime = data['u_prime']
Nt = data['Nt']
Nz = data['Nz']
fs = data['fs']
time = data['time']

print(f"\nData shapes:")
print(f"  tau_prime: {tau_prime.shape} (Nt={Nt}, Nz={Nz})")
print(f"  u_prime: {u_prime.shape}")
print(f"Metadata:")
print(f"  Nt: {Nt}")
print(f"  Nz: {Nz}")
print(f"  Time span: {time[-1] - time[0]:.6f} s")
print(f"  Sampling frequency: {fs:.6f} Hz")

# ============================================================================
# PHASE 2: BASIC VALIDATION
# ============================================================================

print("\n" + "="*70)
print("PHASE 2: BASIC VALIDATION")
print("="*70)

val_stats = validate_signals(data)

print(f"\nSignal validation:")
print(f"  Max absolute mean(tau_prime) over z: {val_stats['tau_max_abs_mean']:.6e}")
print(f"  Max absolute mean(u_prime) over z: {val_stats['u_max_abs_mean']:.6e}")
print(f"  ✓ Data integrity check passed")

# ============================================================================
# PHASE 3: DEFINE BAND SPLIT
# ============================================================================

print("\n" + "="*70)
print("PHASE 3: DEFINE BAND SPLIT")
print("="*70)

f_cutoff = F_STAR_CUTOFF * U_inf / c

print(f"\nBand split cutoff:")
print(f"  Nondimensional cutoff: f* = {F_STAR_CUTOFF}")
print(f"  Dimensional cutoff: f = {f_cutoff:.6e} Hz")
print(f"  Low-pass band: f* <= {F_STAR_CUTOFF}")
print(f"  High-pass band: f* > {F_STAR_CUTOFF}")

# ============================================================================
# PHASE 4: FFT-BASED FILTERING OF u_prime
# ============================================================================

print("\n" + "="*70)
print("PHASE 4: FFT-BASED FILTERING OF u_prime")
print("="*70)

u_prime_lp = np.zeros_like(u_prime)
u_prime_hp = np.zeros_like(u_prime)

# Get FFT frequencies for filtering
fft_freqs = np.fft.rfftfreq(Nt, d=1/fs)

print(f"\nFiltering {Nz} z-positions...")

for iz in range(Nz):
    lp, hp = split_signal_fft(u_prime[:, iz], fft_freqs, f_cutoff)
    u_prime_lp[:, iz] = lp
    u_prime_hp[:, iz] = hp

    if (iz + 1) % max(1, Nz // 5) == 0 or iz == 0:
        print(f"  z-index {iz}/{Nz-1}: Filtered")

print(f"✓ Filtering complete")

# ============================================================================
# PHASE 5: RECONSTRUCTION CHECK WITH VARIANCE FRACTIONS
# ============================================================================

print("\n" + "="*70)
print("PHASE 5: RECONSTRUCTION CHECK WITH VARIANCE FRACTIONS")
print("="*70)

reconstruction = u_prime_lp + u_prime_hp
reconstruction_error = np.max(np.abs(u_prime - reconstruction))

rms_original = np.sqrt(np.mean(u_prime**2))
rms_lp = np.sqrt(np.mean(u_prime_lp**2))
rms_hp = np.sqrt(np.mean(u_prime_hp**2))

# Variance-based energy fractions (these sum to ~100%)
var_original = np.mean(u_prime**2)
var_lp = np.mean(u_prime_lp**2)
var_hp = np.mean(u_prime_hp**2)

lp_variance_fraction = 100.0 * var_lp / var_original
hp_variance_fraction = 100.0 * var_hp / var_original
total_variance_fraction = lp_variance_fraction + hp_variance_fraction

print(f"\nReconstruction verification:")
print(f"  Max reconstruction error: {reconstruction_error:.6e}")

print(f"\nRMS values:")
print(f"  RMS of original signal: {rms_original:.6e}")
print(f"  RMS of low-pass part: {rms_lp:.6e}")
print(f"  RMS of high-pass part: {rms_hp:.6e}")

print(f"\nVariance-based energy fractions (CORRECT - sum to ~100%):")
print(f"  Low-pass variance fraction: {lp_variance_fraction:.2f}%")
print(f"  High-pass variance fraction: {hp_variance_fraction:.2f}%")
print(f"  Total: {total_variance_fraction:.2f}%")

# Orthogonality checks for FFT-based split
mean_lp = np.mean(u_prime_lp)
mean_hp = np.mean(u_prime_hp)
cross_term = np.mean(u_prime_lp * u_prime_hp)

print(f"\nOrthogonality checks:")
print(f"  Mean of u_prime_lp: {mean_lp:.6e} (expect ~0)")
print(f"  Mean of u_prime_hp: {mean_hp:.6e} (expect ~0)")
print(f"  Mean(u_prime_lp * u_prime_hp): {cross_term:.6e} (expect ~0 for orthogonal split)")

# ============================================================================
# PHASE 6: QUICK TIME-SERIES VISUALIZATION
# ============================================================================

print("\n" + "="*70)
print("PHASE 6: QUICK TIME-SERIES VISUALIZATION")
print("="*70)

z_sample_indices = [0, Nz // 4, Nz // 2]
print(f"Plotting time series at z-indices: {z_sample_indices}")

plot_sample_timeseries(data, u_prime_lp, u_prime_hp, z_sample_indices, OUTPUT_DIR)

# ============================================================================
# PHASE 7: PSD CHECK OF FILTERED SIGNALS
# ============================================================================

print("\n" + "="*70)
print("PHASE 7: PSD CHECK OF FILTERED SIGNALS")
print("="*70)

nperseg, noverlap = get_welch_params(Nt)
print(f"Welch parameters: nperseg={nperseg}, noverlap={noverlap}")
print(f"Plotting PSD validation at z-indices: {z_sample_indices}")

# Compute Welch frequency arrays early (needed for plots and spectral analysis)
f_welch_temp, _ = compute_psd_welch(u_prime[:, 0], fs, nperseg, noverlap, WINDOW)
f_star_welch_temp = f_welch_temp * c / U_inf

plot_psd_validation(data, u_prime_lp, u_prime_hp, z_sample_indices,
                   fs, nperseg, noverlap, f_star_welch_temp, OUTPUT_DIR)

# ============================================================================
# PHASE 8: BAND-SPECIFIC SPECTRAL ANALYSIS FOR EACH z
# ============================================================================

print("\n" + "="*70)
print("PHASE 8: BAND-SPECIFIC SPECTRAL ANALYSIS FOR EACH z")
print("="*70)

psd_tau_all_z = []
psd_u_all_z = []
psd_u_lp_all_z = []
psd_u_hp_all_z = []
csd_mag_broadband_all_z = []
csd_mag_lp_all_z = []
csd_mag_hp_all_z = []
coherence_broadband_all_z = []
coherence_lp_all_z = []
coherence_hp_all_z = []

# Use frequency arrays computed in PHASE 7
f_welch = f_welch_temp
f_star_welch = f_star_welch_temp

print(f"Computing spectral quantities for {Nz} z-positions...")

for iz in range(Nz):
    tau_z = tau_prime[:, iz]
    u_z = u_prime[:, iz]
    u_lp_z = u_prime_lp[:, iz]
    u_hp_z = u_prime_hp[:, iz]

    # PSD (verify frequency array consistency)
    f, psd_tau = compute_psd_welch(tau_z, fs, nperseg, noverlap, WINDOW)
    verify_frequency_consistency(f_welch, f, "PSD tau")

    _, psd_u = compute_psd_welch(u_z, fs, nperseg, noverlap, WINDOW)
    _, psd_u_lp = compute_psd_welch(u_lp_z, fs, nperseg, noverlap, WINDOW)
    _, psd_u_hp = compute_psd_welch(u_hp_z, fs, nperseg, noverlap, WINDOW)

    psd_tau_all_z.append(psd_tau)
    psd_u_all_z.append(psd_u)
    psd_u_lp_all_z.append(psd_u_lp)
    psd_u_hp_all_z.append(psd_u_hp)

    # Cross-spectrum magnitude
    f_csd, csd_mag_bb = compute_cross_spectrum_welch(tau_z, u_z, fs, nperseg, noverlap, WINDOW)
    verify_frequency_consistency(f_welch, f_csd, "CSD broadband")

    _, csd_mag_lp = compute_cross_spectrum_welch(tau_z, u_lp_z, fs, nperseg, noverlap, WINDOW)
    _, csd_mag_hp = compute_cross_spectrum_welch(tau_z, u_hp_z, fs, nperseg, noverlap, WINDOW)

    csd_mag_broadband_all_z.append(csd_mag_bb)
    csd_mag_lp_all_z.append(csd_mag_lp)
    csd_mag_hp_all_z.append(csd_mag_hp)

    # Coherence
    f_coh, coh_bb = compute_coherence_welch(tau_z, u_z, fs, nperseg, noverlap, WINDOW)
    verify_frequency_consistency(f_welch, f_coh, "Coherence broadband")

    _, coh_lp = compute_coherence_welch(tau_z, u_lp_z, fs, nperseg, noverlap, WINDOW)
    _, coh_hp = compute_coherence_welch(tau_z, u_hp_z, fs, nperseg, noverlap, WINDOW)

    coherence_broadband_all_z.append(coh_bb)
    coherence_lp_all_z.append(coh_lp)
    coherence_hp_all_z.append(coh_hp)

    if (iz + 1) % max(1, Nz // 5) == 0 or iz == 0:
        print(f"  z-index {iz}/{Nz-1}: Spectral quantities computed")

# Convert to arrays
psd_tau_all_z = np.array(psd_tau_all_z)
psd_u_all_z = np.array(psd_u_all_z)
psd_u_lp_all_z = np.array(psd_u_lp_all_z)
psd_u_hp_all_z = np.array(psd_u_hp_all_z)
csd_mag_broadband_all_z = np.array(csd_mag_broadband_all_z)
csd_mag_lp_all_z = np.array(csd_mag_lp_all_z)
csd_mag_hp_all_z = np.array(csd_mag_hp_all_z)
coherence_broadband_all_z = np.array(coherence_broadband_all_z)
coherence_lp_all_z = np.array(coherence_lp_all_z)
coherence_hp_all_z = np.array(coherence_hp_all_z)

print(f"✓ Spectral analysis complete")
print(f"  Frequency array from Welch: {len(f_welch)} points")

# ============================================================================
# PHASE 9: ZERO-LAG COVARIANCE / CORRELATION PER BAND
# ============================================================================

print("\n" + "="*70)
print("PHASE 9: ZERO-LAG COVARIANCE / CORRELATION")
print("="*70)

covariance_broadband_all_z = np.zeros(Nz)
covariance_lp_all_z = np.zeros(Nz)
covariance_hp_all_z = np.zeros(Nz)
correlation_broadband_all_z = np.zeros(Nz)
correlation_lp_all_z = np.zeros(Nz)
correlation_hp_all_z = np.zeros(Nz)

for iz in range(Nz):
    tau_z = tau_prime[:, iz]
    u_z = u_prime[:, iz]
    u_lp_z = u_prime_lp[:, iz]
    u_hp_z = u_prime_hp[:, iz]

    # Covariance
    covariance_broadband_all_z[iz] = np.mean(tau_z * u_z)
    covariance_lp_all_z[iz] = np.mean(tau_z * u_lp_z)
    covariance_hp_all_z[iz] = np.mean(tau_z * u_hp_z)

    # Correlation
    correlation_broadband_all_z[iz] = np.corrcoef(tau_z, u_z)[0, 1]
    correlation_lp_all_z[iz] = np.corrcoef(tau_z, u_lp_z)[0, 1]
    correlation_hp_all_z[iz] = np.corrcoef(tau_z, u_hp_z)[0, 1]

# Average over z
covariance_stats = {
    'broadband': np.mean(covariance_broadband_all_z),
    'lp': np.mean(covariance_lp_all_z),
    'hp': np.mean(covariance_hp_all_z)
}

correlation_stats = {
    'broadband': np.mean(correlation_broadband_all_z),
    'lp': np.mean(correlation_lp_all_z),
    'hp': np.mean(correlation_hp_all_z)
}

print(f"\nZero-lag statistics (averaged over z):")
print(f"  Covariance(τ, u) broadband: {covariance_stats['broadband']:.6e}")
print(f"  Covariance(τ, u_lp) low-pass: {covariance_stats['lp']:.6e}")
print(f"  Covariance(τ, u_hp) high-pass: {covariance_stats['hp']:.6e}")
if covariance_stats['hp'] < 0:
    print(f"    ⚠ Note: High-pass covariance is NEGATIVE")

print(f"\n  Correlation(τ, u) broadband: {correlation_stats['broadband']:.6f}")
print(f"  Correlation(τ, u_lp) low-pass: {correlation_stats['lp']:.6f}")
print(f"  Correlation(τ, u_hp) high-pass: {correlation_stats['hp']:.6f}")
if correlation_stats['hp'] < 0:
    print(f"    ⚠ Note: High-pass correlation is NEGATIVE")

# ============================================================================
# PHASE 10: AVERAGE OVER z AND CREATE COHERENCE MASKS
# ============================================================================

print("\n" + "="*70)
print("PHASE 10: AVERAGE OVER z AND CREATE COHERENCE MASKS")
print("="*70)

psd_tau_mean = np.mean(psd_tau_all_z, axis=0)
psd_u_mean = np.mean(psd_u_all_z, axis=0)
psd_u_lp_mean = np.mean(psd_u_lp_all_z, axis=0)
psd_u_hp_mean = np.mean(psd_u_hp_all_z, axis=0)
csd_mag_broadband_mean = np.mean(csd_mag_broadband_all_z, axis=0)
csd_mag_lp_mean = np.mean(csd_mag_lp_all_z, axis=0)
csd_mag_hp_mean = np.mean(csd_mag_hp_all_z, axis=0)
coherence_broadband_mean = np.mean(coherence_broadband_all_z, axis=0)
coherence_lp_mean = np.mean(coherence_lp_all_z, axis=0)
coherence_hp_mean = np.mean(coherence_hp_all_z, axis=0)

# Create masks for coherence plots
# Coherence is only plotted where the filtered signal carries meaningful energy
mask_lp_valid = psd_u_lp_mean > EPS_COHERENCE_MASK * np.max(psd_u_lp_mean)
mask_hp_valid = psd_u_hp_mean > EPS_COHERENCE_MASK * np.max(psd_u_hp_mean)

# Find frequency ranges
f_lp_min = f_welch[np.where(mask_lp_valid)[0][0]] if np.any(mask_lp_valid) else f_welch[0]
f_lp_max = f_welch[np.where(mask_lp_valid)[0][-1]] if np.any(mask_lp_valid) else f_welch[0]
f_hp_min = f_welch[np.where(mask_hp_valid)[0][0]] if np.any(mask_hp_valid) else f_welch[0]
f_hp_max = f_welch[np.where(mask_hp_valid)[0][-1]] if np.any(mask_hp_valid) else f_welch[-1]

f_star_lp_min = f_lp_min * c / U_inf
f_star_lp_max = f_lp_max * c / U_inf
f_star_hp_min = f_hp_min * c / U_inf
f_star_hp_max = f_hp_max * c / U_inf

print(f"✓ Averaging over z completed")

print(f"\nCoherence masking (eps_mask = {EPS_COHERENCE_MASK}):")
print(f"  Low-pass coherence valid range: f* ∈ [{f_star_lp_min:.4e}, {f_star_lp_max:.4e}]")
print(f"  High-pass coherence valid range: f* ∈ [{f_star_hp_min:.4e}, {f_star_hp_max:.4e}]")
print(f"  (Coherence plotted only where filtered signal PSD is above threshold)")

# ============================================================================
# PHASE 11: FINAL PLOTS
# ============================================================================

print("\n" + "="*70)
print("PHASE 11: FINAL PLOTS")
print("="*70)

# Update sample validation plot to use computed Welch frequency array
plot_psd_validation(data, u_prime_lp, u_prime_hp, z_sample_indices,
                   fs, nperseg, noverlap, f_star_welch, OUTPUT_DIR)

plot_final_results(f_star_welch, psd_tau_mean, psd_u_mean, psd_u_lp_mean, psd_u_hp_mean,
                   csd_mag_broadband_mean, csd_mag_lp_mean, csd_mag_hp_mean,
                   coherence_broadband_mean, coherence_lp_mean, coherence_hp_mean,
                   mask_lp_valid, mask_hp_valid,
                   covariance_stats, correlation_stats, OUTPUT_DIR)

# ============================================================================
# PHASE 12: SAVE RESULTS TO HDF5
# ============================================================================

print("\n" + "="*70)
print("PHASE 12: SAVE RESULTS TO HDF5")
print("="*70)

output_file = os.path.join(OUTPUT_DIR, "band_split_spectral_analysis_slice_9.h5")

save_results_hdf5(output_file, data, u_prime_lp, u_prime_hp, f_cutoff,
                  psd_tau_mean, psd_u_mean, psd_u_lp_mean, psd_u_hp_mean,
                  csd_mag_broadband_mean, csd_mag_lp_mean, csd_mag_hp_mean,
                  coherence_broadband_mean, coherence_lp_mean, coherence_hp_mean,
                  mask_lp_valid, mask_hp_valid,
                  covariance_stats, correlation_stats, nperseg, noverlap,
                  f_welch, f_star_welch)

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print(f"\nBand-split analysis complete (improved version):")
print(f"  Input file: {INPUT_FILE}")
print(f"  Output file: {output_file}")

print(f"\nConfiguration:")
print(f"  Filter type: FFT-based sharp cutoff")
print(f"  Cutoff f*: {F_STAR_CUTOFF}")
print(f"  Cutoff f: {f_cutoff:.6e} Hz")
print(f"  Nt: {Nt}")
print(f"  Nz: {Nz}")
print(f"  nperseg: {nperseg}")

print(f"\nBand energy distribution (variance-based, CORRECT):")
print(f"  Total variance: {var_original:.6e}")
print(f"  Low-pass variance fraction: {lp_variance_fraction:.2f}%")
print(f"  High-pass variance fraction: {hp_variance_fraction:.2f}%")
print(f"  Total fraction: {total_variance_fraction:.2f}%")

print(f"\nCross-spectrum results:")
print(f"  Broadband peak: {np.max(csd_mag_broadband_mean):.6e} at f* = {f_star_welch[np.argmax(csd_mag_broadband_mean)]:.6e}")
print(f"  Low-pass peak: {np.max(csd_mag_lp_mean):.6e} at f* = {f_star_welch[np.argmax(csd_mag_lp_mean)]:.6e}")
print(f"  High-pass peak: {np.max(csd_mag_hp_mean):.6e} at f* = {f_star_welch[np.argmax(csd_mag_hp_mean)]:.6e}")

print(f"\nMax coherence:")
print(f"  Broadband: {np.max(coherence_broadband_mean):.6f}")
print(f"  Low-pass (valid range f* ∈ [{f_star_lp_min:.4e}, {f_star_lp_max:.4e}]): {np.max(coherence_lp_mean[mask_lp_valid]):.6f}")
print(f"  High-pass (valid range f* ∈ [{f_star_hp_min:.4e}, {f_star_hp_max:.4e}]): {np.max(coherence_hp_mean[mask_hp_valid]):.6f}")

print(f"\nZero-lag statistics:")
print(f"  Covariance(τ, u) broadband: {covariance_stats['broadband']:.6e}")
print(f"  Covariance(τ, u_lp) low-pass: {covariance_stats['lp']:.6e}")
print(f"  Covariance(τ, u_hp) high-pass: {covariance_stats['hp']:.6e}", end="")
if covariance_stats['hp'] < 0:
    print(" [NEGATIVE]")
else:
    print()

print(f"  Correlation(τ, u) broadband: {correlation_stats['broadband']:.6f}")
print(f"  Correlation(τ, u_lp) low-pass: {correlation_stats['lp']:.6f}")
print(f"  Correlation(τ, u_hp) high-pass: {correlation_stats['hp']:.6f}", end="")
if correlation_stats['hp'] < 0:
    print(" [NEGATIVE]")
else:
    print()

print("="*70)
