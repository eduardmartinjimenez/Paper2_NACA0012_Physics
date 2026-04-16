"""
PSD Analysis - Compute and Plot Power Spectral Density
=======================================================

This script:
1. Loads time series data from HDF5 file (created by signal_extraction.py)
2. Identifies surface probe (tau_prime) and velocity probe (u_prime)
3. Computes Power Spectral Density (PSD) for both signals
4. Computes cross-spectrum and coherence between tau_prime and u_prime
5. Converts frequency to nondimensional frequency f* = f * c / U_inf
6. Plots PSDs, cross-spectrum magnitude, coherence, and phase in log-log/semilog formats
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to HDF5 file created by signal_extraction.py
SAVE_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Signal_correlation/"
HDF5_FILE = os.path.join(SAVE_DIR, "velocity_timeseries_slice_9_test.h5")

# Physical constants for nondimensionalization
U_inf = 1.0  # Free-stream velocity [m/s]
c = 1.0      # Chord length [m]

# PSD computation parameters
# Using Welch's method for more stable PSD estimation
nperseg = 4096  # Segment length
noverlap = nperseg // 2  # Overlap (50% of nperseg)
window = 'hann'  # Window function

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_hdf5_data(filename):
    """
    Load time series data from HDF5 file.

    Returns:
        probes_data: Dictionary with probe data
        dt_iteration: Time step per iteration [s]
    """
    print(f"\nLoading HDF5 file: {filename}")

    with h5py.File(filename, 'r') as f:
        # Get metadata
        dt_iteration = f.attrs['dt_iteration']

        # Load probe data
        probes_data = {}
        probes_group = f['probes']

        for probe_name in probes_group.keys():
            probe_group = probes_group[probe_name]

            # Get label to identify probe type
            label = probe_group.attrs['label']

            # Load common data
            iterations = probe_group['iterations'][...]
            time = probe_group['time'][...]

            probe_data = {
                'label': label,
                'iterations': iterations,
                'time': time,
                'y_target': probe_group.attrs['y_target'],
                'y_actual': probe_group.attrs['y_actual'],
                'y_index': probe_group.attrs['y_index']
            }

            # Load probe-specific data
            if label == 'surface':
                # Surface probe: load tau_prime
                tau_prime = probe_group['tau_prime'][...]
                probe_data['tau_prime'] = tau_prime
            else:
                # Velocity probe: load u_prime
                u_prime = probe_group['u_prime'][...]
                probe_data['u_prime'] = u_prime

            probes_data[probe_name] = probe_data

    print(f"✓ File loaded successfully")
    print(f"  Number of probes: {len(probes_data)}")

    return probes_data, dt_iteration


def compute_sampling_frequency(time_array):
    """
    Compute sampling frequency from time array.

    Args:
        time_array: Time points [s]

    Returns:
        fs: Sampling frequency [Hz]
    """
    # Compute time differences
    dt = np.diff(time_array)

    # Get mean time step
    dt_mean = np.mean(dt)

    # Compute sampling frequency
    fs = 1.0 / dt_mean

    return fs


def compute_psd_welch(signal_data, fs, window='hann', nperseg=None, noverlap=None):
    """
    Compute Power Spectral Density using Welch's method.

    Args:
        signal_data: Input signal (time series)
        fs: Sampling frequency [Hz]
        window: Window function ('hann', 'hamming', etc.)
        nperseg: Segment length for Welch's method
        noverlap: Overlap between segments

    Returns:
        frequencies: Frequency array [Hz]
        psd: Power Spectral Density
    """
    # Remove NaN values if present
    valid_idx = ~np.isnan(signal_data)
    signal_clean = signal_data[valid_idx]

    # Remove mean (zero-center the signal)
    signal_centered = signal_clean - np.mean(signal_clean)

    # Use Welch's method for PSD computation
    frequencies, psd = signal.welch(
        signal_centered,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'  # Power spectral density (V**2/Hz)
    )

    return frequencies, psd


def compute_cross_spectrum_welch(signal1, signal2, fs, window='hann', nperseg=None, noverlap=None):
    """
    Compute Cross-Spectrum between two signals using Welch's method.

    Args:
        signal1: First signal (time series)
        signal2: Second signal (time series)
        fs: Sampling frequency [Hz]
        window: Window function ('hann', 'hamming', etc.)
        nperseg: Segment length for Welch's method
        noverlap: Overlap between segments

    Returns:
        frequencies: Frequency array [Hz]
        cross_spectrum: Complex cross-spectrum
    """
    # Remove NaN values if present
    valid_idx = ~(np.isnan(signal1) | np.isnan(signal2))
    signal1_clean = signal1[valid_idx]
    signal2_clean = signal2[valid_idx]

    # Remove mean (zero-center both signals)
    signal1_centered = signal1_clean - np.mean(signal1_clean)
    signal2_centered = signal2_clean - np.mean(signal2_clean)

    # Compute cross-spectrum using Welch's method
    frequencies, cross_spectrum = signal.csd(
        signal1_centered,
        signal2_centered,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'
    )

    return frequencies, cross_spectrum


def compute_coherence_welch(signal1, signal2, fs, window='hann', nperseg=None, noverlap=None):
    """
    Compute Magnitude-Squared Coherence between two signals using Welch's method.

    Args:
        signal1: First signal (time series)
        signal2: Second signal (time series)
        fs: Sampling frequency [Hz]
        window: Window function ('hann', 'hamming', etc.)
        nperseg: Segment length for Welch's method
        noverlap: Overlap between segments

    Returns:
        frequencies: Frequency array [Hz]
        coherence: Magnitude-squared coherence (0 to 1)
    """
    # Remove NaN values if present
    valid_idx = ~(np.isnan(signal1) | np.isnan(signal2))
    signal1_clean = signal1[valid_idx]
    signal2_clean = signal2[valid_idx]

    # Remove mean (zero-center both signals)
    signal1_centered = signal1_clean - np.mean(signal1_clean)
    signal2_centered = signal2_clean - np.mean(signal2_clean)

    # Compute coherence using Welch's method
    frequencies, coherence = signal.coherence(
        signal1_centered,
        signal2_centered,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap
    )

    return frequencies, coherence


def nondimensionalize_frequency(frequency_array, U_inf, c):
    """
    Convert dimensional frequency to nondimensional frequency.

    f* = f * c / U_inf

    Args:
        frequency_array: Dimensional frequency [Hz]
        U_inf: Free-stream velocity [m/s]
        c: Characteristic length scale (chord) [m]

    Returns:
        f_star: Nondimensional frequency [-]
    """
    f_star = frequency_array * c / U_inf
    return f_star


# ============================================================================
# MAIN EXECUTION
# ============================================================================

print("="*70)
print("PSD ANALYSIS: TAU_PRIME AND U_PRIME")
print("="*70)

# Verify file exists
if not os.path.exists(HDF5_FILE):
    raise FileNotFoundError(f"HDF5 file not found: {HDF5_FILE}")

# Load data
probes_data, dt_iteration = load_hdf5_data(HDF5_FILE)

# ============================================================================
# IDENTIFY PROBES
# ============================================================================

print("\n" + "="*70)
print("IDENTIFYING PROBES")
print("="*70)

surface_probe = None
velocity_probe = None

for probe_name, probe_data in probes_data.items():
    label = probe_data['label']
    print(f"\n{probe_name}:")
    print(f"  Label: {label}")
    print(f"  Y position: {probe_data['y_actual']:.6e}")
    print(f"  Number of samples: {len(probe_data['time'])}")
    print(f"  Time span: {probe_data['time'][0]:.6f} to {probe_data['time'][-1]:.6f} s")

    if label == 'surface':
        surface_probe = probe_name
        print(f"  ✓ Identified as SURFACE PROBE")
    else:
        velocity_probe = probe_name
        print(f"  ✓ Identified as VELOCITY PROBE")

if surface_probe is None or velocity_probe is None:
    raise ValueError("Could not identify both surface and velocity probes in HDF5 file")

print(f"\nIdentified probes:")
print(f"  Surface probe: {surface_probe}")
print(f"  Velocity probe: {velocity_probe}")

# ============================================================================
# EXTRACT SIGNALS AND TIME
# ============================================================================

print("\n" + "="*70)
print("EXTRACTING SIGNALS")
print("="*70)

# Get signals
tau_prime = probes_data[surface_probe]['tau_prime']
u_prime = probes_data[velocity_probe]['u_prime']
time = probes_data[velocity_probe]['time']

print(f"\nTau_prime (surface probe):")
print(f"  Shape: {tau_prime.shape}")
print(f"  Min: {np.nanmin(tau_prime):.6e}")
print(f"  Max: {np.nanmax(tau_prime):.6e}")
print(f"  Mean: {np.nanmean(tau_prime):.6e}")
print(f"  Std: {np.nanstd(tau_prime):.6e}")

print(f"\nU_prime (velocity probe):")
print(f"  Shape: {u_prime.shape}")
print(f"  Min: {np.min(u_prime):.6e}")
print(f"  Max: {np.max(u_prime):.6e}")
print(f"  Mean: {np.mean(u_prime):.6e}")
print(f"  Std: {np.std(u_prime):.6e}")

# ============================================================================
# COMPUTE SAMPLING FREQUENCY
# ============================================================================

print("\n" + "="*70)
print("COMPUTING SAMPLING FREQUENCY")
print("="*70)

fs = compute_sampling_frequency(time)

print(f"\nSampling frequency: {fs:.6f} Hz")
print(f"Mean time step: {1/fs:.6e} s")
print(f"Total time span: {time[-1] - time[0]:.6f} s")
print(f"Number of samples: {len(time)}")

# Estimate frequency resolution based on Welch parameters
freq_resolution = fs / nperseg
print(f"Welch segment length (nperseg): {nperseg}")
print(f"Frequency resolution: {freq_resolution:.6e} Hz")

# ============================================================================
# COMPUTE PSDs
# ============================================================================

print("\n" + "="*70)
print("COMPUTING POWER SPECTRAL DENSITIES (WELCH'S METHOD)")
print("="*70)

# Compute PSD for tau_prime
print(f"\nComputing PSD for tau_prime...")
f_tau, psd_tau = compute_psd_welch(tau_prime, fs, window=window, nperseg=nperseg, noverlap=noverlap)
print(f"  Frequency range: {f_tau[1]:.6e} to {f_tau[-1]:.6f} Hz")
print(f"  Number of frequency points: {len(f_tau)}")
print(f"  PSD range: {np.min(psd_tau):.6e} to {np.max(psd_tau):.6e}")

# Compute PSD for u_prime
print(f"\nComputing PSD for u_prime...")
f_u, psd_u = compute_psd_welch(u_prime, fs, window=window, nperseg=nperseg, noverlap=noverlap)
print(f"  Frequency range: {f_u[1]:.6e} to {f_u[-1]:.6f} Hz")
print(f"  Number of frequency points: {len(f_u)}")
print(f"  PSD range: {np.min(psd_u):.6e} to {np.max(psd_u):.6e}")

# ============================================================================
# COMPUTE CROSS-SPECTRUM AND COHERENCE
# ============================================================================

print("\n" + "="*70)
print("COMPUTING CROSS-SPECTRUM AND COHERENCE (WELCH'S METHOD)")
print("="*70)

# Compute cross-spectrum between tau_prime and u_prime
print(f"\nComputing cross-spectrum...")
f_csd, cross_spectrum = compute_cross_spectrum_welch(
    tau_prime, u_prime, fs, window=window, nperseg=nperseg, noverlap=noverlap
)
print(f"  Frequency range: {f_csd[1]:.6e} to {f_csd[-1]:.6f} Hz")
print(f"  Number of frequency points: {len(f_csd)}")
print(f"  Cross-spectrum magnitude range: {np.min(np.abs(cross_spectrum)):.6e} to {np.max(np.abs(cross_spectrum)):.6e}")

# Compute coherence between tau_prime and u_prime
print(f"\nComputing coherence...")
f_coh, coherence = compute_coherence_welch(
    tau_prime, u_prime, fs, window=window, nperseg=nperseg, noverlap=noverlap
)
print(f"  Frequency range: {f_coh[1]:.6e} to {f_coh[-1]:.6f} Hz")
print(f"  Number of frequency points: {len(f_coh)}")
print(f"  Coherence range: {np.min(coherence):.6e} to {np.max(coherence):.6e}")

# Extract magnitude and phase from cross-spectrum
magnitude_cross_spectrum = np.abs(cross_spectrum)
phase_cross_spectrum = np.angle(cross_spectrum)  # Phase in radians

print(f"\nCross-spectrum and coherence computation complete")
print(f"  Mean coherence: {np.mean(coherence):.6f}")
print(f"  Max coherence: {np.max(coherence):.6f}")
print(f"  Min coherence: {np.min(coherence):.6f}")

# ============================================================================
# CONVERT TO NONDIMENSIONAL FREQUENCY
# ============================================================================

print("\n" + "="*70)
print("CONVERTING TO NONDIMENSIONAL FREQUENCY")
print("="*70)
print(f"\nUsing nondimensionalization:")
print(f"  U_inf = {U_inf} m/s")
print(f"  c = {c} m")
print(f"  f* = f * c / U_inf")

# Convert frequencies
f_tau_star = nondimensionalize_frequency(f_tau, U_inf, c)
f_u_star = nondimensionalize_frequency(f_u, U_inf, c)
f_csd_star = nondimensionalize_frequency(f_csd, U_inf, c)
f_coh_star = nondimensionalize_frequency(f_coh, U_inf, c)

print(f"\nNondimensional frequency ranges:")
print(f"  tau_prime: {f_tau_star[1]:.6e} to {f_tau_star[-1]:.6e}")
print(f"  u_prime: {f_u_star[1]:.6e} to {f_u_star[-1]:.6e}")
print(f"  cross_spectrum: {f_csd_star[1]:.6e} to {f_csd_star[-1]:.6e}")
print(f"  coherence: {f_coh_star[1]:.6e} to {f_coh_star[-1]:.6e}")

# ============================================================================
# PLOTTING
# ============================================================================

print("\n" + "="*70)
print("PLOTTING PSDs")
print("="*70)

# Create figure with log-log plot
fig, ax = plt.subplots(figsize=(12, 8))

# Plot PSD for tau_prime
ax.loglog(f_tau_star[1:], psd_tau[1:], 'o-', linewidth=1.5, markersize=3,
         label="τ' (Surface Probe)", color='#d62728', alpha=0.8)

# Plot PSD for u_prime
ax.loglog(f_u_star[1:], psd_u[1:], 's-', linewidth=1.5, markersize=3,
         label="u' (Velocity Probe)", color='#1f77b4', alpha=0.8)

# Labels and formatting
ax.set_xlabel("Nondimensional Frequency f* = f·c/U∞ [-]", fontsize=12, fontweight='bold')
ax.set_ylabel("PSD [signal²/Hz]", fontsize=12, fontweight='bold')
ax.set_title("Power Spectral Density: τ' and u'", fontsize=13, fontweight='bold')

# Grid
ax.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)

# Legend
ax.legend(fontsize=11, loc='upper right', framealpha=0.9)

# Add information box
info_text = (
    f"Sampling frequency: {fs:.2f} Hz\n"
    f"Total time: {time[-1] - time[0]:.2f} s\n"
    f"Samples: {len(time)}\n"
    f"Window: {window}"
)
ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
       fontsize=10, verticalalignment='bottom', horizontalalignment='left',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
       family='monospace')

plt.tight_layout()

# Display the plot
print(f"\nDisplaying PSD plot...")
plt.show()

# ============================================================================
# PLOTTING CROSS-SPECTRUM, COHERENCE, AND PHASE
# ============================================================================

print("\n" + "="*70)
print("PLOTTING CROSS-SPECTRUM, COHERENCE, AND PHASE")
print("="*70)

# Create figure with 3 vertically stacked subplots
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Skip zero-frequency point for better visualization
freq_start_idx = 1

# Subplot 1: Magnitude of cross-spectrum
ax1 = axes[0]
ax1.loglog(f_csd_star[freq_start_idx:], magnitude_cross_spectrum[freq_start_idx:],
           'o-', linewidth=1.5, markersize=3, color='#2ca02c', alpha=0.8)
ax1.set_ylabel("|S_τu(f)| [signal product / Hz]", fontsize=11, fontweight='bold')
ax1.set_title("Cross-Spectrum Magnitude: τ' and u'", fontsize=12, fontweight='bold')
ax1.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
ax1.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)

# Subplot 2: Magnitude-squared coherence
ax2 = axes[1]
ax2.semilogx(f_coh_star[freq_start_idx:], coherence[freq_start_idx:],
             'o-', linewidth=1.5, markersize=3, color='#ff7f0e', alpha=0.8)
ax2.set_ylim([0, 1.05])
ax2.set_ylabel("γ²(f) [-]", fontsize=11, fontweight='bold')
ax2.set_title("Coherence: τ' and u'", fontsize=12, fontweight='bold')
ax2.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
ax2.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)
ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

# Subplot 3: Phase of cross-spectrum
ax3 = axes[2]
ax3.semilogx(f_csd_star[freq_start_idx:], phase_cross_spectrum[freq_start_idx:],
             'o-', linewidth=1.5, markersize=3, color='#d62728', alpha=0.8)
ax3.set_xlabel("Nondimensional Frequency f* = f·c/U∞ [-]", fontsize=11, fontweight='bold')
ax3.set_ylabel("Phase(S_τu) [rad]", fontsize=11, fontweight='bold')
ax3.set_title("Cross-Spectrum Phase: τ' and u'", fontsize=12, fontweight='bold')
ax3.grid(True, which='both', alpha=0.3, linestyle='-', linewidth=0.5)
ax3.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.3)
ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

# Add common title
fig.suptitle("Cross-Spectrum and Coherence Analysis: τ' and u'",
             fontsize=13, fontweight='bold', y=1.00)

plt.tight_layout()

# Display the plot
print(f"\nDisplaying cross-spectrum/coherence plot...")
plt.show()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\nSummary:")
print(f"  Surface probe (tau_prime):")
print(f"    - Probed at y = {probes_data[surface_probe]['y_actual']:.6e}")
print(f"    - {len(tau_prime)} samples")
print(f"    - PSD peak: {np.max(psd_tau):.6e} at f* = {f_tau_star[np.argmax(psd_tau)]:.6e}")
print(f"\n  Velocity probe (u_prime):")
print(f"    - Probed at y = {probes_data[velocity_probe]['y_actual']:.6e}")
print(f"    - {len(u_prime)} samples")
print(f"    - PSD peak: {np.max(psd_u):.6e} at f* = {f_u_star[np.argmax(psd_u)]:.6e}")
print(f"\n  Cross-spectrum and coherence:")
print(f"    - Number of frequency points: {len(f_csd)}")
print(f"    - Maximum coherence: {np.max(coherence):.6f}")
print(f"    - Mean coherence: {np.mean(coherence):.6f}")
print(f"    - Peak cross-spectrum magnitude: {np.max(magnitude_cross_spectrum):.6e} at f* = {f_csd_star[np.argmax(magnitude_cross_spectrum)]:.6e}")
print(f"\nSpectral analysis parameters:")
print(f"  Welch segment length (nperseg): {nperseg}")
print(f"  Welch overlap (noverlap): {noverlap}")
print(f"  Frequency resolution: {freq_resolution:.6e} Hz")
print(f"  Window function: {window}")
print(f"\nPhysical parameters:")
print(f"  Sampling frequency: {fs:.6f} Hz")
print(f"  Nondimensionalization: f* = f × {c} / {U_inf} = f × {c/U_inf}")
print("="*70)
