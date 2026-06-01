import os
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
import logging

# ---------------------------------------------------------------------------
# Matplotlib font configuration
# ---------------------------------------------------------------------------
# Use only bundled Matplotlib fonts to avoid findfont warnings on remote/HPC
# environments where Computer Modern fonts may not be installed.
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.serif": ["DejaVu Serif"],
    "font.monospace": ["DejaVu Sans Mono"],

    # Use Matplotlib mathtext, not external LaTeX.
    "text.usetex": False,
    "mathtext.fontset": "dejavusans",
    "mathtext.rm": "DejaVu Sans",
    "mathtext.it": "DejaVu Sans:italic",
    "mathtext.bf": "DejaVu Sans:bold",

    # Avoid missing glyph issues for minus signs.
    "axes.unicode_minus": False,

    # Do not accidentally request unavailable Computer Modern fonts.
    "font.family": "sans-serif",
})

# Suppress residual font-manager warnings if the font cache still contains
# stale font entries.
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Configuration

aoa = 5  # Angle of attack in degrees

CACHE_FILE = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/Coherence/timeseries_both_xc_0.900.h5"

OUT_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/Freq_correlations/"

TARGET_N_OVER_C = [
    0.0005, 0.001, 0.002, 0.003, 0.005,
    0.0075, 0.01, 0.015, 0.02, 0.03,
    0.05, 0.075, 0.10
    ]  # Target wall-normal distances

# aoa = 12  # Angle of attack in degrees

# CACHE_FILE = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Coherence/timeseries_both_xc_0.900.h5"

# OUT_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Freq_correlations/"

# TARGET_N_OVER_C = [
#     0.0005, 0.001, 0.002, 0.003, 0.005,
#     0.0075, 0.01, 0.015, 0.02, 0.03,
#     0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20
#     ]  # Target wall-normal distances

# Create output directory if it doesn't exist
os.makedirs(OUT_PATH, exist_ok=True)

# Load metadata and time arrays
with h5py.File(CACHE_FILE, 'r') as f:
    meta = f['_metadata']
    metadata = {}
    for key in meta.attrs:
        metadata[key] = meta.attrs[key]
    for key in meta.keys():
        metadata[key] = meta[key][...]

    time = f['time'][...]

    # Load raw wall signals from cache
    tau_w_z = f['tau_w_z'][...].astype(np.float64)
    pressure_z = f['pressure_z'][...].astype(np.float64)

    # Recompute fluctuations by subtracting temporal mean at each z-plane
    tau_w_mean_z = np.mean(tau_w_z, axis=0, keepdims=True, dtype=np.float64)
    pressure_mean_z = np.mean(pressure_z, axis=0, keepdims=True, dtype=np.float64)

    tau_w_prime = tau_w_z - tau_w_mean_z
    pressure_prime = pressure_z - pressure_mean_z

    num_probes = metadata['num_probes']
    nz = metadata['nz']

    # Get probe locations first
    probe_n = metadata.get('probe_n_actual', None)  # Wall distance

    # Select probe indices based on target wall-normal distances
    selected_probe_indices = []
    if probe_n is not None:
        for n_target in TARGET_N_OVER_C:
            idx = np.argmin(np.abs(probe_n - n_target))
            selected_probe_indices.append(idx)
        selected_probe_indices = sorted(set(selected_probe_indices))
    else:
        selected_probe_indices = list(range(min(3, num_probes)))

    # Load velocity data for selected probes
    velocity_raw = {}
    velocity_fluct = {}

    for i in selected_probe_indices:
        key_raw = f'u_s_z_{i}'

        if key_raw in f:
            u_s_z = f[key_raw][...].astype(np.float64)

            u_mean_z = np.mean(u_s_z, axis=0, keepdims=True, dtype=np.float64)

            velocity_raw[i] = u_s_z
            velocity_fluct[i] = u_s_z - u_mean_z
        else:
            print(f"WARNING: raw velocity dataset {key_raw} not found in cache")

    # ============================================================================
    # EXTRACT CONSISTENT METADATA FOR FILENAMES AND TITLES
    # ============================================================================

    aoa = metadata.get('AOA_deg', metadata.get('aoa', aoa))
    xc = metadata.get('xc_actual', 0.5)

    aoa_str = f"AOA{int(round(float(aoa)))}"
    xc_str = f"xc_{float(xc):.3f}"

    print("Data loaded successfully")
    print(f"Time steps: {len(time)}")
    print(f"Spanwise planes (nz): {nz}")
    print(f"Number of probes: {num_probes}")
    print(f"Loaded velocity probes: {list(velocity_fluct.keys())}")

# ============================================================================
# ZERO-MEAN VALIDATION
# ============================================================================

def check_zero_temporal_mean(name, q):
    """
    Check residual temporal mean at each spanwise plane.

    q is expected to have shape (Nt, Nz).
    """
    mean_z = np.mean(q, axis=0)
    dc_rms = np.sqrt(np.mean(mean_z**2))
    full_rms = np.sqrt(np.mean(q**2))
    ratio = dc_rms / (full_rms + 1e-30)

    print(f"{name:<20} DC/full RMS = {ratio:.6e}")


print("\n" + "="*60)
print("ZERO-MEAN VALIDATION")
print("="*60)

check_zero_temporal_mean("tau_w_prime", tau_w_prime)
check_zero_temporal_mean("pressure_prime", pressure_prime)

for probe_id in sorted(velocity_fluct.keys()):
    check_zero_temporal_mean(f"u_s_prime_{probe_id}", velocity_fluct[probe_id])

# Get probe locations
probe_y = metadata.get('probe_y_actual', None)  # Wall-normal coordinate y/c
xc_pos = metadata.get('xc_actual', None)  # Chord position (x-coordinate)

if probe_n is not None and probe_y is not None:
    print(f"\nProbe Locations for visualized probes:")
    print(f"Chord position (x): {xc_pos:.3f}")
    print(f"{'Probe':<8} {'Wall distance n/c':<20} {'y/c':<20}")
    print("-" * 50)
    for probe_id in sorted(velocity_fluct.keys()):
        print(f"{probe_id:<8} {probe_n[probe_id]:<20.6f} {probe_y[probe_id]:<20.6f}")
else:
    print("\nNote: probe coordinates not found in metadata")

# Create comprehensive visualization
num_velocity_probes = len(velocity_fluct)
total_subplots = 2 + num_velocity_probes  # tau_w + pressure + velocity probes
fig = plt.figure(figsize=(14, 2 + 2.5 * num_velocity_probes))
gs = GridSpec(total_subplots, 1, figure=fig, hspace=0.4)

# Select first z-plane (z=0)
z_idx = 0

# Plot 1: Wall shear stress fluctuations
ax1 = fig.add_subplot(gs[0])
ax1.plot(time, tau_w_prime[:, z_idx], 'b-', linewidth=0.8)
ax1.set_ylabel('τ\'_w', fontsize=11, fontweight='bold')
ax1.set_title('Wall Shear Stress Fluctuations vs Time (First Z-Plane)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot 2: Pressure fluctuations
ax2 = fig.add_subplot(gs[1])
ax2.plot(time, pressure_prime[:, z_idx], 'r-', linewidth=0.8)
ax2.set_ylabel('p\'_w', fontsize=11, fontweight='bold')
ax2.set_title('Pressure Fluctuations vs Time (First Z-Plane)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plots for velocity fluctuations at selected probes
colors = ['g', 'orange', 'purple', 'brown', 'pink', 'gray']
for plot_idx, probe_id in enumerate(sorted(velocity_fluct.keys())):
    ax = fig.add_subplot(gs[2 + plot_idx])
    color = colors[plot_idx % len(colors)]
    ax.plot(time, velocity_fluct[probe_id][:, z_idx], color=color, linewidth=0.8)
    ax.set_ylabel(f'u\'_probe{probe_id}', fontsize=11, fontweight='bold')

    # Get probe position
    probe_y_pos = probe_y[probe_id] if probe_y is not None else None
    probe_n_pos = probe_n[probe_id] if probe_n is not None else None

    if probe_y_pos is not None and probe_n_pos is not None:
        title = f'Velocity Fluctuation at Probe {probe_id} (Z-Plane 0) | y={probe_y_pos:.6f}, n={probe_n_pos:.6f}'
    else:
        title = f'Velocity Fluctuation at Probe {probe_id} (Z-Plane 0)'

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if plot_idx == num_velocity_probes - 1:
        ax.set_xlabel(r"$tU_\infty/c$", fontsize=11)

plt.suptitle(
    f"Time Series Visualization - Z-Plane {z_idx} | "
    f"NACA 0012 {aoa_str} Re50000 | "
    f"xc={float(xc):.3f}, y_wall={metadata['y_wall_actual']:.6f}",
    fontsize=12,
    fontweight="bold"
)


fig_filename = f"timeseries_visualization_{aoa_str}_{xc_str}.png"

plt.savefig(
    os.path.join(OUT_PATH, fig_filename),
    dpi=150,
    bbox_inches="tight"
)

print(f"Plot saved to: {fig_filename}")
plt.show()

# Statistical summary
print("\n" + "="*60)
print("STATISTICAL SUMMARY (First Z-Plane)")
print("="*60)
print(f"\nWall Shear Stress Fluctuations:")
print(f"  Mean: {np.mean(tau_w_prime[:, z_idx]):.6f}")
print(f"  Std:  {np.std(tau_w_prime[:, z_idx]):.6f}")
print(f"  RMS:  {np.sqrt(np.mean(tau_w_prime[:, z_idx]**2)):.6f}")
print(f"  Min:  {np.min(tau_w_prime[:, z_idx]):.6f}")
print(f"  Max:  {np.max(tau_w_prime[:, z_idx]):.6f}")

print(f"\nPressure Fluctuations:")
print(f"  Mean: {np.mean(pressure_prime[:, z_idx]):.6f}")
print(f"  Std:  {np.std(pressure_prime[:, z_idx]):.6f}")
print(f"  RMS:  {np.sqrt(np.mean(pressure_prime[:, z_idx]**2)):.6f}")
print(f"  Min:  {np.min(pressure_prime[:, z_idx]):.6f}")
print(f"  Max:  {np.max(pressure_prime[:, z_idx]):.6f}")

for probe_idx in velocity_fluct.keys():
    print(f"\nVelocity Fluctuations at Probe {probe_idx}:")
    data = velocity_fluct[probe_idx][:, z_idx]
    print(f"  Mean: {np.mean(data):.6f}")
    print(f"  Std:  {np.std(data):.6f}")
    print(f"  RMS:  {np.sqrt(np.mean(data**2)):.6f}")
    print(f"  Min:  {np.min(data):.6f}")
    print(f"  Max:  {np.max(data):.6f}")


# ============================================================================
# PSD PARAMETERS AND SAMPLING INFO
# ============================================================================

from scipy import signal

dt = np.mean(np.diff(time))
fs = 1.0 / dt
f_nyquist = 0.5 * fs
T = time[-1] - time[0]
df = 1.0 / T

nperseg = min(4096, len(time) // 4)
noverlap = nperseg // 2
window = "hann"
detrend_type = "linear"

print("\n" + "="*60)
print("SAMPLING INFORMATION")
print("="*60)
print(f"dt              = {dt:.8e} s")
print(f"fs              = {fs:.6f} Hz")
print(f"Nyquist freq    = {f_nyquist:.6f} Hz")
print(f"Total duration  = {T:.6f} s")
print(f"Frequency res.  = {df:.6f} Hz")
print(f"Welch nperseg   = {nperseg}")
print(f"Welch noverlap  = {noverlap}")
print(f"Relative dt variation = {np.std(np.diff(time)) / np.mean(np.diff(time)):.3e}")

# ============================================================================
# FREQUENCY BANDS
# ============================================================================

FREQUENCY_BANDS = {
    "low": (None, 2.0),          # 0 < f <= 2
    "mid": (2.0, 12.5),          # 2 < f <= 12.5
    "high": (12.5, None),        # 12.5 < f <= Nyquist
}

print("\n" + "="*60)
print("FREQUENCY BANDS")
print("="*60)
for band_name, (f_low, f_high) in FREQUENCY_BANDS.items():
    print(f"{band_name:<10}: f_low={f_low}, f_high={f_high}")


def fft_band_filter(q_tz, time, f_low=None, f_high=None, axis=0):
    """
    Ideal FFT-domain band filter.

    Parameters
    ----------
    q_tz : ndarray
        Signal array. For wall signals, expected shape is (Nt, Nz).
    time : ndarray
        Time array of shape (Nt,).
    f_low : float or None
        Lower cutoff frequency. If None, keep all positive frequencies below f_high.
    f_high : float or None
        Upper cutoff frequency. If None, keep all frequencies above f_low up to Nyquist.
    axis : int
        Time axis.

    Returns
    -------
    q_filtered : ndarray
        Band-filtered signal with same shape as q_tz.
    freq : ndarray
        Frequencies associated with the rFFT.
    mask : ndarray
        Boolean frequency mask used for filtering.
    """
    q_tz = np.asarray(q_tz, dtype=float)

    Nt = q_tz.shape[axis]
    dt = np.mean(np.diff(time))
    freq = np.fft.rfftfreq(Nt, d=dt)

    Q = np.fft.rfft(q_tz, axis=axis)

    mask = np.ones_like(freq, dtype=bool)

    # Always remove zero-frequency component
    mask &= freq > 0.0

    if f_low is not None:
        mask &= freq > f_low

    if f_high is not None:
        mask &= freq <= f_high

    Q_filtered = np.zeros_like(Q)

    # Build slicer that selects retained frequencies along the FFT axis
    slicer = [slice(None)] * Q.ndim
    slicer[axis] = mask

    Q_filtered[tuple(slicer)] = Q[tuple(slicer)]

    q_filtered = np.fft.irfft(Q_filtered, n=Nt, axis=axis)

    return q_filtered, freq, mask

# ============================================================================
# FILTER WALL SIGNALS INTO FREQUENCY BANDS
# ============================================================================

filtered_wall_signals = {
    "tau_w": {},
    "pressure": {},
}

filter_masks = {}

for band_name, (f_low, f_high) in FREQUENCY_BANDS.items():
    tau_band, freq_fft, mask = fft_band_filter(
        tau_w_prime,
        time,
        f_low=f_low,
        f_high=f_high,
        axis=0
    )

    p_band, _, _ = fft_band_filter(
        pressure_prime,
        time,
        f_low=f_low,
        f_high=f_high,
        axis=0
    )

    filtered_wall_signals["tau_w"][band_name] = tau_band
    filtered_wall_signals["pressure"][band_name] = p_band
    filter_masks[band_name] = mask

    print(f"\nBand: {band_name}")
    print(f"  Number of retained frequency bins: {np.sum(mask)}")
    if np.sum(mask) > 0:
        print(f"  Frequency range retained: {freq_fft[mask][0]:.6f} to {freq_fft[mask][-1]:.6f}")
    else:
        print("  WARNING: no frequency bins retained")


# ============================================================================
# FILTER STREAMWISE VELOCITY PROBES INTO FREQUENCY BANDS
# ============================================================================

filtered_velocity = {}

for probe_id in sorted(velocity_fluct.keys()):

    filtered_velocity[probe_id] = {}

    u_probe = velocity_fluct[probe_id]

    for band_name, (f_low, f_high) in FREQUENCY_BANDS.items():

        u_band, _, _ = fft_band_filter(
            u_probe,
            time,
            f_low=f_low,
            f_high=f_high,
            axis=0
        )

        filtered_velocity[probe_id][band_name] = u_band

print("\nFiltered streamwise velocity probes into frequency bands.")

# ============================================================================
# RECONSTRUCTION CHECK FOR FILTERED VELOCITY PROBES
# ============================================================================

print("\n" + "="*60)
print("RECONSTRUCTION CHECK FOR FILTERED VELOCITY PROBES")
print("="*60)

for probe_id in sorted(velocity_fluct.keys()):

    u_full = velocity_fluct[probe_id]

    u_rec = (
        filtered_velocity[probe_id]["low"]
        + filtered_velocity[probe_id]["mid"]
        + filtered_velocity[probe_id]["high"]
    )

    u_full_rms = np.sqrt(np.mean(u_full**2))
    rec_error = np.sqrt(np.mean((u_full - u_rec)**2)) / (u_full_rms + 1e-30)

    print(
        f"probe={probe_id:<5} "
        f"n/c={probe_n[probe_id]:.6e} "
        f"velocity reconstruction error={rec_error:.6e}"
    )

# ============================================================================
# RMS CONTRIBUTION OF EACH BAND
# ============================================================================

print("\n" + "="*60)
print("RMS CONTRIBUTION OF FILTERED WALL SIGNALS")
print("="*60)

for signal_name, q_full in {
    "tau_w": tau_w_prime,
    "pressure": pressure_prime,
}.items():

    q_full_rms = np.sqrt(np.mean(q_full**2))

    print(f"\nSignal: {signal_name}")
    print(f"  Full RMS: {q_full_rms:.6e}")

    for band_name in FREQUENCY_BANDS.keys():
        q_band = filtered_wall_signals[signal_name][band_name]
        q_band_rms = np.sqrt(np.mean(q_band**2))
        ratio = q_band_rms / (q_full_rms + 1e-30)

        print(f"  {band_name:<10} RMS: {q_band_rms:.6e}   ratio: {ratio:.4f}")

# ============================================================================
# RECONSTRUCTION CHECK
# ============================================================================

print("\n" + "="*60)
print("RECONSTRUCTION CHECK")
print("="*60)

for signal_name, q_full in {
    "tau_w": tau_w_prime,
    "pressure": pressure_prime,
}.items():

    q_rec = (
        filtered_wall_signals[signal_name]["low"]
        + filtered_wall_signals[signal_name]["mid"]
        + filtered_wall_signals[signal_name]["high"]
    )

    q_full_rms = np.sqrt(np.mean(q_full**2))
    rec_error = np.sqrt(np.mean((q_full - q_rec)**2)) / (q_full_rms + 1e-30)

    print(f"{signal_name:<10} reconstruction relative RMS error: {rec_error:.6e}")

# ============================================================================
# VISUALIZE FILTERED WALL SIGNALS AT ONE Z-PLANE
# ============================================================================

z_idx_plot = 0

for signal_name, q_full in {
    "tau_w": tau_w_prime,
    "pressure": pressure_prime,
}.items():

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(time, q_full[:, z_idx_plot], color="k", linewidth=0.8)
    axes[0].set_ylabel("full")
    axes[0].set_title(f"{signal_name}: full and band-filtered signals, z-index={z_idx_plot}")

    for ax, band_name in zip(axes[1:], FREQUENCY_BANDS.keys()):
        q_band = filtered_wall_signals[signal_name][band_name]
        ax.plot(time, q_band[:, z_idx_plot], linewidth=0.8)
        ax.set_ylabel(band_name)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (t U∞/c)")
    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = os.path.join(OUT_PATH, f"filtered_{signal_name}_signals_{aoa_str}_{xc_str}_z{z_idx_plot}.png")
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    print(f"Saved filtered signal plot: {output_file}")
    plt.show()

# ============================================================================
# COMPUTE FILTERED WALL-SIGNAL / VELOCITY CORRELATIONS
# ============================================================================

correlations_wall_only = {
    "tau_w": {},
    "pressure": {},
}

wall_signal_labels = {
    "tau_w": r"$\tau_w'$",
    "pressure": r"$p_w'$",
}

for signal_name in ["tau_w", "pressure"]:

    print("\n" + "="*60)
    print(f"FILTERED CORRELATIONS FOR {signal_name}")
    print("="*60)

    for band_name in FREQUENCY_BANDS.keys():

        q_band = filtered_wall_signals[signal_name][band_name]

        q_rms = np.sqrt(np.mean(q_band**2))

        probe_ids = []
        n_values = []
        R_values = []

        for probe_id in sorted(velocity_fluct.keys()):

            u_probe = velocity_fluct[probe_id]

            if u_probe.shape != q_band.shape:
                raise ValueError(
                    f"Shape mismatch for probe {probe_id}: "
                    f"q_band shape = {q_band.shape}, "
                    f"u_probe shape = {u_probe.shape}"
                )

            u_rms = np.sqrt(np.mean(u_probe**2))
            numerator = np.mean(q_band * u_probe)

            R = numerator / (q_rms * u_rms + 1e-30)

            probe_ids.append(probe_id)
            n_values.append(probe_n[probe_id])
            R_values.append(R)

            print(
                f"band={band_name:<10} "
                f"probe={probe_id:<5} "
                f"n/c={probe_n[probe_id]:.6e} "
                f"R={R:.6e}"
            )

        correlations_wall_only[signal_name][band_name] = {
            "probe_ids": np.array(probe_ids),
            "n_over_c": np.array(n_values),
            "R": np.array(R_values),
        }

# ============================================================================
# COMPUTE BAND-LIMITED WALL-SIGNAL / BAND-LIMITED VELOCITY CORRELATIONS
# ============================================================================

correlations_both_filtered = {
    "tau_w": {},
    "pressure": {},
}

for signal_name in ["tau_w", "pressure"]:

    print("\n" + "="*60)
    print(f"BAND-LIMITED CORRELATIONS FOR {signal_name}")
    print("="*60)

    for band_name in FREQUENCY_BANDS.keys():

        q_band = filtered_wall_signals[signal_name][band_name]
        q_rms = np.sqrt(np.mean(q_band**2))

        probe_ids = []
        n_values = []
        R_values = []

        for probe_id in sorted(velocity_fluct.keys()):

            u_band = filtered_velocity[probe_id][band_name]

            if u_band.shape != q_band.shape:
                raise ValueError(
                    f"Shape mismatch for probe {probe_id}, band {band_name}: "
                    f"q_band shape = {q_band.shape}, "
                    f"u_band shape = {u_band.shape}"
                )

            u_band_rms = np.sqrt(np.mean(u_band**2))
            numerator = np.mean(q_band * u_band)

            R = numerator / (q_rms * u_band_rms + 1e-30)

            probe_ids.append(probe_id)
            n_values.append(probe_n[probe_id])
            R_values.append(R)

            print(
                f"band={band_name:<10} "
                f"probe={probe_id:<5} "
                f"n/c={probe_n[probe_id]:.6e} "
                f"R_band_limited={R:.6e}"
            )

        correlations_both_filtered[signal_name][band_name] = {
            "probe_ids": np.array(probe_ids),
            "n_over_c": np.array(n_values),
            "R": np.array(R_values),
        }



# ============================================================================
# PLOT COMPARISON: WALL-ONLY FILTERED VS BOTH FILTERED
# ============================================================================

for signal_name in ["tau_w", "pressure"]:

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    for band_name in FREQUENCY_BANDS.keys():

        data_wall_only = correlations_wall_only[signal_name][band_name]
        data_both = correlations_both_filtered[signal_name][band_name]

        ax.plot(
            data_wall_only["R"],
            data_wall_only["n_over_c"],
            marker="o",
            linewidth=1.4,
            linestyle="-",
            label=rf"{band_name}, $q_B$-$u_s$"
        )

        ax.plot(
            data_both["R"],
            data_both["n_over_c"],
            marker="s",
            linewidth=1.4,
            linestyle="--",
            label=rf"{band_name}, $q_B$-$u_{{s,B}}$"
        )

    ax.axvline(0.0, color="k", linewidth=0.8)
    ax.set_xlabel(r"$R$", fontsize=12)
    ax.set_ylabel(r"$n/c$", fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)

    ax.set_title(
        rf"Frequency-filtered correlations, "
        rf"{wall_signal_labels[signal_name]}, "
        rf"$x/c={metadata['xc_actual']:.3f}$"
    )

    plt.tight_layout()

    output_file = os.path.join(
        OUT_PATH,
        f"filtered_correlation_comparison_{signal_name}_{aoa_str}_{xc_str}.png"
    )

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved correlation comparison plot: {output_file}")
    plt.close()



# ============================================================================
# PLOT DIFFERENCE BETWEEN BOTH-FILTERED AND WALL-ONLY CORRELATIONS
# ============================================================================

for signal_name in ["tau_w", "pressure"]:

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for band_name in FREQUENCY_BANDS.keys():

        data_wall = correlations_wall_only[signal_name][band_name]
        data_both = correlations_both_filtered[signal_name][band_name]

        delta_R = data_both["R"] - data_wall["R"]

        ax.plot(
            delta_R,
            data_wall["n_over_c"],
            marker="o",
            linewidth=1.4,
            label=band_name
        )

    ax.axvline(0.0, color="k", linewidth=0.8)
    ax.set_xlabel(r"$R_{q_B u_{s,B}} - R_{q_B u_s}$", fontsize=12)
    ax.set_ylabel(r"$n/c$", fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)

    ax.set_title(
        rf"Effect of filtering velocity, "
        rf"{wall_signal_labels[signal_name]}, "
        rf"$x/c={float(xc):.3f}$"
    )

    plt.tight_layout()

    output_file = os.path.join(
        OUT_PATH,
        f"filtered_correlation_difference_{signal_name}_{aoa_str}_{xc_str}.png"
    )

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved correlation difference plot: {output_file}")
    plt.close()

# ============================================================================
# DELTA-Z CORRELATION FUNCTIONS
# ============================================================================

def compute_delta_z_correlation_fft(q_tz, u_tz, normalize=True):
    """
    q_tz, u_tz shape: (Nt, Nz)
    Return R_dz shape: (Nz,)
    Raw index 0 is Delta z = 0.
    Use FFT circular cross-correlation along axis=1.
    """
    if q_tz.shape != u_tz.shape:
        raise ValueError(
            f"Shape mismatch: q_tz shape = {q_tz.shape}, "
            f"u_tz shape = {u_tz.shape}"
        )

    Nt, Nz = q_tz.shape

    q_fft = np.fft.rfft(q_tz, axis=1)
    u_fft = np.fft.rfft(u_tz, axis=1)

    corr_t_dz = np.fft.irfft(
        np.conj(q_fft) * u_fft,
        n=Nz,
        axis=1
    )

    numerator_dz = np.mean(corr_t_dz, axis=0) / Nz

    if not normalize:
        return numerator_dz

    q_rms = np.sqrt(np.mean(q_tz**2))
    u_rms = np.sqrt(np.mean(u_tz**2))

    return numerator_dz / (q_rms * u_rms + 1e-30)


def center_delta_z_correlation(R):
    """Center Delta-z correlation for visualization using fftshift."""
    return np.fft.fftshift(R, axes=-1)


# Define delta_z index
delta_z_index = np.arange(nz) - nz // 2

print("\n" + "="*60)
print("DELTA-Z INDEX DEFINITION")
print("="*60)
print(f"nz = {nz}")
print(f"delta_z_index range: {delta_z_index[0]} to {delta_z_index[-1]}")
print(f"Raw Delta z = 0 at index: 0")

# ============================================================================
# COMPUTE SPANWISE-SEPARATION CORRELATIONS
# ============================================================================

delta_z_correlations_wall_only = {
    "tau_w": {},
    "pressure": {},
}

delta_z_correlations_both_filtered = {
    "tau_w": {},
    "pressure": {},
}

print("\n" + "="*60)
print("COMPUTING SPANWISE-SEPARATION CORRELATIONS")
print("="*60)

for signal_name in ["tau_w", "pressure"]:

    print(f"\nSignal: {signal_name}")

    for band_name in FREQUENCY_BANDS.keys():

        print(f"  Band: {band_name}")

        q_band = filtered_wall_signals[signal_name][band_name]

        probe_ids = []
        n_values = []
        R_wall_only_list = []
        R_both_filtered_list = []

        for probe_id in sorted(velocity_fluct.keys()):

            u_full = velocity_fluct[probe_id]
            u_band = filtered_velocity[probe_id][band_name]

            # Compute Delta-z correlations
            R_wall_only_dz = compute_delta_z_correlation_fft(q_band, u_full, normalize=True)
            R_both_dz = compute_delta_z_correlation_fft(q_band, u_band, normalize=True)

            probe_ids.append(probe_id)
            n_values.append(probe_n[probe_id])
            R_wall_only_list.append(R_wall_only_dz)
            R_both_filtered_list.append(R_both_dz)

            print(
                f"    probe={probe_id:<5} "
                f"n/c={probe_n[probe_id]:.6e} "
                f"R_wall_only[0]={R_wall_only_dz[0]:.6e} "
                f"R_both[0]={R_both_dz[0]:.6e}"
            )

        delta_z_correlations_wall_only[signal_name][band_name] = {
            "probe_ids": np.array(probe_ids),
            "n_over_c": np.array(n_values),
            "R": np.array(R_wall_only_list),  # (Nprobe, Nz)
        }

        delta_z_correlations_both_filtered[signal_name][band_name] = {
            "probe_ids": np.array(probe_ids),
            "n_over_c": np.array(n_values),
            "R": np.array(R_both_filtered_list),  # (Nprobe, Nz)
        }

# ============================================================================
# VALIDATION: DELTA Z = 0 REPRODUCES SCALAR CORRELATIONS
# ============================================================================

print("\n" + "="*60)
print("VALIDATION: DELTA Z = 0 CONSISTENCY CHECK")
print("="*60)

for signal_name in ["tau_w", "pressure"]:
    print(f"\nSignal: {signal_name}")

    for band_name in FREQUENCY_BANDS.keys():

        print(f"  Band: {band_name}")

        # Raw index 0 is Delta z = 0
        R_dz_wall = delta_z_correlations_wall_only[signal_name][band_name]["R"][:, 0]
        R_scalar_wall = correlations_wall_only[signal_name][band_name]["R"]

        R_dz_both = delta_z_correlations_both_filtered[signal_name][band_name]["R"][:, 0]
        R_scalar_both = correlations_both_filtered[signal_name][band_name]["R"]

        error_wall = np.max(np.abs(R_dz_wall - R_scalar_wall))
        error_both = np.max(np.abs(R_dz_both - R_scalar_both))

        print(f"    wall_only max error: {error_wall:.6e}")
        print(f"    both_filtered max error: {error_both:.6e}")

        if error_wall > 1e-10:
            raise RuntimeError(
                f"Delta z=0 validation failed for {signal_name}/{band_name} (wall_only): "
                f"max error = {error_wall:.6e}"
            )

        if error_both > 1e-10:
            raise RuntimeError(
                f"Delta z=0 validation failed for {signal_name}/{band_name} (both_filtered): "
                f"max error = {error_both:.6e}"
            )

print("\nValidation passed: All Delta z=0 correlations match scalar correlations.")

# ============================================================================
# HEATMAP PLOTS FOR DELTA-Z CORRELATIONS
# ============================================================================

print("\n" + "="*60)
print("CREATING DELTA-Z HEATMAP PLOTS")
print("="*60)

for corr_type, corr_dict in [
    ("wall_only", delta_z_correlations_wall_only),
    ("both_filtered", delta_z_correlations_both_filtered),
]:

    for signal_name in ["tau_w", "pressure"]:

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for ax_idx, band_name in enumerate(FREQUENCY_BANDS.keys()):

            ax = axes[ax_idx]

            data = corr_dict[signal_name][band_name]
            R_raw = data["R"]  # (Nprobe, Nz)
            n_vals = data["n_over_c"]

            # Center for plotting
            R_map = center_delta_z_correlation(R_raw)

            # Compute symmetric limits
            vmax = np.max(np.abs(R_map))
            vmin = -vmax

            # Create heatmap
            im = ax.pcolormesh(
                delta_z_index,
                n_vals,
                R_map,
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                shading="auto"
            )

            ax.set_xlabel("Δz index (centered)", fontsize=11)
            ax.set_ylabel(r"$n/c$", fontsize=11)
            ax.set_yscale("log")
            ax.set_title(f"{band_name}", fontsize=12, fontweight="bold")

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("R", fontsize=10)

        plt.suptitle(
            f"Spanwise Correlation R(Δz, n/c) - {corr_type}\n"
            f"{wall_signal_labels[signal_name]}, {aoa_str}, {xc_str}",
            fontsize=13,
            fontweight="bold"
        )

        plt.tight_layout()

        output_file = os.path.join(
            OUT_PATH,
            f"delta_z_map_{corr_type}_{signal_name}_{aoa_str}_{xc_str}.png"
        )

        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_file}")
        plt.close()

# ============================================================================
# CURVE PLOTS FOR SELECTED DELTA-Z SEPARATIONS
# ============================================================================

print("\n" + "="*60)
print("CREATING DELTA-Z CURVE PLOTS")
print("="*60)

SELECTED_CURVE_N = [0.001, 0.005, 0.01, 0.05, 0.10]

for corr_type, corr_dict in [
    ("wall_only", delta_z_correlations_wall_only),
    ("both_filtered", delta_z_correlations_both_filtered),
]:

    for signal_name in ["tau_w", "pressure"]:

        for band_name in FREQUENCY_BANDS.keys():

            fig, ax = plt.subplots(figsize=(10, 6))

            data = corr_dict[signal_name][band_name]
            R_raw = data["R"]  # (Nprobe, Nz)
            n_vals = data["n_over_c"]

            # Center for plotting
            R_centered = center_delta_z_correlation(R_raw)

            # Find nearest n/c values for selected targets
            for n_target in SELECTED_CURVE_N:
                idx_n = np.argmin(np.abs(n_vals - n_target))
                n_actual = n_vals[idx_n]

                ax.plot(
                    delta_z_index,
                    R_centered[idx_n, :],
                    marker="o",
                    markersize=4,
                    linewidth=1.5,
                    label=f"n/c = {n_actual:.6e}"
                )

            ax.axhline(0.0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.set_xlabel("Δz index (centered)", fontsize=12)
            ax.set_ylabel("R(Δz)", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, frameon=False)

            ax.set_title(
                f"Spanwise Correlation Curves - {corr_type}\n"
                f"{wall_signal_labels[signal_name]}, {band_name}, {aoa_str}, {xc_str}",
                fontsize=12,
                fontweight="bold"
            )

            plt.tight_layout()

            output_file = os.path.join(
                OUT_PATH,
                f"delta_z_curves_{corr_type}_{signal_name}_{band_name}_{aoa_str}_{xc_str}.png"
            )

            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"Saved: {output_file}")
            plt.close()

# ============================================================================
# SAVE RESULTS TO HDF5
# ============================================================================

# Extract metadata for file naming and storage
Nt = metadata.get('Nt', len(time))
Nz_dim = metadata.get('nz', nz)

# Create HDF5 filename
h5_filename = f"filtered_slice_correlations_with_delta_z_{aoa_str}_{xc_str}.h5"
h5_filepath = os.path.join(OUT_PATH, h5_filename)

print("\n" + "="*60)
print("SAVING RESULTS TO HDF5")
print("="*60)

with h5py.File(h5_filepath, 'w') as hf:

    # ========================================================================
    # Metadata group
    # ========================================================================
    meta_grp = hf.create_group('_metadata')

    # Scalar metadata
    meta_grp.attrs['aoa'] = aoa
    meta_grp.attrs['xc_actual'] = xc
    meta_grp.attrs['fs'] = fs
    meta_grp.attrs['dt'] = dt
    meta_grp.attrs['Nt'] = Nt
    meta_grp.attrs['Nz'] = Nz_dim
    meta_grp.attrs['T_total'] = T
    meta_grp.attrs['df'] = df
    meta_grp.attrs['f_nyquist'] = f_nyquist
    meta_grp.attrs['nperseg_welch'] = nperseg
    meta_grp.attrs['noverlap_welch'] = noverlap
    meta_grp.attrs['window_type'] = window
    meta_grp.attrs['detrend_type'] = detrend_type

    # Additional metadata from original file
    for key in ['num_probes', 'y_wall_actual']:
        if key in metadata:
            meta_grp.attrs[key] = metadata[key]

    # ========================================================================
    # Probe information group
    # ========================================================================
    probe_grp = hf.create_group('probes')

    probe_grp.create_dataset('selected_indices', data=selected_probe_indices, dtype=np.int32)
    probe_grp.create_dataset('n_over_c', data=probe_n[selected_probe_indices], dtype=np.float64)
    probe_grp.create_dataset('y_actual', data=probe_y[selected_probe_indices], dtype=np.float64)

    # ========================================================================
    # Frequency bands group
    # ========================================================================
    freq_grp = hf.create_group('frequency_bands')

    for band_name, (f_low, f_high) in FREQUENCY_BANDS.items():
        band_subgrp = freq_grp.create_group(band_name)
        band_subgrp.attrs['f_low'] = f_low if f_low is not None else -1  # -1 indicates None
        band_subgrp.attrs['f_high'] = f_high if f_high is not None else -1
        band_subgrp.attrs['num_freq_bins_retained'] = int(np.sum(filter_masks[band_name]))

    # ========================================================================
    # RMS ratios for wall signals by band
    # ========================================================================
    rms_grp = hf.create_group('rms_ratios')

    for signal_name in ["tau_w", "pressure"]:
        signal_grp = rms_grp.create_group(signal_name)

        q_full = tau_w_prime if signal_name == "tau_w" else pressure_prime
        q_full_rms = np.sqrt(np.mean(q_full**2))

        signal_grp.attrs['full_rms'] = q_full_rms

        for band_name in FREQUENCY_BANDS.keys():
            q_band = filtered_wall_signals[signal_name][band_name]
            q_band_rms = np.sqrt(np.mean(q_band**2))
            ratio = q_band_rms / (q_full_rms + 1e-30)

            signal_grp.attrs[f'{band_name}_rms'] = q_band_rms
            signal_grp.attrs[f'{band_name}_ratio'] = ratio

    # ========================================================================
    # Reconstruction errors
    # ========================================================================
    recon_grp = hf.create_group('reconstruction_errors')

    for signal_name in ["tau_w", "pressure"]:
        q_full = tau_w_prime if signal_name == "tau_w" else pressure_prime

        q_rec = (
            filtered_wall_signals[signal_name]["low"]
            + filtered_wall_signals[signal_name]["mid"]
            + filtered_wall_signals[signal_name]["high"]
        )

        q_full_rms = np.sqrt(np.mean(q_full**2))
        rec_error = np.sqrt(np.mean((q_full - q_rec)**2)) / (q_full_rms + 1e-30)

        recon_grp.attrs[f'{signal_name}_relative_rms_error'] = rec_error

    # ========================================================================
    # Correlation results: wall-only filtered
    # ========================================================================
    corr_wall_grp = hf.create_group('correlations_wall_only')

    for signal_name in ["tau_w", "pressure"]:
        signal_grp = corr_wall_grp.create_group(signal_name)

        for band_name in FREQUENCY_BANDS.keys():
            data = correlations_wall_only[signal_name][band_name]

            band_subgrp = signal_grp.create_group(band_name)
            band_subgrp.create_dataset('probe_ids', data=data['probe_ids'], dtype=np.int32)
            band_subgrp.create_dataset('n_over_c', data=data['n_over_c'], dtype=np.float64)
            band_subgrp.create_dataset('R', data=data['R'], dtype=np.float64)

    # Direct access arrays
    corr_wall_grp.create_dataset('R_tau_w_low', data=correlations_wall_only['tau_w']['low']['R'], dtype=np.float64)
    corr_wall_grp.create_dataset('R_tau_w_mid', data=correlations_wall_only['tau_w']['mid']['R'], dtype=np.float64)
    corr_wall_grp.create_dataset('R_tau_w_high', data=correlations_wall_only['tau_w']['high']['R'], dtype=np.float64)

    corr_wall_grp.create_dataset('R_pressure_low', data=correlations_wall_only['pressure']['low']['R'], dtype=np.float64)
    corr_wall_grp.create_dataset('R_pressure_mid', data=correlations_wall_only['pressure']['mid']['R'], dtype=np.float64)
    corr_wall_grp.create_dataset('R_pressure_high', data=correlations_wall_only['pressure']['high']['R'], dtype=np.float64)


    # ========================================================================
    # Correlation results: both wall signal and velocity filtered
    # ========================================================================
    corr_both_grp = hf.create_group('correlations_both_filtered')

    for signal_name in ["tau_w", "pressure"]:
        signal_grp = corr_both_grp.create_group(signal_name)

        for band_name in FREQUENCY_BANDS.keys():
            data = correlations_both_filtered[signal_name][band_name]

            band_subgrp = signal_grp.create_group(band_name)
            band_subgrp.create_dataset('probe_ids', data=data['probe_ids'], dtype=np.int32)
            band_subgrp.create_dataset('n_over_c', data=data['n_over_c'], dtype=np.float64)
            band_subgrp.create_dataset('R', data=data['R'], dtype=np.float64)

    # Direct access arrays
    corr_both_grp.create_dataset('R_tau_w_low', data=correlations_both_filtered['tau_w']['low']['R'], dtype=np.float64)
    corr_both_grp.create_dataset('R_tau_w_mid', data=correlations_both_filtered['tau_w']['mid']['R'], dtype=np.float64)
    corr_both_grp.create_dataset('R_tau_w_high', data=correlations_both_filtered['tau_w']['high']['R'], dtype=np.float64)

    corr_both_grp.create_dataset('R_pressure_low', data=correlations_both_filtered['pressure']['low']['R'], dtype=np.float64)
    corr_both_grp.create_dataset('R_pressure_mid', data=correlations_both_filtered['pressure']['mid']['R'], dtype=np.float64)
    corr_both_grp.create_dataset('R_pressure_high', data=correlations_both_filtered['pressure']['high']['R'], dtype=np.float64)

    # ========================================================================
    # Delta-z correlations group
    # ========================================================================
    delta_z_grp = hf.create_group('delta_z')

    delta_z_grp.create_dataset('delta_z_index', data=delta_z_index, dtype=np.int32)
    delta_z_grp.attrs['axis_type'] = 'index'
    delta_z_grp.attrs['zero_separation_index_raw'] = 0
    delta_z_grp.attrs['note'] = 'Raw Delta-z correlations have Delta z = 0 at index 0. Use fftshift only for centered plotting.'

    # Delta-z correlations: wall-only
    delta_z_wall_grp = hf.create_group('delta_z_correlations_wall_only')

    for signal_name in ["tau_w", "pressure"]:
        signal_grp = delta_z_wall_grp.create_group(signal_name)

        for band_name in FREQUENCY_BANDS.keys():
            data = delta_z_correlations_wall_only[signal_name][band_name]

            band_subgrp = signal_grp.create_group(band_name)
            band_subgrp.create_dataset('probe_ids', data=data['probe_ids'], dtype=np.int32)
            band_subgrp.create_dataset('n_over_c', data=data['n_over_c'], dtype=np.float64)
            band_subgrp.create_dataset('R', data=data['R'], dtype=np.float64)

    # Delta-z correlations: both filtered
    delta_z_both_grp = hf.create_group('delta_z_correlations_both_filtered')

    for signal_name in ["tau_w", "pressure"]:
        signal_grp = delta_z_both_grp.create_group(signal_name)

        for band_name in FREQUENCY_BANDS.keys():
            data = delta_z_correlations_both_filtered[signal_name][band_name]

            band_subgrp = signal_grp.create_group(band_name)
            band_subgrp.create_dataset('probe_ids', data=data['probe_ids'], dtype=np.int32)
            band_subgrp.create_dataset('n_over_c', data=data['n_over_c'], dtype=np.float64)
            band_subgrp.create_dataset('R', data=data['R'], dtype=np.float64)

    # ========================================================================
    # Sampling information
    # ========================================================================
    samp_grp = hf.create_group('sampling')

    samp_grp.attrs['dt'] = dt
    samp_grp.attrs['fs'] = fs
    samp_grp.attrs['f_nyquist'] = f_nyquist
    samp_grp.attrs['T_total'] = T
    samp_grp.attrs['df'] = df
    samp_grp.attrs['num_time_steps'] = len(time)
    samp_grp.attrs['num_z_planes'] = Nz_dim
    samp_grp.attrs['dt_relative_std'] = np.std(np.diff(time)) / np.mean(np.diff(time))

print(f"\nFile structure:")
print(f"  /_metadata                              : Simulation and processing parameters")
print(f"  /probes                                 : Selected probe indices and coordinates")
print(f"  /frequency_bands                        : Frequency band definitions")
print(f"  /rms_ratios                             : Wall-signal RMS values and ratios")
print(f"  /reconstruction_errors                  : Reconstruction errors for wall signals")
print(f"  /correlations_wall_only                 : R(q_B, u_s)")
print(f"  /correlations_both_filtered             : R(q_B, u_s_B)")
print(f"  /delta_z                                : Delta-z index and metadata")
print(f"  /delta_z_correlations_wall_only         : R(q_B, u_s, Δz)")
print(f"  /delta_z_correlations_both_filtered     : R(q_B, u_s_B, Δz)")
print(f"  /sampling                               : Sampling parameters")

print(f"\nResults saved to: {h5_filepath}")
print("="*60)
