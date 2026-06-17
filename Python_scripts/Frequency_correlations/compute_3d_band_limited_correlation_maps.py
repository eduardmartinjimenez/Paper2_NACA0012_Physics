import os
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import StrMethodFormatter
import warnings
import logging

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.serif": ["DejaVu Serif"],
    "font.monospace": ["DejaVu Sans Mono"],
    "text.usetex": False,
    "mathtext.fontset": "dejavusans",
    "mathtext.rm": "DejaVu Sans",
    "mathtext.it": "DejaVu Sans:italic",
    "mathtext.bf": "DejaVu Sans:bold",
    "axes.unicode_minus": False,
    "font.family": "sans-serif",
})

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# ============================================================================
# Configuration
# ============================================================================

SOLVER_DT = 2.0e-6

FREQUENCY_BANDS = {
    "low": (None, 2.0),
    "mid": (2.0, 12.5),
}

# CACHE_FILE = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Signal_correlation/3D_time_series/3D_time_series_AoA12_Re50000_all_snapshots_20260514_022417.h5"
CACHE_FILE = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Signal_correlation/3D_time_series/new/3D_time_series_AoA12_Re50000_all_snapshots_20260605_194150.h5"

OUT_BASE_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Freq_correlations_3D/"

AOA = 12
aoa_str = f"AOA{AOA}"

# # CACHE_FILE = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/Signal_correlation/3D_time_series/3D_time_series_AoA5_Re50000_all_snapshots_20260603_011744.h5"
# CACHE_FILE = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/Signal_correlation/3D_time_series/3D_time_series_AoA5_Re50000_all_snapshots_20260604_222641.h5" 

# OUT_BASE_PATH = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/Freq_correlations_3D/"

# AOA = 5
# aoa_str = f"AOA{AOA}"


# ============================================================================
# Create Output Directories
# ============================================================================

os.makedirs(OUT_BASE_PATH, exist_ok=True)

print("=" * 70)
print("3D BAND-LIMITED CORRELATION MAPS")
print("=" * 70)
print(f"\nCache file: {CACHE_FILE}")
print(f"Output base: {OUT_BASE_PATH}")

# ============================================================================
# Load Cached 3D Time-Series
# ============================================================================

print("\n" + "=" * 70)
print("LOADING CACHED 3D TIME-SERIES")
print("=" * 70)

with h5py.File(CACHE_FILE, 'r') as f:
    # Extract list of x_c groups
    x_c_groups = sorted([key for key in f.keys() if key.startswith('x_c_')])
    print(f"Found {len(x_c_groups)} x/c locations: {x_c_groups}")

    # Load iteration_step if available
    iteration_step = f.attrs.get('iteration_step', 20000)
    print(f"Iteration step: {iteration_step}")

    dt = iteration_step * SOLVER_DT
    fs = 1.0 / dt
    f_nyquist = fs / 2.0

    print(f"dt = {dt:.6e} s")
    print(f"fs = {fs:.6f} Hz")
    print(f"f_nyquist = {f_nyquist:.6f} Hz")

    # Validate Nyquist frequency against mid band
    if f_nyquist < 12.5:
        print(f"\n⚠ Warning: f_nyquist ({f_nyquist:.2f}) < 12.5 Hz")
        print(f"  Adjusting mid band upper bound to f_nyquist")
        FREQUENCY_BANDS["mid"] = (2.0, f_nyquist)
    elif abs(f_nyquist - 12.5) < 0.01:
        print(f"\n✓ f_nyquist = {f_nyquist:.6f} Hz ≈ 12.5 Hz")
    else:
        print(f"\n✓ f_nyquist = {f_nyquist:.6f} Hz > 12.5 Hz")

# ============================================================================
# Band Filtering Function
# ============================================================================

def fft_band_filter(q, dt, f_low=None, f_high=None, axis=0):
    """
    Ideal FFT-domain band filter.

    Parameters
    ----------
    q : ndarray
        Signal array. Shape can be (Nt, Nz) for wall signals or
        (Nt, N_valid_points, Nz) for velocity fields.
    dt : float
        Time step.
    f_low : float or None
        Lower cutoff frequency. If None, keep all positive frequencies below f_high.
    f_high : float or None
        Upper cutoff frequency. If None, keep all frequencies above f_low up to Nyquist.
    axis : int
        Time axis (default 0).

    Returns
    -------
    q_filtered : ndarray
        Band-filtered signal with same shape as q.
    freq : ndarray
        Frequencies associated with the rFFT.
    mask : ndarray
        Boolean frequency mask used for filtering.
    """
    q = np.asarray(q, dtype=np.float64)

    Nt = q.shape[axis]
    freq = np.fft.rfftfreq(Nt, d=dt)

    Q = np.fft.rfft(q, axis=axis)

    mask = np.ones_like(freq, dtype=bool)
    mask &= freq > 0.0

    if f_low is not None:
        mask &= freq > f_low

    if f_high is not None:
        mask &= freq <= f_high

    Q_filtered = np.zeros_like(Q)

    slicer = [slice(None)] * Q.ndim
    slicer[axis] = mask

    Q_filtered[tuple(slicer)] = Q[tuple(slicer)]

    q_filtered = np.fft.irfft(Q_filtered, n=Nt, axis=axis)

    return q_filtered, freq, mask

# ============================================================================
# 3D Delta-z Correlation Function
# ============================================================================

def compute_delta_z_correlation_fft_3d(q_tz, u_tnz, normalize=True):
    """
    Compute FFT-based circular cross-correlation along z axis between
    wall signal and 3D velocity field.

    Parameters
    ----------
    q_tz : ndarray
        Wall signal with shape (Nt, Nz).
    u_tnz : ndarray
        3D velocity field with shape (Nt, N_valid_points, Nz).
    normalize : bool
        Whether to normalize by RMS values.

    Returns
    -------
    R : ndarray
        Correlation map with shape (N_valid_points, Nz).
        Raw index 0 is Delta z = 0.
    """
    q_tz = np.asarray(q_tz, dtype=np.float64)
    u_tnz = np.asarray(u_tnz, dtype=np.float64)

    Nt, Nz = q_tz.shape

    if u_tnz.shape[0] != Nt or u_tnz.shape[2] != Nz:
        raise ValueError(
            f"Shape mismatch: q_tz shape = {q_tz.shape}, "
            f"u_tnz shape = {u_tnz.shape}"
        )

    # FFT along z axis
    q_fft = np.fft.rfft(q_tz, axis=1)  # (Nt, Nf)
    u_fft = np.fft.rfft(u_tnz, axis=2)  # (Nt, N_valid_points, Nf)

    # Cross-correlation via inverse FFT
    corr_t = np.fft.irfft(
        np.conj(q_fft)[:, None, :] * u_fft,
        n=Nz,
        axis=2
    )  # (Nt, N_valid_points, Nz)

    numerator = np.mean(corr_t, axis=0) / Nz  # (N_valid_points, Nz)

    if not normalize:
        return numerator

    q_rms = np.sqrt(np.mean(q_tz**2))
    u_rms = np.sqrt(np.mean(u_tnz**2, axis=(0, 2)))  # (N_valid_points,)

    R = numerator / (q_rms * u_rms[:, None] + 1e-30)

    return R

def center_delta_z_correlation(R):
    """Center Delta-z correlation for visualization using fftshift."""
    return np.fft.fftshift(R, axes=-1)

# ============================================================================
# Process Each x/c Location
# ============================================================================

all_results = {}

with h5py.File(CACHE_FILE, 'r') as f:

    for x_c_group in sorted(x_c_groups):
        grp = f[x_c_group]

        # Extract x_c value from group name
        x_c_actual = float(x_c_group.split('_')[2])
        x_c_str = f"x_c_{x_c_actual:.2f}"

        print(f"\n" + "=" * 70)
        print(f"Processing {x_c_group} (x/c = {x_c_actual:.2f})")
        print("=" * 70)

        # Load data
        wall_pressure = grp['wall_pressure'][:].astype(np.float64)
        wall_shear_stress = grp['wall_shear_stress'][:].astype(np.float64)
        fluid_u_streamwise = grp['fluid_u_streamwise'][:].astype(np.float64)

        valid_x = grp['valid_x'][:] if 'valid_x' in grp else None
        valid_y = grp['valid_y'][:] if 'valid_y' in grp else None
        valid_ix = grp['valid_ix'][:] if 'valid_ix' in grp else None
        valid_iy = grp['valid_iy'][:] if 'valid_iy' in grp else None

        Nt, Nz = wall_pressure.shape
        N_valid_points = fluid_u_streamwise.shape[1]

        print(f"  Nt={Nt}, Nz={Nz}, N_valid_points={N_valid_points}")
        print(f"  dt={dt:.6e}, fs={fs:.6f}, f_nyquist={f_nyquist:.6f}")

        # ====================================================================
        # Compute Fluctuations
        # ====================================================================

        pressure_prime = wall_pressure - np.mean(wall_pressure, axis=0, keepdims=True)
        tau_prime = wall_shear_stress - np.mean(wall_shear_stress, axis=0, keepdims=True)
        u_prime = fluid_u_streamwise - np.mean(fluid_u_streamwise, axis=0, keepdims=True)

        # ====================================================================
        # Validate Zero Mean
        # ====================================================================

        print("\n  Zero-mean validation:")

        # Wall signals
        p_mean_z = np.mean(pressure_prime, axis=0)
        p_dc_rms = np.sqrt(np.mean(p_mean_z**2))
        p_full_rms = np.sqrt(np.mean(pressure_prime**2))
        p_ratio = p_dc_rms / (p_full_rms + 1e-30)
        print(f"    pressure: DC/full RMS = {p_ratio:.6e}")

        tau_mean_z = np.mean(tau_prime, axis=0)
        tau_dc_rms = np.sqrt(np.mean(tau_mean_z**2))
        tau_full_rms = np.sqrt(np.mean(tau_prime**2))
        tau_ratio = tau_dc_rms / (tau_full_rms + 1e-30)
        print(f"    tau_w:    DC/full RMS = {tau_ratio:.6e}")

        # Velocity field: compute residual temporal mean per point and z
        u_mean = np.mean(u_prime, axis=0)  # (N_valid_points, Nz)
        u_dc_rms = np.sqrt(np.mean(u_mean**2))
        u_full_rms = np.sqrt(np.mean(u_prime**2))
        u_ratio = u_dc_rms / (u_full_rms + 1e-30)
        print(f"    u:        DC/full RMS = {u_ratio:.6e}")

        # ====================================================================
        # Filter Wall Signals
        # ====================================================================

        print("\n  Filtering wall signals:")

        filtered_wall_signals = {"tau_w": {}, "pressure": {}}
        filter_masks = {}

        for band_name, (f_low, f_high) in FREQUENCY_BANDS.items():
            tau_band, freq_fft, mask = fft_band_filter(
                tau_prime, dt, f_low=f_low, f_high=f_high, axis=0
            )

            p_band, _, _ = fft_band_filter(
                pressure_prime, dt, f_low=f_low, f_high=f_high, axis=0
            )

            filtered_wall_signals["tau_w"][band_name] = tau_band
            filtered_wall_signals["pressure"][band_name] = p_band
            filter_masks[band_name] = mask

            n_bins = np.sum(mask)
            print(f"    {band_name:<10}: {n_bins} frequency bins retained")

        # ====================================================================
        # Filter Velocity Field
        # ====================================================================

        print("  Filtering velocity field:")

        filtered_velocity = {}

        for band_name, (f_low, f_high) in FREQUENCY_BANDS.items():
            u_band, _, _ = fft_band_filter(
                u_prime, dt, f_low=f_low, f_high=f_high, axis=0
            )
            filtered_velocity[band_name] = u_band
            print(f"    {band_name:<10}: ✓")

        # ====================================================================
        # Reconstruction Checks
        # ====================================================================

        print("\n  Reconstruction checks:")

        # Wall signals
        for signal_name, q_full in [
            ("pressure", pressure_prime),
            ("tau_w", tau_prime),
        ]:
            q_rec = (
                filtered_wall_signals[signal_name]["low"]
                + filtered_wall_signals[signal_name]["mid"]
            )

            q_full_rms = np.sqrt(np.mean(q_full**2))
            rec_error = np.sqrt(np.mean((q_full - q_rec)**2)) / (q_full_rms + 1e-30)
            print(f"    {signal_name:<10}: relative RMS error = {rec_error:.6e}")

        # Velocity
        u_rec = filtered_velocity["low"] + filtered_velocity["mid"]
        u_full_rms = np.sqrt(np.mean(u_prime**2))
        u_rec_error = np.sqrt(np.mean((u_prime - u_rec)**2)) / (u_full_rms + 1e-30)
        print(f"    u_streamwise: relative RMS error = {u_rec_error:.6e}")

        # ====================================================================
        # Compute Correlations
        # ====================================================================

        print("\n  Computing 3D correlations:")

        correlations_wall_only = {"tau_w": {}, "pressure": {}}
        correlations_both_filtered = {"tau_w": {}, "pressure": {}}
        correlation_maxima = {
            "wall_only": {"tau_w": {}, "pressure": {}},
            "both_filtered": {"tau_w": {}, "pressure": {}},
        }

        for signal_name in ["tau_w", "pressure"]:
            q_full = tau_prime if signal_name == "tau_w" else pressure_prime

            for band_name in FREQUENCY_BANDS.keys():
                q_band = filtered_wall_signals[signal_name][band_name]

                # Wall-only: wall signal filtered, velocity full
                R_wall_only = compute_delta_z_correlation_fft_3d(
                    q_band, u_prime, normalize=True
                )
                correlations_wall_only[signal_name][band_name] = R_wall_only

                # Both-filtered: both wall signal and velocity filtered
                u_band = filtered_velocity[band_name]
                R_both = compute_delta_z_correlation_fft_3d(
                    q_band, u_band, normalize=True
                )
                correlations_both_filtered[signal_name][band_name] = R_both

                print(
                    f"    {signal_name} / {band_name:<5}: "
                    f"R_wall_only shape={R_wall_only.shape}, "
                    f"R_both shape={R_both.shape}"
                )

        print("\n  Correlation maxima for plot limits:")
        for corr_type, corr_dict in [
            ("wall_only", correlations_wall_only),
            ("both_filtered", correlations_both_filtered),
        ]:
            for signal_name in ["tau_w", "pressure"]:
                for band_name in FREQUENCY_BANDS.keys():
                    R = corr_dict[signal_name][band_name]
                    R_min = np.nanmin(R)
                    R_max = np.nanmax(R)
                    R_abs_max = np.nanmax(np.abs(R))
                    correlation_maxima[corr_type][signal_name][band_name] = R_abs_max
                    print(
                        f"    {corr_type:<13} / {signal_name:<8} / {band_name:<5}: "
                        f"min R={R_min:.6e}, max R={R_max:.6e}, max|R|={R_abs_max:.6e}"
                    )

        # ====================================================================
        # Validation: Delta z = 0 Consistency
        # ====================================================================

        print("\n  Validation: Delta z = 0 consistency:")

        validation_passed = True

        for signal_name in ["tau_w", "pressure"]:
            q_band_low = filtered_wall_signals[signal_name]["low"]
            q_band_mid = filtered_wall_signals[signal_name]["mid"]
            q_rms_low = np.sqrt(np.mean(q_band_low**2))
            q_rms_mid = np.sqrt(np.mean(q_band_mid**2))

            for band_name in ["low", "mid"]:
                q_band = filtered_wall_signals[signal_name][band_name]
                q_rms = np.sqrt(np.mean(q_band**2)) if band_name == "low" else q_rms_mid

                if band_name == "low":
                    q_rms = q_rms_low
                else:
                    q_rms = q_rms_mid

                # Wall-only: Delta z=0 should match scalar correlation
                R_dz0_wall = correlations_wall_only[signal_name][band_name][:, 0]
                u_rms = np.sqrt(np.mean(u_prime**2, axis=(0, 2)))
                R_scalar_wall = np.mean(q_band[:, None, :] * u_prime, axis=(0, 2)) / (q_rms * u_rms + 1e-30)

                error_wall = np.max(np.abs(R_dz0_wall - R_scalar_wall))

                if error_wall > 1e-10:
                    print(f"    ✗ {signal_name} / {band_name} (wall_only): error={error_wall:.6e}")
                    validation_passed = False
                else:
                    print(f"    ✓ {signal_name} / {band_name} (wall_only): error={error_wall:.6e}")

                # Both-filtered
                R_dz0_both = correlations_both_filtered[signal_name][band_name][:, 0]
                u_band = filtered_velocity[band_name]
                u_band_rms = np.sqrt(np.mean(u_band**2, axis=(0, 2)))
                R_scalar_both = np.mean(q_band[:, None, :] * u_band, axis=(0, 2)) / (q_rms * u_band_rms + 1e-30)

                error_both = np.max(np.abs(R_dz0_both - R_scalar_both))

                if error_both > 1e-10:
                    print(f"    ✗ {signal_name} / {band_name} (both_filtered): error={error_both:.6e}")
                    validation_passed = False
                else:
                    print(f"    ✓ {signal_name} / {band_name} (both_filtered): error={error_both:.6e}")

        if not validation_passed:
            print("\n  ✗ Validation failed! Skipping this x/c location.")
            continue

        # ====================================================================
        # Create Output Directories
        # ====================================================================

        case_fig_path = os.path.join(OUT_BASE_PATH, "figures", x_c_str)
        case_data_path = os.path.join(OUT_BASE_PATH, "data")

        os.makedirs(case_fig_path, exist_ok=True)
        os.makedirs(case_data_path, exist_ok=True)

        print(f"\n  Figures: {case_fig_path}")
        print(f"  Data:    {case_data_path}")

        # ====================================================================
        # Create Scatter Plots
        # ====================================================================

        print("\n  Creating scatter plots:")

        delta_z_index = np.arange(Nz) - Nz // 2

        for corr_type, corr_dict in [
            ("wall_only", correlations_wall_only),
            ("both_filtered", correlations_both_filtered),
        ]:

            for signal_name in ["tau_w", "pressure"]:

                for band_name in FREQUENCY_BANDS.keys():

                    R = corr_dict[signal_name][band_name]  # (N_valid_points, Nz)
                    corr_abs_max = correlation_maxima[corr_type][signal_name][band_name]

                    # A. Delta z = 0 map
                    fig, ax = plt.subplots(figsize=(8, 6))
                    R_dz0 = R[:, 0]

                    vmax = corr_abs_max
                    vmin = -vmax

                    scatter = ax.scatter(
                        valid_x, valid_y, c=R_dz0,
                        cmap="RdBu_r", vmin=vmin, vmax=vmax,
                        s=30, edgecolors='none', alpha=0.7
                    )

                    ax.set_xlabel("x/c", fontsize=11)
                    ax.set_ylabel("y/c", fontsize=11)
                    ax.set_title(
                        f"R(Δz=0) - {signal_name} - {band_name}\n{corr_type}",
                        fontsize=12, fontweight="bold"
                    )
                    ax.set_aspect('equal', adjustable='box')
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label("R", fontsize=10)
                    cbar.ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
                    ax.grid(True, alpha=0.3)

                    fname = f"map_dz0_{corr_type}_{signal_name}_{band_name}_{x_c_str}.png"
                    fpath = os.path.join(case_fig_path, fname)
                    plt.savefig(fpath, dpi=150, bbox_inches="tight")
                    plt.close()
                    print(f"      Saved: {fname}")

                    # B. Peak absolute correlation
                    R_peak = np.max(np.abs(R), axis=1)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    scatter = ax.scatter(
                        valid_x, valid_y, c=R_peak,
                        cmap="viridis", vmin=0.0, vmax=corr_abs_max,
                        s=30, edgecolors='none', alpha=0.7
                    )

                    ax.set_xlabel("x/c", fontsize=11)
                    ax.set_ylabel("y/c", fontsize=11)
                    ax.set_title(
                        f"Max|R(Δz)| - {signal_name} - {band_name}\n{corr_type}",
                        fontsize=12, fontweight="bold"
                    )
                    ax.set_aspect('equal', adjustable='box')
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label("Max|R|", fontsize=10)
                    cbar.ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
                    ax.grid(True, alpha=0.3)

                    fname = f"map_peak_abs_{corr_type}_{signal_name}_{band_name}_{x_c_str}.png"
                    fpath = os.path.join(case_fig_path, fname)
                    plt.savefig(fpath, dpi=150, bbox_inches="tight")
                    plt.close()
                    print(f"      Saved: {fname}")

                    # C. Peak Delta-z index (centered)
                    idx_peak = np.argmax(np.abs(R), axis=1)
                    idx_peak_centered = ((idx_peak + Nz // 2) % Nz) - Nz // 2

                    fig, ax = plt.subplots(figsize=(8, 6))
                    scatter = ax.scatter(
                        valid_x, valid_y, c=idx_peak_centered,
                        cmap="RdBu_r", s=30, edgecolors='none', alpha=0.7
                    )

                    ax.set_xlabel("x/c", fontsize=11)
                    ax.set_ylabel("y/c", fontsize=11)
                    ax.set_title(
                        f"Δz Index at Peak|R| - {signal_name} - {band_name}\n{corr_type}",
                        fontsize=12, fontweight="bold"
                    )
                    ax.set_aspect('equal', adjustable='box')
                    cbar = plt.colorbar(scatter, ax=ax)
                    cbar.set_label("Δz Index", fontsize=10)
                    cbar.ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
                    ax.grid(True, alpha=0.3)

                    fname = f"map_peak_delta_z_index_{corr_type}_{signal_name}_{band_name}_{x_c_str}.png"
                    fpath = os.path.join(case_fig_path, fname)
                    plt.savefig(fpath, dpi=150, bbox_inches="tight")
                    plt.close()
                    print(f"      Saved: {fname}")

                    # D. Selected Delta-z maps
                    selected_raw_dz_indices = [0, Nz // 8, Nz // 4]

                    for raw_idx in selected_raw_dz_indices:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        R_sel = R[:, raw_idx]

                        vmax = corr_abs_max
                        vmin = -vmax

                        scatter = ax.scatter(
                            valid_x, valid_y, c=R_sel,
                            cmap="RdBu_r", vmin=vmin, vmax=vmax,
                            s=30, edgecolors='none', alpha=0.7
                        )

                        ax.set_xlabel("x/c", fontsize=11)
                        ax.set_ylabel("y/c", fontsize=11)
                        ax.set_title(
                            f"R(raw_Δz={raw_idx}) - {signal_name} - {band_name}\n{corr_type}",
                            fontsize=12, fontweight="bold"
                        )
                        ax.set_aspect('equal', adjustable='box')
                        cbar = plt.colorbar(scatter, ax=ax)
                        cbar.set_label("R", fontsize=10)
                        cbar.ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
                        ax.grid(True, alpha=0.3)

                        fname = f"map_dz_raw{raw_idx}_{corr_type}_{signal_name}_{band_name}_{x_c_str}.png"
                        fpath = os.path.join(case_fig_path, fname)
                        plt.savefig(fpath, dpi=150, bbox_inches="tight")
                        plt.close()
                        print(f"      Saved: {fname}")

        # ====================================================================
        # Store Results for HDF5 Saving
        # ====================================================================

        all_results[x_c_actual] = {
            'x_c_actual': x_c_actual,
            'x_c_str': x_c_str,
            'Nt': Nt,
            'Nz': Nz,
            'N_valid_points': N_valid_points,
            'valid_x': valid_x,
            'valid_y': valid_y,
            'valid_ix': valid_ix,
            'valid_iy': valid_iy,
            'delta_z_index': delta_z_index,
            'correlations_wall_only': correlations_wall_only,
            'correlations_both_filtered': correlations_both_filtered,
            'correlation_maxima': correlation_maxima,
            'filter_masks': filter_masks,
        }

# ============================================================================
# Save HDF5 Output
# ============================================================================

print("\n" + "=" * 70)
print("SAVING HDF5 OUTPUT")
print("=" * 70)

output_file = os.path.join(
    os.path.dirname(OUT_BASE_PATH.rstrip("/")),
    "Freq_correlations_3D",
    f"band_limited_3d_correlation_maps_{aoa_str}_Re50000.h5"
)

os.makedirs(os.path.dirname(output_file), exist_ok=True)

with h5py.File(output_file, 'w') as hf:

    # Metadata
    meta_grp = hf.create_group('_metadata')
    meta_grp.attrs['AOA'] = AOA
    meta_grp.attrs['Re_c'] = 50000
    meta_grp.attrs['dt'] = dt
    meta_grp.attrs['fs'] = fs
    meta_grp.attrs['f_nyquist'] = f_nyquist
    meta_grp.attrs['source_time_series_file'] = CACHE_FILE
    meta_grp.attrs['SOLVER_DT'] = SOLVER_DT

    # Frequency bands
    freq_grp = hf.create_group('frequency_bands')
    for band_name, (f_low, f_high) in FREQUENCY_BANDS.items():
        band_subgrp = freq_grp.create_group(band_name)
        band_subgrp.attrs['f_low'] = f_low if f_low is not None else -1.0
        band_subgrp.attrs['f_high'] = f_high if f_high is not None else -1.0

    # Results for each x/c location
    for x_c_actual in sorted(all_results.keys()):
        result = all_results[x_c_actual]
        x_c_str = result['x_c_str']

        loc_grp = hf.create_group(x_c_str)
        loc_grp.attrs['x_c_target'] = x_c_actual
        loc_grp.attrs['x_c_actual'] = x_c_actual
        loc_grp.attrs['Nt'] = result['Nt']
        loc_grp.attrs['Nz'] = result['Nz']
        loc_grp.attrs['N_valid_points'] = result['N_valid_points']

        # Spatial coordinates
        loc_grp.create_dataset('valid_x', data=result['valid_x'], dtype=np.float64)
        loc_grp.create_dataset('valid_y', data=result['valid_y'], dtype=np.float64)
        loc_grp.create_dataset('valid_ix', data=result['valid_ix'], dtype=np.int32)
        loc_grp.create_dataset('valid_iy', data=result['valid_iy'], dtype=np.int32)
        loc_grp.create_dataset('delta_z_raw_index', data=result['delta_z_index'], dtype=np.int32)
        loc_grp.create_dataset(
            'delta_z_plot_index',
            data=np.fft.fftshift(result['delta_z_index']),
            dtype=np.int32
        )

        # Wall-only correlations
        wall_only_grp = loc_grp.create_group('wall_only')
        for signal_name in ["tau_w", "pressure"]:
            sig_grp = wall_only_grp.create_group(signal_name)
            for band_name in FREQUENCY_BANDS.keys():
                R = result['correlations_wall_only'][signal_name][band_name]
                ds = sig_grp.create_dataset(
                    f'{band_name}_R',
                    data=R,
                    dtype=np.float64
                )
                ds.attrs['plot_abs_max'] = result['correlation_maxima']['wall_only'][signal_name][band_name]

        # Both-filtered correlations
        both_grp = loc_grp.create_group('both_filtered')
        for signal_name in ["tau_w", "pressure"]:
            sig_grp = both_grp.create_group(signal_name)
            for band_name in FREQUENCY_BANDS.keys():
                R = result['correlations_both_filtered'][signal_name][band_name]
                ds = sig_grp.create_dataset(
                    f'{band_name}_R',
                    data=R,
                    dtype=np.float64
                )
                ds.attrs['plot_abs_max'] = result['correlation_maxima']['both_filtered'][signal_name][band_name]

print(f"Output saved to: {output_file}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nProcessed {len(all_results)} x/c locations")
for x_c_actual in sorted(all_results.keys()):
    result = all_results[x_c_actual]
    print(f"\n  x/c = {x_c_actual:.2f}:")
    print(f"    Nt={result['Nt']}, Nz={result['Nz']}, N_valid_points={result['N_valid_points']}")
    print(f"    Sampling: dt={dt:.6e} s, fs={fs:.6f} Hz, f_nyquist={f_nyquist:.6f} Hz")
    print(f"    Frequency bands: low={FREQUENCY_BANDS['low']}, mid={FREQUENCY_BANDS['mid']}")
    x_c_str = result['x_c_str']
    print(f"    Figures: {os.path.join(OUT_BASE_PATH, 'figures', x_c_str)}/")
    print(f"    HDF5 output: {output_file}")

print("\n✓ Processing complete!")
