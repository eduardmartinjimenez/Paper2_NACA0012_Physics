"""
HDF5 Data Inspection Script
============================

Load and inspect the saved velocity time series HDF5 file.
Displays complete structure, metadata, and data summaries.
"""

import os
import sys
import h5py
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to the HDF5 file to inspect
DATA_FILE = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data/Signal_correlation/velocity_timeseries_slice_9_test.h5"

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n  {title}")
    print("  " + "-" * (len(title)))

def inspect_group(group, level=0):
    """Recursively inspect HDF5 group contents."""
    indent = "    " * level

    # List all items in this group
    items = sorted(group.keys())

    for item_name in items:
        item = group[item_name]

        if isinstance(item, h5py.Dataset):
            # It's a dataset
            shape = item.shape
            dtype = item.dtype
            size_mb = item.nbytes / (1024 ** 2)
            compression = item.compression if item.compression else "None"

            print(f"{indent}📊 {item_name}")
            print(f"{indent}   └─ Shape: {shape}, Dtype: {dtype}, Size: {size_mb:.2f} MB, Compression: {compression}")

        elif isinstance(item, h5py.Group):
            # It's a group
            print(f"{indent}📁 {item_name}/")
            inspect_group(item, level + 1)

def print_attributes(obj, indent="  "):
    """Print all attributes of an HDF5 object."""
    if len(obj.attrs) == 0:
        print(f"{indent}[No attributes]")
        return

    for attr_name in sorted(obj.attrs.keys()):
        attr_value = obj.attrs[attr_name]

        if isinstance(attr_value, bytes):
            attr_value = attr_value.decode('utf-8')
        elif isinstance(attr_value, np.ndarray):
            if attr_value.size > 5:
                attr_value = f"array with {attr_value.size} elements"
            else:
                attr_value = str(attr_value)

        print(f"{indent}• {attr_name}: {attr_value}")

# ============================================================================
# MAIN INSPECTION
# ============================================================================

print("\n")
print("╔" + "=" * 78 + "╗")
print("║" + " " * 15 + "HDF5 DATA INSPECTION SCRIPT" + " " * 36 + "║")
print("║" + " " * 15 + "Velocity Time Series Data" + " " * 37 + "║")
print("╚" + "=" * 78 + "╝")

# Check if file exists
if not os.path.exists(DATA_FILE):
    print(f"\n❌ ERROR: File not found at {DATA_FILE}")
    sys.exit(1)

print(f"\n✓ File found: {DATA_FILE}")

# Get file size
file_size_mb = os.path.getsize(DATA_FILE) / (1024 ** 2)
print(f"✓ File size: {file_size_mb:.2f} MB")

# ============================================================================
# OPEN AND INSPECT FILE
# ============================================================================

with h5py.File(DATA_FILE, 'r') as f:

    # ========================================================================
    # 1. ROOT-LEVEL ATTRIBUTES
    # ========================================================================
    print_section("ROOT-LEVEL METADATA (Global Attributes)")
    print_attributes(f)

    # ========================================================================
    # 2. FILE STRUCTURE
    # ========================================================================
    print_section("FILE STRUCTURE (Hierarchy)")
    inspect_group(f)

    # ========================================================================
    # 3. DETAILED PROBE INFORMATION
    # ========================================================================
    print_section("DETAILED PROBE INFORMATION")

    if 'probes' in f:
        # Check overall temporal spacing validation status
        if 'temporal_spacing_uniform' in f.attrs:
            overall_valid = f.attrs['temporal_spacing_uniform']
            status_icon = "✓" if overall_valid else "✗"
            status_text = "UNIFORM" if overall_valid else "NON-UNIFORM"
            print(f"\nOVERALL TEMPORAL SPACING: {status_icon} {status_text}")
            if 'temporal_spacing_validation' in f.attrs:
                print(f"Status: {f.attrs['temporal_spacing_validation']}")
            if overall_valid:
                print(f"✓ Data is suitable for spectral analysis (FFT, power spectral density)\n")
            else:
                print(f"⚠ Data has non-uniform time stepping - caution with spectral methods!\n")

        probes_group = f['probes']
        probe_ids = sorted([k for k in probes_group.keys() if k.startswith('probe_')])
        print(f"\nTotal probes found: {len(probe_ids)}\n")

        for probe_id in probe_ids:
            probe_group = probes_group[probe_id]
            print_subsection(f"{probe_id.upper()}")

            # Probe metadata
            print(f"\n  Attributes:")
            print_attributes(probe_group, indent="    ")

            # Temporal spacing validation status
            if 'temporal_spacing_uniform' in probe_group.attrs:
                spacing_valid = probe_group.attrs['temporal_spacing_uniform']
                status_icon = "✓" if spacing_valid else "✗"
                status_text = "UNIFORM" if spacing_valid else "NON-UNIFORM"
                print(f"    ⓘ Temporal Spacing: {status_icon} {status_text}")

            # Datasets in this probe
            print(f"\n  Datasets:")
            datasets = sorted([k for k in probe_group.keys() if isinstance(probe_group[k], h5py.Dataset)])

            for dataset_name in datasets:
                dataset = probe_group[dataset_name]
                shape = dataset.shape
                dtype = dataset.dtype
                size_mb = dataset.nbytes / (1024 ** 2)

                print(f"\n    • {dataset_name}")
                print(f"      Shape: {shape}")
                print(f"      Dtype: {dtype}")
                print(f"      Size: {size_mb:.2f} MB")

                # Show first/last few values
                if len(shape) == 1 and shape[0] > 0:
                    data = dataset[:]
                    print(f"      First 3 values:  {data[:3]}")
                    print(f"      Last 3 values:   {data[-3:]}")
                    print(f"      Min: {np.min(data):.6e}, Max: {np.max(data):.6e}")
                    print(f"      Mean: {np.mean(data):.6e}, Std: {np.std(data):.6e}")

    # ========================================================================
    # 4. SUMMARY STATISTICS
    # ========================================================================
    print_section("DATA SUMMARY & STATISTICS")

    if 'probes' in f:
        probes_group = f['probes']
        probe_ids = sorted([k for k in probes_group.keys() if k.startswith('probe_')])

        for probe_id in probe_ids:
            probe_group = probes_group[probe_id]
            probe_label = probe_group.attrs.get('label', 'unknown')
            y_actual = probe_group.attrs.get('y_actual', np.nan)

            print_subsection(f"{probe_id.upper()} - {probe_label} (y = {y_actual})")

            # Get time array
            if 'time' in probe_group:
                time_data = probe_group['time'][:]
                n_samples = len(time_data)
                t_start = time_data[0]
                t_end = time_data[-1]
                dt_avg = (t_end - t_start) / (n_samples - 1) if n_samples > 1 else 0

                print(f"\n  Time Series Information:")
                print(f"    • Number of samples: {n_samples}")
                print(f"    • Time range: {t_start:.6f} to {t_end:.6f} s")
                print(f"    • Total duration: {t_end - t_start:.6f} s")
                print(f"    • Average time step: {dt_avg:.6e} s")
                print(f"    • Sampling frequency: {1/dt_avg:.2f} Hz (approx)")

            # Statistics for each dataset
            datasets = sorted([k for k in probe_group.keys() if isinstance(probe_group[k], h5py.Dataset)])

            for dataset_name in datasets:
                if dataset_name not in ['iterations', 'time']:
                    dataset = probe_group[dataset_name]
                    data = dataset[:]

                    print(f"\n  {dataset_name}:")
                    print(f"    • Mean: {np.mean(data):.6e}")
                    print(f"    • Std:  {np.std(data):.6e}")
                    print(f"    • Min:  {np.min(data):.6e}")
                    print(f"    • Max:  {np.max(data):.6e}")
                    print(f"    • RMS:  {np.sqrt(np.mean(data**2)):.6e}")

                    # Percentiles
                    p25, p50, p75 = np.percentile(data, [25, 50, 75])
                    print(f"    • Percentiles (25, 50, 75): {p25:.6e}, {p50:.6e}, {p75:.6e}")

    # ========================================================================
    # 5. DATA RELATIONSHIPS
    # ========================================================================
    print_section("DATA RELATIONSHIPS & CONSISTENCY CHECKS")

    if 'probes' in f:
        probes_group = f['probes']
        probe_ids = sorted([k for k in probes_group.keys() if k.startswith('probe_')])

        # Check if all probes have same number of timesteps
        timestep_counts = {}
        for probe_id in probe_ids:
            probe_group = probes_group[probe_id]
            if 'time' in probe_group:
                n_samples = len(probe_group['time'])
                timestep_counts[probe_id] = n_samples

        print(f"\nTimestep counts per probe:")
        for probe_id, count in timestep_counts.items():
            probe_label = probes_group[probe_id].attrs.get('label', 'unknown')
            print(f"  • {probe_id} ({probe_label}): {count} samples")

        # Check time alignment
        if len(probe_ids) >= 2:
            probe_0_time = probes_group[probe_ids[0]]['time'][:]
            probe_1_time = probes_group[probe_ids[1]]['time'][:]

            if len(probe_0_time) == len(probe_1_time):
                time_diff = np.max(np.abs(probe_0_time - probe_1_time))
                print(f"\nTime alignment check:")
                print(f"  • Probes have same length: ✓")
                print(f"  • Max time difference: {time_diff:.6e} s")
                if time_diff < 1e-10:
                    print(f"  • Time arrays align: ✓ (identical)")
                else:
                    print(f"  • Time arrays align: ~ (small differences)")
            else:
                print(f"\nTime alignment check:")
                print(f"  • Probes have different lengths: ✗")
                print(f"    Probe {probe_ids[0]}: {len(probe_0_time)} samples")
                print(f"    Probe {probe_ids[1]}: {len(probe_1_time)} samples")

    # ========================================================================
    # 6. MEMORY USAGE
    # ========================================================================
    print_section("MEMORY USAGE BREAKDOWN")

    total_data_size = 0

    def calculate_group_size(group):
        """Recursively calculate total size of group."""
        total = 0
        for item_name in group.keys():
            item = group[item_name]
            if isinstance(item, h5py.Dataset):
                total += item.nbytes
            elif isinstance(item, h5py.Group):
                total += calculate_group_size(item)
        return total

    total_data_size = calculate_group_size(f)

    print(f"\nTotal data size:")
    print(f"  • Dataset size: {total_data_size / (1024**2):.2f} MB")
    print(f"  • File size: {file_size_mb:.2f} MB")
    print(f"  • Compression ratio: {total_data_size / (file_size_mb * (1024**2)):.2f}x")

    # Breakdown by probe
    if 'probes' in f:
        probes_group = f['probes']
        probe_ids = sorted([k for k in probes_group.keys() if k.startswith('probe_')])

        print(f"\nSize by probe:")
        for probe_id in probe_ids:
            probe_size = calculate_group_size(probes_group[probe_id])
            probe_label = probes_group[probe_id].attrs.get('label', 'unknown')
            pct = 100 * probe_size / total_data_size if total_data_size > 0 else 0
            print(f"  • {probe_id} ({probe_label}): {probe_size / (1024**2):.2f} MB ({pct:.1f}%)")

# ============================================================================
# SUMMARY - DATA ACCESS EXAMPLE
# ============================================================================

print_section("DATA ACCESS EXAMPLES")

print("""
  Example Python code to load and work with the data:

  import h5py
  import numpy as np

  with h5py.File('velocity_timeseries_slice_9_test.h5', 'r') as f:

      # Access global metadata
      num_probes = f.attrs['num_probes']
      aoa_deg = f.attrs['aoa_deg']
      dt_iteration = f.attrs['dt_iteration']

      # Access surface probe data (Probe 0)
      time_surface = f['probes/probe_0/time'][:]        # Time array (s)
      tau_prime = f['probes/probe_0/tau_prime'][:]      # Wall shear stress fluctuation (Pa)
      y_surface = f['probes/probe_0'].attrs['y_actual'] # Probe location

      # Access velocity probe data (Probe 1)
      time_velocity = f['probes/probe_1/time'][:]       # Time array (s)
      u_prime = f['probes/probe_1/u_prime'][:]          # Streamwise fluctuation (m/s)
      v_prime = f['probes/probe_1/v_prime'][:]          # Cross-stream fluctuation (m/s)
      w_prime = f['probes/probe_1/w_prime'][:]          # Spanwise fluctuation (m/s)
      tke = f['probes/probe_1/tke'][:]                  # Turbulent kinetic energy ((m/s)²)
      y_velocity = f['probes/probe_1'].attrs['y_actual'] # Probe location

      # Compute correlations, spectral content, etc.
      correlation_tau_u = np.correlate(tau_prime, u_prime, mode='full')

      print(f"Surface probe location: y = {y_surface}")
      print(f"Number of samples: {len(time_surface)}")
      print(f"Time range: {time_surface[0]:.2f} - {time_surface[-1]:.2f} s")
""")

# ============================================================================
# COMPLETION MESSAGE
# ============================================================================

print_section("INSPECTION COMPLETE")
print(f"\n✓ Successfully inspected: {DATA_FILE}\n")
print("=" * 80 + "\n")
