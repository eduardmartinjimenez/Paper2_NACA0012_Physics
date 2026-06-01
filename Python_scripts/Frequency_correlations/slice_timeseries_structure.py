import os
import sys
import h5py
import numpy as np
from pathlib import Path
import pickle

# Data file
CACHE_FILE = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA5_Re50000_1716x1662x128/Mean_data/Coherence/timeseries_both_xc_0.500.h5"

# Physical parameters
rho_ref = 1.0           # Reference density [kg/m³]
u_infty = 1.0           # Free-stream velocity [m/s]
c = 1.0                 # Airfoil chord [m]
Re_c = 50000            # Reynolds number

print("=" * 80)
print("CACHE FILE STRUCTURE ANALYSIS")
print("=" * 80)

with h5py.File(CACHE_FILE, 'r') as f:
    # File overview
    print(f"\nFile: {CACHE_FILE}")
    print(f"File size: {os.path.getsize(CACHE_FILE) / (1024**3):.2f} GB")
    print(f"Total top-level items: {len(f.keys())}")

    # Analyze metadata
    print("\n" + "-" * 80)
    print("METADATA")
    print("-" * 80)
    meta = f['_metadata']
    metadata = {}

    print("\nAttributes:")
    for key in sorted(meta.attrs):
        val = meta.attrs[key]
        metadata[key] = val
        print(f"  {key}: {val}")

    print("\nDatasets in _metadata/:")
    for key in sorted(meta.keys()):
        data = meta[key][...]
        metadata[key] = data
        print(f"  {key}: shape={data.shape}, dtype={data.dtype}")
        if data.size <= 10:
            print(f"    values: {data}")

    # Analyze main datasets
    print("\n" + "-" * 80)
    print("MAIN DATASETS")
    print("-" * 80)

    main_datasets = {}
    for key in ['time', 'iterations', 'tau_w_z', 'tau_w_prime', 'pressure_z', 'pressure_prime']:
        if key in f:
            data = f[key]
            main_datasets[key] = data.shape
            print(f"\n{key}:")
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
            print(f"  Size: {data.nbytes / 1024:.1f} KB")
            if key == 'time':
                print(f"  Range: [{data[0]:.6f}, {data[-1]:.6f}]")
            elif key == 'iterations':
                print(f"  Range: [{data[0]}, {data[-1]}]")
            else:
                print(f"  Stats - Mean: {np.mean(data):.6f}, Std: {np.std(data):.6f}, Min: {np.min(data):.6f}, Max: {np.max(data):.6f}")

    # Analyze velocity datasets
    print("\n" + "-" * 80)
    print("VELOCITY DATASETS")
    print("-" * 80)

    num_probes = metadata['num_probes']
    velocity_keys = {'u_s_z': [], 'u_s_prime': []}

    for key in f.keys():
        if key.startswith('u_s_z_'):
            velocity_keys['u_s_z'].append(int(key.split('_')[-1]))
        elif key.startswith('u_s_prime_'):
            velocity_keys['u_s_prime'].append(int(key.split('_')[-1]))

    print(f"\nTotal probes specified: {num_probes}")
    print(f"u_s_z datasets found: {len(velocity_keys['u_s_z'])}")
    print(f"u_s_prime datasets found: {len(velocity_keys['u_s_prime'])}")

    # Sample a few velocity datasets
    print(f"\nSample velocity datasets:")
    for probe_idx in [0, num_probes//2, num_probes-1]:
        key_z = f'u_s_z_{probe_idx}'
        key_prime = f'u_s_prime_{probe_idx}'
        if key_z in f:
            data_z = f[key_z][...]
            print(f"\n  {key_z}:")
            print(f"    Shape: {data_z.shape}")
            print(f"    Dtype: {data_z.dtype}")
            print(f"    Stats - Mean: {np.mean(data_z):.6f}, Std: {np.std(data_z):.6f}")
        if key_prime in f:
            data_prime = f[key_prime][...]
            print(f"  {key_prime}:")
            print(f"    Shape: {data_prime.shape}")
            print(f"    Dtype: {data_prime.dtype}")
            print(f"    Stats - Mean: {np.mean(data_prime):.6f}, Std: {np.std(data_prime):.6f}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Time steps: {metadata['Nt']}")
    print(f"Spanwise points (nz): {metadata['nz']}")
    print(f"Number of probes: {num_probes}")
    print(f"Chord position: {metadata['xc_actual']}")
    print(f"Wall distance: {metadata['y_wall_actual']}")
    print(f"Reynolds number: {metadata['Re_c']}")
    print(f"Angle of attack: {metadata['AOA_deg']}°")
    print(f"Time step: {metadata['dt_iteration']}")
    print("=" * 80 + "\n")

print("Loading data into timeseries_data dictionary...")
timeseries_data = {}

with h5py.File(CACHE_FILE, 'r') as f:
    # Load metadata
    meta = f['_metadata']
    metadata = {}
    for key in meta.attrs:
        metadata[key] = meta.attrs[key]
    for key in meta.keys():
        metadata[key] = meta[key][...]

    timeseries_data['metadata'] = metadata

    # Load wall signals
    timeseries_data['time'] = f['time'][...]
    timeseries_data['iterations'] = f['iterations'][...]
    timeseries_data['tau_w_z'] = f['tau_w_z'][...]
    timeseries_data['tau_w_prime'] = f['tau_w_prime'][...]
    timeseries_data['pressure_z'] = f['pressure_z'][...]
    timeseries_data['pressure_prime'] = f['pressure_prime'][...]

    # Load velocity data
    num_probes = metadata['num_probes']
    velocity_mean = {}
    velocity_fluct = {}

    for i in range(num_probes):
        key_z = f'u_s_z_{i}'
        key_prime = f'u_s_prime_{i}'

        if key_z in f:
            velocity_mean[i] = f[key_z][...]
        if key_prime in f:
            velocity_fluct[i] = f[key_prime][...]

    timeseries_data['u_s_z'] = velocity_mean
    timeseries_data['u_s_prime'] = velocity_fluct

print(f"Data loaded successfully. Total dictionary keys: {len(timeseries_data)}")
