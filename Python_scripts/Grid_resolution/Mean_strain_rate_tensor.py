import os
import sys
import re
import h5py
import numpy as np
import glob
import gc
import matplotlib.pyplot as plt

module_path = "/home/jofre/Members/Eduard/Paper2/Python_scripts/Data_loader"

if module_path not in sys.path:
    sys.path.append(module_path)

from data_loader_functions import CompressedSnapshotLoader

# Snapshots directory
SNAPSHOTS_ROOT = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Snapshots"

# Data directory
MEAN_DATA_DIR = "/home/jofre/Members/Eduard/Paper2/Simulations/NACA_0012_AOA12_Re50000_1716x1662x128/Mean_data"

# Get all strain tensor files in the directory
strain_files = sorted(glob.glob(os.path.join(MEAN_DATA_DIR, "3d_NACA0012_Re50000_AoA12_strain_rate_tensor_batch_*.h5")))
print(f"Found {len(strain_files)} strain tensor files.")

# Save varibales: Sij_Sij_temporal_avg
SAVE_NAME = "3d_NACA0012_Re50000_AoA12_strain_rate_tensor.h5"
SAVE_PATH = os.path.join(MEAN_DATA_DIR, SAVE_NAME)

# Helper functions
_batch_re = re.compile(r"_batch_(\d+)\.h5$")

def batch_id_from_meanfile(path: str) -> str:
    """Extract XXXX from ..._batch_XXXX.h5 (keeps leading zeros)."""
    m = _batch_re.search(os.path.basename(path))
    if not m:
        raise ValueError(f"Could not parse batch id from filename: {path}")
    return m.group(1)

def count_snapshots_in_batch(batch_dir: str) -> int:
    """
    Count snapshots in a batch folder.
    Priority:
      1) count *.h5 / *.hdf5
      2) else count all non-hidden files (fallback)
    Tune patterns below to match your snapshot naming convention.
    """
    # Adjust these patterns if your snapshots have different extensions/names
    patterns = ["*.h5", "*.hdf5"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(batch_dir, p)))

    if len(files) > 0:
        return len(files)

    # Fallback: count all regular, non-hidden files
    n = 0
    for name in os.listdir(batch_dir):
        if name.startswith("."):
            continue
        full = os.path.join(batch_dir, name)
        if os.path.isfile(full):
            n += 1
    return n

# Weighted accumulation
sum_weighted = None
N_total = 0

missing_batches = []
zero_count_batches = []

for file in strain_files:
    batch_id = batch_id_from_meanfile(file)
    batch_dir = os.path.join(SNAPSHOTS_ROOT, f"batch_{batch_id}")

    if not os.path.isdir(batch_dir):
        missing_batches.append((batch_id, batch_dir))
        continue

    N_batch = count_snapshots_in_batch(batch_dir)
    if N_batch == 0:
        zero_count_batches.append((batch_id, batch_dir))
        continue

    print(f"[batch {batch_id}] snapshots = {N_batch} | mean file = {os.path.basename(file)}")

    with h5py.File(file, "r") as f:
        batch_mean = f["Sij_Sij_temporal_avg"][...].astype(np.float64, copy=False)

    if sum_weighted is None:
        sum_weighted = np.zeros_like(batch_mean, dtype=np.float64)

    sum_weighted += batch_mean * N_batch
    N_total += N_batch

# Safety checks
if missing_batches:
    print("\nWARNING: some batch snapshot folders were not found:")
    for bid, bdir in missing_batches:
        print(f"  - batch {bid}: {bdir}")

if zero_count_batches:
    print("\nWARNING: some batches had 0 snapshots (skipped):")
    for bid, bdir in zero_count_batches:
        print(f"  - batch {bid}: {bdir}")

if N_total == 0 or sum_weighted is None:
    raise RuntimeError("No snapshots were counted. Cannot compute global mean.")

mean_strain = (sum_weighted / N_total).astype(np.float32)

print(f"\nTotal snapshots used: {N_total}")
print(f"Mean strain tensor shape: {mean_strain.shape}")
print(f"Saving temporal average to: {SAVE_PATH}")

with h5py.File(SAVE_PATH, "w") as f:
    f.create_dataset("Sij_Sij_temporal_avg", data=mean_strain, dtype="float32")

print("Completed successfully!")