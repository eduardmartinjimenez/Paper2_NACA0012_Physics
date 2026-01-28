import argparse
import h5py
import numpy as np


def compare_attrs(a, b, path, diffs):
    a_keys = set(a.keys())
    b_keys = set(b.keys())
    for key in sorted(a_keys | b_keys):
        if key not in a_keys:
            diffs.append(f"Missing attr in first at {path}: {key}")
        elif key not in b_keys:
            diffs.append(f"Missing attr in second at {path}: {key}")
        else:
            if np.array_equal(np.asarray(a[key]), np.asarray(b[key])):
                continue
            diffs.append(f"Attr mismatch at {path}: {key} -> {a[key]} vs {b[key]}")


def compare_items(a, b, path, atol, rtol, diffs):
    # Attributes
    compare_attrs(a.attrs, b.attrs, path, diffs)

    # Groups vs datasets
    if isinstance(a, h5py.Group) and isinstance(b, h5py.Group):
        a_keys = set(a.keys())
        b_keys = set(b.keys())
        for key in sorted(a_keys | b_keys):
            sub_path = f"{path}/{key}" if path else key
            if key not in a_keys:
                diffs.append(f"Missing in first: {sub_path}")
            elif key not in b_keys:
                diffs.append(f"Missing in second: {sub_path}")
            else:
                compare_items(a[key], b[key], sub_path, atol, rtol, diffs)
    elif isinstance(a, h5py.Dataset) and isinstance(b, h5py.Dataset):
        if a.shape != b.shape:
            diffs.append(f"Shape mismatch at {path}: {a.shape} vs {b.shape}")
            return
        if a.dtype != b.dtype:
            diffs.append(f"Dtype mismatch at {path}: {a.dtype} vs {b.dtype}")
        # Numeric compare with tolerances; fall back to exact for non-float
        data_a = a[...]
        data_b = b[...]
        if np.issubdtype(a.dtype, np.floating) or np.issubdtype(a.dtype, np.complexfloating):
            if not np.allclose(data_a, data_b, atol=atol, rtol=rtol, equal_nan=True):
                diffs.append(f"Data mismatch at {path}: not within atol={atol}, rtol={rtol}")
        else:
            if not np.array_equal(data_a, data_b):
                diffs.append(f"Data mismatch at {path}: values differ")
    else:
        diffs.append(f"Type mismatch at {path}: {type(a)} vs {type(b)}")


def main():
    parser = argparse.ArgumentParser(description="Compare two HDF5 files for equality")
    parser.add_argument("file1", help="First HDF5 file")
    parser.add_argument("file2", help="Second HDF5 file")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance for floats")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for floats")
    args = parser.parse_args()

    diffs = []
    with h5py.File(args.file1, "r") as f1, h5py.File(args.file2, "r") as f2:
        compare_items(f1, f2, path="", atol=args.atol, rtol=args.rtol, diffs=diffs)

    if diffs:
        print("Files differ:")
        for d in diffs:
            print(" -", d)
        print(f"Total differences: {len(diffs)}")
        raise SystemExit(1)
    print("Files are equal within tolerances.")


if __name__ == "__main__":
    main()
