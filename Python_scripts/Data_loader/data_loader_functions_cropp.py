import h5py
import numpy as np
import os

class CompressedSnapshotLoader:
    def __init__(self, mesh_file_path, region=None, exclude_z_ghosts=True):
        """
        Initialize the loader by loading the mesh and topology once.

        Parameters
        ----------
        mesh_file_path : str
            Path to HDF5 mesh file containing mesh coordinates and topology.
        region : tuple of 6 floats, optional
            Spatial region to extract: (x_min, x_max, y_min, y_max, z_min, z_max)
            in physical coordinates. If None (default), loads full domain.
        exclude_z_ghosts : bool, default=True
            If True, automatically excludes ghost cells at z[0] and z[-1].
        """
        if not os.path.exists(mesh_file_path):
            raise FileNotFoundError(f"Mesh file not found: {mesh_file_path}")

        print(f"Loading compressed mesh from: {mesh_file_path}")

        # Load full mesh arrays
        with h5py.File(mesh_file_path, "r") as f:
            x_full = f["x"][:, :, :]
            y_full = f["y"][:, :, :]
            z_full = f["z"][:, :, :]
            tag_ibm_full = f["tag_IBM"][:, :, :]
            topo_full = f["compressed_topology"][:, :]  # (N, 3)

        # Fast path: no region, just ghost cell exclusion (most common case)
        if region is None and exclude_z_ghosts:
            # Direct slicing for ghost cell exclusion (MUCH faster than full pipeline)
            self.x = x_full[1:-1, :, :]
            self.y = y_full[1:-1, :, :]
            self.z = z_full[1:-1, :, :]
            self.tag_ibm = tag_ibm_full[1:-1, :, :]

            # Filter topology to exclude ghost cells
            zi = topo_full[:, 0]
            mask = (zi >= 1) & (zi < z_full.shape[0] - 1)
            self.topo = topo_full[mask].copy()
            self.topo[:, 0] -= 1  # Remap z indices after removing first ghost layer
            self._global_indices = np.where(mask)[0]

            self.shape = self.x.shape
            self.N_points = self.topo.shape[0]
            self.region = None
            self.index_ranges = None
            self.exclude_z_ghosts = True

        # Fast path: no region, no ghost cell exclusion
        elif region is None and not exclude_z_ghosts:
            # Direct assignment (fastest - just like original code)
            self.x = x_full
            self.y = y_full
            self.z = z_full
            self.tag_ibm = tag_ibm_full
            self.topo = topo_full
            self._global_indices = np.arange(topo_full.shape[0])

            self.shape = self.x.shape
            self.N_points = self.topo.shape[0]
            self.region = None
            self.index_ranges = None
            self.exclude_z_ghosts = False

        # Full filtering path: custom region specified
        else:
            # Handle ghost cell exclusion for custom regions
            if exclude_z_ghosts:
                z_line = z_full[:, 0, 0]
                x_min, x_max, y_min, y_max, z_min, z_max = region
                z_min = max(z_min, z_line[1])
                z_max = min(z_max, z_line[-2])
                region = (x_min, x_max, y_min, y_max, z_min, z_max)

            # Convert region to index ranges
            coord_arrays = {'x': x_full, 'y': y_full, 'z': z_full}
            region = self._validate_region(region, coord_arrays)
            index_ranges = self._physical_coords_to_index_ranges(coord_arrays, region)

            # Crop mesh arrays
            z_start, z_end = index_ranges['z']
            y_start, y_end = index_ranges['y']
            x_start, x_end = index_ranges['x']

            self.x = x_full[z_start:z_end, y_start:y_end, x_start:x_end]
            self.y = y_full[z_start:z_end, y_start:y_end, x_start:x_end]
            self.z = z_full[z_start:z_end, y_start:y_end, x_start:x_end]
            self.tag_ibm = tag_ibm_full[z_start:z_end, y_start:y_end, x_start:x_end]

            # Filter and remap topology
            self.topo, self._global_indices = self._filter_topology_by_region(
                topo_full, index_ranges
            )

            # Update attributes
            self.shape = self.x.shape
            self.N_points = self.topo.shape[0]
            self.region = region
            self.index_ranges = index_ranges
            self.exclude_z_ghosts = exclude_z_ghosts

        # # Add informative logging
        # if region is not None and index_ranges is not None:
        #     x_min, x_max, y_min, y_max, z_min, z_max = region
        #     print(f"Filtering to region:")
        #     print(f"  X: [{x_min:.4f}, {x_max:.4f}]")
        #     print(f"  Y: [{y_min:.4f}, {y_max:.4f}]")
        #     print(f"  Z: [{z_min:.4f}, {z_max:.4f}]")
        #     print(f"Filtered mesh shape: {self.shape}")
        #     print(f"Filtered points: {self.N_points} / {topo_full.shape[0]} "
        #           f"({100*self.N_points/topo_full.shape[0]:.1f}%)")

    def load_snapshot(self, snapshot_file_path):
        """
        Load a compressed snapshot file and return the fields as 1D arrays.
        Automatically filters to region if specified during initialization.
        """
        if not os.path.exists(snapshot_file_path):
            raise FileNotFoundError(f"Snapshot file not found: {snapshot_file_path}")

        print(f"Loading snapshot from: {snapshot_file_path}")
        with h5py.File(snapshot_file_path, "r") as f:
            u_full = f["u_compressed"][:]
            v_full = f["v_compressed"][:]
            w_full = f["w_compressed"][:]
            p_full = f["p_compressed"][:]

        # Filter to region if applicable
        if hasattr(self, '_global_indices') and len(self._global_indices) < len(u_full):
            u = u_full[self._global_indices]
            v = v_full[self._global_indices]
            w = w_full[self._global_indices]
            p = p_full[self._global_indices]
        else:
            u, v, w, p = u_full, v_full, w_full, p_full

        return {
            "u": u,
            "v": v,
            "w": w,
            "p": p,
            "topo": self.topo
        }

    def load_snapshot_avg(self, snapshot_file_path):
        """
        Load a compressed snapshot file and return the fields as 1D arrays.
        Averages are loaded only if present in the file.
        Handles both naming conventions: avg_u_compressed and u_avg_compressed.
        Automatically filters to region if specified during initialization.
        """
        if not os.path.exists(snapshot_file_path):
            raise FileNotFoundError(f"Snapshot file not found: {snapshot_file_path}")

        print(f"Loading averaged snapshot from: {snapshot_file_path}")
        with h5py.File(snapshot_file_path, "r") as f:
            u_full = f["u_compressed"][:]
            v_full = f["v_compressed"][:]
            w_full = f["w_compressed"][:]
            p_full = f["p_compressed"][:]

            # Handle both naming conventions for averaged fields
            if "avg_u_compressed" in f.keys():
                avg_u_full = f["avg_u_compressed"][:]
            elif "u_avg_compressed" in f.keys():
                avg_u_full = f["u_avg_compressed"][:]
            else:
                avg_u_full = None

            if "avg_v_compressed" in f.keys():
                avg_v_full = f["avg_v_compressed"][:]
            elif "v_avg_compressed" in f.keys():
                avg_v_full = f["v_avg_compressed"][:]
            else:
                avg_v_full = None

            if "avg_w_compressed" in f.keys():
                avg_w_full = f["avg_w_compressed"][:]
            elif "w_avg_compressed" in f.keys():
                avg_w_full = f["w_avg_compressed"][:]
            else:
                avg_w_full = None

            if "avg_p_compressed" in f.keys():
                avg_p_full = f["avg_p_compressed"][:]
            elif "p_avg_compressed" in f.keys():
                avg_p_full = f["p_avg_compressed"][:]
            else:
                avg_p_full = None

        # Filter to region if applicable
        if hasattr(self, '_global_indices') and len(self._global_indices) < len(u_full):
            u = u_full[self._global_indices]
            v = v_full[self._global_indices]
            w = w_full[self._global_indices]
            p = p_full[self._global_indices]

            # Filter averaged fields if they exist
            avg_u = avg_u_full[self._global_indices] if avg_u_full is not None else None
            avg_v = avg_v_full[self._global_indices] if avg_v_full is not None else None
            avg_w = avg_w_full[self._global_indices] if avg_w_full is not None else None
            avg_p = avg_p_full[self._global_indices] if avg_p_full is not None else None
        else:
            u, v, w, p = u_full, v_full, w_full, p_full
            avg_u, avg_v, avg_w, avg_p = avg_u_full, avg_v_full, avg_w_full, avg_p_full

        result = {
            "u": u,
            "v": v,
            "w": w,
            "p": p,
            "topo": self.topo
        }

        # Only add averages if they were found
        if avg_u is not None:
            result["avg_u"] = avg_u
        if avg_v is not None:
            result["avg_v"] = avg_v
        if avg_w is not None:
            result["avg_w"] = avg_w
        if avg_p is not None:
            result["avg_p"] = avg_p

        return result

    def get_coordinates(self):
        """
        Return the coordinates at the fluid points (compressed topology).
        """
        zi, yi, xi = self.topo[:, 0], self.topo[:, 1], self.topo[:, 2]
        return self.x[zi, yi, xi], self.y[zi, yi, xi], self.z[zi, yi, xi]

    def reconstruct_field(self, compressed_field):
        """
        Reconstruct a full 3D array with fluid values at topology points and NaN elsewhere.
        """
        field_full = np.full(self.shape, np.nan, dtype=np.float32)
        zi, yi, xi = self.topo[:, 0], self.topo[:, 1], self.topo[:, 2]
        field_full[zi, yi, xi] = compressed_field
        return field_full

    def _validate_region(self, region, coord_arrays):
        """
        Validate region bounds are sensible and within domain.

        Parameters
        ----------
        region : tuple
            (x_min, x_max, y_min, y_max, z_min, z_max)
        coord_arrays : dict
            Dictionary with 'x', 'y', 'z' keys containing 3D coordinate arrays

        Returns
        -------
        tuple
            Validated and potentially corrected region bounds
        """
        import warnings

        x_min, x_max, y_min, y_max, z_min, z_max = region

        # Check bounds order and swap if inverted
        if x_min > x_max:
            warnings.warn(f"x_min > x_max, swapping: {x_min} <-> {x_max}")
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            warnings.warn(f"y_min > y_max, swapping: {y_min} <-> {y_max}")
            y_min, y_max = y_max, y_min
        if z_min > z_max:
            warnings.warn(f"z_min > z_max, swapping: {z_min} <-> {z_max}")
            z_min, z_max = z_max, z_min

        # Check bounds are within domain
        x_domain = (coord_arrays['x'].min(), coord_arrays['x'].max())
        y_domain = (coord_arrays['y'].min(), coord_arrays['y'].max())
        z_domain = (coord_arrays['z'].min(), coord_arrays['z'].max())

        if x_max < x_domain[0] or x_min > x_domain[1]:
            raise ValueError(f"X region [{x_min}, {x_max}] outside domain {x_domain}")
        if y_max < y_domain[0] or y_min > y_domain[1]:
            raise ValueError(f"Y region [{y_min}, {y_max}] outside domain {y_domain}")
        if z_max < z_domain[0] or z_min > z_domain[1]:
            raise ValueError(f"Z region [{z_min}, {z_max}] outside domain {z_domain}")

        return (x_min, x_max, y_min, y_max, z_min, z_max)

    def _physical_coords_to_index_ranges(self, coord_arrays, region):
        """
        Convert physical coordinate bounds to grid index ranges.

        Parameters
        ----------
        coord_arrays : dict
            Dictionary with 'x', 'y', 'z' keys containing 3D coordinate arrays
        region : tuple
            (x_min, x_max, y_min, y_max, z_min, z_max) in physical coordinates

        Returns
        -------
        dict
            Dictionary with 'x', 'y', 'z' keys containing (start_idx, end_idx) tuples
        """
        x_min, x_max, y_min, y_max, z_min, z_max = region

        # Extract 1D coordinate lines from structured grid
        # Assumes: x varies along axis=2, y along axis=1, z along axis=0
        z_line = coord_arrays['z'][:, 0, 0]  # (Nz,)
        y_line = coord_arrays['y'][0, :, 0]  # (Ny,)
        x_line = coord_arrays['x'][0, 0, :]  # (Nx,)

        def find_range(coord_1d, min_val, max_val):
            """Find indices where min_val <= coord <= max_val"""
            mask = (coord_1d >= min_val) & (coord_1d <= max_val)
            if not np.any(mask):
                raise ValueError(
                    f"Region [{min_val}, {max_val}] selects no points. "
                    f"Coordinate range: [{coord_1d.min()}, {coord_1d.max()}]"
                )
            indices = np.where(mask)[0]
            return (indices[0], indices[-1] + 1)  # +1 for Python slice convention

        x_range = find_range(x_line, x_min, x_max)
        y_range = find_range(y_line, y_min, y_max)
        z_range = find_range(z_line, z_min, z_max)

        return {'x': x_range, 'y': y_range, 'z': z_range}

    def _filter_topology_by_region(self, topo, index_ranges):
        """
        Filter topology to only include points within index ranges.

        Parameters
        ----------
        topo : ndarray
            (N_points, 3) array where topo[:, 0]=zi, topo[:, 1]=yi, topo[:, 2]=xi
        index_ranges : dict
            Dictionary with 'x', 'y', 'z' keys containing (start, end) tuples

        Returns
        -------
        filtered_topo : ndarray
            (N_filtered, 3) array with topology indices remapped to local coordinates
        global_indices : ndarray
            (N_filtered,) array mapping filtered point index to original index
        """
        z_start, z_end = index_ranges['z']
        y_start, y_end = index_ranges['y']
        x_start, x_end = index_ranges['x']

        # Extract topology indices
        zi = topo[:, 0]
        yi = topo[:, 1]
        xi = topo[:, 2]

        # Create mask for points within region
        mask = (
            (zi >= z_start) & (zi < z_end) &
            (yi >= y_start) & (yi < y_end) &
            (xi >= x_start) & (xi < x_end)
        )

        # Filter topology
        filtered_topo = topo[mask].copy()

        # Remap indices to local coordinate system (relative to cropped region)
        filtered_topo[:, 0] -= z_start  # zi
        filtered_topo[:, 1] -= y_start  # yi
        filtered_topo[:, 2] -= x_start  # xi

        # Keep track of original indices for snapshot data extraction
        global_indices = np.where(mask)[0]

        return filtered_topo, global_indices
