import h5py
import numpy as np
import os

class CompressedSnapshotLoader:
    def __init__(self, mesh_file_path):
        """
        Initialize the loader by loading the mesh and topology once.
        """
        if not os.path.exists(mesh_file_path):
            raise FileNotFoundError(f"Mesh file not found: {mesh_file_path}")

        print(f"Loading compressed mesh from: {mesh_file_path}")
        with h5py.File(mesh_file_path, "r") as f:
            self.x = f["x"][:, :, :]
            self.y = f["y"][:, :, :]
            self.z = f["z"][:, :, :]
            self.tag_ibm = f["tag_IBM"][:, :, :]
            self.topo = f["compressed_topology"][:, :]  # (N, 3)
        
        self.shape = self.x.shape  # shape of full (cropped) domain
        self.N_points = self.topo.shape[0]

    def load_snapshot(self, snapshot_file_path):
        """
        Load a compressed snapshot file and return the fields as 1D arrays.
        """
        if not os.path.exists(snapshot_file_path):
            raise FileNotFoundError(f"Snapshot file not found: {snapshot_file_path}")
        
        print(f"Loading snapshot from: {snapshot_file_path}")
        with h5py.File(snapshot_file_path, "r") as f:
            u = f["u_compressed"][:]
            v = f["v_compressed"][:]
            w = f["w_compressed"][:]
            p = f["p_compressed"][:]

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
        """
        if not os.path.exists(snapshot_file_path):
            raise FileNotFoundError(f"Snapshot file not found: {snapshot_file_path}")
        
        print(f"Loading averaged snapshot from: {snapshot_file_path}")
        with h5py.File(snapshot_file_path, "r") as f:
            u = f["u_compressed"][:]
            v = f["v_compressed"][:]
            w = f["w_compressed"][:]
            p = f["p_compressed"][:]

            avg_u = f["avg_u_compressed"][:] 
            avg_v = f["avg_v_compressed"][:] 
            avg_w = f["avg_w_compressed"][:] 
            avg_p = f["avg_p_compressed"][:] 
        
        return {
            "u": u,
            "v": v,
            "w": w,
            "p": p,
            "avg_u": avg_u,
            "avg_v": avg_v,
            "avg_w": avg_w,
            "avg_p": avg_p,
            "topo": self.topo
        }

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

