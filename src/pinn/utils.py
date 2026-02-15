import numpy as np
import pyproj
import torch


class CoordinateTransformer:
    """
    Handles coordinate projection (Lat/Lon -> UTM) and Normalization ([-1, 1]).
    Critical for PINN to ensure derivatives are well-scaled and physically meaningful (isotropic).
    """

    def __init__(self, lats, longs, utm_zone=39):
        """
        Initialize transformer based on data bounds.
        Args:
            lats (np.array): Latitude array of data.
            longs (np.array): Longitude array of data.
            utm_zone (int): UTM Zone (Central Iran is mostly 39N and 40N).
        """
        # 1. Project to UTM (Meters)
        # EPSG:32639 is WGS 84 / UTM zone 39N.
        # We can auto-detect zone, but fixing it ensures consistency for local study.
        # Central Iran (Tehran) is roughly 51E -> Zone 39.
        self.proj = pyproj.Transformer.from_crs(
            "epsg:4326", f"epsg:326{utm_zone}", always_xy=True
        )

        self.x_meters, self.y_meters = self.proj.transform(longs, lats)

        # 2. Determine Bounds (Physical Domain)
        self.min_x = np.min(self.x_meters)
        self.max_x = np.max(self.x_meters)
        self.min_y = np.min(self.y_meters)
        self.max_y = np.max(self.y_meters)

        # Center and Scale
        self.center_x = (self.min_x + self.max_x) / 2.0
        self.center_y = (self.min_y + self.max_y) / 2.0

        # Scale to [-1, 1]. Use the larger dimension to preserve aspect ratio (Conformal).
        self.scale = max(self.max_x - self.min_x, self.max_y - self.min_y) / 2.0

        print(f"CoordinateTransformer Initialized:")
        print(
            f"  Bounds (UTM): X[{self.min_x:.1f}, {self.max_x:.1f}] Y[{self.min_y:.1f}, {self.max_y:.1f}]"
        )
        print(f"  Scale Factor: {self.scale:.2f} meters")

    def to_normalized(self, lats, longs):
        """
        Convert Lat/Lon -> Normalized [-1, 1] tensor.
        """
        x_m, y_m = self.proj.transform(longs, lats)

        x_norm = (x_m - self.center_x) / self.scale
        y_norm = (y_m - self.center_y) / self.scale

        # Prepare for PyTorch
        x_norm = torch.tensor(x_norm, dtype=torch.float32)
        y_norm = torch.tensor(y_norm, dtype=torch.float32)

        return torch.stack([x_norm, y_norm], dim=1)  # (N, 2)

    def to_physical(self, x_norm_tensor):
        """
        Convert Normalized Tensor -> UTM Meters (numpy).
        """
        if isinstance(x_norm_tensor, torch.Tensor):
            xv = x_norm_tensor.detach().cpu().numpy()
        else:
            xv = x_norm_tensor

        x_m = xv[:, 0] * self.scale + self.center_x
        y_m = xv[:, 1] * self.scale + self.center_y

        return x_m, y_m
