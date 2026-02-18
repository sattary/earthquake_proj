import numpy as np
import pyproj
import torch


class CoordinateTransformer:
    """
    Handles coordinate projection (Lat/Lon -> UTM) and Normalization ([-1, 1]).
    Ensures isotropic scaling for the PINN domain.
    """

    def __init__(self, lats, longs, utm_zone=39):
        """
        Initialize transformer based on data bounds.
        """
        self.utm_zone = utm_zone
        self.proj = pyproj.Transformer.from_crs(
            "epsg:4326", f"epsg:326{utm_zone}", always_xy=True
        )
        self.inv_proj = pyproj.Transformer.from_crs(
            f"epsg:326{utm_zone}", "epsg:4326", always_xy=True
        )

        self.x_meters, self.y_meters = self.proj.transform(longs, lats)

        self.min_x = np.min(self.x_meters)
        self.max_x = np.max(self.x_meters)
        self.min_y = np.min(self.y_meters)
        self.max_y = np.max(self.y_meters)

        self.center_x = (self.min_x + self.max_x) / 2.0
        self.center_y = (self.min_y + self.max_y) / 2.0

        # Conformal scaling (isotropic)
        self.scale = max(self.max_x - self.min_x, self.max_y - self.min_y) / 2.0

        print("CoordinateTransformer Initialized:")
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

        return torch.tensor(np.stack([x_norm, y_norm], axis=1), dtype=torch.float32)

    def to_utm(self, x_norm_tensor):
        """
        Convert Normalized -> UTM Meters.
        """
        if isinstance(x_norm_tensor, torch.Tensor):
            xv = x_norm_tensor.detach().cpu().numpy()
        else:
            xv = x_norm_tensor

        x_m = xv[:, 0] * self.scale + self.center_x
        y_m = xv[:, 1] * self.scale + self.center_y
        return x_m, y_m

    def to_physical(self, x_norm_tensor):
        """
        Convert Normalized Tensor -> Lat/Lon (numpy).
        """
        x_m, y_m = self.to_utm(x_norm_tensor)
        longs, lats = self.inv_proj.transform(x_m, y_m)
        return lats, longs
