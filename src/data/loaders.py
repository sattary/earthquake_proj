import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from src.data.transformers import CoordinateTransformer


class GPSDataset(Dataset):
    """
    Dataset for Sparse GPS Strain Rate Azimuths.
    Loads data from CSVs containing [Longitude, Latitude, azimuth_value].
    """

    def __init__(self, csv_files):
        """
        Args:
            csv_files (list): List of paths to GPS CSV files.
        """
        self.data = pd.DataFrame()
        for f in csv_files:
            df = pd.read_csv(f)
            self.data = pd.concat([self.data, df], ignore_index=True)

        # Init Transformer
        self.transformer = CoordinateTransformer(
            self.data["latitude"].values, self.data["longitude"].values
        )

        # Normalize coordinates
        self.coords = self.transformer.to_normalized(
            self.data["latitude"].values, self.data["longitude"].values
        )

        self.azimuths = self.data["azimuth_value"].values.astype(np.float32)
        # Convert azimuth to radians
        self.azimuths_rad = np.deg2rad(self.azimuths)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # x_norm, y_norm
        x = self.coords[idx, 0]
        y = self.coords[idx, 1]
        theta = self.azimuths_rad[idx]
        return torch.tensor([x, y]), torch.tensor(theta)


class CatalogDataset(Dataset):
    """
    Dataset for Earthquake Catalog.
    Loads data from cleaned_historical_Eq.csv.
    """

    def __init__(
        self,
        csv_file,
        transformer: CoordinateTransformer = None,
        min_magnitude: float = 0.0,
        max_depth_km: float = 30.0,
    ):
        """
        Args:
            csv_file: Path to earthquake catalog CSV.
            transformer: CoordinateTransformer for lat/lon normalization.
            min_magnitude: Magnitude of completeness (Mc). Events below
                this threshold are excluded.
            max_depth_km: Maximum depth for depth normalization (km).
        """
        self.data = pd.read_csv(csv_file)

        # Filter by magnitude of completeness (Mc)
        if min_magnitude > 0.0:
            self.data = self.data[self.data["mw_unified"] >= min_magnitude].reset_index(
                drop=True
            )

        self.long = self.data["long"].values.astype(np.float32)
        self.lat = self.data["lat"].values.astype(np.float32)
        self.depth = self.data["fd"].fillna(0).values.astype(np.float32)
        self.mag = self.data["mw_unified"].values.astype(np.float32)

        # Normalize depth to [0, 1] where 0=surface, 1=max_depth
        self.max_depth_km = max_depth_km
        self.depth_norm = np.clip(self.depth / max_depth_km, 0.0, 1.0)

        self.transformer = transformer
        if self.transformer:
            self.coords = self.transformer.to_normalized(self.lat, self.long)
        else:
            # Fallback: No normalization (Raw Lat/Lon)
            # WARNING: This will break if used with PINN.
            self.coords = torch.stack(
                [torch.tensor(self.long), torch.tensor(self.lat)], dim=1
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Returns: x, y, z(depth_normalized), mag
        x = self.coords[idx, 0]
        y = self.coords[idx, 1]
        z = self.depth_norm[idx]
        mag = self.mag[idx]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(z, dtype=torch.float32),
            torch.tensor(mag, dtype=torch.float32),
        )
