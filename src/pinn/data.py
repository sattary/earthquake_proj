import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class KinematicData(Dataset):
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
            # Ensure standard columns. Adjust if files differ.
            # Expecting: Longitude, Latitude, azimuth_value
            self.data = pd.concat([self.data, df], ignore_index=True)

        # Normalize coordinates (Simple Min-Max for now, or load a scaler)
        self.long = self.data["longitude"].values.astype(np.float32)
        self.lat = self.data["latitude"].values.astype(np.float32)
        self.azimuths = self.data["azimuth_value"].values.astype(np.float32)

        # Convert azimuth to radians
        self.azimuths_rad = np.deg2rad(self.azimuths)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.long[idx]
        y = self.lat[idx]
        theta = self.azimuths_rad[idx]
        return torch.tensor([x, y]), torch.tensor(theta)


class EarthquakeCatalog(Dataset):
    """
    Dataset for Earthquake Catalog.
    Loads data from cleaned_historical_Eq.csv.
    """

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        # Mapping columns based on cleaned_historical_Eq.csv header:
        # date,origine time,lat,long,mi,mb,ms,mw,fd,mw_unified
        self.long = self.data["long"].values.astype(np.float32)
        self.lat = self.data["lat"].values.astype(np.float32)
        # Assuming 'fd' is Focal Depth. Fill NaNs if necessary.
        self.depth = self.data["fd"].fillna(0).values.astype(np.float32)
        self.mag = self.data["mw_unified"].values.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(
            [self.long[idx], self.lat[idx], self.depth[idx], self.mag[idx]]
        )
