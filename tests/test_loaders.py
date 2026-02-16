import pandas as pd
from src.data.loaders import GPSDataset, CatalogDataset


def test_gps_dataset_loading(tmp_path):
    # Create dummy GPS CSV
    df = pd.DataFrame(
        {
            "latitude": [35.5, 36.0],
            "longitude": [51.5, 52.0],
            "azimuth_value": [10.0, 20.0],
        }
    )
    csv_path = tmp_path / "dummy_gps.csv"
    df.to_csv(csv_path, index=False)

    dataset = GPSDataset([str(csv_path)])
    assert len(dataset) == 2

    x, y = dataset[0]
    assert x.shape == (2,)  # x, y in 2D KinematicData
    assert y.shape == ()  # theta scalar


def test_catalog_dataset_filtering(tmp_path):
    # Dummy catalog
    df = pd.DataFrame(
        {
            "lat": [35.5, 36.0, 37.0],
            "long": [51.5, 52.0, 53.0],
            "fd": [10.0, 20.0, 40.0],
            "mw_unified": [4.0, 5.0, 3.0],
        }
    )
    txt_path = tmp_path / "dummy_catalog.txt"
    df.to_csv(txt_path, index=False)

    dataset = CatalogDataset(str(txt_path))
    assert len(dataset) == 3


def test_catalog_dataset_normalization(tmp_path):
    # Dummy catalog
    df = pd.DataFrame(
        {
            "long": [51.5, 52.5],
            "lat": [35.5, 36.5],
            "fd": [10.0, 20.0],
            "mw_unified": [4.0, 5.0],
        }
    )
    txt_path = tmp_path / "dummy_norm_catalog.csv"
    df.to_csv(txt_path, index=False)

    # Manual transformer mock
    from src.data.transformers import CoordinateTransformer

    # Create a transformer based on these points
    transformer = CoordinateTransformer(df["lat"].values, df["long"].values)

    dataset = CatalogDataset(str(txt_path), transformer=transformer)

    x, y, z, mag = dataset[0]

    # Check bounds - should be within [-1, 1]
    assert -1.05 <= x <= 1.05
    assert -1.05 <= y <= 1.05
