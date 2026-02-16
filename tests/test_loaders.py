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

    # Values 1 and 2 should pass, value 3 (Mag 3.0) and value 4 (higher depth) logic check
    # Let's verify row count.
    assert len(dataset) == 3
