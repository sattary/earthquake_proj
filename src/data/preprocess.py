import pandas as pd
from pathlib import Path


def preprocess_catalog(
    input_file: str = "data/files/historical_Eq.txt",
    output_file: str = "data/cleaned_historical_Eq.csv",
):
    """
    Parses the messy tab-separated historical earthquake catalog.
    Unifies the multiple magnitude columns into a single 'mw_unified' column
    and renames columns to match the PINN loaders.py expectations.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw catalog from {input_file}...")

    try:
        # The file is tab-separated and contains missing values
        df = pd.read_csv(input_path, sep="\t", skipinitialspace=True)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    # Strip whitespace from column names just in case
    df.columns = df.columns.str.strip()

    print(f"Found {len(df)} initial events. Columns: {list(df.columns)}")

    # 1. Unify Magnitude
    # There are 'MI', 'mb', 'ms', 'mw'.
    # Let's take 'mw' if it exists, else 'ms', else 'mb', else 'MI'.
    # We create a new column 'mw_unified' by taking the max across the magnitude columns.
    # If they are strings with letters, we need to coerce them to numeric first.

    mag_cols = ["MI", "mb", "ms", "mw"]
    available_mags = [c for c in mag_cols if c in df.columns]

    for col in available_mags:
        # Force numeric, replacing errors with NaN
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Take the maximum available magnitude as the unified magnitude
    df["mw_unified"] = df[available_mags].max(axis=1)

    # 2. Rename Core Columns for loaders.py
    # `loaders.py` expects: 'long', 'lat', 'fd' (depth), 'mw_unified'
    rename_map = {"Lat": "lat", "Long": "long", "FD": "fd"}
    df = df.rename(columns=rename_map)

    # Clean up Depth (FD) if it has letters like '10f', '144', '33N'
    # Force to numeric
    if "fd" in df.columns:
        df["fd"] = df["fd"].astype(str).str.extract(r"(\d+\.?\d*)")[0]
        df["fd"] = pd.to_numeric(df["fd"], errors="coerce")
        # Default depth to 10.0km if missing
        df["fd"] = df["fd"].fillna(10.0)
    else:
        df["fd"] = 10.0

    # Drop rows without lat/lon or magnitude
    df_clean = df.dropna(subset=["lat", "long", "mw_unified"])

    # Format the final columns
    final_cols = ["lat", "long", "fd", "mw_unified"]
    df_final = df_clean[final_cols].copy()

    df_final.to_csv(output_path, index=False)

    print(f"✅ Saved cleaned catalog to {output_path}")
    print(f"Cleaned events: {len(df_final)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/files/historical_Eq.txt")
    parser.add_argument("--output", default="data/cleaned_historical_Eq.csv")
    args = parser.parse_args()

    preprocess_catalog(input_file=args.input, output_file=args.output)
