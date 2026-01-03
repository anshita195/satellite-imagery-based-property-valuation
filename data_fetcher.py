import os
import time
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def fetch_satellite_image(
    lat: float,
    lon: float,
    save_path: str,
    zoom: int = 18,
    size: str = "256x256",
    api_key: Optional[str] = None,
    provider: str = "google",
) -> bool:
    """
    Download a satellite image for a single (lat, lon) pair.

    Parameters
    ----------
    lat : float
        Latitude of the property.
    lon : float
        Longitude of the property.
    save_path : str
        Full file path (including filename) where the image will be saved.
    zoom : int, default 18
        Zoom level for the satellite image.
    size : str, default "256x256"
        Image size in the format "WIDTHxHEIGHT".
    api_key : str, optional
        API key for the selected provider.
    provider : {"google", "mapbox"}
        Which static maps provider to use.

    Returns
    -------
    bool
        True if the image was downloaded successfully, else False.
    """
    if api_key is None:
        raise ValueError("api_key must be provided to fetch satellite images.")

    if provider == "google":
        # Google Maps Static API
        base_url = "https://maps.googleapis.com/maps/api/staticmap"
        params = {
            "center": f"{lat},{lon}",
            "zoom": zoom,
            "size": size,
            "maptype": "satellite",
            "key": api_key,
        }
        url = base_url
    elif provider == "mapbox":
        # Mapbox Static Images API
        # NOTE: Replace "satellite-v9" with another style if needed.
        style_id = "satellite-v9"
        # size is in format "WIDTHxHEIGHT"
        width, height = size.split("x")
        base_url = (
            f"https://api.mapbox.com/styles/v1/mapbox/{style_id}/static/"
            f"{lon},{lat},{zoom}/{width}x{height}"
        )
        params = {
            "access_token": api_key,
        }
        url = base_url
    else:
        raise ValueError("provider must be one of {'google', 'mapbox'}.")

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        with open(save_path, "wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"Failed to fetch image for ({lat}, {lon}): {e}")
        return False


def download_images_for_dataset(
    csv_path: str,
    id_column: str,
    lat_column: str = "lat",
    lon_column: str = "long",
    output_dir: str = "images",
    zoom: int = 18,
    size: str = "256x256",
    api_key: Optional[str] = None,
    provider: str = "google",
    rate_limit_sec: float = 0.2,
) -> None:
    """
    Download satellite images for all rows in a CSV dataset.

    The function expects latitude and longitude columns, and will save one image
    per row using the row's ID for the filename.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the data.
    id_column : str
        Column name to be used as the unique identifier for each property.
    lat_column : str, default "lat"
        Name of the column containing latitude values.
    lon_column : str, default "long"
        Name of the column containing longitude values.
    output_dir : str, default "images"
        Directory where images will be saved.
    zoom : int, default 18
        Zoom level for the satellite image.
    size : str, default "256x256"
        Image size in the format "WIDTHxHEIGHT".
    api_key : str, optional
        API key for the selected provider.
    provider : {"google", "mapbox"}
        Which static maps provider to use.
    rate_limit_sec : float, default 0.2
        Sleep time (in seconds) between requests to avoid hitting rate limits.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    if id_column not in df.columns:
        raise ValueError(f"id_column '{id_column}' not found in CSV.")
    if lat_column not in df.columns or lon_column not in df.columns:
        raise ValueError(f"lat_column '{lat_column}' or lon_column '{lon_column}' not found in CSV.")

    for _, row in df.iterrows():
        prop_id = row[id_column]
        lat = row[lat_column]
        lon = row[lon_column]

        filename = f"{prop_id}.png"
        save_path = os.path.join(output_dir, filename)

        if os.path.exists(save_path):
            # Skip if already downloaded
            continue

        success = fetch_satellite_image(
            lat=lat,
            lon=lon,
            save_path=save_path,
            zoom=zoom,
            size=size,
            api_key=api_key,
            provider=provider,
        )

        if success:
            print(f"Downloaded image for ID {prop_id} -> {save_path}")
        time.sleep(rate_limit_sec)


if __name__ == "__main__":
    """
    Example usage:

    1. Create a .env file in the project root with:
       MAPS_API_KEY=YOUR_API_KEY_HERE

       OR set it as an environment variable:
       - On Windows PowerShell:
         $env:MAPS_API_KEY = "YOUR_API_KEY_HERE"

    2. Run:
       python data_fetcher.py

    Adjust paths and column names below as needed.
    """
    API_KEY = os.getenv("MAPS_API_KEY", None)
    if API_KEY is None:
        raise RuntimeError(
            "Please set the MAPS_API_KEY in a .env file or as an environment variable.\n"
            "Create a .env file with: MAPS_API_KEY=YOUR_API_KEY_HERE"
        )

    # Update these paths/names based on your actual files
    TRAIN_CSV = "train(1)(train(1)).csv"
    TEST_CSV = "test2(test(1)).csv"

    # Download training images
    download_images_for_dataset(
        csv_path=TRAIN_CSV,
        id_column="id",
        lat_column="lat",
        lon_column="long",
        output_dir="images_train",
        api_key=API_KEY,
        provider="mapbox",
    )

    # Download test images
    download_images_for_dataset(
        csv_path=TEST_CSV,
        id_column="id",
        lat_column="lat",
        lon_column="long",
        output_dir="images_test",
        api_key=API_KEY,
        provider="mapbox",
    )



