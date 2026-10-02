"""Download scripts for all datasets with caching and checksums."""

import os
import logging
from pathlib import Path
from typing import Tuple, List, Optional
import urllib.request
import hashlib

logger = logging.getLogger(__name__)

def compute_checksum(file_path: Path) -> str:
    """Compute MD5 checksum of a file."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def _download_file(url: str, out_path: Path, expected_md5: Optional[str] = None):
    if out_path.exists():
        if expected_md5:
            if compute_checksum(out_path) == expected_md5:
                logger.info(f"File {out_path} already exists and checksum matches.")
                return
        else:
            logger.info(f"File {out_path} already exists. Skipping download.")
            return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {url} to {out_path}")
    try:
        urllib.request.urlretrieve(url, out_path)
    except Exception as e:
        logger.error(f"Download failed for {url}: {e}")

def download_sen1floods11(root: str):
    """Download from GCS bucket"""
    root_path = Path(root) / "sen1floods11"
    # Placeholder URLs
    _download_file("https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand.tar.gz", root_path / "S1Hand.tar.gz")

def download_copernicus_dem(root: str, bbox: Tuple[float, float, float, float]):
    """Download GLO-30 DEM from AWS"""
    root_path = Path(root) / "copernicus_dem"
    logger.info(f"Mock downloading DEM for bbox {bbox} to {root_path}")
    root_path.mkdir(parents=True, exist_ok=True)

def download_worldcover(root: str, bbox: Tuple[float, float, float, float]):
    """Download ESA WorldCover"""
    root_path = Path(root) / "worldcover"
    logger.info(f"Mock downloading WorldCover for bbox {bbox} to {root_path}")
    root_path.mkdir(parents=True, exist_ok=True)

def download_chirps(root: str, bbox: Tuple[float, float, float, float], dates: List[str]):
    """Download CHIRPS daily rainfall (no login needed)"""
    root_path = Path(root) / "chirps"
    logger.info(f"Mock downloading CHIRPS for dates {dates} to {root_path}")
    root_path.mkdir(parents=True, exist_ok=True)

def download_worldpop(root: str, country: str):
    """Download WorldPop population grid"""
    root_path = Path(root) / "worldpop"
    url = f"https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/2020/{country}/{country}_ppp_2020_1km_Aggregated_UNadj.tif"
    _download_file(url, root_path / f"{country}_pop.tif")
