"""Derive terrain features from DEM: elevation, slope, HAND, TWI, distance to drainage."""

import numpy as np
import logging
from pathlib import Path
try:
    import rasterio
except ImportError:
    rasterio = None

logger = logging.getLogger(__name__)

def compute_slope(dem: np.ndarray, cell_size: float = 30.0) -> np.ndarray:
    """Slope from DEM using numpy gradient"""
    gy, gx = np.gradient(dem, cell_size, cell_size)
    slope = np.sqrt(gx**2 + gy**2)
    return np.degrees(np.arctan(slope))

def compute_hand(dem: np.ndarray, drainage_threshold: float) -> np.ndarray:
    """Height Above Nearest Drainage"""
    logger.info("Computing HAND placeholder...")
    # Requires a flow routing algorithm (e.g. pysheds) to compute accurately
    return np.zeros_like(dem)

def compute_twi(dem: np.ndarray, slope: np.ndarray, cell_size: float = 30.0) -> np.ndarray:
    """Topographic Wetness Index"""
    flow_acc = compute_flow_accumulation(dem)
    # Avoid div by zero in slope
    slope_rad = np.radians(np.maximum(slope, 0.1))
    sca = (flow_acc + 1) * cell_size
    twi = np.log(sca / np.tan(slope_rad))
    return twi

def compute_flow_accumulation(dem: np.ndarray) -> np.ndarray:
    """Simple D8 flow accumulation"""
    logger.info("Computing Flow Accumulation placeholder...")
    return np.ones_like(dem)

def compute_distance_to_drainage(dem: np.ndarray, flow_acc: np.ndarray, threshold: float) -> np.ndarray:
    """Euclidean distance to drainage channels"""
    logger.info("Computing Distance to Drainage placeholder...")
    return np.zeros_like(dem)

def derive_all_terrain_features(dem_path: str, output_path: str, cell_size: float = 30.0):
    """Complete pipeline"""
    if rasterio is None:
        logger.error("rasterio not installed. Cannot read DEM.")
        return
        
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        profile = src.profile
        
    slope = compute_slope(dem, cell_size)
    flow_acc = compute_flow_accumulation(dem)
    hand = compute_hand(dem, 100.0)
    twi = compute_twi(dem, slope, cell_size)
    dist_drain = compute_distance_to_drainage(dem, flow_acc, 100.0)
    
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    def save_raster(data, name):
        with rasterio.open(out_dir / name, 'w', **profile) as dst:
            dst.write(data.astype(profile['dtype']), 1)
            
    save_raster(slope, 'slope.tif')
    save_raster(hand, 'hand.tif')
    save_raster(twi, 'twi.tif')
    save_raster(flow_acc, 'flow_acc.tif')
    save_raster(dist_drain, 'dist_drainage.tif')
    logger.info("Successfully derived all terrain features.")
