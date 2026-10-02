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

def compute_flow_accumulation(dem: np.ndarray) -> np.ndarray:
    """Compute D8 flow accumulation using pure NumPy topological sort.

    Finds the steepest downslope neighbor for each pixel and accumulates flow.
    """
    from scipy import ndimage

    H, W = dem.shape
    acc = np.ones((H, W), dtype=np.float32)

    # 8 neighbor offsets: (dy, dx, distance)
    neighbors = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)),
        (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))
    ]

    # Compute steepest descent receiver for each cell
    receiver_r = np.full((H, W), -1, dtype=np.int32)
    receiver_c = np.full((H, W), -1, dtype=np.int32)
    max_drop = np.zeros((H, W), dtype=np.float32)

    for dy, dx, dist in neighbors:
        # Shifted neighbor elevations
        r_slice = slice(max(0, dy), min(H, H + dy))
        c_slice = slice(max(0, dx), min(W, W + dx))
        orig_r = slice(max(0, -dy), min(H, H - dy))
        orig_c = slice(max(0, -dx), min(W, W - dx))

        drop = (dem[orig_r, orig_c] - dem[r_slice, c_slice]) / dist
        better = drop > max_drop[orig_r, orig_c]
        if np.any(better):
            idx_r = np.arange(H)[orig_r][:, None]
            idx_c = np.arange(W)[orig_c][None, :]
            receiver_r[orig_r, orig_c] = np.where(better, idx_r + dy, receiver_r[orig_r, orig_c])
            receiver_c[orig_r, orig_c] = np.where(better, idx_c + dx, receiver_c[orig_r, orig_c])
            max_drop[orig_r, orig_c] = np.maximum(max_drop[orig_r, orig_c], drop)

    # Topological order: sort by descending elevation to route flow
    flat_indices = np.argsort(-dem.ravel())
    coords = np.unravel_index(flat_indices, (H, W))

    for r, c in zip(coords[0], coords[1]):
        rr = receiver_r[r, c]
        cc = receiver_c[r, c]
        if 0 <= rr < H and 0 <= cc < W:
            acc[rr, cc] += acc[r, c]

    return acc

def compute_hand(dem: np.ndarray, drainage_threshold: float = 100.0) -> np.ndarray:
    """Height Above Nearest Drainage (HAND) using D8 drainage network."""
    from scipy import ndimage

    acc = compute_flow_accumulation(dem)
    is_channel = acc >= drainage_threshold

    if not np.any(is_channel):
        # Fallback: minimum 10% highest accumulation cells
        thresh = np.percentile(acc, 95)
        is_channel = acc >= thresh

    # Channel elevation grid with inf elsewhere
    channel_elev = np.where(is_channel, dem, np.nan)

    # Nearest channel elevation via distance transform indices
    indices = ndimage.distance_transform_edt(~is_channel, return_indices=True)[1]
    nearest_elev = dem[indices[0], indices[1]]

    hand = np.maximum(0.0, dem - nearest_elev)
    return hand.astype(np.float32)

def compute_distance_to_drainage(dem: np.ndarray, flow_acc: np.ndarray, threshold: float = 100.0, cell_size: float = 30.0) -> np.ndarray:
    """Euclidean distance to nearest drainage channel in meters."""
    from scipy import ndimage

    is_channel = flow_acc >= threshold
    if not np.any(is_channel):
        thresh = np.percentile(flow_acc, 95)
        is_channel = flow_acc >= thresh

    dist_cells = ndimage.distance_transform_edt(~is_channel)
    return (dist_cells * cell_size).astype(np.float32)

def compute_twi(dem: np.ndarray, slope: np.ndarray, cell_size: float = 30.0) -> np.ndarray:
    """Topographic Wetness Index (TWI = ln(a / tan(beta)))."""
    flow_acc = compute_flow_accumulation(dem)
    slope_rad = np.radians(np.maximum(slope, 0.1))
    sca = (flow_acc + 1.0) * cell_size
    twi = np.log(np.maximum(sca / np.tan(slope_rad), 1e-4))
    return twi.astype(np.float32)

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
