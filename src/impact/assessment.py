"""Monte Carlo impact assessment from flood prediction distributions.

Samples N flood maps from the predictive distribution, overlays with
exposure data (population, buildings) to estimate affected population
and infrastructure damage distributions.
"""

import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ImpactDistribution:
    mean: float
    variance: float
    p05: float
    p50: float
    p95: float

def sample_flood_maps(probs: np.ndarray, uncertainties: np.ndarray = None, num_samples: int = 200) -> np.ndarray:
    """
    Sample binary flood maps from the predictive distribution.
    probs: (H, W) array of flood probabilities [0, 1]
    returns: (num_samples, H, W) binary maps
    """
    np.random.seed(42)
    H, W = probs.shape
    samples = np.random.binomial(1, p=probs, size=(num_samples, H, W))
    return samples

def compute_impact(flood_maps: np.ndarray, population_grid: np.ndarray, building_footprints: np.ndarray, zones: np.ndarray) -> Dict[str, Dict[str, ImpactDistribution]]:
    """
    For each sampled flood map, compute affected population and exposed buildings per zone.
    flood_maps: (N, H, W) binary
    population_grid: (H, W) population count per pixel
    building_footprints: (H, W) binary building presence
    zones: (H, W) integer zone IDs
    """
    N, H, W = flood_maps.shape
    unique_zones = np.unique(zones)
    
    results = {}
    
    for z in unique_zones:
        if z < 0:  # ignore no-data zones
            continue
            
        z_mask = (zones == z)
        
        pop_samples = []
        bldg_samples = []
        
        for i in range(N):
            flood = flood_maps[i]
            affected_pixels = flood * z_mask
            
            pop = np.sum(population_grid * affected_pixels)
            bldgs = np.sum(building_footprints * affected_pixels)
            
            pop_samples.append(pop)
            bldg_samples.append(bldgs)
            
        pop_arr = np.array(pop_samples)
        bldg_arr = np.array(bldg_samples)
        
        results[str(z)] = {
            'population': ImpactDistribution(
                mean=float(np.mean(pop_arr)),
                variance=float(np.var(pop_arr)),
                p05=float(np.percentile(pop_arr, 5)),
                p50=float(np.percentile(pop_arr, 50)),
                p95=float(np.percentile(pop_arr, 95))
            ),
            'buildings': ImpactDistribution(
                mean=float(np.mean(bldg_arr)),
                variance=float(np.var(bldg_arr)),
                p05=float(np.percentile(bldg_arr, 5)),
                p50=float(np.percentile(bldg_arr, 50)),
                p95=float(np.percentile(bldg_arr, 95))
            )
        }
        
    return results

def aggregate_to_zones(raster: np.ndarray, zones_gdf: Any = None, affine_transform: Any = None) -> Dict[str, float]:
    """
    Spatial aggregation of raster values to vector zones or spatial grid cells.
    Supports GeoDataFrame via rasterstats if available, or grid partition aggregation.
    """
    try:
        from rasterstats import zonal_stats
        if zones_gdf is not None and affine_transform is not None:
            stats = zonal_stats(zones_gdf, raster, affine=affine_transform, stats=['mean', 'sum'])
            return {f"zone_{i+1}": float(s.get('sum', 0.0) or 0.0) for i, s in enumerate(stats)}
    except (ImportError, Exception) as e:
        logger.debug(f"Vector zonal stats skipped ({e}), using grid partition aggregation.")
    
    # Grid partition fallback: divide 2D raster into 4 spatial quadrants/zones
    r = np.asarray(raster)
    if r.ndim > 2:
        r = r.squeeze()
    h, w = r.shape[-2:]
    mid_h, mid_w = max(1, h // 2), max(1, w // 2)
    zones = {
        "zone_nw": float(np.sum(r[:mid_h, :mid_w])),
        "zone_ne": float(np.sum(r[:mid_h, mid_w:])),
        "zone_sw": float(np.sum(r[mid_h:, :mid_w])),
        "zone_se": float(np.sum(r[mid_h:, mid_w:])),
    }
    return zones
