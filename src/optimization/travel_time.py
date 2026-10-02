"""Travel time computation from OpenStreetMap road networks."""

import logging
import math
from typing import List, Dict, Tuple, Any
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import networkx as nx
except ImportError:
    nx = None

try:
    import osmnx as ox
    HAS_OSMNX = True
except (ImportError, Exception):
    ox = None
    HAS_OSMNX = False

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance between two points in km."""
    r = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c

def _haversine_travel_times(zones: List[Dict[str, Any]], facilities: List[Dict[str, Any]]) -> Dict[Tuple[str, str], float]:
    travel_times = {}
    for z in zones:
        for f in facilities:
            dist = haversine_distance_km(z.get('lat', 0.0), z.get('lon', 0.0), f.get('lat', 0.0), f.get('lon', 0.0))
            # Disaster travel speed ~30 km/h -> 0.5 km/min, minimum 2 minutes
            tt_min = max(2.0, dist / 0.5)
            travel_times[(z['id'], f['id'])] = round(tt_min, 2)
    return travel_times

def compute_travel_times(zones: List[Dict[str, Any]], facilities: List[Dict[str, Any]], city_name: str) -> Dict[Tuple[str, str], float]:
    """
    Uses osmnx to download road network, computes shortest path travel times 
    between zone centroids and facility locations.
    Falls back to Haversine distance-based travel time (assuming 30 km/h) if OSM is unavailable.
    
    zones: List of dicts with 'id', 'lat', 'lon'
    facilities: List of dicts with 'id', 'lat', 'lon'
    """
    if not HAS_OSMNX:
        logger.info(f"osmnx not installed. Using Haversine distance fallback for {city_name}.")
        return _haversine_travel_times(zones, facilities)

    try:
        G = ox.graph_from_place(city_name, network_type='drive', simplify=True)
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
    except Exception as e:
        logger.warning(f"Failed to download OSM graph for {city_name} ({e}). Using Haversine distance fallback.")
        return _haversine_travel_times(zones, facilities)

    travel_times = {}
    
    for z in zones:
        for f in facilities:
            try:
                orig_node = ox.nearest_nodes(G, z['lon'], z['lat'])
                dest_node = ox.nearest_nodes(G, f['lon'], f['lat'])
                
                tt = nx.shortest_path_length(G, orig_node, dest_node, weight='travel_time')
                # convert seconds to minutes
                travel_times[(z['id'], f['id'])] = tt / 60.0
            except nx.NetworkXNoPath:
                travel_times[(z['id'], f['id'])] = 9999.0
            except Exception as e:
                logger.error(f"Error computing path from {z['id']} to {f['id']}: {e}")
                travel_times[(z['id'], f['id'])] = 9999.0
                
    return travel_times

def get_osm_facilities(bbox: Tuple[float, float, float, float], amenity_types: List[str]) -> List[Dict[str, Any]]:
    """
    Query OSM for facilities (hospitals, shelters, fire stations) within bbox.
    bbox: (west, south, east, north)
    """
    if not HAS_OSMNX:
        logger.warning("osmnx not installed, cannot fetch OSM facilities.")
        return []

    west, south, east, north = bbox
    tags = {'amenity': amenity_types}
    try:
        gdf = ox.geometries_from_bbox(north, south, east, west, tags)
        
        facilities = []
        for idx, row in gdf.iterrows():
            if row.geometry.geom_type == 'Point':
                lat, lon = row.geometry.y, row.geometry.x
            else:
                centroid = row.geometry.centroid
                lat, lon = centroid.y, centroid.x
                
            facilities.append({
                'id': str(idx),
                'type': row.get('amenity', 'unknown'),
                'lat': lat,
                'lon': lon
            })
        return facilities
    except Exception as e:
        logger.error(f"Failed to fetch amenities from OSM: {e}")
        return []
