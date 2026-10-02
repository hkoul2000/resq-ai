"""Travel time computation from OpenStreetMap road networks."""

import logging
import networkx as nx
import osmnx as ox
from typing import List, Dict, Tuple, Any
from pathlib import Path

logger = logging.getLogger(__name__)

def compute_travel_times(zones: List[Dict[str, Any]], facilities: List[Dict[str, Any]], city_name: str) -> Dict[Tuple[str, str], float]:
    """
    Uses osmnx to download road network, computes shortest path travel times 
    between zone centroids and facility locations.
    
    zones: List of dicts with 'id', 'lat', 'lon'
    facilities: List of dicts with 'id', 'lat', 'lon'
    """
    try:
        G = ox.graph_from_place(city_name, network_type='drive', simplify=True)
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
    except Exception as e:
        logger.error(f"Failed to download graph for {city_name}: {e}")
        return {}

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
