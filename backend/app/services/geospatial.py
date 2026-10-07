"""
Geospatial utilities for H3 spatial indexing.
"""
import h3

H3_RESOLUTION = 9  # per schema.json / spec convention


def location_to_h3_cell(lat: float, lng: float) -> str:
    """Convert latitude/longitude coordinates to an H3 index at resolution 9."""
    return h3.latlng_to_cell(lat, lng, H3_RESOLUTION)
