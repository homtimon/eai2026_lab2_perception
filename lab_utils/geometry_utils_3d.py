"""
3D Geometry Utility Functions

This module contains pure 3D geometry and depth processing functions
that are used across multiple lab levels.
"""

import numpy as np
from typing import List


def extract_depth_region(bbox_2d: List[float], depth_image: np.ndarray, depth_scale: float = 1000.0) -> tuple:
    """Extract and validate depth values within bounding box."""
    x1, y1, x2, y2 = [int(coord) for coord in bbox_2d]
    h, w = depth_image.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w-1, x2), min(h-1, y2)
    
    depth_region = depth_image[y1:y2, x1:x2] / depth_scale
    valid_mask = (depth_region > 0.1) & (depth_region < 5.0)
    valid_depths = depth_region[valid_mask]
    
    if len(valid_depths) == 0:
        return None, {'min': 0, 'max': 0, 'mean': 0, 'valid_pixels': 0}
    
    return valid_depths, {
        'min': float(np.min(valid_depths)),
        'max': float(np.max(valid_depths)), 
        'mean': float(np.mean(valid_depths)),
        'valid_pixels': len(valid_depths)
    }


def create_3d_bbox_corners(center_3d: List[float], width_3d: float, height_3d: float, depth_3d: float) -> List[List[float]]:
    """Generate 8 corners of 3D bounding box around center."""
    half_w, half_h, half_d = width_3d/2, height_3d/2, depth_3d/2
    
    corners_relative = np.array([
        [-half_w, -half_h, -half_d],
        [+half_w, -half_h, -half_d],
        [-half_w, +half_h, -half_d],
        [+half_w, +half_h, -half_d],
        [-half_w, -half_h, +half_d],
        [+half_w, -half_h, +half_d],
        [-half_w, +half_h, +half_d],
        [+half_w, +half_h, +half_d],
    ])
    
    bbox_3d = corners_relative + np.array(center_3d)
    return bbox_3d.tolist()
