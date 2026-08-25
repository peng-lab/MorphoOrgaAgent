# organoid_agent/tools/morphology_tools.py

import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional


def _get_contours_for_all_objects(mask: np.ndarray) -> List[Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]]:
    """
    Internal helper function to process instance segmentation label masks.
    Background is expected to be 0, while each distinct organoid object is assigned a unique positive integer ID (1, 2, 3... N).
    
    Returns a sorted list of tuples: (object_id, outer_contour, extra_contour_data_tree)
    ordered sequentially by object_id from 1 to N.
    """
    # Dynamically extract all unique object IDs present in the label mask, excluding background (0)
    object_ids = sorted([int(x) for x in np.unique(mask) if x > 0])
    
    cached_contours = []
    
    for obj_id in object_ids:
        # 1. Isolate the target organoid object into a temporary binary slice
        obj_slice = (mask == obj_id).astype(np.uint8) * 255
        
        # 2. Extract contour data hierarchy tree using 2-level component mode (RETR_CCOMP)
        contours, hierarchy = cv2.findContours(obj_slice, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        
        if hierarchy is None or len(contours) == 0:
            cached_contours.append((obj_id, None, None))
            continue
            
        hierarchy = hierarchy[0]
        
        # Find the main exterior contour for the current object (where parent index is -1)
        outer_cnt = None
        outer_idx = -1
        for idx, h in enumerate(hierarchy):
            if h[3] == -1:  # Parent index == -1 signifies the topmost external boundary
                outer_cnt = contours[idx]
                outer_idx = idx
                break
                
        if outer_cnt is None:
            # Fallback mechanism: if no explicit root node is tracked, default to the first contour slice
            outer_cnt = contours[0]
            outer_idx = 0
            
        cached_contours.append((obj_id, outer_cnt, (contours, hierarchy, outer_idx)))
        
    return cached_contours


# =========================================================================
# Phase 1: Atomic Pixel-Level Metric Extraction (Returns Sequential Lists)
# =========================================================================

def get_organoid_ids(mask: np.ndarray) -> List[int]:
    """
    Extracts all unique tracked organoid target label IDs from the mask [1, 2, 3... N].
    """
    return sorted([int(x) for x in np.unique(mask) if x > 0])


def calculate_area_outer(mask: np.ndarray) -> List[float]:
    """
    Calculates the raw pixel area contained within the external boundary (Area_outer) 
    for each detected organoid sequentially.
    """
    results = []
    for _, cnt, _ in _get_contours_for_all_objects(mask):
        if cnt is not None:
            results.append(float(cv2.contourArea(cnt)))
        else:
            results.append(0.0)
    return results


def calculate_perimeter_outer(mask: np.ndarray) -> List[float]:
    """
    Calculates the spatial curve length of the external boundary outline (Perimeter_outer) 
    for each detected organoid sequentially.
    """
    results = []
    for _, cnt, _ in _get_contours_for_all_objects(mask):
        if cnt is not None:
            results.append(float(cv2.arcLength(cnt, True)))
        else:
            results.append(0.0)
    return results


def calculate_inner_hole_metrics(mask: np.ndarray) -> Tuple[List[float], List[float]]:
    """
    Traverses the topological component tree hierarchy to compute the cumulative area 
    and total perimeter length of all inner cavities/lumens nested within each organoid shape.
    
    Returns:
        Tuple(areas_inner_list, perimeters_inner_list) mapped sequentially by object ID.
    """
    areas_inner = []
    perimeters_inner = []
    
    for _, cnt, extra in _get_contours_for_all_objects(mask):
        if cnt is None or extra is None:
            areas_inner.append(0.0)
            perimeters_inner.append(0.0)
            continue
            
        contours, hierarchy, outer_idx = extra
        
        # Navigate to the first embedded sub-cavity boundary (First Child)
        hole_index = hierarchy[outer_idx][2]
        cum_hole_area = 0.0
        cum_hole_perimeter = 0.0
        
        # Traverse the horizontal hole chain link list using the [Next] pointer attribute
        while hole_index != -1:
            cum_hole_area += cv2.contourArea(contours[hole_index])
            cum_hole_perimeter += cv2.arcLength(contours[hole_index], True)
            hole_index = hierarchy[hole_index][0]  # Move seamlessly to the adjacent sibling lumen
            
        areas_inner.append(float(cum_hole_area))
        perimeters_inner.append(float(cum_hole_perimeter))
        
    return areas_inner, perimeters_inner


def calculate_centroids(mask: np.ndarray) -> Tuple[List[float], List[float]]:
    """
    Extracts the geometric spatial centroids (Center of Mass) using image moments equations.
    
    Returns:
        Tuple(x_coordinates_list, y_coordinates_list) mapped sequentially by object ID.
    """
    x_coords = []
    y_coords = []
    
    for _, cnt, _ in _get_contours_for_all_objects(mask):
        if cnt is None:
            x_coords.append(0.0)
            y_coords.append(0.0)
            continue
            
        m = cv2.moments(cnt)
        if m['m00'] != 0:
            x_coords.append(float(m['m10'] / m['m00']))
            y_coords.append(float(m['m01'] / m['m00']))
        else:
            x_coords.append(0.0)
            y_coords.append(0.0)
            
    return x_coords, y_coords


def check_is_border(mask: np.ndarray) -> List[bool]:
    """
    Determines whether any segment node of the organoid body touches the physical matrix border 
    of the microscopy capture slice. Used to filter out incomplete/truncated shapes downstream.
    """
    results = []
    h_max, w_max = mask.shape[:2]
    
    for _, cnt, _ in _get_contours_for_all_objects(mask):
        if cnt is None:
            results.append(False)
            continue
            
        is_border = (np.any(cnt[:, :, 0] == 0) or np.any(cnt[:, :, 0] == w_max - 1) or 
                     np.any(cnt[:, :, 1] == 0) or np.any(cnt[:, :, 1] == h_max - 1))
        results.append(bool(is_border))
        
    return results


# =========================================================================
# Phase 2: High-Level Morphological Derivations (Supports Dependency Injection)
# =========================================================================

def calculate_ideal_perimeter(mask: np.ndarray, area_outer: Optional[List[float]] = None) -> List[float]:
    """
    Calculates the theoretical ideal circle perimeter given the outer surface area of the organoid.
    Supports dependency injection of pre-calculated 'area_outer' list to save CPU cycles.
    """
    if area_outer is None:
        area_outer = calculate_area_outer(mask)
        
    return [float(2 * np.sqrt(np.pi * val)) for val in area_outer]


def calculate_perimeter_diff(mask: np.ndarray, 
                             perimeter_outer: Optional[List[float]] = None, 
                             ideal_perimeter: Optional[List[float]] = None) -> List[float]:
    """
    Calculates the raw morphological discrepancy between the actual external outline 
    and the theoretical perfect circle perimeter (perimeter_diff). Higher values indicate higher rugosity.
    """
    if perimeter_outer is None:
        perimeter_outer = calculate_perimeter_outer(mask)
    if ideal_perimeter is None:
        ideal_perimeter = calculate_ideal_perimeter(mask)
        
    return [float(p - idl) for p, idl in zip(perimeter_outer, ideal_perimeter)]


def calculate_roundness(mask: np.ndarray, 
                        area_outer: Optional[List[float]] = None, 
                        perimeter_outer: Optional[List[float]] = None) -> List[float]:
    """
    Calculates the isoperimetric circularity quotient (roundness value scale 0.0 to 1.0) 
    for each mapped organoid body.
    """
    if area_outer is None:
        area_outer = calculate_area_outer(mask)
    if perimeter_outer is None:
        perimeter_outer = calculate_perimeter_outer(mask)
        
    results = []
    for a, p in zip(area_outer, perimeter_outer):
        if p > 0:
            results.append(float((4 * np.pi * a) / (p ** 2)))
        else:
            results.append(0.0)
    return results


def calculate_lumen_ratio(mask: np.ndarray, 
                          area_outer: Optional[List[float]] = None, 
                          area_inner: Optional[List[float]] = None) -> List[float]:
    """
    Calculates the ratio of internal cavity hollow spaces relative to the overall outer dimensions 
    (lumen_ratio). Vital parameter for measuring differentiation and tracking internal cavitation events.
    """
    if area_outer is None:
        area_outer = calculate_area_outer(mask)
    if area_inner is None:
        area_inner, _ = calculate_inner_hole_metrics(mask)
        
    return [float(inner / (outer + 1e-5)) for outer, inner in zip(area_outer, area_inner)]


# =========================================================================
# Phase 3: Spatial Metrology & Physical Unit Conversions
# =========================================================================

def calculate_area_outer_mm(mask: np.ndarray, 
                            area_outer: Optional[List[float]] = None, 
                            pixel_size_um: float = 2.0) -> List[float]:
    """
    Converts dimensions dynamically from digital pixel spaces into tangible metric parameters (mm^2).
    
    Args:
        mask: The labelled mask ndarray matrix.
        area_outer: Pre-extracted raw pixel area list.
        pixel_size_um: Linear physical metric size calibration index (defaults to 2.0 um/pixel).
    """
    if area_outer is None:
        area_outer = calculate_area_outer(mask)
        
    MICRON_TO_MM = 1_000_000
    pixel_area_factor = pixel_size_um * pixel_size_um  # Map quadratic micron area per digital node
    
    return [float((px_area * pixel_area_factor) / MICRON_TO_MM) for px_area in area_outer]


def calculate_metrics_from_perimeter(mask: np.ndarray, perimeter_outer: Optional[List[float]] = None) -> Tuple[List[float], List[float]]:
    """
    Derives comparative morphological baseline dimensions reversed from the primary perimeter parameter.
    
    Returns:
        Tuple(radius_from_peri_list, area_from_peri_list) mapped sequentially by object ID.
    """
    if perimeter_outer is None:
        perimeter_outer = calculate_perimeter_outer(mask)
        
    radii = [float(p / (np.pi * 2)) for p in perimeter_outer]
    areas = [float((r ** 2) * np.pi) for r in radii]
    return radii, areas