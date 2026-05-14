"""
feature_extraction.py
=====================
Fusion Feature Extraction Layer.
"""

import os
import cv2
import csv
import numpy as np
from geometry_engine import GeometryEngine
from physics_severity import PhysicsSeverityEngine

def load_triplet(image_path: str, mask_path: str, depth_path: str):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Original image not found: {image_path}")

    mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        raise FileNotFoundError(f"Segmentation mask not found: {mask_path}")

    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Depth map not found: {depth_path}")

    if len(depth_raw.shape) == 3 and depth_raw.shape[2] == 3:
        depth_gray = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        depth_gray = depth_raw.astype(np.float32)

    h, w = image.shape[:2]
    if mask_raw.shape[:2] != (h, w):
        mask_raw = cv2.resize(mask_raw, (w, h), interpolation=cv2.INTER_NEAREST)
    if depth_gray.shape[:2] != (h, w):
        depth_gray = cv2.resize(depth_gray, (w, h), interpolation=cv2.INTER_LINEAR)

    _, mask_thresh = cv2.threshold(mask_raw, 127, 255, cv2.THRESH_BINARY)
    mask_bin = (mask_thresh / 255.0).astype(np.float32)
    return image, mask_bin, depth_gray

def apply_depth_mask(depth_gray: np.ndarray, mask_bin: np.ndarray) -> np.ndarray:
    return depth_gray * mask_bin

def extract_features(
    image_name: str,
    image:      np.ndarray,
    mask_bin:   np.ndarray,
    depth_gray: np.ndarray,
    pothole_depth: np.ndarray,
    min_area:   int = 50,
) -> list:
    mask = (mask_bin > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = []
    pothole_id = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        pothole_id += 1
        x, y, w, h = cv2.boundingRect(cnt)

        # Draw contour on blank mask to get depth correctly
        region_mask = np.zeros_like(mask)
        cv2.drawContours(region_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        region_depths = pothole_depth[region_mask == 255]
        valid_depths = region_depths[region_depths > 0]

        if valid_depths.size > 0:
            avg_depth = float(np.mean(valid_depths))
        else:
            avg_depth = 0.0

        # PHASES 1, 2, 3 logic:
        # Phase 1: Real-world area
        area_m2 = round(area * 0.0005, 2)

        # New Physics-Based Integration
        geom_engine = GeometryEngine()
        phys_engine = PhysicsSeverityEngine()

        # Try to load metric depth if available, otherwise use proxy
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_name = os.path.splitext(image_name)[0]
        metric_path = os.path.join(base_dir, 'outputs', 'metric_depth', f"{base_name}_metric.npy")
        
        if os.path.exists(metric_path):
            metric_depth_map = np.load(metric_path)
        else:
            # Proxy metric depth from grayscale
            metric_depth_map = (pothole_depth / 255.0) * 25.0

        volume_cm3 = geom_engine.compute_volume(metric_depth_map, region_mask / 255.0)
        edge_sharpness = geom_engine.compute_edge_sharpness(region_mask / 255.0, metric_depth_map)
        
        max_depth_cm = float(np.max(metric_depth_map[region_mask == 255])) if np.any(region_mask == 255) else 0.0

        # Phase 3: Severity Logic
        priority_score, severity = phys_engine.calculate_severity(area_m2, max_depth_cm, volume_cm3, edge_sharpness)

        # Legacy variables for compatibility
        norm_depth = avg_depth / 255.0
        depth_cat = "Shallow" if norm_depth < 0.3 else ("Moderate" if norm_depth <= 0.6 else "Deep")
        depth_cm = f"~{max_depth_cm:.1f} cm"

        # Phase 7: Debug Print
        print(f"ID {pothole_id} -> area: {area_m2}m2, depth: {max_depth_cm:.1f}cm, vol: {volume_cm3:.1f}cm3, priority: {priority_score}, severity: {severity}")


        features.append({
            'pothole_id': pothole_id,
            'area_px': area,
            'area_m2': area_m2,
            'avg_depth': avg_depth,
            'norm_depth': round(norm_depth, 2),
            'depth_cat': depth_cat,
            'depth_cm': depth_cm,
            'volume_cm3': round(volume_cm3, 1),
            'edge_sharpness': round(edge_sharpness, 2),
            'priority_score': priority_score,
            'severity': severity,
            'severity_score': priority_score, # For legacy UI compat
            'bbox_x': x,
            'bbox_y': y,
            'bbox_w': w,
            'bbox_h': h
        })

    if len(features) > 0:
        print("potholes detected")
    else:
        print("no potholes")

    return features

def save_features_csv(all_features: list, output_path: str):
    if not all_features:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = list(all_features[0].keys())
    write_header = not os.path.exists(output_path)
    with open(output_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(all_features)
