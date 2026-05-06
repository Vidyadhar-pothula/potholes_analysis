"""
feature_extraction.py
=====================
Fusion Feature Extraction Layer.
"""

import os
import cv2
import csv
import numpy as np

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

        # Phase 2: Relative Depth
        norm_depth = avg_depth / 255.0
        
        if norm_depth < 0.3:
            depth_cat = "Shallow"
            depth_cm = "~3 cm"
        elif norm_depth <= 0.6:
            depth_cat = "Moderate"
            depth_cm = "~8 cm"
        else:
            depth_cat = "Deep"
            depth_cm = "~18 cm"

        # Phase 3: Severity Logic
        severity_score = area_m2 * norm_depth
        if severity_score < 0.2:
            severity = "Low"
        elif severity_score <= 0.6:
            severity = "Medium"
        else:
            severity = "High"

        # Phase 7: Debug Print
        print(f"ID {pothole_id} -> pixel_area: {area}, normalized_depth: {norm_depth:.2f}, computed area_m2: {area_m2}, severity_score: {severity_score:.2f}")

        features.append({
            'pothole_id': pothole_id,
            'area_px': area,
            'area_m2': area_m2,
            'avg_depth': avg_depth,
            'norm_depth': round(norm_depth, 2),
            'depth_cat': depth_cat,
            'depth_cm': depth_cm,
            'severity': severity,
            'severity_score': round(severity_score, 2),
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
