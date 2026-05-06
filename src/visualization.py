import matplotlib
matplotlib.use("Agg")

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from feature_extraction import extract_features, apply_depth_mask

def create_final_visualization(image_filename, upload_dir, mask_dir, depth_dir, fusion_dir, features_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    orig_path = os.path.join(upload_dir, image_filename)
    mask_path = os.path.join(mask_dir, image_filename)
    depth_path = os.path.join(depth_dir, image_filename)
    fusion_path = os.path.join(fusion_dir, 'masked_depth_color.png')
    if not os.path.exists(fusion_path):
        fusion_path = os.path.join(fusion_dir, 'masked_depth.png')
        if not os.path.exists(fusion_path):
            fusion_path = os.path.join(fusion_dir, image_filename)

    orig_img = cv2.imread(orig_path)
    if orig_img is not None:
        pass
    else:
        orig_img = np.zeros((256, 256, 3), dtype=np.uint8)

    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        mask_img = np.zeros(orig_img.shape[:2], dtype=np.uint8)

    depth_img = cv2.imread(depth_path)
    if depth_img is not None:
        depth_img_rgb = cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB)
    else:
        depth_img_rgb = np.zeros(orig_img.shape[:2], dtype=np.uint8)

    fusion_img = cv2.imread(fusion_path)
    if fusion_img is not None:
        fusion_img_rgb = cv2.cvtColor(fusion_img, cv2.COLOR_BGR2RGB)
    else:
        fusion_img_rgb = np.zeros(orig_img.shape[:2], dtype=np.uint8)

    mask_bin = (mask_img > 127).astype(np.float32)
    depth_gray = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if depth_gray is None:
        depth_gray = np.zeros(orig_img.shape[:2], dtype=np.float32)
    else:
        depth_gray = depth_gray.astype(np.float32)

    pothole_depth = apply_depth_mask(depth_gray, mask_bin)

    extracted_features = extract_features(
        image_name=image_filename,
        image=orig_img,
        mask_bin=mask_bin,
        depth_gray=depth_gray,
        pothole_depth=pothole_depth,
        min_area=50
    )

    # Sort potholes by severity (High -> Medium -> Low)
    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    extracted_features.sort(key=lambda x: (severity_order.get(x['severity'], 3), -x['severity_score']))

    for feat in extracted_features:
        x, y, w, h = feat['bbox_x'], feat['bbox_y'], feat['bbox_w'], feat['bbox_h']
        pid = feat['pothole_id']
        severity = feat['severity']

        # Choose color based on severity
        if severity == "High":
            color = (0, 0, 255) # Red (BGR)
        elif severity == "Medium":
            color = (0, 165, 255) # Orange
        else:
            color = (0, 255, 0) # Green
            
        cv2.rectangle(orig_img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            orig_img,
            f"ID {pid}",
            (x, max(y - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    orig_img_bgr_with_boxes = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs[0, 0].imshow(orig_img_bgr_with_boxes)
    axs[0, 0].set_title('Original + BBoxes')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(mask_img, cmap='gray')
    axs[0, 1].set_title('Segmentation Mask')
    axs[0, 1].axis('off')

    axs[1, 0].imshow(depth_img_rgb)
    axs[1, 0].set_title('Depth Heatmap')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(fusion_img_rgb)
    axs[1, 1].set_title('Masked Depth Fusion')
    axs[1, 1].axis('off')

    plt.tight_layout()
    output_filepath = os.path.join(output_dir, image_filename)
    plt.savefig(output_filepath, bbox_inches='tight')
    plt.close()

    return {
        "num_potholes": len(extracted_features),
        "features": extracted_features
    }
