"""
fusion.py
=========
Fusion Layer: Applies binary segmentation mask onto the depth map
to produce a masked depth output — isolating depth values ONLY in
pothole regions.

Inputs:
    - Segmentation mask (grayscale binary PNG: white=pothole, black=bg)
    - Depth map (grayscale or colored heatmap PNG from MiDaS)

Outputs:
    - outputs/fusion/masked_depth.png       (grayscale)
    - outputs/fusion/masked_depth_color.png (INFERNO heatmap)
"""

import os
import cv2
import numpy as np


# =========================================================
# CORE FUSION FUNCTION
# =========================================================
def apply_mask(depth_path: str, mask_path: str, output_dir: str = None) -> str:
    """
    Applies the binary segmentation mask onto the depth map.

    Args:
        depth_path  : Path to the depth map image (grayscale or colored).
        mask_path   : Path to the segmentation mask (grayscale 0–255).
        output_dir  : Directory to save outputs. Defaults to outputs/fusion/.

    Returns:
        Path to the saved grayscale masked depth PNG.
    """
    print("\n--- Initializing Fusion Layer ---")

    # --------------------------------------------------
    # STEP 1: Load inputs
    # --------------------------------------------------
    if not os.path.exists(depth_path):
        print(f"[ERROR] Depth map not found: {depth_path}")
        return None
    if not os.path.exists(mask_path):
        print(f"[ERROR] Segmentation mask not found: {mask_path}")
        return None

    # Load depth map — keep as-is (could be grayscale or BGR colored)
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    # Load mask always as grayscale
    mask_raw  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if depth_raw is None:
        print(f"[ERROR] Could not read depth map: {depth_path}")
        return None
    if mask_raw is None:
        print(f"[ERROR] Could not read mask: {mask_path}")
        return None

    print(f"  Depth map shape : {depth_raw.shape}")
    print(f"  Mask shape      : {mask_raw.shape}")

    # --------------------------------------------------
    # STEP 2: Preprocess
    # --------------------------------------------------
    # If depth map is colored (3 channels), convert to grayscale
    if len(depth_raw.shape) == 3 and depth_raw.shape[2] == 3:
        depth_gray = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)
        print("  Depth map was colored — converted to grayscale.")
    else:
        depth_gray = depth_raw.copy()

    # Ensure both are same size — resize mask to match depth if needed
    h_d, w_d = depth_gray.shape[:2]
    h_m, w_m = mask_raw.shape[:2]
    if (h_d, w_d) != (h_m, w_m):
        print(f"  Size mismatch — resizing mask from ({w_m}x{h_m}) to ({w_d}x{h_d})")
        mask_raw = cv2.resize(mask_raw, (w_d, h_d), interpolation=cv2.INTER_NEAREST)

    # Convert mask to binary float (0.0 or 1.0)
    # Threshold at 127 to handle any anti-aliasing artifacts
    _, mask_binary = cv2.threshold(mask_raw, 127, 1, cv2.THRESH_BINARY)
    mask_binary = mask_binary.astype(np.float32)

    # --------------------------------------------------
    # STEP 3: Apply mask onto depth map
    # --------------------------------------------------
    # depth values retained ONLY where mask = 1 (pothole region)
    depth_float   = depth_gray.astype(np.float32)
    masked_depth  = depth_float * mask_binary  # element-wise multiply

    # --------------------------------------------------
    # STEP 5: Debug — print stats
    # --------------------------------------------------
    pothole_pixels = int(mask_binary.sum())
    total_pixels   = mask_binary.size

    print(f"\n  Pothole pixels  : {pothole_pixels:,} / {total_pixels:,} "
          f"({pothole_pixels/total_pixels*100:.2f}%)")

    # Edge case: empty mask
    if pothole_pixels == 0:
        print("  [WARNING] Empty mask detected — no pothole region found!")
        print("            Masked depth output will be entirely black.")

    active_depths = masked_depth[mask_binary == 1]
    if active_depths.size > 0:
        print(f"  Masked depth min: {active_depths.min():.2f}")
        print(f"  Masked depth max: {active_depths.max():.2f}")
        print(f"  Masked depth avg: {active_depths.mean():.2f}")
    else:
        print("  [WARNING] No active depth values in pothole region.")

    # --------------------------------------------------
    # STEP 4: Save outputs
    # --------------------------------------------------
    if output_dir is None:
        # Default: project_root/outputs/fusion/
        base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, 'outputs', 'fusion')
    os.makedirs(output_dir, exist_ok=True)

    # Save grayscale masked depth
    masked_depth_uint8 = np.clip(masked_depth, 0, 255).astype(np.uint8)
    gray_out_path = os.path.join(output_dir, 'masked_depth.png')
    cv2.imwrite(gray_out_path, masked_depth_uint8)
    print(f"\n  Saved grayscale : {gray_out_path}")

    # Save INFERNO colormap version
    colored = cv2.applyColorMap(masked_depth_uint8, cv2.COLORMAP_INFERNO)
    # Zero out background pixels in the colored version too
    mask_3ch = np.stack([mask_binary, mask_binary, mask_binary], axis=-1)
    colored  = (colored * mask_3ch).astype(np.uint8)
    color_out_path = os.path.join(output_dir, 'masked_depth_color.png')
    cv2.imwrite(color_out_path, colored)
    print(f"  Saved colormap  : {color_out_path}")

    print("\n[Success] Fusion Layer Complete!")
    return gray_out_path


# =========================================================
# STANDALONE TEST
# =========================================================
if __name__ == '__main__':
    import sys
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Use command-line args or fall back to defaults for quick testing
    depth_p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base, 'app', 'static', 'depth',  'test.jpg')
    mask_p  = sys.argv[2] if len(sys.argv) > 2 else os.path.join(base, 'app', 'static', 'masks',  'test.jpg')

    apply_mask(depth_path=depth_p, mask_path=mask_p)
