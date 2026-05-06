"""
Merges the two new dataset folders into the existing dataset/images and dataset/masks directories.

Strategy:
- my dataset/train/Pothole/  -> full-white masks (entire image is pothole)
- my dataset/train/Plain/    -> full-black masks (no pothole, negative samples)
- pothole_image_data/        -> full-white masks (entire image is pothole)

Images are copied and renamed with a prefix to avoid filename collisions.
"""

import os
import shutil
from PIL import Image
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, 'dataset', 'images')
MASKS_DIR  = os.path.join(BASE_DIR, 'dataset', 'masks')

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')

def generate_mask(img_path, mask_path, is_pothole):
    """Generate a solid white (pothole) or solid black (plain) mask matching the image size."""
    try:
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        fill = 255 if is_pothole else 0
        mask = Image.fromarray(np.full((h, w), fill, dtype=np.uint8), mode='L')
        mask.save(mask_path)
        return True
    except Exception as e:
        print(f"  [WARN] Skipping {img_path}: {e}")
        return False

def merge_folder(src_folder, prefix, is_pothole):
    count = 0
    for fname in os.listdir(src_folder):
        if not any(fname.lower().endswith(ext.lower()) for ext in IMG_EXTENSIONS):
            continue
        src_path = os.path.join(src_folder, fname)

        # Ensure unique name using prefix
        base, _ = os.path.splitext(fname)
        new_name = f"{prefix}_{base}.jpg"
        dst_img  = os.path.join(IMAGES_DIR, new_name)
        dst_mask = os.path.join(MASKS_DIR,  new_name.replace('.jpg', '.png'))

        # Skip if already merged
        if os.path.exists(dst_img):
            continue

        # Copy and convert image to JPEG
        try:
            img = Image.open(src_path).convert('RGB')
            img.save(dst_img, 'JPEG', quality=90)
        except Exception as e:
            print(f"  [WARN] Could not copy {fname}: {e}")
            continue

        ok = generate_mask(dst_img, dst_mask, is_pothole)
        if ok:
            count += 1

    return count

print("=" * 55)
print("   Merging New Datasets into Training Pool")
print("=" * 55)

# 1. my dataset -> Pothole images (full-white masks)
pothole_src = os.path.join(BASE_DIR, 'dataset', 'my dataset', 'train', 'Pothole')
n1 = merge_folder(pothole_src, 'mydataset_pothole', is_pothole=True)
print(f"[1] my dataset / Pothole  -> {n1} images added")

# 2. my dataset -> Plain images (full-black masks)
plain_src = os.path.join(BASE_DIR, 'dataset', 'my dataset', 'train', 'Plain')
n2 = merge_folder(plain_src, 'mydataset_plain', is_pothole=False)
print(f"[2] my dataset / Plain    -> {n2} images added")

# 3. pothole_image_data (full-white masks)
potholedata_src = os.path.join(BASE_DIR, 'dataset', 'pothole_image_data', 'Pothole_Image_Data')
n3 = merge_folder(potholedata_src, 'potholedata', is_pothole=True)
print(f"[3] pothole_image_data    -> {n3} images added")

total = n1 + n2 + n3
total_images = len([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
print(f"\n  New images added  : {total}")
print(f"  Total in dataset  : {total_images}")
print("\n[Done] Dataset merge complete. Ready to retrain!")
