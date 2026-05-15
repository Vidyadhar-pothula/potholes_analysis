import os
import shutil
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOT_BROKEN_DIR = os.path.join(BASE_DIR, 'Road Classification', 'Not Broken')
IMAGES_DIR = os.path.join(BASE_DIR, 'dataset', 'images')
MASKS_DIR  = os.path.join(BASE_DIR, 'dataset', 'masks')

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)

print("--- Importing 'Not Broken' Negative Samples ---")
count = 0

for img_file in os.listdir(NOT_BROKEN_DIR):
    if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
        
    src_path = os.path.join(NOT_BROKEN_DIR, img_file)
    
    # Prefix to avoid collisions
    base_name = os.path.splitext(img_file)[0]
    new_name = f"negative_notbroken_{base_name}"
    
    dst_img_path = os.path.join(IMAGES_DIR, new_name + '.jpg')
    dst_mask_path = os.path.join(MASKS_DIR, new_name + '.png')
    
    # Read image to get dimensions and convert to standard JPEG
    img = cv2.imread(src_path)
    if img is None:
        continue
        
    h, w, _ = img.shape
    
    # Generate pure black mask (0 potholes)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Save files
    cv2.imwrite(dst_img_path, img)
    cv2.imwrite(dst_mask_path, mask)
    
    count += 1

print(f"[Success] Imported {count} negative samples into the training dataset.")
print("These will heavily penalize the model for False Positives and boost Precision.")
