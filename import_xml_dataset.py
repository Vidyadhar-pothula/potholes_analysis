import os
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.join(BASE_DIR, 'dataset_1', 'archive-2')
IMAGES_DIR = os.path.join(BASE_DIR, 'dataset', 'images')
MASKS_DIR  = os.path.join(BASE_DIR, 'dataset', 'masks')

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)

ann_dir = os.path.join(ARCHIVE_DIR, 'annotations')
img_dir = os.path.join(ARCHIVE_DIR, 'images')

print("--- Importing Archive 2 XML Dataset ---")
count = 0

for xml_file in os.listdir(ann_dir):
    if not xml_file.endswith('.xml'): continue
    
    xml_path = os.path.join(ann_dir, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    filename = root.find('filename').text
    img_path = os.path.join(img_dir, filename)
    
    if not os.path.exists(img_path):
        # Sometime extension mismatches happen in VOC
        base = os.path.splitext(filename)[0]
        img_path = os.path.join(img_dir, base + '.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(img_dir, base + '.png')
            if not os.path.exists(img_path):
                print(f"[WARN] Image not found for {xml_file}")
                continue

    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)
    
    # Create black mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    has_pothole = False
    for obj in root.findall('object'):
        name = obj.find('name').text.lower()
        if 'pothole' in name:
            has_pothole = True
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            # Draw solid white rectangle for the pothole area
            cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, -1)
            
    if has_pothole:
        # Save mask and copy image
        prefix_name = f"archive2_{os.path.splitext(filename)[0]}"
        dst_img_path = os.path.join(IMAGES_DIR, prefix_name + '.jpg')
        dst_mask_path = os.path.join(MASKS_DIR, prefix_name + '.png')
        
        # Save image as JPEG
        img = cv2.imread(img_path)
        cv2.imwrite(dst_img_path, img)
        
        # Save mask
        cv2.imwrite(dst_mask_path, mask)
        count += 1

print(f"\n[Success] Imported {count} new images with bounding-box pseudo-masks into the training pool.")
