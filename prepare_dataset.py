import os
import glob
import cv2
import numpy as np
import xml.etree.ElementTree as ET

images_dir = "dataset/images"
annotations_dir = "dataset/annotations"
out_masks_dir = "dataset/masks"

os.makedirs(out_masks_dir, exist_ok=True)

xml_files = sorted(glob.glob(os.path.join(annotations_dir, "*.xml")))
print(f"Found {len(xml_files)} XML annotation files.")

count = 0
for xml_path in xml_files:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        base = os.path.splitext(os.path.basename(xml_path))[0]
        
        img_path = os.path.join(images_dir, base + ".png")
        if not os.path.exists(img_path):
            img_path = os.path.join(images_dir, base + ".jpg")
            
        if not os.path.exists(img_path):
            print(f"Image not found for {base}")
            continue
            
        # Fallback to cv2 image shape if XML size is missing/incorrect
        img = cv2.imread(img_path)
        if img is None:
            continue
        height, width = img.shape[:2]
        
        # Create blank mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Draw bounding boxes
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            if bndbox is not None:
                xmin = int(float(bndbox.find("xmin").text))
                ymin = int(float(bndbox.find("ymin").text))
                xmax = int(float(bndbox.find("xmax").text))
                ymax = int(float(bndbox.find("ymax").text))
                
                # Draw filled white rectangle
                cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, -1)
            
        # Save as PNG
        cv2.imwrite(os.path.join(out_masks_dir, base + ".png"), mask)
        count += 1
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")

print(f"Dataset prepared successfully! Generated {count} masks.")
