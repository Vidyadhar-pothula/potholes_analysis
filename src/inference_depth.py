import os
import cv2
import torch
import numpy as np

def generate_depth_maps(image_dir, output_dir):
    print("--- Initializing MiDaS Depth Estimation ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Target: {device}")
    
    model_type = "MiDaS_small"
    print(f"Retrieving {model_type} from torch.hub (This may take a moment on first run)...")
    
    try:
        midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        midas.to(device)
        midas.eval()
        
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        transform = midas_transforms.small_transform
    except Exception as e:
        print(f"[ERROR] Failed to load MiDaS from torch hub. Details: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(image_dir):
        print(f"[ERROR] Directory not found: {image_dir}")
        return

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(image_files) == 0:
        print(f"No valid images found in {image_dir}.")
        return

    print(f"Found {len(image_files)} images. Beginning depth mapping...\n")

    with torch.no_grad():
        for filename in image_files:
            img_path = os.path.join(image_dir, filename)
            
            img = cv2.imread(img_path)
            if img is None:
                print(f"[Warning] Could not read {filename}, skipping.")
                continue
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_batch = transform(img_rgb).to(device)
            
            prediction = midas(input_batch)
            
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
            output_tensor = prediction.cpu().numpy()
            
            print(f"[{filename}] Raw depth min: {output_tensor.min():.4f}, max: {output_tensor.max():.4f}")
            
            # Normalize Depth correctly using cv2.normalize
            normalized_depth = np.zeros_like(output_tensor, dtype=np.uint8)
            cv2.normalize(output_tensor, normalized_depth, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            # Save raw depth map before colormap
            raw_out_path = os.path.join(output_dir, f"raw_{filename}")
            cv2.imwrite(raw_out_path, normalized_depth)
            
            # Apply OpenCV Colormap
            heatmap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_INFERNO)
            out_path = os.path.join(output_dir, filename)
            cv2.imwrite(out_path, heatmap)
            print(f"  --> Processed and saved depth map for: {filename}")
            
    print("\n[Success] MiDaS Depth Generation Complete!")

if __name__ == "__main__":
    base_proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eval_image_dir = os.path.join(base_proj_dir, 'dataset', 'images')
    eval_output_dir = os.path.join(base_proj_dir, 'outputs', 'depth')
    
    generate_depth_maps(image_dir=eval_image_dir, output_dir=eval_output_dir)
