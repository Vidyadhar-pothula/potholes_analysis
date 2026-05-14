import os
import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from train import get_model
from inference_depth import generate_depth_maps

def generate_segmentation_masks(image_dir, output_dir, model_path):
    print("--- Initializing RGBD DeepLabV3+ Inference ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Targeting: {device}")
    
    model = get_model(num_classes=1)
    
    # Load RGBD model weights
    rgbd_model_path = model_path.replace('deeplab_model.pth', 'deeplab_rgbd_best.pth')
    if not os.path.exists(rgbd_model_path):
        print(f"[Warning] RGBD model not found at {rgbd_model_path}, trying original {model_path}")
        rgbd_model_path = model_path

    if os.path.exists(rgbd_model_path):
        model.load_state_dict(torch.load(rgbd_model_path, map_location=device))
        print(f"Loaded weights from {rgbd_model_path}")
    else:
        print(f"[ERROR] Trained model weights not found.")
        return
        
    model.to(device)
    model.eval()
    
    img_size = (512, 512)
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(image_dir):
        print(f"[ERROR] Directory not found: {image_dir}")
        return

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(image_files) == 0:
        print(f"No images in {image_dir}.")
        return

    # Generate depth maps on the fly for RGBD since Flask calls this before depth generation
    temp_depth_dir = os.path.join(output_dir, "temp_depth")
    os.makedirs(temp_depth_dir, exist_ok=True)
    generate_depth_maps(image_dir, temp_depth_dir)

    with torch.no_grad():
        for filename in image_files:
            img_path = os.path.join(image_dir, filename)
            base_name = os.path.splitext(filename)[0]
            
            # 1. Load RGB
            image = cv2.imread(img_path)
            if image is None:
                continue
            original_size = (image.shape[1], image.shape[0]) # (w, h)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image, img_size)
            
            # 2. Load Depth
            depth_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                candidate = os.path.join(temp_depth_dir, base_name + ext)
                if os.path.exists(candidate):
                    depth_path = candidate
                    break
                    
            if depth_path and os.path.exists(depth_path):
                depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
                if depth is None:
                    depth = np.zeros(img_size, dtype=np.uint8)
                else:
                    depth = cv2.resize(depth, img_size)
            else:
                depth = np.zeros(img_size, dtype=np.uint8)
                
            # 3. Normalize
            image_resized = image_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_resized = (image_resized - mean) / std
            
            depth = depth.astype(np.float32) / 255.0
            depth = np.expand_dims(depth, axis=-1)
            
            # 4-Channel Concatenation
            rgbd = np.concatenate([image_resized, depth], axis=-1)
            input_tensor = torch.from_numpy(rgbd).permute(2, 0, 1).unsqueeze(0).float().to(device)
            
            # 4. Infer
            output = model(input_tensor)['out']
            
            output = torch.nn.functional.interpolate(
                output,
                size=(original_size[1], original_size[0]), # (h, w)
                mode="bilinear",
                align_corners=False,
            )
            
            # Lower threshold to improve Recall (0.35)
            probs = torch.sigmoid(output).squeeze(0).squeeze(0)
            binary_mask = (probs > 0.35).float()
            
            mask_np = (binary_mask.cpu().numpy() * 255).astype(np.uint8)
            
            out_path = os.path.join(output_dir, filename)
            Image.fromarray(mask_np).save(out_path)
            print(f"Processed: {filename}")
            
    # Cleanup temp depth
    import shutil
    shutil.rmtree(temp_depth_dir, ignore_errors=True)
            
    print("\n[Success] RGBD Inference Complete!")

if __name__ == "__main__":
    base_proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eval_image_dir = os.path.join(base_proj_dir, 'dataset', 'images')
    eval_output_dir = os.path.join(base_proj_dir, 'outputs', 'masks')
    trained_model_path = os.path.join(base_proj_dir, 'models', 'deeplab_rgbd_best.pth')
    
    generate_segmentation_masks(
        image_dir=eval_image_dir, 
        output_dir=eval_output_dir,
        model_path=trained_model_path
    )
