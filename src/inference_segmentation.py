import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from train import get_model

def generate_segmentation_masks(image_dir, output_dir, model_path):
    print("--- Initializing DeepLabV3+ Pothole Segmentation ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Target: {device}")
    
    print(f"Loading trained model weights from: {model_path}")
    model = get_model(num_classes=1)
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model weights not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Same normalization as training
    img_size = (512, 512)
    img_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(image_dir):
        print(f"[ERROR] Directory not found: {image_dir}")
        return

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if len(image_files) == 0:
        print(f"No valid images found in {image_dir}.")
        return

    print(f"Found {len(image_files)} images. Beginning segmentation inference...\n")

    with torch.no_grad():
        for filename in image_files:
            img_path = os.path.join(image_dir, filename)
            try:
                image = Image.open(img_path).convert("RGB")
                original_size = image.size # (width, height)
            except Exception as e:
                print(f"[Warning] Could not read {filename}, skipping. Details: {e}")
                continue
                
            input_tensor = img_transform(image).unsqueeze(0).to(device)
            output = model(input_tensor)['out']
            
            print(f"[{filename}] Model output raw min: {output.min().item():.4f}, max: {output.max().item():.4f}")
            
            output = torch.nn.functional.interpolate(
                output,
                size=(original_size[1], original_size[0]), # (height, width)
                mode="bilinear",
                align_corners=False,
            )
            
            # Apply sigmoid and threshold (0.5)
            probs = torch.sigmoid(output).squeeze(0).squeeze(0)
            binary_mask = (probs > 0.5).float()
            
            # Convert to binary mask (0-255)
            mask_np = (binary_mask.cpu().numpy() * 255).astype(np.uint8)
            
            out_path = os.path.join(output_dir, filename)
            Image.fromarray(mask_np).save(out_path)
            print(f"  --> Processed and saved mask for: {filename}")
            
    print("\n[Success] Segmentation Mask Generation Complete!")

if __name__ == "__main__":
    base_proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eval_image_dir = os.path.join(base_proj_dir, 'dataset', 'images')
    eval_output_dir = os.path.join(base_proj_dir, 'outputs', 'masks')
    trained_model_path = os.path.join(base_proj_dir, 'models', 'deeplab_model.pth')
    
    generate_segmentation_masks(
        image_dir=eval_image_dir, 
        output_dir=eval_output_dir,
        model_path=trained_model_path
    )
