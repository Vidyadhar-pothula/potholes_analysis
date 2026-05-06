import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

class PotholeDataset(Dataset):
    """
    Custom PyTorch Dataset for loading pothole images and their corresponding masks.
    """
    def __init__(self, image_paths, mask_dir, img_size=(512, 512), is_train=False):
        self.image_paths = image_paths
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.is_train = is_train
        
        # Build transform list
        transform_list = [transforms.Resize(self.img_size)]
        
        # Apply photometric augmentations for training data
        if self.is_train:
            transform_list.extend([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                transforms.RandomAutocontrast(p=0.5)
            ])
            
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        self.img_transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image_name = os.path.basename(img_path)
        
        # 1. Load and transform the image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Fallback to an empty image if failed
            image = Image.new('RGB', self.img_size)
            
        image = self.img_transform(image)
        
        # 2. Attempt to load the mask; gracefully handle if missing
        base_name = os.path.splitext(image_name)[0]
        mask_path = os.path.join(self.mask_dir, base_name + '.png')
        
        if os.path.exists(mask_path):
            try:
                # Load mask as grayscale
                mask = Image.open(mask_path).convert("L")
                mask = mask.resize(self.img_size, Image.NEAREST)
                
                # Convert to binary tensor (0 or 1)
                mask_np = np.array(mask)
                mask_tensor = torch.from_numpy(mask_np).float()
                mask_tensor = (mask_tensor > 0).float() # Positive values become 1
            except Exception as e:
                print(f"Error loading mask {mask_path}, using empty mask. Error: {e}")
                mask_tensor = torch.zeros((self.img_size[0], self.img_size[1]), dtype=torch.float32)
        else:
            # Fallback if mask is missing: empty mask (meaning no pothole / background)
            mask_tensor = torch.zeros((self.img_size[0], self.img_size[1]), dtype=torch.float32)

        # Add channel dimension (1, H, W)
        mask_tensor = mask_tensor.unsqueeze(0)

        # 3. Output image tensor and mask tensor
        return image, mask_tensor

def get_dataloaders(image_dir='dataset/images', mask_dir='dataset/masks', batch_size=8):
    """
    Scans the directory for images, splits them (70/20/10), and returns DataLoaders.
    """
    all_images = []
    
    # Collect all valid image locations
    if os.path.exists(image_dir):
        for f in os.listdir(image_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                all_images.append(os.path.join(image_dir, f))
    else:
        print(f"Warning: Directory not found -> {image_dir}")
                
    if len(all_images) == 0:
        print(f"Warning: No valid images found in {image_dir}")
        return None, None, None
        
    # Split: Train=70%, Val=20%, Test=10%
    if len(all_images) > 2:
        # Separate 70% for train, 30% for remainder (val+test)
        train_paths, temp_paths = train_test_split(all_images, test_size=0.3, random_state=42)
        
        # Split the 30% temp into 20% validation and 10% test sets
        # 1/3 of 30% is 10% (test size), 2/3 of 30% is 20% (val size)
        if len(temp_paths) > 1:
            val_paths, test_paths = train_test_split(temp_paths, test_size=(1/3), random_state=42)
        else:
            val_paths, test_paths = temp_paths, []
    else:
        # Gracefully handle extremely small directories (e.g. <3 files)
        train_paths, val_paths, test_paths = all_images, [], []

    print(f"Dataset split finalized -> Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

    # Initialize PyTorch Datasets
    train_dataset = PotholeDataset(train_paths, mask_dir, is_train=True) if train_paths else None
    val_dataset = PotholeDataset(val_paths, mask_dir, is_train=False) if val_paths else None
    test_dataset = PotholeDataset(test_paths, mask_dir, is_train=False) if test_paths else None

    # num_workers=0 is safer across operating systems for basic scripts
    num_workers = 0 
    
    # Initialize PyTorch DataLoaders
    # Using drop_last=True for train to prevent potential BN issues with singular batches
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, drop_last=(len(train_paths) > batch_size)) if train_dataset else None
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers) if test_dataset else None

    return train_loader, val_loader, test_loader

# =========================================================
# TEST SNIPPET
# =========================================================
if __name__ == "__main__":
    print("--- Running Dataset Pipeline Test ---")
    
    # Configure mock directories relying on project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_image_dir = os.path.join(base_dir, 'dataset', 'images')
    test_mask_dir = os.path.join(base_dir, 'dataset', 'masks')
    
    try:
        os.makedirs(test_image_dir, exist_ok=True)
        os.makedirs(test_mask_dir, exist_ok=True)
        
        # Populate with dummy data if needed specifically for the test
        if len(os.listdir(test_image_dir)) == 0:
            print("Dataset directory is empty. Generating 10 dummy images and 5 masks for local test...")
            for i in range(10):
                img_path = os.path.join(test_image_dir, f'test_image_{i}.jpg')
                # Generate random solid-color background
                Image.new('RGB', (800, 600), color=(int(np.random.randint(255)), 100, 100)).save(img_path)
                
                # Assign mock masks to only 5 images (Testing fallback functionality)
                if i < 5:
                    mask_path = os.path.join(test_mask_dir, f'test_image_{i}.jpg')
                    # Generate random binary 'pothole' mask (0 vs 255)
                    color = 255 if np.random.rand() > 0.5 else 0
                    Image.new('L', (800, 600), color=color).save(mask_path)
        
        print("\nAttempting to initialize DataLoaders...")
        train_dl, val_dl, test_dl = get_dataloaders(image_dir=test_image_dir, mask_dir=test_mask_dir, batch_size=4)
        
        if train_dl:
            # Yield single batch
            images, masks = next(iter(train_dl))
            print("\n[SUCCESS] Batch loaded successfully via PotholeDataset!")
            print(f"-> Image tensor shape: {images.shape} (dtype: {images.dtype})")
            print(f"-> Mask tensor shape:  {masks.shape} (dtype: {masks.dtype})")
            print(f"-> Unique mask values: {torch.unique(masks).tolist()}     (Expected combinations of 0.0 and 1.0)")
            
        else:
            print("\n[FAILED] Failed to boot dataloaders.")
    except Exception as e:
        print(f"\n[ERROR] An error occurred during testing: {e}")
