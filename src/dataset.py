import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

class PotholeRGBDDataset(Dataset):
    """
    Custom PyTorch Dataset for loading pothole RGB images, Depth maps, and their corresponding masks.
    Outputs a 4-channel [R, G, B, D] tensor.
    """
    def __init__(self, image_paths, mask_dir, depth_dir, img_size=(512, 512), is_train=False):
        self.image_paths = image_paths
        self.mask_dir = mask_dir
        self.depth_dir = depth_dir
        self.img_size = img_size
        self.is_train = is_train
        
        # Albumentations setup
        if self.is_train:
            self.transform = A.Compose([
                A.Resize(self.img_size[0], self.img_size[1]),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                # A.RandomCrop can be added here if images are larger than img_size
            ], additional_targets={'depth': 'image'})
        else:
            self.transform = A.Compose([
                A.Resize(self.img_size[0], self.img_size[1])
            ], additional_targets={'depth': 'image'})

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image_name = os.path.basename(img_path)
        base_name = os.path.splitext(image_name)[0]
        
        # 1. Load RGB image
        image = cv2.imread(img_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
            
        # 2. Load Depth Map
        # Find matching depth map (can be .png, .jpg, etc.)
        depth_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            candidate = os.path.join(self.depth_dir, base_name + ext)
            if os.path.exists(candidate):
                depth_path = candidate
                break
                
        if depth_path and os.path.exists(depth_path):
            depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            if depth is None:
                depth = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            depth = np.zeros(image.shape[:2], dtype=np.uint8)
            
        # 3. Load Mask
        mask_path = os.path.join(self.mask_dir, base_name + '.png')
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
            else:
                mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.float32)

        # 4. Apply Augmentations
        augmented = self.transform(image=image, mask=mask, depth=depth)
        aug_img = augmented['image']
        aug_mask = augmented['mask']
        aug_depth = augmented['depth']
        
        # 5. Normalize and create 4-channel tensor
        # RGB Normalization
        aug_img = aug_img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        aug_img = (aug_img - mean) / std
        
        # Depth Normalization (0 to 1)
        aug_depth = aug_depth.astype(np.float32) / 255.0
        # Expand depth dims
        aug_depth = np.expand_dims(aug_depth, axis=-1)
        
        # Concatenate [R, G, B, D]
        rgbd = np.concatenate([aug_img, aug_depth], axis=-1)
        
        # To Tensor (C, H, W)
        rgbd_tensor = torch.from_numpy(rgbd).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(aug_mask).unsqueeze(0).float()
        
        return rgbd_tensor, mask_tensor

def get_dataloaders(image_dir='dataset/images', mask_dir='dataset/masks', depth_dir='outputs/depth', batch_size=8):
    all_images = []
    if os.path.exists(image_dir):
        for f in os.listdir(image_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                all_images.append(os.path.join(image_dir, f))
                
    if len(all_images) == 0:
        print(f"Warning: No valid images found in {image_dir}")
        return None, None, None
        
    if len(all_images) > 2:
        train_paths, temp_paths = train_test_split(all_images, test_size=0.3, random_state=42)
        if len(temp_paths) > 1:
            val_paths, test_paths = train_test_split(temp_paths, test_size=(1/3), random_state=42)
        else:
            val_paths, test_paths = temp_paths, []
    else:
        train_paths, val_paths, test_paths = all_images, [], []

    print(f"RGBD Dataset split -> Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

    train_dataset = PotholeRGBDDataset(train_paths, mask_dir, depth_dir, is_train=True) if train_paths else None
    val_dataset = PotholeRGBDDataset(val_paths, mask_dir, depth_dir, is_train=False) if val_paths else None
    test_dataset = PotholeRGBDDataset(test_paths, mask_dir, depth_dir, is_train=False) if test_paths else None

    num_workers = 0 
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, drop_last=(len(train_paths) > batch_size)) if train_dataset else None
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers) if test_dataset else None

    return train_loader, val_loader, test_loader
