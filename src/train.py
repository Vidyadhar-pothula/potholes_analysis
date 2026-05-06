import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from dataset import get_dataloaders
import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# Debug Visualization
# =========================================================
def save_debug_visualization(epoch, image, true_mask, pred_mask, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # Undo normalization for image
    img = image.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

    t_mask = true_mask.cpu().numpy().squeeze()
    p_mask = pred_mask.cpu().numpy().squeeze()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title('Input Image')
    axes[1].imshow(t_mask, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth')
    axes[2].imshow(p_mask, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Predicted Mask')
    
    plt.savefig(os.path.join(save_dir, f'debug_epoch_{epoch}.png'))
    plt.close()

# =========================================================
# Dice Loss Implementation
# =========================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice

# =========================================================
# Combined BCE + Dice Loss
# =========================================================
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return (self.bce_weight * bce_loss) + ((1.0 - self.bce_weight) * dice_loss)

# =========================================================
# Model Setup Layer
# =========================================================
def get_model(num_classes=1):
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model

# =========================================================
# Evaluation Metrics
# =========================================================
def compute_metrics(pred_logits, targets):
    probs = torch.sigmoid(pred_logits)
    preds = (probs > 0.5).float()
    targets = targets.float()

    eps = 1e-6
    TP = (preds * targets).sum().item()
    FP = (preds * (1 - targets)).sum().item()
    FN = ((1 - preds) * targets).sum().item()

    dice = (2 * TP + eps) / (2 * TP + FP + FN + eps)
    iou = (TP + eps) / (TP + FP + FN + eps)
    precision = (TP + eps) / (TP + FP + eps)
    recall = (TP + eps) / (TP + FN + eps)

    return dice, iou, precision, recall

# =========================================================
# Main Training Routine
# =========================================================
def train_model(epochs=5, batch_size=4, lr=5e-5, resume=True):
    print("--- Initializing Pothole Segmentation Training ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Execution Target: {device}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_dir = os.path.join(base_dir, 'dataset', 'images')
    mask_dir = os.path.join(base_dir, 'dataset', 'masks')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, 'deeplab_model.pth')

    print("\n[1] Indexing Dataset Pipeline...")
    train_loader, val_loader, _ = get_dataloaders(image_dir=image_dir, mask_dir=mask_dir, batch_size=batch_size)
    if train_loader is None or val_loader is None:
        print("[ERROR] Cannot initiate training. Valid datasets could not be loaded.")
        return

    print("\n[2] Loading Pretrained DeepLabV3+ Component...")
    model = get_model(num_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = BCEDiceLoss(bce_weight=0.5)

    # Resume from existing checkpoint if available
    if resume and os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"  => Resumed from checkpoint: {save_path}")
    else:
        print("  => Starting fresh (no checkpoint found)")

    print("\n[3] Launching Training Loop...")
    best_dice = 0.0
    metrics_history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_iou': [], 'val_precision': [], 'val_recall': []}

    from tqdm import tqdm
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Train"):
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        epoch_dice, epoch_iou, epoch_precision, epoch_recall = 0.0, 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for i, (images, masks) in enumerate(val_loader):
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                
                d, iou, p, r = compute_metrics(outputs, masks)
                b_size = images.size(0)
                epoch_dice += d * b_size
                epoch_iou += iou * b_size
                epoch_precision += p * b_size
                epoch_recall += r * b_size
                
                if i == 0:
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.5).float()
                    save_debug_visualization(epoch+1, images[0], masks[0], preds[0], os.path.join(base_dir, 'outputs', 'samples'))
                    
        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_dice = epoch_dice / len(val_loader.dataset)
        avg_iou = epoch_iou / len(val_loader.dataset)
        avg_precision = epoch_precision / len(val_loader.dataset)
        avg_recall = epoch_recall / len(val_loader.dataset)

        metrics_history['train_loss'].append(avg_train_loss)
        metrics_history['val_loss'].append(avg_val_loss)
        metrics_history['val_dice'].append(avg_dice)
        metrics_history['val_iou'].append(avg_iou)
        metrics_history['val_precision'].append(avg_precision)
        metrics_history['val_recall'].append(avg_recall)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Dice: {avg_dice:.4f}")
        print(f"IoU: {avg_iou:.4f}")
        print(f"Precision: {avg_precision:.4f}")
        print(f"Recall: {avg_recall:.4f}")
        
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), save_path)
            print(f"  => Captured improved metrics. Model saved: {save_path}")

    # Save history to file for plotting
    torch.save(metrics_history, os.path.join(models_dir, 'metrics_history.pt'))
    print("\n[Success] Training Completion Protocol Finished.")

if __name__ == '__main__':
    train_model(epochs=5, batch_size=4, lr=5e-5, resume=True)
