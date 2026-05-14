import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from dataset import get_dataloaders
from metrics import MetricTracker
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# =========================================================
# Custom Losses
# =========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss
        return focal_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice

class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.laplacian = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, logits, targets):
        device = logits.device
        lap = self.laplacian.to(device)
        probs = torch.sigmoid(logits)
        edge_pred = F.conv2d(probs, lap, padding=1)
        edge_target = F.conv2d(targets, lap, padding=1)
        
        edge_pred = torch.clamp(torch.abs(edge_pred), 0, 1)
        edge_target = torch.clamp(torch.abs(edge_target), 0, 1)
        return F.mse_loss(edge_pred, edge_target)

class TotalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()

    def forward(self, logits, targets):
        return self.focal(logits, targets) + self.dice(logits, targets) + 0.3 * self.boundary(logits, targets)

# =========================================================
# Model Setup (RGBD)
# =========================================================
def get_model(num_classes=1):
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
    
    # Modify 1st Conv to accept 4 channels (R, G, B, D)
    old_conv = model.backbone.conv1
    new_conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = old_conv.weight
        new_conv.weight[:, 3:4, :, :] = old_conv.weight[:, :1, :, :] # Initialize D channel
    model.backbone.conv1 = new_conv
    
    # Add Dropout to reduce overfitting
    model.classifier[4] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Conv2d(256, num_classes, kernel_size=1)
    )
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        
    return model

# =========================================================
# Visual Debugging
# =========================================================
def save_debug_visualization(epoch, rgbd, true_mask, pred_mask, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # Undo normalization for RGB
    rgb = rgbd[:3].cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rgb = std * rgb + mean
    rgb = np.clip(rgb, 0, 1)
    
    # Depth
    depth = rgbd[3].cpu().numpy()
    
    t_mask = true_mask.cpu().numpy().squeeze()
    p_mask = pred_mask.cpu().numpy().squeeze()

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(rgb)
    axes[0].set_title('Input RGB')
    axes[1].imshow(depth, cmap='inferno')
    axes[1].set_title('Input Depth')
    axes[2].imshow(t_mask, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Ground Truth')
    axes[3].imshow(p_mask, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title('Predicted Mask')
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'debug_epoch_{epoch}.png'))
    plt.close()

# =========================================================
# Training Loop
# =========================================================
def train_model(epochs=15, batch_size=4, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Targeting: {device}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_dir = os.path.join(base_dir, 'dataset', 'images')
    mask_dir = os.path.join(base_dir, 'dataset', 'masks')
    depth_dir = os.path.join(base_dir, 'outputs', 'depth')
    models_dir = os.path.join(base_dir, 'models')
    debug_dir = os.path.join(base_dir, 'outputs', 'debug')
    metrics_dir = os.path.join(base_dir, 'outputs', 'metrics')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    save_path = os.path.join(models_dir, 'deeplab_rgbd_best.pth')

    train_loader, val_loader, test_loader = get_dataloaders(
        image_dir=image_dir, mask_dir=mask_dir, depth_dir=depth_dir, batch_size=batch_size
    )

    if not train_loader or not val_loader:
        print("[ERROR] Datasets missing.")
        return

    model = get_model(num_classes=1).to(device)
    # AdamW with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = TotalLoss()
    
    # Early Stopping tracking
    best_iou = 0.0
    patience = 5
    patience_counter = 0

    metric_tracker = MetricTracker()

    print("\n[Starting Training...]")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for rgbd, masks in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Train"):
            rgbd, masks = rgbd.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(rgbd)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * rgbd.size(0)
            
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        metric_tracker.reset()
        
        with torch.no_grad():
            for i, (rgbd, masks) in enumerate(val_loader):
                rgbd, masks = rgbd.to(device), masks.to(device)
                outputs = model(rgbd)['out']
                loss = criterion(outputs, masks)
                val_loss += loss.item() * rgbd.size(0)
                
                probs = torch.sigmoid(outputs)
                metric_tracker.update(probs, masks, threshold=0.35)
                
                # Save visual debugging
                if i == 0:
                    preds = (probs > 0.35).float()
                    save_debug_visualization(epoch+1, rgbd[0], masks[0], preds[0], debug_dir)
                    
        avg_val_loss = val_loss / len(val_loader.dataset)
        dice, iou, precision, recall, f1 = metric_tracker.compute_metrics()
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Dice: {dice:.4f} | IoU: {iou:.4f} | Recall: {recall:.4f} | Precision: {precision:.4f} | F1: {f1:.4f}")
        
        # Save best validation model
        if iou > best_iou:
            best_iou = iou
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f" => New best model saved (IoU: {iou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered. No improvement for {patience} epochs.")
                break
                
    # Final Evaluation & Metrics Save
    print("\nGenerating final metrics (PR Curve & Severity metrics)...")
    ap = metric_tracker.save_pr_curve(metrics_dir)
    print(f"Average Precision (AP) saved: {ap:.4f}")
    
    sev_metrics = metric_tracker.compute_severity_metrics()
    print("\nSeverity-wise Evaluation:")
    for sev, m in sev_metrics.items():
        print(f"[{sev}] IoU: {m['iou']:.4f} | Recall: {m['recall']:.4f} | Prec: {m['precision']:.4f} | F1: {m['f1']:.4f}")
        
if __name__ == '__main__':
    train_model(epochs=15, batch_size=4, lr=1e-4)
