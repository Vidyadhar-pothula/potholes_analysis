import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from dataset import get_dataloaders
from train import get_model, BCEDiceLoss

def full_evaluation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, 'dataset', 'images')
    mask_dir  = os.path.join(base_dir, 'dataset', 'masks')
    model_path = os.path.join(base_dir, 'models', 'deeplab_model.pth')

    print("Loading dataset...")
    _, val_loader, _ = get_dataloaders(image_dir=image_dir, mask_dir=mask_dir, batch_size=4)
    model = get_model(num_classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    criterion = BCEDiceLoss(bce_weight=0.5)

    val_loss = 0.0
    correct_pixels = 0
    total_pixels   = 0
    eps = 1e-6
    TP = FP = FN = TN = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks  = masks.to(device)
            outputs = model(images)['out']

            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            correct_pixels += (preds == masks).sum().item()
            total_pixels   += torch.numel(preds)

            TP += (preds * masks).sum().item()
            FP += (preds * (1 - masks)).sum().item()
            FN += ((1 - preds) * masks).sum().item()
            TN += ((1 - preds) * (1 - masks)).sum().item()

    avg_loss   = val_loss / len(val_loader.dataset)
    accuracy   = correct_pixels / total_pixels * 100
    dice       = (2 * TP + eps) / (2 * TP + FP + FN + eps)
    iou        = (TP + eps) / (TP + FP + FN + eps)
    precision  = (TP + eps) / (TP + FP + eps)
    recall     = (TP + eps) / (TP + FN + eps)
    f1         = dice   # Dice = F1
    specificity = (TN + eps) / (TN + FP + eps)

    print("\n" + "="*52)
    print("   FINAL MODEL EVALUATION REPORT")
    print("="*52)
    print(f"  Validation Loss      :  {avg_loss:.4f}  ({avg_loss*100:.2f}%)")
    print(f"  Pixel Accuracy       :  {accuracy:.2f}%")
    print("-"*52)
    print(f"  Dice Coefficient(F1) :  {dice:.4f}  ({dice*100:.2f}%)")
    print(f"  IoU (Jaccard Index)  :  {iou:.4f}  ({iou*100:.2f}%)")
    print(f"  Precision            :  {precision:.4f}  ({precision*100:.2f}%)")
    print(f"  Recall (Sensitivity) :  {recall:.4f}  ({recall*100:.2f}%)")
    print(f"  Specificity          :  {specificity:.4f}  ({specificity*100:.2f}%)")
    print("-"*52)
    print(f"  True Positives  (TP) :  {int(TP):,}")
    print(f"  True Negatives  (TN) :  {int(TN):,}")
    print(f"  False Positives (FP) :  {int(FP):,}")
    print(f"  False Negatives (FN) :  {int(FN):,}")
    print("="*52)

if __name__ == '__main__':
    full_evaluation()
