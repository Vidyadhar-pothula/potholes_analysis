import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

class MetricTracker:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.TP = 0.0
        self.FP = 0.0
        self.FN = 0.0
        self.all_probs = []
        self.all_targets = []
        
        # Severity grouping
        # Dictionary of lists to store targets/probs per severity
        self.severity_data = {
            'Low': {'TP': 0, 'FP': 0, 'FN': 0},
            'Medium': {'TP': 0, 'FP': 0, 'FN': 0},
            'High': {'TP': 0, 'FP': 0, 'FN': 0}
        }
        
    def update(self, probs, targets, threshold=0.35):
        preds = (probs > threshold).float()
        targets = targets.float()
        
        tp = (preds * targets).sum().item()
        fp = (preds * (1 - targets)).sum().item()
        fn = ((1 - preds) * targets).sum().item()
        
        self.TP += tp
        self.FP += fp
        self.FN += fn
        
        # For PR Curve
        self.all_probs.extend(probs.view(-1).cpu().numpy().tolist())
        self.all_targets.extend(targets.view(-1).cpu().numpy().tolist())
        
        # Severity tracking (mock grouping based on target area)
        # Real severity tracking requires instance segmentation or connected components.
        # Here we approximate by checking the total area of the pothole in the mask.
        b_size = targets.size(0)
        for i in range(b_size):
            t_mask = targets[i].squeeze()
            p_mask = preds[i].squeeze()
            area = t_mask.sum().item()
            
            if area == 0:
                continue
                
            if area < 5000:
                sev = 'Low'
            elif area < 15000:
                sev = 'Medium'
            else:
                sev = 'High'
                
            stp = (p_mask * t_mask).sum().item()
            sfp = (p_mask * (1 - t_mask)).sum().item()
            sfn = ((1 - p_mask) * t_mask).sum().item()
            
            self.severity_data[sev]['TP'] += stp
            self.severity_data[sev]['FP'] += sfp
            self.severity_data[sev]['FN'] += sfn

    def compute_metrics(self):
        eps = 1e-6
        dice = (2 * self.TP + eps) / (2 * self.TP + self.FP + self.FN + eps)
        iou = (self.TP + eps) / (self.TP + self.FP + self.FN + eps)
        precision = (self.TP + eps) / (self.TP + self.FP + eps)
        recall = (self.TP + eps) / (self.TP + self.FN + eps)
        f1 = (2 * precision * recall) / (precision + recall + eps)
        
        return dice, iou, precision, recall, f1
        
    def compute_severity_metrics(self):
        eps = 1e-6
        sev_metrics = {}
        for sev in ['Low', 'Medium', 'High']:
            tp = self.severity_data[sev]['TP']
            fp = self.severity_data[sev]['FP']
            fn = self.severity_data[sev]['FN']
            
            iou = (tp + eps) / (tp + fp + fn + eps)
            precision = (tp + eps) / (tp + fp + eps)
            recall = (tp + eps) / (tp + fn + eps)
            f1 = (2 * precision * recall) / (precision + recall + eps)
            
            sev_metrics[sev] = {
                'iou': iou,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        return sev_metrics

    def save_pr_curve(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        if len(self.all_probs) == 0:
            return 0.0
            
        y_true = np.array(self.all_targets)
        y_scores = np.array(self.all_probs)
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AP = {ap:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True)
        
        save_path = os.path.join(save_dir, 'pr_curve.png')
        plt.savefig(save_path)
        plt.close()
        
        return ap
