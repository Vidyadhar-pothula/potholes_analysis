import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12

output_dir = 'outputs/metrics'

# =========================================================
# Graph 1: Architecture Comparison Bar Chart
# =========================================================
metrics = ['Pixel\nAccuracy', 'Precision', 'Recall', 'F1-Score', 'Dice', 'mIoU']

deeplabv3_rgbd = [96.2, 89.80, 91.10, 90.45, 90.45, 82.56]
unet           = [89.2, 87.4,  86.0,  86.7,  84.9,  83.2]
segnet         = [90.5, 86.8,  85.3,  86.0,  85.5,  84.5]

x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 7))
bars1 = ax.bar(x - width, deeplabv3_rgbd, width, label='DeepLabV3+ RGBD (Ours)', color='#6c47ff', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x, unet, width, label='U-Net', color='#f59e0b', edgecolor='white', linewidth=0.5)
bars3 = ax.bar(x + width, segnet, width, label='SegNet', color='#34d399', edgecolor='white', linewidth=0.5)

# Add value labels on top of bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
ax.set_title('Segmentation Model Comparison', fontsize=16, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(75, 100)
ax.legend(fontsize=11, loc='lower right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_facecolor('#fafafa')
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig(f'{output_dir}/architecture_comparison.png', dpi=200, bbox_inches='tight')
plt.close()
print("[1/3] Architecture comparison chart saved.")


# =========================================================
# Graph 2: Training Phase Progression Line Chart
# =========================================================
phases = ['Phase 1\n(RGB, 665 imgs)', 'Phase 2\n(RGBD, 1330 imgs)', 'Phase 3\n(RGBD+Neg, 1424 imgs)']

dice_vals      = [79.25, 90.45, 85.75]
iou_vals       = [66.37, 82.56, 75.05]
precision_vals = [83.11, 89.80, 85.63]
recall_vals    = [77.87, 91.10, 85.87]

fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(phases))

ax.plot(x, dice_vals, 'o-', color='#6c47ff', linewidth=3, markersize=12, label='Dice', zorder=5)
ax.plot(x, iou_vals, 's-', color='#f87171', linewidth=3, markersize=12, label='IoU', zorder=5)
ax.plot(x, precision_vals, '^-', color='#f59e0b', linewidth=3, markersize=12, label='Precision', zorder=5)
ax.plot(x, recall_vals, 'D-', color='#34d399', linewidth=3, markersize=12, label='Recall', zorder=5)

# Add data labels
for vals, color in [(dice_vals, '#6c47ff'), (iou_vals, '#f87171'), (precision_vals, '#f59e0b'), (recall_vals, '#34d399')]:
    for i, v in enumerate(vals):
        ax.annotate(f'{v:.1f}%', (i, v), textcoords="offset points", xytext=(0, 12),
                    ha='center', fontsize=10, fontweight='bold', color=color)

ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
ax.set_title('Training Phase Progression — Metric Evolution', fontsize=16, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(phases, fontsize=11)
ax.set_ylim(55, 100)
ax.legend(fontsize=12, loc='lower right')
ax.grid(alpha=0.3, linestyle='--')
ax.set_facecolor('#fafafa')
fig.patch.set_facecolor('white')

# Highlight best phase
ax.axvspan(0.8, 1.2, alpha=0.08, color='#6c47ff')
ax.text(1, 58, 'BEST', ha='center', fontsize=11, fontweight='bold', color='#6c47ff', alpha=0.6)

plt.tight_layout()
plt.savefig(f'{output_dir}/training_progression.png', dpi=200, bbox_inches='tight')
plt.close()
print("[2/3] Training progression chart saved.")


# =========================================================
# Graph 3: Severity-Wise Performance Radar / Grouped Bar
# =========================================================
severity_labels = ['Low', 'Medium', 'High']

sev_iou       = [77.74, 72.52, 83.03]
sev_precision = [82.61, 83.96, 90.07]
sev_recall    = [92.95, 84.19, 91.39]
sev_f1        = [87.47, 84.07, 90.73]

x = np.arange(len(severity_labels))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 7))

bars1 = ax.bar(x - 1.5*width, sev_iou, width, label='IoU', color='#6c47ff', edgecolor='white')
bars2 = ax.bar(x - 0.5*width, sev_precision, width, label='Precision', color='#f59e0b', edgecolor='white')
bars3 = ax.bar(x + 0.5*width, sev_recall, width, label='Recall', color='#34d399', edgecolor='white')
bars4 = ax.bar(x + 1.5*width, sev_f1, width, label='F1-Score', color='#f87171', edgecolor='white')

for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

# Color-code severity labels
colors_sev = ['#34d399', '#f59e0b', '#f87171']
for i, (label, color) in enumerate(zip(severity_labels, colors_sev)):
    ax.text(i, -4, f'● {label}', ha='center', fontsize=13, fontweight='bold', color=color)

ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
ax.set_title('Severity-Wise Model Performance (Best Model — Phase 2)', fontsize=16, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(['', '', ''])
ax.set_ylim(60, 100)
ax.legend(fontsize=11, loc='lower right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_facecolor('#fafafa')
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig(f'{output_dir}/severity_performance.png', dpi=200, bbox_inches='tight')
plt.close()
print("[3/3] Severity performance chart saved.")

print("\nAll 3 graphs saved to outputs/metrics/!")
