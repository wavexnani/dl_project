"""
=================================================================
  Clash Royale AI Bot — Model Analysis & Presentation Dashboard
=================================================================
Generates publication-quality charts and tables for all 3 AI models:
  1. Card Classifier (MobileNetV2)
  2. Enemy Detector (YOLOv9)
  3. Decision Transformer (RL Agent)

Output: Saves all figures to 'presentation/' folder.
Usage:  python model_analysis.py
=================================================================
"""

import os, sys, pickle, warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONIOENCODING"] = "utf-8"

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
for sub in ['core', 'drivers', 'training', 'evaluation']:
    sp = os.path.join(ROOT_DIR, sub)
    if sp not in sys.path:
        sys.path.insert(0, sp)
import config

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, models, transforms
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image

# ── Style Setup ──────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
})

ACCENT = '#58a6ff'
GREEN = '#3fb950'
RED = '#f85149'
ORANGE = '#d29922'
PURPLE = '#bc8cff'
CYAN = '#39d2c0'
PALETTE = ['#58a6ff', '#3fb950', '#f85149', '#d29922', '#bc8cff',
           '#39d2c0', '#f778ba', '#79c0ff', '#7ee787', '#ffa657', '#ff7b72']

OUTPUT_DIR = os.path.join(ROOT_DIR, "presentation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*65}")
print(f"  CLASH ROYALE AI — MODEL ANALYSIS DASHBOARD")
print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print(f"{'='*65}\n")


# ═══════════════════════════════════════════════════════════════
#  SECTION 1: CARD CLASSIFIER (MobileNetV2) EVALUATION
# ═══════════════════════════════════════════════════════════════
print("━" * 65)
print("  [1/4] CARD CLASSIFIER — MobileNetV2 Evaluation")
print("━" * 65)

DATA_PATH = os.path.join(config.SORTED_DATA_PATH, "cards") if os.path.exists(os.path.join(config.SORTED_DATA_PATH, "cards")) else os.path.join(ROOT_DIR, "sorted_data", "cards")
IMG_SIZE = getattr(config, 'IMG_SIZE', 128)
MODEL_PATH = config.MODEL_PATH
CLASS_NAMES_PATH = config.CLASS_NAMES_PATH

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = [line.strip() for line in f if line.strip()]
num_classes = len(class_names)
print(f"  Classes ({num_classes}): {class_names}")

# Load model
card_model = models.mobilenet_v2()
card_model.classifier[1] = torch.nn.Linear(card_model.classifier[1].in_features, num_classes)
card_model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
card_model.to(device)
card_model.eval()

# Load dataset
eval_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE), transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
dataset = datasets.ImageFolder(DATA_PATH, transform=eval_transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

# Evaluate
all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        outputs = card_model(inputs)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

overall_acc = np.mean(all_preds == all_labels) * 100
print(f"  Overall Accuracy: {overall_acc:.1f}%")

# Confusion matrix
conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
for t, p in zip(all_labels, all_preds):
    conf_matrix[t][p] += 1

# Per-class metrics
per_class_acc = []
per_class_precision = []
per_class_recall = []
per_class_f1 = []
per_class_count = []

for i in range(num_classes):
    tp = conf_matrix[i][i]
    fn = np.sum(conf_matrix[i]) - tp
    fp = np.sum(conf_matrix[:, i]) - tp
    total = np.sum(conf_matrix[i])
    
    acc = tp / total * 100 if total > 0 else 0
    prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    per_class_acc.append(acc)
    per_class_precision.append(prec)
    per_class_recall.append(rec)
    per_class_f1.append(f1)
    per_class_count.append(total)
    print(f"    {class_names[i]:>15s}: Acc={acc:5.1f}%  Prec={prec:5.1f}%  Rec={rec:5.1f}%  F1={f1:5.1f}%  (n={total})")


# ── Figure 1: Confusion Matrix ──────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(conf_matrix, cmap='Blues', aspect='auto')
ax.set_xticks(range(num_classes))
ax.set_yticks(range(num_classes))
display_names = [n.replace('_', '\n') for n in class_names]
ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(display_names, fontsize=9)
ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax.set_title(f'Card Classifier — Confusion Matrix\nOverall Accuracy: {overall_acc:.1f}%', 
             fontsize=15, fontweight='bold', pad=15)

for i in range(num_classes):
    for j in range(num_classes):
        val = conf_matrix[i][j]
        color = 'white' if val > conf_matrix.max() * 0.5 else '#c9d1d9'
        ax.text(j, i, str(val), ha='center', va='center', color=color, fontsize=10, fontweight='bold')

plt.colorbar(im, ax=ax, shrink=0.8, label='Count')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '1_confusion_matrix.png'), dpi=200, bbox_inches='tight')
plt.close()
print(f"  [SAVED] 1_confusion_matrix.png")


# ── Figure 2: Per-Class Accuracy Bars ────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(num_classes)
bars = ax.bar(x, per_class_acc, color=PALETTE[:num_classes], edgecolor='none', width=0.7, alpha=0.9)

for bar, acc, count in zip(bars, per_class_acc, per_class_count):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, 
            f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#c9d1d9')
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
            f'n={count}', ha='center', va='center', fontsize=8, color='white', alpha=0.8)

ax.axhline(y=overall_acc, color=ACCENT, linestyle='--', alpha=0.7, linewidth=1.5, label=f'Overall: {overall_acc:.1f}%')
ax.set_xticks(x)
ax.set_xticklabels([n.replace('_', '\n') for n in class_names], fontsize=9)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_ylim(0, 115)
ax.set_title('Card Classifier — Per-Class Accuracy', fontsize=15, fontweight='bold', pad=15)
ax.legend(loc='upper right', fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '2_per_class_accuracy.png'), dpi=200, bbox_inches='tight')
plt.close()
print(f"  [SAVED] 2_per_class_accuracy.png")


# ── Figure 3: Precision / Recall / F1 Grouped Bars ──────────
fig, ax = plt.subplots(figsize=(14, 6))
bar_width = 0.25
x = np.arange(num_classes)

b1 = ax.bar(x - bar_width, per_class_precision, bar_width, label='Precision', color=ACCENT, alpha=0.85)
b2 = ax.bar(x, per_class_recall, bar_width, label='Recall', color=GREEN, alpha=0.85)
b3 = ax.bar(x + bar_width, per_class_f1, bar_width, label='F1 Score', color=PURPLE, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels([n.replace('_', '\n') for n in class_names], fontsize=9)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_ylim(0, 115)
ax.set_title('Card Classifier — Precision, Recall & F1 Score', fontsize=15, fontweight='bold', pad=15)
ax.legend(loc='upper right', fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '3_precision_recall_f1.png'), dpi=200, bbox_inches='tight')
plt.close()
print(f"  [SAVED] 3_precision_recall_f1.png")


# ── Figure 4: Confidence Distribution ────────────────────────
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
axes = axes.flatten()
for i in range(num_classes):
    ax = axes[i]
    mask = all_labels == i
    correct_mask = mask & (all_preds == i)
    wrong_mask = mask & (all_preds != i)
    
    if correct_mask.any():
        correct_confs = all_probs[correct_mask, i]
        ax.hist(correct_confs, bins=20, range=(0,1), alpha=0.8, color=GREEN, label='Correct')
    if wrong_mask.any():
        wrong_confs = np.max(all_probs[wrong_mask], axis=1)
        ax.hist(wrong_confs, bins=20, range=(0,1), alpha=0.6, color=RED, label='Wrong')
    
    ax.set_title(class_names[i].replace('_', ' ').title(), fontsize=10, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    if i == 0:
        ax.legend(fontsize=7)

# Hide unused subplot
if num_classes < len(axes):
    for j in range(num_classes, len(axes)):
        axes[j].set_visible(False)

fig.suptitle('Card Classifier — Prediction Confidence Distribution', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '4_confidence_distribution.png'), dpi=200, bbox_inches='tight')
plt.close()
print(f"  [SAVED] 4_confidence_distribution.png")


# ── Figure 5: Dataset Distribution ──────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
colors_sorted = PALETTE[:num_classes]
wedges, texts, autotexts = ax.pie(
    per_class_count, labels=[n.replace('_', '\n') for n in class_names],
    autopct='%1.1f%%', colors=colors_sorted, pctdistance=0.82,
    wedgeprops={'linewidth': 2, 'edgecolor': '#0d1117'},
    textprops={'fontsize': 9, 'color': '#c9d1d9'}
)
for at in autotexts:
    at.set_fontsize(8)
    at.set_color('white')
    at.set_fontweight('bold')

ax.set_title(f'Training Dataset Distribution\nTotal: {sum(per_class_count)} images', 
             fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '5_dataset_distribution.png'), dpi=200, bbox_inches='tight')
plt.close()
print(f"  [SAVED] 5_dataset_distribution.png")


# ═══════════════════════════════════════════════════════════════
#  SECTION 2: MODEL ARCHITECTURE COMPARISON
# ═══════════════════════════════════════════════════════════════
print(f"\n{'━'*65}")
print("  [2/4] MODEL ARCHITECTURE OVERVIEW")
print("━" * 65)

import config

# Count parameters
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# Card Classifier params
cc_total, cc_train = count_params(card_model)

# Decision Transformer params
x_steps, y_steps = 18, 30
NUM_GRID = x_steps * y_steps
NUM_CARD_TYPES = len(config.CARD_COSTS)
ACTION_DIM = (NUM_CARD_TYPES + 1) * NUM_GRID
# INC-1 FIX: Use centralized STATE_DIM constant from config
STATE_DIM = getattr(config, 'STATE_DIM', 1 + 6 + (4 * NUM_CARD_TYPES) + (20 * 4))

sys.path.insert(0, '.')
from agent import DecisionTransformer
dt_model = DecisionTransformer(
    state_dim=STATE_DIM, act_dim=ACTION_DIM, n_blocks=3, h_dim=128,
    context_len=10, n_heads=1, drop_p=0.1
).to(device)
# BUG-M1 FIX: Explicit weights_only=True for secure model checkpoint loading
dt_model.load_state_dict(torch.load(config.RL_MODEL_PATH, map_location=device, weights_only=True))
dt_total, dt_train = count_params(dt_model)

# BUG-M2 FIX: Dynamically measure YOLOv9 parameters from checkpoint when available
yolo_total = 51_003_198
if os.path.exists(config.YOLO_MODEL_PATH):
    try:
        yolo_ckpt = torch.load(config.YOLO_MODEL_PATH, map_location='cpu', weights_only=False)
        if isinstance(yolo_ckpt, dict) and 'model' in yolo_ckpt:
            yolo_total = sum(p.numel() for p in yolo_ckpt['model'].parameters())
        elif hasattr(yolo_ckpt, 'parameters'):
            yolo_total = sum(p.numel() for p in yolo_ckpt.parameters())
    except Exception:
        pass
yolo_train = yolo_total

model_data = {
    'Card Classifier\n(MobileNetV2)': {
        'total': cc_total, 'trainable': cc_train,
        'input': f'{IMG_SIZE}×{IMG_SIZE}×3\nRGB Image',
        'output': f'{num_classes} classes',
        'task': 'Classification',
        'file': os.path.basename(config.MODEL_PATH),
        'size_mb': os.path.getsize(config.MODEL_PATH) / 1e6 if os.path.exists(config.MODEL_PATH) else 0,
    },
    'Enemy Detector\n(YOLOv9-c)': {
        'total': yolo_total, 'trainable': yolo_train,
        'input': '640×640×3\nRGB Frame',
        'output': 'Bounding Boxes\n+ Classes',
        'task': 'Object Detection',
        'file': os.path.basename(config.YOLO_MODEL_PATH),
        'size_mb': os.path.getsize(config.YOLO_MODEL_PATH) / 1e6 if os.path.exists(config.YOLO_MODEL_PATH) else 0,
    },
    'RL Agent\n(Decision Transformer)': {
        'total': dt_total, 'trainable': dt_train,
        'input': f'State({STATE_DIM})\n+ Action({ACTION_DIM})\n+ Reward',
        'output': f'{ACTION_DIM} actions',
        'task': 'Reinforcement\nLearning',
        'file': os.path.basename(config.RL_MODEL_PATH),
        'size_mb': os.path.getsize(config.RL_MODEL_PATH) / 1e6 if os.path.exists(config.RL_MODEL_PATH) else 0,
    }
}

for name, d in model_data.items():
    print(f"  {name.replace(chr(10),' ')}: {d['total']:,} params ({d['size_mb']:.1f} MB)")


# ── Figure 6: Parameter Count Comparison ─────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
names = list(model_data.keys())
totals = [model_data[n]['total'] for n in names]
colors = [ACCENT, GREEN, PURPLE]

bars = ax.barh(range(len(names)), totals, color=colors, height=0.5, edgecolor='none', alpha=0.9)
for bar, total, name in zip(bars, totals, names):
    label = f'{total/1e6:.1f}M' if total > 1e6 else f'{total/1e3:.0f}K'
    ax.text(bar.get_width() + max(totals)*0.01, bar.get_y() + bar.get_height()/2,
            label, va='center', fontsize=13, fontweight='bold', color='#c9d1d9')

ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=11)
ax.set_xlabel('Total Parameters', fontsize=12)
ax.set_title('Model Complexity — Parameter Count Comparison', fontsize=15, fontweight='bold', pad=15)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.0f}M'))
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '6_parameter_comparison.png'), dpi=200, bbox_inches='tight')
plt.close()
print(f"  [SAVED] 6_parameter_comparison.png")


# ── Figure 7: Model Size on Disk ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
sizes = [model_data[n]['size_mb'] for n in names]
wedges, texts, autotexts = ax.pie(
    sizes, labels=[n.replace('\n', ' ') for n in names],
    autopct=lambda p: f'{p*sum(sizes)/100:.1f} MB',
    colors=colors, pctdistance=0.75,
    wedgeprops={'linewidth': 3, 'edgecolor': '#0d1117'},
    textprops={'fontsize': 11, 'color': '#c9d1d9'}
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_color('white')
    at.set_fontweight('bold')

total_size = sum(sizes)
ax.set_title(f'Model File Sizes — Total: {total_size:.1f} MB', fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '7_model_sizes.png'), dpi=200, bbox_inches='tight')
plt.close()
print(f"  [SAVED] 7_model_sizes.png")


# ═══════════════════════════════════════════════════════════════
#  SECTION 3: DECISION TRANSFORMER ARCHITECTURE DEEP DIVE
# ═══════════════════════════════════════════════════════════════
print(f"\n{'━'*65}")
print("  [3/4] DECISION TRANSFORMER — Architecture Analysis")
print("━" * 65)

# Layer-by-layer parameter breakdown
layer_names = []
layer_params = []
for name, param in dt_model.named_parameters():
    layer_names.append(name)
    layer_params.append(param.numel())

# Group by module
module_groups = OrderedDict()
for name, count in zip(layer_names, layer_params):
    module = name.split('.')[0]
    if module.startswith('embed'):
        key = 'Embedding\nLayers'
    elif module == 'transformer_blocks':
        block_num = name.split('.')[1]
        key = f'Transformer\nBlock {block_num}'
    elif module.startswith('predict'):
        key = 'Action\nPredictor'
    else:
        key = module
    module_groups[key] = module_groups.get(key, 0) + count

# ── Figure 8: DT Layer Breakdown ─────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Bar chart of module groups
mod_names = list(module_groups.keys())
mod_counts = list(module_groups.values())
mod_colors = [ACCENT if 'Embed' in n else GREEN if 'Block' in n else PURPLE for n in mod_names]

bars = ax1.bar(range(len(mod_names)), mod_counts, color=mod_colors, edgecolor='none', alpha=0.9)
for bar, count in zip(bars, mod_counts):
    label = f'{count/1e6:.2f}M' if count > 1e6 else f'{count/1e3:.1f}K'
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mod_counts)*0.02,
             label, ha='center', fontsize=9, fontweight='bold', color='#c9d1d9')

ax1.set_xticks(range(len(mod_names)))
ax1.set_xticklabels(mod_names, fontsize=9)
ax1.set_ylabel('Parameters')
ax1.set_title('Decision Transformer — Parameter Distribution', fontsize=13, fontweight='bold')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
ax1.grid(axis='y', alpha=0.3)

# Architecture info table
arch_info = [
    ['State Dimension', str(STATE_DIM)],
    ['Action Dimension', f'{ACTION_DIM:,}'],
    ['Embedding Dimension', '128'],
    ['Transformer Blocks', '3'],
    ['Attention Heads', '1'],
    ['Context Length', '10 steps'],
    ['Dropout', '0.1'],
    ['Optimizer', 'AdamW'],
    ['Learning Rate', '1e-4'],
    ['Placement Grid', f'{x_steps}×{y_steps} = {NUM_GRID}'],
    ['Card Types', str(NUM_CARD_TYPES)],
    ['ε (exploration)', '1.0 → 0.1'],
    ['Replay Buffer', '100,000 cap'],
]

ax2.axis('off')
table = ax2.table(cellText=arch_info, colLabels=['Hyperparameter', 'Value'],
                  loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#30363d')
    if row == 0:
        cell.set_facecolor(ACCENT)
        cell.set_text_props(color='white', fontweight='bold')
    else:
        cell.set_facecolor('#161b22' if row % 2 == 0 else '#0d1117')
        cell.set_text_props(color='#c9d1d9')

ax2.set_title('Architecture Hyperparameters', fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '8_decision_transformer_analysis.png'), dpi=200, bbox_inches='tight')
plt.close()
print(f"  [SAVED] 8_decision_transformer_analysis.png")


# ── Figure 9: Action Space Visualization ─────────────────────
fig, ax = plt.subplots(figsize=(12, 7))

# Show the placement grid overlaid on a schematic arena
arena_bbox = config.ARENA_BBOX
min_x, min_y, max_x, max_y = arena_bbox

# Draw arena
arena = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                        linewidth=2, edgecolor=ACCENT, facecolor=ACCENT, alpha=0.1)
ax.add_patch(arena)

# Draw grid points
from agent import PLACEMENT_GRID
grid_x = [p[0] for p in PLACEMENT_GRID]
grid_y = [p[1] for p in PLACEMENT_GRID]
ax.scatter(grid_x, grid_y, c=CYAN, s=8, alpha=0.6, zorder=5)

# Draw bridge line (roughly at y=0.45)
ax.axhline(y=(min_y + max_y)/2, color=ORANGE, linestyle='--', alpha=0.5, linewidth=1)
ax.text(max_x + 0.01, (min_y + max_y)/2, 'Bridge', fontsize=9, color=ORANGE, va='center')

# Card slot annotations
card_y = 0.85
for i in range(4):
    cx = min_x + (max_x - min_x) * (i + 0.5) / 4
    card_rect = plt.Rectangle((cx - 0.025, card_y), 0.05, 0.06,
                                linewidth=1.5, edgecolor=GREEN, facecolor=GREEN, alpha=0.3)
    ax.add_patch(card_rect)
    ax.text(cx, card_y + 0.03, f'Card\n{i+1}', ha='center', va='center', fontsize=7, color=GREEN)

# Labels
ax.text((min_x + max_x)/2, min_y - 0.02, 'Enemy Side', ha='center', fontsize=11, color=RED, fontweight='bold')
ax.text((min_x + max_x)/2, max_y + 0.02, 'Your Side', ha='center', fontsize=11, color=GREEN, fontweight='bold')

ax.set_xlim(0.35, 0.8)
ax.set_ylim(0.05, 0.95)
ax.set_aspect('equal')
ax.invert_yaxis()
ax.set_xlabel('X Position (% of screen)', fontsize=11)
ax.set_ylabel('Y Position (% of screen)', fontsize=11)
ax.set_title(f'RL Action Space — {x_steps}×{y_steps} Placement Grid ({NUM_GRID} locations × {NUM_CARD_TYPES+1} cards = {ACTION_DIM:,} actions)',
             fontsize=13, fontweight='bold', pad=15)
ax.grid(True, alpha=0.15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '9_action_space.png'), dpi=200, bbox_inches='tight')
plt.close()
print(f"  [SAVED] 9_action_space.png")


# ═══════════════════════════════════════════════════════════════
#  SECTION 4: SYSTEM OVERVIEW SUMMARY
# ═══════════════════════════════════════════════════════════════
print(f"\n{'━'*65}")
print("  [4/4] SYSTEM OVERVIEW — Summary Dashboard")
print("━" * 65)

# ── Figure 10: Full System Overview Table ────────────────────
fig, ax = plt.subplots(figsize=(16, 8))
ax.axis('off')

summary_data = [
    ['Card Classifier', 'MobileNetV2', 'Hand Card\nIdentification',
     f'{cc_total/1e6:.1f}M', f'{overall_acc:.1f}%', f'{os.path.getsize(MODEL_PATH)/1e6:.1f} MB',
     'Transfer Learning\n(ImageNet)', f'{sum(per_class_count)} images\n{num_classes} classes'],
    ['Enemy Detector', 'YOLOv9-c', 'Enemy Unit\nDetection',
     f'{yolo_total/1e6:.1f}M', 'N/A\n(needs test set)', f'{(os.path.getsize(config.YOLO_MODEL_PATH)/1e6):.1f} MB' if os.path.exists(config.YOLO_MODEL_PATH) else 'N/A',
     'Fine-tuned\n(Custom dataset)', '640×640 input\n238.3 GFLOPs'],
    ['RL Agent', 'Decision\nTransformer', 'Card Play\nStrategy',
     f'{dt_total/1e6:.1f}M', 'Online\nLearning', f'{(os.path.getsize(config.RL_MODEL_PATH)/1e6):.1f} MB' if os.path.exists(config.RL_MODEL_PATH) else 'N/A',
     f'ε-greedy + Self-play\n(ε: 1.0→0.1)', f'State: {STATE_DIM}D\nAction: {ACTION_DIM:,}D'],
    ['OCR Engine', 'EasyOCR\n(CRAFT+LSTM)', 'Tower HP &\nElixir Reading',
     '~30M', 'Pre-trained', '93.7 MB', 'Pre-trained\n(No fine-tuning)', 'English text\nrecognition'],
]

col_labels = ['Component', 'Architecture', 'Task', 'Parameters', 'Accuracy', 'File Size', 'Training\nMethod', 'Key Details']

table = ax.table(cellText=summary_data, colLabels=col_labels,
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.8)

header_colors = [ACCENT, '#1f6feb', GREEN, PURPLE, ORANGE, CYAN, '#f778ba', '#7ee787']
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#30363d')
    cell.set_linewidth(1.5)
    if row == 0:
        cell.set_facecolor(header_colors[col % len(header_colors)])
        cell.set_text_props(color='white', fontweight='bold', fontsize=10)
        cell.set_height(0.08)
    else:
        cell.set_facecolor('#161b22' if row % 2 == 0 else '#0d1117')
        cell.set_text_props(color='#c9d1d9')

fig.suptitle('Clash Royale AI Bot — Complete System Overview', 
             fontsize=18, fontweight='bold', color='#c9d1d9', y=0.95)
fig.text(0.5, 0.08, f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"} | '
         f'PyTorch {torch.__version__} | Total Model Size: {total_size + 93.7:.0f} MB',
         ha='center', fontsize=11, color='#8b949e')

plt.savefig(os.path.join(OUTPUT_DIR, '10_system_overview.png'), dpi=200, bbox_inches='tight')
plt.close()
print(f"  [SAVED] 10_system_overview.png")


# ── Figure 11: Sample Card Predictions ───────────────────────
fig, axes = plt.subplots(2, 5, figsize=(16, 7))
axes = axes.flatten()

# Get one sample per class
shown = set()
idx = 0
for i in range(len(dataset)):
    img, label = dataset[i]
    if label not in shown and idx < 10:
        ax = axes[idx]
        # Denormalize for display
        img_display = img.cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_display = (img_display * std + mean).clip(0, 1)
        
        ax.imshow(img_display)
        pred = all_preds[i]
        correct = pred == label
        title_color = GREEN if correct else RED
        
        ax.set_title(f'True: {class_names[label]}\nPred: {class_names[pred]}',
                     fontsize=9, fontweight='bold', color=title_color)
        ax.axis('off')
        
        conf = all_probs[i][pred] * 100
        ax.text(0.5, -0.05, f'Conf: {conf:.0f}%', transform=ax.transAxes,
                ha='center', fontsize=8, color='#8b949e')
        
        shown.add(label)
        idx += 1

fig.suptitle('Card Classifier — Sample Predictions', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '11_sample_predictions.png'), dpi=200, bbox_inches='tight')
plt.close()
print(f"  [SAVED] 11_sample_predictions.png")


# ═══════════════════════════════════════════════════════════════
#  DONE
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  ALL DONE! {len(os.listdir(OUTPUT_DIR))} figures saved to: {os.path.abspath(OUTPUT_DIR)}/")
print(f"{'='*65}")
print(f"\n  Generated figures:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith('.png'):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"    {f} ({size:.0f} KB)")
print()
