"""
Run the quantum GAT evaluation and comparison from 07_eval_quantum.ipynb as a script
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from pathlib import Path

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from src.models import GAT
from src.config import MODEL_CONFIG, QUANTUM_MODEL_CONFIG, TRAINING_CONFIG, ARTIFACTS_DIR, FIGURES_DIR, ARTIFACT_FILES, FIGURE_FILES
from src.utils import get_device

device = get_device()
print("✓ Setup complete!")
print(f"Using device: {device}")

# Load baseline and quantum metrics
print("\n" + "="*60)
print("LOADING METRICS")
print("="*60)

baseline_path = ARTIFACTS_DIR / ARTIFACT_FILES['baseline_metrics']
with open(baseline_path, 'r') as f:
    baseline_metrics = json.load(f)

quantum_path = ARTIFACTS_DIR / ARTIFACT_FILES['quantum_metrics']
with open(quantum_path, 'r') as f:
    quantum_metrics_raw = json.load(f)

# Handle different metric file formats
if 'performance' not in quantum_metrics_raw:
    quantum_metrics = {
        'performance': {
            'test': quantum_metrics_raw
        }
    }
else:
    quantum_metrics = quantum_metrics_raw

print(f"✓ Baseline test accuracy: {baseline_metrics['performance']['test']['accuracy']:.4f}")
print(f"✓ Quantum test accuracy: {quantum_metrics['performance']['test']['accuracy']:.4f}")

# Compare baseline vs quantum performance
print("\n" + "="*60)
print("BASELINE vs QUANTUM GAT COMPARISON")
print("="*60)

metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

for split in ['test']:
    print(f"\n{split.upper()} SET:")
    print(f"{'Metric':<15} {'Baseline':<12} {'Quantum':<12} {'Improvement':<12}")
    print("-" * 60)
    
    for metric in metrics_to_compare:
        baseline_val = baseline_metrics['performance'][split][metric]
        quantum_val = quantum_metrics['performance'][split][metric]
        improvement = quantum_val - baseline_val
        
        print(f"{metric:<15} {baseline_val:<12.4f} {quantum_val:<12.4f} {improvement:+.4f}")

# Visualize performance comparison
print("\n" + "="*60)
print("GENERATING COMPARISON CHART")
print("="*60)

metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
baseline_vals = [baseline_metrics['performance']['test'][m] for m in metrics_to_compare]
quantum_vals = [quantum_metrics['performance']['test'][m] for m in metrics_to_compare]

x = np.arange(len(metrics_names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline GAT', alpha=0.8, color='#1f77b4')
bars2 = ax.bar(x + width/2, quantum_vals, width, label='Quantum GAT', alpha=0.8, color='#ff7f0e')

ax.set_xlabel('Metric', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Baseline vs Quantum GAT Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0, 1.05])

plt.tight_layout()
save_path = FIGURES_DIR / FIGURE_FILES['comparison']
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"✓ Comparison chart saved to {save_path}")
plt.close()

# Load quantum model and data
print("\n" + "="*60)
print("LOADING QUANTUM MODEL")
print("="*60)

graph_path = ARTIFACTS_DIR / ARTIFACT_FILES['quantum_graph']
data = torch.load(graph_path, map_location=device, weights_only=False)

# Create test mask if it doesn't exist
if not hasattr(data, 'test_mask'):
    print("Creating train/val/test splits...")
    labeled_indices = torch.where(data.labeled_mask)[0].cpu().numpy()
    labeled_y = data.y[data.labeled_mask].cpu().numpy()
    
    train_val_idx, test_idx = train_test_split(
        labeled_indices, test_size=0.2, 
        random_state=TRAINING_CONFIG['random_seed'],
        stratify=labeled_y
    )
    
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask[test_idx] = True
    data.test_mask = test_mask.to(device)
    print(f"  Test set size: {test_mask.sum()}")

model = GAT(
    in_channels=data.num_node_features,
    hidden_channels=QUANTUM_MODEL_CONFIG['hidden_channels'],
    out_channels=QUANTUM_MODEL_CONFIG['out_channels'],
    num_heads=QUANTUM_MODEL_CONFIG['num_heads'],
    num_layers=QUANTUM_MODEL_CONFIG['num_layers'],
    dropout=QUANTUM_MODEL_CONFIG['dropout']
).to(device)

model_path = ARTIFACTS_DIR / ARTIFACT_FILES['quantum_model']
checkpoint = torch.load(model_path, map_location=device, weights_only=False)

# Handle different checkpoint formats
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
    
model.eval()

print(f"✓ Quantum model loaded with {data.num_node_features} features")
print(f"  Architecture: {QUANTUM_MODEL_CONFIG['num_layers']} layers, {QUANTUM_MODEL_CONFIG['num_heads']} heads, {QUANTUM_MODEL_CONFIG['hidden_channels']} hidden channels")

# Generate predictions
print("\nGenerating predictions...")
@torch.no_grad()
def get_predictions():
    out = model(data.x, data.edge_index)
    probs = F.softmax(out, dim=1)
    preds = out.argmax(dim=1)
    return preds, probs

predictions, probabilities = get_predictions()
print(f"✓ Generated predictions for {data.num_nodes} nodes")

# Check for NaN/Inf values
if torch.isnan(probabilities).any():
    print(f"⚠️  Warning: {torch.isnan(probabilities).sum().item()} NaN values found in probabilities")
if torch.isinf(probabilities).any():
    print(f"⚠️  Warning: {torch.isinf(probabilities).sum().item()} Inf values found in probabilities")

# Generate confusion matrix for quantum model
print("\n" + "="*60)
print("QUANTUM MODEL CONFUSION MATRIX")
print("="*60)

y_test = data.y[data.test_mask].cpu().numpy()
y_pred = predictions[data.test_mask].cpu().numpy()

cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=['Licit', 'Illicit'],
            yticklabels=['Licit', 'Illicit'],
            cbar_kws={'label': 'Count'})
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.title('Confusion Matrix - Quantum GAT', fontsize=14, fontweight='bold')
plt.tight_layout()
save_path = FIGURES_DIR / FIGURE_FILES['quantum_confusion']
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"✓ Confusion matrix saved to {save_path}")
plt.close()

# Compare ROC curves
print("\n" + "="*60)
print("LOADING BASELINE MODEL FOR ROC COMPARISON")
print("="*60)

baseline_graph_path = ARTIFACTS_DIR / ARTIFACT_FILES['baseline_graph']
data_baseline = torch.load(baseline_graph_path, map_location=device, weights_only=False)

# Create test mask for baseline if it doesn't exist
if not hasattr(data_baseline, 'test_mask'):
    labeled_indices = torch.where(data_baseline.labeled_mask)[0].cpu().numpy()
    labeled_y = data_baseline.y[data_baseline.labeled_mask].cpu().numpy()
    
    train_val_idx, test_idx = train_test_split(
        labeled_indices, test_size=0.2, 
        random_state=TRAINING_CONFIG['random_seed'],
        stratify=labeled_y
    )
    
    test_mask = torch.zeros(data_baseline.num_nodes, dtype=torch.bool)
    test_mask[test_idx] = True
    data_baseline.test_mask = test_mask.to(device)

model_baseline = GAT(
    in_channels=data_baseline.num_node_features,
    hidden_channels=MODEL_CONFIG['hidden_channels'],
    out_channels=MODEL_CONFIG['out_channels'],
    num_heads=MODEL_CONFIG['num_heads'],
    num_layers=MODEL_CONFIG['num_layers'],
    dropout=MODEL_CONFIG['dropout']
).to(device)

baseline_model_path = ARTIFACTS_DIR / ARTIFACT_FILES['baseline_model']
checkpoint_baseline = torch.load(baseline_model_path, map_location=device, weights_only=False)

# Handle different checkpoint formats
if isinstance(checkpoint_baseline, dict) and 'model_state_dict' in checkpoint_baseline:
    model_baseline.load_state_dict(checkpoint_baseline['model_state_dict'])
else:
    model_baseline.load_state_dict(checkpoint_baseline)
    
model_baseline.eval()
print("✓ Baseline model loaded")

@torch.no_grad()
def get_baseline_probs():
    out = model_baseline(data_baseline.x, data_baseline.edge_index)
    return F.softmax(out, dim=1)

baseline_probs = get_baseline_probs()

# Get test set probabilities
y_test_baseline = data_baseline.y[data_baseline.test_mask].cpu().numpy()
y_prob_baseline = baseline_probs[data_baseline.test_mask][:, 1].cpu().numpy()

y_test_quantum = data.y[data.test_mask].cpu().numpy()
y_prob_quantum = probabilities[data.test_mask][:, 1].cpu().numpy()

# Check for and handle NaN/Inf values
def clean_arrays(*arrays):
    """Remove NaN and Inf values from all arrays at the same indices"""
    masks = [~(np.isnan(arr) | np.isinf(arr)) for arr in arrays]
    combined_mask = np.logical_and.reduce(masks)
    return tuple(arr[combined_mask] for arr in arrays)

y_test_baseline, y_prob_baseline = clean_arrays(y_test_baseline, y_prob_baseline)
y_test_quantum, y_prob_quantum = clean_arrays(y_test_quantum, y_prob_quantum)

# Calculate ROC curves
print("\n" + "="*60)
print("GENERATING ROC CURVES")
print("="*60)

fpr_baseline, tpr_baseline, _ = roc_curve(y_test_baseline, y_prob_baseline)
fpr_quantum, tpr_quantum, _ = roc_curve(y_test_quantum, y_prob_quantum)

auc_baseline = auc(fpr_baseline, tpr_baseline)
auc_quantum = auc(fpr_quantum, tpr_quantum)

# Plot ROC curves
plt.figure(figsize=(10, 6))
plt.plot(fpr_baseline, tpr_baseline, lw=2, label=f'Baseline GAT (AUC = {auc_baseline:.4f})', color='#1f77b4')
plt.plot(fpr_quantum, tpr_quantum, lw=2, label=f'Quantum GAT (AUC = {auc_quantum:.4f})', color='#ff7f0e')
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
save_path = FIGURES_DIR / FIGURE_FILES['roc_comparison']
plt.savefig(save_path, dpi=150, bbox_inches='tight')

print(f"Baseline AUC: {auc_baseline:.4f}")
print(f"Quantum AUC:  {auc_quantum:.4f}")
print(f"Improvement:  {auc_quantum - auc_baseline:+.4f}")
print(f"✓ ROC comparison saved to {save_path}")
plt.close()

# Summary
print("\n" + "="*60)
print("EVALUATION COMPLETE")
print("="*60)
print("\n📊 Key Findings:")
print(f"  1. Quantum feature expansion: {data_baseline.num_node_features} → {data.num_node_features} features")
print(f"  2. Test F1 improvement: {quantum_metrics['performance']['test']['f1'] - baseline_metrics['performance']['test']['f1']:+.4f}")
print(f"  3. Test AUC improvement: {auc_quantum - auc_baseline:+.4f}")
print(f"  4. Test accuracy improvement: {quantum_metrics['performance']['test']['accuracy'] - baseline_metrics['performance']['test']['accuracy']:+.4f}")
print("\n✅ Next step: Proceed to 08_explain_llm.ipynb for explainability analysis")
