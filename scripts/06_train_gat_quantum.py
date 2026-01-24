"""
IMPROVED Quantum Model Training - With Better Hyperparameters
This script retrains ONLY the quantum model with optimized settings for better performance.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import GAT
from src.config import QUANTUM_MODEL_CONFIG, TRAINING_CONFIG, ARTIFACTS_DIR, ARTIFACT_FILES
from src.utils import set_random_seeds, get_device

# Set random seeds
set_random_seeds(TRAINING_CONFIG['random_seed'])
device = get_device()

print("="*70)
print("IMPROVED QUANTUM GAT TRAINING")
print("="*70)
print(f"Device: {device}\n")

# Load quantum graph
print("Loading quantum graph...")
data = torch.load(ARTIFACTS_DIR / ARTIFACT_FILES['quantum_graph'], weights_only=False).to(device)

# Check if masks exist, if not create them from baseline
if not hasattr(data, 'train_mask'):
    print("Creating train/val/test splits...")
    from sklearn.model_selection import train_test_split
    
    labeled_indices = torch.where(data.labeled_mask)[0].cpu().numpy()
    labeled_y = data.y[data.labeled_mask].cpu().numpy()
    
    train_val_idx, test_idx = train_test_split(
        labeled_indices, 
        test_size=TRAINING_CONFIG['train_test_split'], 
        random_state=TRAINING_CONFIG['random_seed'], 
        stratify=labeled_y
    )
    
    train_val_y = data.y[train_val_idx].cpu().numpy()
    train_idx, val_idx = train_test_split(
        train_val_idx, 
        test_size=TRAINING_CONFIG['train_val_split'], 
        random_state=TRAINING_CONFIG['random_seed'], 
        stratify=train_val_y
    )
    
    # Create masks
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    print(f"✓ Created splits - Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")

print(f"✓ Graph: {data}")
print(f"  Feature stats - Mean: {data.x.mean():.4f}, Std: {data.x.std():.4f}")

# IMPROVED MODEL CONFIGURATION
# The issue: quantum model was too complex and undertrained
# Solution: Better balance of architecture and training
IMPROVED_CONFIG = {
    'hidden_channels': 96,      # Reduced from 128 (was too large)
    'num_heads': 4,             # Reduced from 6 (more stable)
    'num_layers': 2,            # Reduced from 3 (simpler is better)
    'dropout': 0.3,             # Reduced from 0.4 (less regularization)
    'learning_rate': 0.001,     # Slightly lower for stability
    'weight_decay': 5e-4,       # Lower regularization
}

print("\nImproved Model Configuration:")
print(f"  Hidden channels: {IMPROVED_CONFIG['hidden_channels']}")
print(f"  Attention heads: {IMPROVED_CONFIG['num_heads']}")
print(f"  Layers: {IMPROVED_CONFIG['num_layers']}")
print(f"  Dropout: {IMPROVED_CONFIG['dropout']}")
print(f"  Learning rate: {IMPROVED_CONFIG['learning_rate']}")
print(f"  Weight decay: {IMPROVED_CONFIG['weight_decay']}")

# Initialize model
print("\nInitializing improved quantum GAT model...")
model = GAT(
    in_channels=data.num_node_features,
    hidden_channels=IMPROVED_CONFIG['hidden_channels'],
    out_channels=2,
    num_heads=IMPROVED_CONFIG['num_heads'],
    num_layers=IMPROVED_CONFIG['num_layers'],
    dropout=IMPROVED_CONFIG['dropout']
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=IMPROVED_CONFIG['learning_rate'], 
    weight_decay=IMPROVED_CONFIG['weight_decay']
)

# Use weighted loss to handle class imbalance
n_licit = (data.y[data.train_mask] == 0).sum().item()
n_illicit = (data.y[data.train_mask] == 1).sum().item()
weight = torch.tensor([1.0, n_licit / n_illicit], device=device)
criterion = nn.CrossEntropyLoss(weight=weight)

print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"✓ Class weights: Licit=1.0, Illicit={weight[1].item():.2f}")

# Training functions
def train_epoch():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(mask):
    model.eval()
    out = model(data.x, data.edge_index)
    
    # Check for NaN/Inf
    if torch.isnan(out).any() or torch.isinf(out).any():
        out = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
    
    pred = out[mask].argmax(dim=1)
    prob = F.softmax(out[mask], dim=1)[:, 1]
    
    y_true = data.y[mask].cpu().numpy()
    y_pred = pred.cpu().numpy()
    y_prob = prob.cpu().numpy()
    
    y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except (ValueError, RuntimeError):
        roc_auc = 0.0
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc
    }

# Training loop with better patience
print("\nTraining improved quantum model...")
print("-" * 70)

history = {'train_loss': [], 'val_metrics': []}
best_val_f1 = 0
patience_counter = 0
EPOCHS = 300  # More epochs allowed
PATIENCE = 50  # More patience for convergence

start_time = time.time()
for epoch in range(1, EPOCHS + 1):
    loss = train_epoch()
    val_metrics = evaluate(data.val_mask)
    
    history['train_loss'].append(loss)
    history['val_metrics'].append(val_metrics)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val F1: {val_metrics['f1']:.4f} | "
              f"Val AUC: {val_metrics['roc_auc']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f}")
    
    if val_metrics['f1'] > best_val_f1:
        best_val_f1 = val_metrics['f1']
        patience_counter = 0
        # Save best model
        save_path = ARTIFACTS_DIR / ARTIFACT_FILES['quantum_model']
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_f1': best_val_f1,
            'val_metrics': val_metrics,
            'config': IMPROVED_CONFIG
        }, save_path)
        print(f"         ✓ New best Val F1: {best_val_f1:.4f} (saved)")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

training_time = time.time() - start_time
print(f"\n✓ Training completed in {training_time:.2f}s")
print(f"✓ Best Val F1: {best_val_f1:.4f}")

# Load best model and evaluate
print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

checkpoint = torch.load(ARTIFACTS_DIR / ARTIFACT_FILES['quantum_model'], weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

train_metrics = evaluate(data.train_mask)
val_metrics = evaluate(data.val_mask)
test_metrics = evaluate(data.test_mask)

print("\nImproved Quantum GAT Results:")
print(f"  Train - Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['roc_auc']:.4f}")
print(f"  Val   - Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['roc_auc']:.4f}")
print(f"  Test  - Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['roc_auc']:.4f}")

# Load baseline metrics for comparison
with open(ARTIFACTS_DIR / ARTIFACT_FILES['baseline_metrics'], 'r') as f:
    baseline_metrics = json.load(f)

print("\n" + "="*70)
print("COMPARISON: BASELINE vs IMPROVED QUANTUM")
print("="*70)
print(f"\n{'Metric':<15} {'Baseline':<12} {'Quantum':<12} {'Improvement':<12}")
print("-" * 60)

for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
    base_val = baseline_metrics['performance']['test'][metric]
    quant_val = test_metrics[metric]
    improvement = quant_val - base_val
    symbol = "✓" if improvement > 0 else "✗"
    print(f"{metric:<15} {base_val:<12.4f} {quant_val:<12.4f} {improvement:+.4f} {symbol}")

# Save improved metrics
metrics_dict = {
    'model_type': 'GAT_Quantum_Improved',
    'training_time': training_time,
    'best_epoch': checkpoint['epoch'],
    'config': IMPROVED_CONFIG,
    'performance': {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics
    }
}

metrics_path = ARTIFACTS_DIR / ARTIFACT_FILES['quantum_metrics']
with open(metrics_path, 'w') as f:
    json.dump(metrics_dict, f, indent=2)
print(f"\n✓ Metrics saved to {metrics_path}")

# Summary
print("\n" + "="*70)
if test_metrics['f1'] > baseline_metrics['performance']['test']['f1']:
    print("✅ SUCCESS! Quantum model outperforms baseline!")
    print(f"   F1 improvement: +{test_metrics['f1'] - baseline_metrics['performance']['test']['f1']:.4f}")
else:
    print("⚠️  Quantum model still needs improvement")
    print(f"   F1 difference: {test_metrics['f1'] - baseline_metrics['performance']['test']['f1']:.4f}")
print("="*70)
