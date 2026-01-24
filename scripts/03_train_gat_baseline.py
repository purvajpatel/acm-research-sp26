"""
Notebook 03: Train Baseline GAT Model
This script trains a Graph Attention Network for fraud detection.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models import GAT
from src.config import MODEL_CONFIG, TRAINING_CONFIG, ARTIFACTS_DIR, ARTIFACT_FILES
from src.utils import set_random_seeds, get_device

set_random_seeds(TRAINING_CONFIG['random_seed'])
device = get_device()

print("="*70)
print("BASELINE GAT MODEL TRAINING")
print("="*70)
print(f"Device: {device}\n")

# Load graph
print("Loading baseline graph...")
graph_path = ARTIFACTS_DIR / ARTIFACT_FILES['baseline_graph']
data = torch.load(graph_path, weights_only=False).to(device)
print(f"✓ Graph: {data}")

# Create splits
print("\nCreating train/val/test splits...")
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

print(f"✓ Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")

# Preprocess features
print("\nPreprocessing features...")
nan_mask = torch.isnan(data.x)
if nan_mask.any():
    data.x = torch.where(nan_mask, torch.zeros_like(data.x), data.x)
    print(f"✓ Replaced {nan_mask.sum().item()} NaN values")

inf_mask = torch.isinf(data.x)
if inf_mask.any():
    data.x = torch.where(inf_mask, torch.zeros_like(data.x), data.x)
    print(f"✓ Replaced {inf_mask.sum().item()} Inf values")

# Normalize
train_x = data.x[data.train_mask]
mean = train_x.mean(dim=0, keepdim=True)
std = train_x.std(dim=0, keepdim=True)
std = torch.where(std == 0, torch.ones_like(std), std)
data.x = (data.x - mean) / std
data.x = torch.clamp(data.x, min=-10, max=10)
print(f"✓ Features normalized and clipped")

# Initialize model
print("\nInitializing baseline GAT model...")
model = GAT(
    in_channels=data.num_node_features,
    hidden_channels=MODEL_CONFIG['hidden_channels'],
    out_channels=MODEL_CONFIG['out_channels'],
    num_heads=MODEL_CONFIG['num_heads'],
    num_layers=MODEL_CONFIG['num_layers'],
    dropout=MODEL_CONFIG['dropout']
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=TRAINING_CONFIG['learning_rate'], 
    weight_decay=TRAINING_CONFIG['weight_decay']
)
criterion = nn.CrossEntropyLoss()

print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training functions
def train_epoch():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out[mask].argmax(dim=1)
    prob = F.softmax(out[mask], dim=1)[:, 1]
    
    y_true = data.y[mask].cpu().numpy()
    y_pred = pred.cpu().numpy()
    y_prob = prob.cpu().numpy()
    
    y_prob = np.nan_to_num(y_prob, nan=0.5)
    
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

# Train model
print("\nTraining baseline model...")
print("-" * 70)

history = {'train_loss': [], 'val_metrics': []}
best_val_f1 = 0
patience_counter = 0
EPOCHS = TRAINING_CONFIG['epochs']
PATIENCE = TRAINING_CONFIG['patience']

start_time = time.time()
for epoch in range(1, EPOCHS + 1):
    loss = train_epoch()
    val_metrics = evaluate(data.val_mask)
    
    history['train_loss'].append(loss)
    history['val_metrics'].append(val_metrics)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val F1: {val_metrics['f1']:.4f} | "
              f"Val AUC: {val_metrics['roc_auc']:.4f}")
    
    if val_metrics['f1'] > best_val_f1:
        best_val_f1 = val_metrics['f1']
        patience_counter = 0
        save_path = ARTIFACTS_DIR / ARTIFACT_FILES['baseline_model']
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_f1': best_val_f1,
            'val_metrics': val_metrics
        }, save_path)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

training_time = time.time() - start_time
print(f"\n✓ Training completed in {training_time:.2f}s")
print(f"✓ Best Val F1: {best_val_f1:.4f}")

# Evaluate
print("\n" + "="*70)
print("FINAL EVALUATION")
print("="*70)

checkpoint = torch.load(ARTIFACTS_DIR / ARTIFACT_FILES['baseline_model'], weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

train_metrics = evaluate(data.train_mask)
val_metrics = evaluate(data.val_mask)
test_metrics = evaluate(data.test_mask)

print("\nBaseline GAT Results:")
print(f"  Train - Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['roc_auc']:.4f}")
print(f"  Val   - Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['roc_auc']:.4f}")
print(f"  Test  - Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['roc_auc']:.4f}")

# Save metrics
metrics_dict = {
    'model_type': 'GAT_Baseline',
    'training_time': training_time,
    'best_epoch': checkpoint['epoch'],
    'performance': {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics
    }
}

metrics_path = ARTIFACTS_DIR / ARTIFACT_FILES['baseline_metrics']
with open(metrics_path, 'w') as f:
    json.dump(metrics_dict, f, indent=2)
print(f"\n✓ Metrics saved to {metrics_path}")

print("\n" + "="*70)
print("✅ BASELINE TRAINING COMPLETE!")
print("="*70)
print("\nNext: Run 04_eval_baseline.py")
