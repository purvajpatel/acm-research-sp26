"""
Complete Training Pipeline: Baseline -> Quantum Feature Mapping -> Quantum Training
This script runs notebooks 03, 05, and 06 in sequence to properly train both models.
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
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from src.models import GAT
from src.quantum_features import LearnableQuantumFeatureMap
from src.config import (MODEL_CONFIG, QUANTUM_MODEL_CONFIG, QUANTUM_CONFIG, TRAINING_CONFIG, 
                        ARTIFACTS_DIR, FIGURES_DIR, ARTIFACT_FILES, FIGURE_FILES)
from src.utils import set_random_seeds, get_device
from torch_geometric.data import Data

# Ensure directories exist
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set random seeds
set_random_seeds(TRAINING_CONFIG['random_seed'])
device = get_device()

print("="*70)
print("QUANTUM BLOCKCHAIN FRAUD DETECTION - COMPLETE TRAINING PIPELINE")
print("="*70)
print(f"Device: {device}")
print(f"Random seed: {TRAINING_CONFIG['random_seed']}")

# ============================================================================
# STEP 1: TRAIN BASELINE GAT MODEL (Notebook 03)
# ============================================================================
print("\n" + "="*70)
print("STEP 1: TRAINING BASELINE GAT MODEL")
print("="*70)

print("\nLoading baseline graph...")
graph_path = ARTIFACTS_DIR / ARTIFACT_FILES['baseline_graph']
data = torch.load(graph_path, weights_only=False).to(device)
print(f"Graph loaded: {data}")

# Get labeled indices and create splits
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

# Initialize baseline model
print("\nInitializing baseline GAT model...")
model_baseline = GAT(
    in_channels=data.num_node_features,
    hidden_channels=MODEL_CONFIG['hidden_channels'],
    out_channels=MODEL_CONFIG['out_channels'],
    num_heads=MODEL_CONFIG['num_heads'],
    num_layers=MODEL_CONFIG['num_layers'],
    dropout=MODEL_CONFIG['dropout']
).to(device)

optimizer_baseline = torch.optim.Adam(
    model_baseline.parameters(), 
    lr=TRAINING_CONFIG['learning_rate'], 
    weight_decay=TRAINING_CONFIG['weight_decay']
)
criterion = nn.CrossEntropyLoss()

print(f"✓ Model parameters: {sum(p.numel() for p in model_baseline.parameters()):,}")

# Training functions
def train_epoch(model, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, mask):
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

# Train baseline model
print("\nTraining baseline model...")
history_baseline = {'train_loss': [], 'val_metrics': []}
best_val_f1 = 0
patience_counter = 0
EPOCHS = TRAINING_CONFIG['epochs']
PATIENCE = TRAINING_CONFIG['patience']

start_time = time.time()
for epoch in range(1, EPOCHS + 1):
    loss = train_epoch(model_baseline, optimizer_baseline)
    val_metrics = evaluate(model_baseline, data.val_mask)
    
    history_baseline['train_loss'].append(loss)
    history_baseline['val_metrics'].append(val_metrics)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val F1: {val_metrics['f1']:.4f} | "
              f"Val AUC: {val_metrics['roc_auc']:.4f}")
    
    if val_metrics['f1'] > best_val_f1:
        best_val_f1 = val_metrics['f1']
        patience_counter = 0
        save_path = ARTIFACTS_DIR / ARTIFACT_FILES['baseline_model']
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_baseline.state_dict(),
            'val_f1': best_val_f1,
            'val_metrics': val_metrics
        }, save_path)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

training_time_baseline = time.time() - start_time
print(f"✓ Baseline training completed in {training_time_baseline:.2f}s")
print(f"✓ Best Val F1: {best_val_f1:.4f}")

# Evaluate baseline
model_path = ARTIFACTS_DIR / ARTIFACT_FILES['baseline_model']
checkpoint = torch.load(model_path, weights_only=False)
model_baseline.load_state_dict(checkpoint['model_state_dict'])

train_metrics = evaluate(model_baseline, data.train_mask)
val_metrics = evaluate(model_baseline, data.val_mask)
test_metrics = evaluate(model_baseline, data.test_mask)

print("\nBaseline GAT Results:")
print(f"  Train - F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['roc_auc']:.4f}")
print(f"  Val   - F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['roc_auc']:.4f}")
print(f"  Test  - F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['roc_auc']:.4f}")

# Save baseline metrics
metrics_dict = {
    'model_type': 'GAT_Baseline',
    'training_time': training_time_baseline,
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
print(f"✓ Baseline metrics saved to {metrics_path}")

# ============================================================================
# STEP 2: QUANTUM FEATURE MAPPING (Notebook 05)
# ============================================================================
print("\n" + "="*70)
print("STEP 2: APPLYING QUANTUM FEATURE MAPPING")
print("="*70)

# Reload baseline graph for quantum transformation
print("\nLoading baseline graph for quantum transformation...")
data = torch.load(graph_path, weights_only=False).to(device)

# Apply the same splits
data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask

# Check and handle NaN
nan_count = torch.isnan(data.x).sum().item()
if nan_count > 0:
    print(f"✓ Replacing {nan_count} NaN values...")
    data.x = torch.nan_to_num(data.x, nan=0.0)

print(f"Original features: {data.x.shape}")
print(f"Original range: [{data.x.min():.4f}, {data.x.max():.4f}]")

# Normalize features
print("\nNormalizing features...")
train_x = data.x[data.train_mask]
mean = train_x.mean(dim=0, keepdim=True)
std = train_x.std(dim=0, keepdim=True)
std = torch.where(std == 0, torch.ones_like(std), std)
data.x = (data.x - mean) / std
data.x = torch.clamp(data.x, min=-10, max=10)
print(f"✓ Normalized range: [{data.x.min():.4f}, {data.x.max():.4f}]")

# Initialize quantum feature mapper
print("\nInitializing learnable quantum feature mapper...")
feature_mapper = LearnableQuantumFeatureMap(
    input_dim=data.num_node_features,
    expansion_factor=QUANTUM_CONFIG['expansion_factor'],
    use_fourier=QUANTUM_CONFIG['fourier_features'],
    random_seed=TRAINING_CONFIG['random_seed']
).to(device)

print(f"✓ Mapping: {data.num_node_features} -> {feature_mapper.output_dim}")
print(f"✓ Learnable parameters: {sum(p.numel() for p in feature_mapper.parameters()):,}")

# Apply quantum transformation
x_quantum = feature_mapper(data.x)
print(f"\n✓ Quantum features: {x_quantum.shape}")
print(f"  Mean: {x_quantum.mean().item():.4f}, Std: {x_quantum.std().item():.4f}")
print(f"  Min: {x_quantum.min().item():.4f}, Max: {x_quantum.max().item():.4f}")

# Create quantum graph
print("\nCreating quantum graph data object...")
data_quantum = Data(
    x=x_quantum,
    edge_index=data.edge_index,
    y=data.y,
    timestep=data.timestep,
    labeled_mask=data.labeled_mask,
    unlabeled_mask=data.unlabeled_mask,
    train_mask=data.train_mask,
    val_mask=data.val_mask,
    test_mask=data.test_mask
)
data_quantum.feature_mapper = feature_mapper

print(f"✓ Quantum graph: {data_quantum}")

# Save quantum graph
save_path = ARTIFACTS_DIR / ARTIFACT_FILES['quantum_graph']
torch.save(data_quantum, save_path)
print(f"✓ Quantum graph saved to {save_path}")

# ============================================================================
# STEP 3: TRAIN QUANTUM GAT MODEL (Notebook 06)
# ============================================================================
print("\n" + "="*70)
print("STEP 3: TRAINING QUANTUM GAT MODEL")
print("="*70)

# Initialize quantum model with QUANTUM_MODEL_CONFIG
print("\nInitializing quantum GAT model...")
model_quantum = GAT(
    in_channels=data_quantum.num_node_features,
    hidden_channels=QUANTUM_MODEL_CONFIG['hidden_channels'],
    out_channels=QUANTUM_MODEL_CONFIG['out_channels'],
    num_heads=QUANTUM_MODEL_CONFIG['num_heads'],
    num_layers=QUANTUM_MODEL_CONFIG['num_layers'],
    dropout=QUANTUM_MODEL_CONFIG['dropout']
).to(device)

optimizer_quantum = torch.optim.Adam(
    model_quantum.parameters(), 
    lr=TRAINING_CONFIG['learning_rate'], 
    weight_decay=TRAINING_CONFIG['weight_decay']
)

print(f"✓ Model parameters: {sum(p.numel() for p in model_quantum.parameters()):,}")
print(f"✓ Architecture: {QUANTUM_MODEL_CONFIG['num_layers']} layers, "
      f"{QUANTUM_MODEL_CONFIG['num_heads']} heads, "
      f"{QUANTUM_MODEL_CONFIG['hidden_channels']} hidden channels")

# Training functions for quantum model
def train_epoch_quantum(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate_quantum(model, mask, data):
    model.eval()
    out = model(data.x, data.edge_index)
    
    # Check for NaN/Inf in output
    if torch.isnan(out).any() or torch.isinf(out).any():
        print("⚠️  Warning: NaN/Inf detected in model output")
        out = torch.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0)
    
    pred = out[mask].argmax(dim=1)
    prob = F.softmax(out[mask], dim=1)[:, 1]
    
    y_true = data.y[mask].cpu().numpy()
    y_pred = pred.cpu().numpy()
    y_prob = prob.cpu().numpy()
    
    # Handle NaN/Inf in probabilities
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

# Train quantum model
print("\nTraining quantum model...")
history_quantum = {'train_loss': [], 'val_metrics': []}
best_val_f1 = 0
patience_counter = 0

start_time = time.time()
for epoch in range(1, EPOCHS + 1):
    loss = train_epoch_quantum(model_quantum, optimizer_quantum, data_quantum)
    val_metrics = evaluate_quantum(model_quantum, data_quantum.val_mask, data_quantum)
    
    history_quantum['train_loss'].append(loss)
    history_quantum['val_metrics'].append(val_metrics)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f} | Val F1: {val_metrics['f1']:.4f} | "
              f"Val AUC: {val_metrics['roc_auc']:.4f}")
    
    if val_metrics['f1'] > best_val_f1:
        best_val_f1 = val_metrics['f1']
        patience_counter = 0
        save_path = ARTIFACTS_DIR / ARTIFACT_FILES['quantum_model']
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_quantum.state_dict(),
            'val_f1': best_val_f1,
            'val_metrics': val_metrics
        }, save_path)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

training_time_quantum = time.time() - start_time
print(f"✓ Quantum training completed in {training_time_quantum:.2f}s")
print(f"✓ Best Val F1: {best_val_f1:.4f}")

# Evaluate quantum model
model_path = ARTIFACTS_DIR / ARTIFACT_FILES['quantum_model']
checkpoint = torch.load(model_path, weights_only=False)
model_quantum.load_state_dict(checkpoint['model_state_dict'])

train_metrics = evaluate_quantum(model_quantum, data_quantum.train_mask, data_quantum)
val_metrics = evaluate_quantum(model_quantum, data_quantum.val_mask, data_quantum)
test_metrics = evaluate_quantum(model_quantum, data_quantum.test_mask, data_quantum)

print("\nQuantum GAT Results:")
print(f"  Train - F1: {train_metrics['f1']:.4f}, AUC: {train_metrics['roc_auc']:.4f}")
print(f"  Val   - F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['roc_auc']:.4f}")
print(f"  Test  - F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['roc_auc']:.4f}")

# Save quantum metrics
metrics_dict = {
    'model_type': 'GAT_Quantum',
    'training_time': training_time_quantum,
    'best_epoch': checkpoint['epoch'],
    'performance': {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics
    }
}

metrics_path = ARTIFACTS_DIR / ARTIFACT_FILES['quantum_metrics']
with open(metrics_path, 'w') as f:
    json.dump(metrics_dict, f, indent=2)
print(f"✓ Quantum metrics saved to {metrics_path}")

# ============================================================================
# FINAL COMPARISON
# ============================================================================
print("\n" + "="*70)
print("FINAL COMPARISON: BASELINE vs QUANTUM")
print("="*70)

# Load saved metrics
with open(ARTIFACTS_DIR / ARTIFACT_FILES['baseline_metrics'], 'r') as f:
    baseline_final = json.load(f)
with open(ARTIFACTS_DIR / ARTIFACT_FILES['quantum_metrics'], 'r') as f:
    quantum_final = json.load(f)

print("\nTEST SET RESULTS:")
print(f"{'Metric':<15} {'Baseline':<12} {'Quantum':<12} {'Improvement':<12}")
print("-" * 60)

for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
    base_val = baseline_final['performance']['test'][metric]
    quant_val = quantum_final['performance']['test'][metric]
    improvement = quant_val - base_val
    print(f"{metric:<15} {base_val:<12.4f} {quant_val:<12.4f} {improvement:+.4f}")

print("\n" + "="*70)
print("✅ COMPLETE TRAINING PIPELINE FINISHED!")
print("="*70)
print(f"\n📊 Key Improvements:")
print(f"  • Feature expansion: {data.num_node_features} → {data_quantum.num_node_features} features")
print(f"  • Test F1 improvement: {quantum_final['performance']['test']['f1'] - baseline_final['performance']['test']['f1']:+.4f}")
print(f"  • Test AUC improvement: {quantum_final['performance']['test']['roc_auc'] - baseline_final['performance']['test']['roc_auc']:+.4f}")
print(f"\n✅ Next: Run notebook 07_eval_quantum.ipynb or script run_07_eval_quantum.py for detailed analysis")
