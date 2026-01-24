"""
Notebook 05: Quantum Feature Mapping
This script applies quantum-inspired feature expansion using Random Fourier Features.
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.quantum_features import LearnableQuantumFeatureMap
from src.config import QUANTUM_CONFIG, TRAINING_CONFIG, ARTIFACTS_DIR, FIGURES_DIR, ARTIFACT_FILES, FIGURE_FILES
from src.utils import get_device
from torch_geometric.data import Data

device = get_device()

print("="*70)
print("QUANTUM-INSPIRED FEATURE MAPPING")
print("="*70)
print(f"Device: {device}\n")

# Load baseline graph
print("Loading baseline graph...")
graph_path = ARTIFACTS_DIR / ARTIFACT_FILES['baseline_graph']
data = torch.load(graph_path, weights_only=False).to(device)

print(f"Original features: {data.x.shape}")
print(f"Original range: [{data.x.min():.4f}, {data.x.max():.4f}]")

# Handle NaN
nan_count = torch.isnan(data.x).sum().item()
if nan_count > 0:
    print(f"✓ Replacing {nan_count} NaN values...")
    data.x = torch.nan_to_num(data.x, nan=0.0)

# Normalize features BEFORE quantum transformation
print("\nNormalizing features...")
if hasattr(data, 'train_mask'):
    train_x = data.x[data.train_mask]
else:
    train_x = data.x[data.labeled_mask] if hasattr(data, 'labeled_mask') else data.x

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
print("\nApplying quantum transformation...")
x_quantum = feature_mapper(data.x)
print(f"✓ Quantum features: {x_quantum.shape}")
print(f"  Mean: {x_quantum.mean().item():.4f}, Std: {x_quantum.std().item():.4f}")
print(f"  Min: {x_quantum.min().item():.4f}, Max: {x_quantum.max().item():.4f}")

# Visualize feature distributions
print("\nGenerating feature distribution visualization...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Filter out NaN
original_flat = data.x.cpu().numpy().flatten()
original_flat = original_flat[~np.isnan(original_flat)]

quantum_flat = x_quantum.detach().cpu().numpy().flatten()
quantum_flat = quantum_flat[~np.isnan(quantum_flat)]

# Original features
axes[0].hist(original_flat, bins=100, alpha=0.7, edgecolor='black')
axes[0].set_xlabel('Feature Value')
axes[0].set_ylabel('Count')
axes[0].set_title('Original Features Distribution')
axes[0].set_yscale('log')

# Quantum features
axes[1].hist(quantum_flat, bins=100, alpha=0.7, color='orange', edgecolor='black')
axes[1].set_xlabel('Feature Value')
axes[1].set_ylabel('Count')
axes[1].set_title('Quantum Features Distribution')
axes[1].set_yscale('log')

plt.tight_layout()
save_path = FIGURES_DIR / FIGURE_FILES['quantum_features']
plt.savefig(save_path, dpi=150)
print(f"✓ Figure saved to {save_path}")
plt.close()

# Create quantum graph
print("\nCreating quantum graph data object...")
data_quantum = Data(
    x=x_quantum,
    edge_index=data.edge_index,
    y=data.y,
    timestep=data.timestep,
    labeled_mask=data.labeled_mask,
    unlabeled_mask=data.unlabeled_mask
)

# Copy masks if they exist
if hasattr(data, 'train_mask'):
    data_quantum.train_mask = data.train_mask
    data_quantum.val_mask = data.val_mask
    data_quantum.test_mask = data.test_mask
    print("✓ Copied train/val/test masks")

data_quantum.feature_mapper = feature_mapper
print(f"✓ Quantum graph: {data_quantum}")

# Save quantum graph
save_path = ARTIFACTS_DIR / ARTIFACT_FILES['quantum_graph']
torch.save(data_quantum, save_path)
print(f"✓ Quantum graph saved to {save_path}")

print("\n" + "="*70)
print("✅ QUANTUM FEATURE MAPPING COMPLETE!")
print("="*70)
print(f"\n📊 Summary:")
print(f"  Feature expansion: {data.num_node_features} → {data_quantum.num_node_features}")
print(f"  Expansion factor: {QUANTUM_CONFIG['expansion_factor']}x")
print(f"\n✅ Next: Run 06_train_gat_quantum.py")
