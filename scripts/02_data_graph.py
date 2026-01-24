"""
Run the data graph construction from 02_data_graph.ipynb as a script
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from src.config import DATASET_CONFIG, TRAINING_CONFIG, DATA_DIR, ARTIFACTS_DIR, FIGURES_DIR, ARTIFACT_FILES, FIGURE_FILES
from src.utils import set_random_seeds

# Set random seeds for reproducibility
set_random_seeds(TRAINING_CONFIG['random_seed'])
print("✓ Setup complete!")

# Load features
print("\nLoading features...")
features_file = DATA_DIR / DATASET_CONFIG['features_file']
df_features = pd.read_csv(features_file, header=0)
print(f"Shape: {df_features.shape}")

node_ids = df_features.iloc[:, 0].values
timesteps = df_features.iloc[:, 1].values
features = df_features.iloc[:, 2:].values.astype(np.float32)

node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
print(f"Nodes: {len(node_ids)}, Features: {features.shape[1]}")

# Load labels
print("\nLoading labels...")
classes_file = DATA_DIR / DATASET_CONFIG['classes_file']
df_classes = pd.read_csv(classes_file)
print("Original class distribution:")
print(df_classes[DATASET_CONFIG['label_column']].value_counts())

labels = np.full(len(node_ids), -1, dtype=np.int64)

illicit_label = DATASET_CONFIG['illicit_label']
licit_label = DATASET_CONFIG['licit_label']

for _, row in tqdm(df_classes.iterrows(), total=len(df_classes), desc="Processing labels"):
    node_id = row[DATASET_CONFIG['id_column']]
    if node_id in node_id_to_idx:
        idx = node_id_to_idx[node_id]
        if row[DATASET_CONFIG['label_column']] == illicit_label:
            labels[idx] = 1  # Illicit
        elif row[DATASET_CONFIG['label_column']] == licit_label:
            labels[idx] = 0  # Licit
        # Unknown (3) remains -1

labeled_mask = labels != -1
print(f"\nLabeled: {labeled_mask.sum()}, Unlabeled: {(~labeled_mask).sum()}")
print(f"Licit (0): {(labels==0).sum()}, Illicit (1): {(labels==1).sum()}")

# Load edges
print("\nLoading edges...")
edgelist_file = DATA_DIR / DATASET_CONFIG['edgelist_file']
df_edges = pd.read_csv(edgelist_file)
print(f"Edges in file: {len(df_edges)}")

edge_list = []
for _, row in tqdm(df_edges.iterrows(), total=len(df_edges), desc="Processing edges"):
    src_id = row[DATASET_CONFIG['edge_source_column']]
    dst_id = row[DATASET_CONFIG['edge_target_column']]
    
    if src_id in node_id_to_idx and dst_id in node_id_to_idx:
        src_idx = node_id_to_idx[src_id]
        dst_idx = node_id_to_idx[dst_id]
        edge_list.append([src_idx, dst_idx])
        edge_list.append([dst_idx, src_idx])  # Make undirected

edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
print(f"Edge index shape (before self-loops): {edge_index.shape}")

# Add self-loops for GAT
edge_index, _ = add_self_loops(edge_index, num_nodes=len(node_ids))
print(f"Edge index shape (after self-loops): {edge_index.shape}")

# Create PyTorch Geometric Data object
x = torch.tensor(features, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.long)
timestep_tensor = torch.tensor(timesteps, dtype=torch.long)
labeled_mask_tensor = torch.tensor(labeled_mask, dtype=torch.bool)
unlabeled_mask_tensor = torch.tensor(~labeled_mask, dtype=torch.bool)

data = Data(
    x=x,
    edge_index=edge_index,
    y=y,
    timestep=timestep_tensor,
    labeled_mask=labeled_mask_tensor,
    unlabeled_mask=unlabeled_mask_tensor
)

print("\n" + "="*50)
print(data)
print("="*50)

# Save graph
save_path = ARTIFACTS_DIR / ARTIFACT_FILES['baseline_graph']
torch.save(data, save_path)
print(f"\n✓ Graph saved to {save_path}")

# Visualize data distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Labels
label_counts = [(labels==0).sum(), (labels==1).sum(), (~labeled_mask).sum()]
axes[0].bar(['Licit', 'Illicit', 'Unknown'], label_counts, color=['green', 'red', 'gray'])
axes[0].set_ylabel('Count')
axes[0].set_title('Label Distribution')
axes[0].set_yscale('log')

# Timesteps
axes[1].hist(timesteps, bins=50, edgecolor='black')
axes[1].set_xlabel('Timestep')
axes[1].set_ylabel('Count')
axes[1].set_title('Temporal Distribution')

plt.tight_layout()
save_path = FIGURES_DIR / FIGURE_FILES['data_distribution']
plt.savefig(save_path, dpi=150)
print(f"✓ Figure saved to {save_path}")

print("\n" + "="*50)
print("✅ Graph Construction Complete!")
print("="*50)
print("\nNext: Run 03_train_gat_baseline.ipynb")
