"""
Notebook 01: Setup and Environment Verification
This script sets up the environment and verifies dependencies.
"""
import sys
import os
import torch
from pathlib import Path

print("="*70)
print("AEGIS PROJECT SETUP")
print("="*70)

# Verify Python environment
print("\n1. VERIFYING PYTHON ENVIRONMENT")
print("-" * 70)
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU (no CUDA available)")

# Create project directories
print("\n2. CREATING PROJECT DIRECTORIES")
print("-" * 70)
dirs = ['data', 'artifacts', 'figures', 'src']

for d in dirs:
    Path(d).mkdir(parents=True, exist_ok=True)
    print(f"✓ Created/verified: {d}/")

# Create src package
init_file = Path('src/__init__.py')
init_file.touch()
print(f"✓ Initialized src package")

# Verify dataset files
print("\n3. VERIFYING DATASET FILES")
print("-" * 70)
required_files = [
    'data/txs_features.csv',
    'data/txs_edgelist.csv',
    'data/txs_classes.csv'
]

all_present = True
for file_path in required_files:
    exists = os.path.exists(file_path)
    status = "✓" if exists else "✗"
    print(f"  {status} {file_path}")
    if not exists:
        all_present = False

# Summary
print("\n" + "="*70)
if all_present:
    print("✅ SETUP COMPLETE! All dataset files found.")
    print("✅ Ready to proceed to 02_data_graph.py")
else:
    print("⚠️  WARNING: Some dataset files are missing.")
    print("   Please add them to the data/ directory before proceeding.")
print("="*70)
