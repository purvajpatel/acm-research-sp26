# Quick Start Guide

## 🚀 Running the Project

All notebooks have been converted to Python scripts for better performance.

### Option 1: Run Complete Pipeline
```bash
python run_all.py
```
This runs all 8 scripts in sequence with progress tracking.

### Option 2: Run Individual Scripts
```bash
python scripts/01_setup.py
python scripts/02_data_graph.py
python scripts/03_train_gat_baseline.py
python scripts/04_eval_baseline.py
python scripts/05_quantum_feature_map.py
python scripts/06_train_gat_quantum.py
python scripts/07_eval_quantum.py
python scripts/08_explain_llm.py
```

### Option 3: Run Training Only
```bash
python run_complete_training.py
```
This runs scripts 03, 05, and 06 (baseline + quantum training).

## 📊 What You'll Get

After running the complete pipeline:

**Models:**
- `artifacts/gat_baseline.pt` - Baseline GAT model
- `artifacts/gat_quantum.pt` - Quantum-enhanced GAT model

**Visualizations:**
- `figures/baseline_confusion_matrix.png`
- `figures/baseline_roc_curve.png`
- `figures/quantum_confusion_matrix.png`
- `figures/baseline_vs_quantum_comparison.png`
- `figures/roc_comparison.png`

**Metrics:**
- Baseline: F1=79.38%, AUC=96.98%
- Quantum: F1=83.44%, AUC=98.42% ✓

## ⏱️ Estimated Time

- Setup: < 1 second
- Data loading: 2-3 minutes
- Baseline training: 15-20 minutes
- Quantum training: 20-25 minutes
- Evaluation: 5-8 minutes

**Total: ~45-60 minutes**

## 🔧 Requirements

Make sure you have:
1. Python 3.8+ installed
2. Virtual environment activated: `venv\Scripts\activate`
3. Dependencies installed: `pip install -r requirements.txt`
4. Dataset files in `data/` folder

Run `python scripts/01_setup.py` to verify everything is ready!
