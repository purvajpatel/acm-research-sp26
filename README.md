# Kryptos: Quantum-Inspired Graph Neural Networks for Blockchain Fraud Detection

Complete implementation of GAT-based fraud detection with quantum-inspired feature mapping and explainability.

## Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### 2. Dataset
Your dataset files should be in `data/`:
- `txs_features.csv`
- `txs_edgelist.csv`
- `txs_classes.csv`

The code is configured to work with the Elliptic Bitcoin Dataset format where:
- Class 1 = illicit transactions
- Class 2 = licit transactions  
- Class 3 = unknown/unlabeled transactions

### 3. Run Scripts in Order

All scripts are organized in the `scripts/` folder:

**scripts/01_setup.py** - Verify environment and create directories  
**scripts/02_data_graph.py** - Load data and build graph (5-10 min)  
**scripts/03_train_gat_baseline.py** - Train baseline GAT (10-20 min)  
**scripts/04_eval_baseline.py** - Evaluate baseline model (2-5 min)  
**scripts/05_quantum_feature_map.py** - Apply quantum transformation (1 min)  
**scripts/06_train_gat_quantum.py** - Train quantum GAT (10-20 min)  
**scripts/07_eval_quantum.py** - Compare models (2-5 min)  
**scripts/08_explain_llm.py** - Generate explanations (requires OpenAI API)

**Run individual scripts:**
```bash
python scripts/01_setup.py
python scripts/02_data_graph.py
```

**Or run complete pipeline:**
```bash
python run_all.py
```

**Total time: ~1 hour**

## Optional: LLM Explanations

To use GPT-4 for generating explanations:

1. Create `.env` file in project root:
```
OPENAI_API_KEY=your_key_here
```

2. The last notebook will generate forensic narratives automatically

## What Each Notebook Does

### 01_setup.ipynb
- Checks Python version and packages
- Creates project directories
- Verifies dataset files

### 02_data_graph.ipynb
- Loads 203K+ transactions with 166 features
- Processes ~234K edges
- Maps labels (illicit/licit/unknown)
- Creates PyTorch Geometric graph
- Visualizes data distribution

### 03_train_gat_baseline.ipynb
- Trains 2-layer Graph Attention Network
- 4 attention heads, 64 hidden dimensions
- Train/val/test split: 60/20/20
- Early stopping on validation F1
- Saves best model checkpoint

### 04_eval_baseline.ipynb
- Confusion matrix and classification report
- ROC curve and AUC analysis
- Temporal robustness test
- Identifies top fraudulent transactions

### 05_quantum_feature_map.ipynb
- Applies sinusoidal/Fourier feature expansion
- Doubles feature dimensionality (166 → 332)
- Random Fourier Features: φ(x) = [cos(Wx+b), sin(Wx+b)]
- Creates quantum-enhanced graph

### 06_train_gat_quantum.ipynb
- Trains GAT on quantum-expanded features
- Same architecture as baseline
- Compares training dynamics

### 07_eval_quantum.ipynb
- Side-by-side performance comparison
- ROC curve overlay
- Quantifies improvement from quantum features
- Confusion matrices for both models

### 08_explain_llm.ipynb
- Extracts GAT attention weights
- Identifies influential neighbors
- Generates LLM-powered forensic narratives
- Explains top fraud cases

## Expected Results

### Baseline GAT
- Test F1: ~0.55-0.65
- Test AUC: ~0.75-0.85
- Recall: ~0.50-0.60

### Quantum GAT
- Test F1: ~0.60-0.70 (+5-10% improvement)
- Test AUC: ~0.80-0.90
- Better feature representation

### Key Findings
1. Quantum feature expansion improves fraud detection
2. Temporal shift affects performance (early vs late timesteps)
3. Attention weights reveal transaction relationships
4. LLM explanations make models interpretable

## Project Structure

```
Kryptos---Quantum-Blockchain-Fraud-Detection/
├── README.md
├── requirements.txt
├── .env (create this)
├── data/
│   └── elliptic/          # Place dataset here
├── artifacts/             # Saved models and metrics
│   ├── elliptic_graph.pt
│   ├── elliptic_graph_quantum.pt
│   ├── gat_baseline.pt
│   ├── gat_quantum.pt
│   ├── gat_baseline_metrics.json
│   ├── gat_quantum_metrics.json
│   └── fraud_explanations.json
├── figures/               # Generated plots
│   ├── data_distribution.png
│   ├── baseline_confusion_matrix.png
│   ├── baseline_roc_curve.png
│   ├── quantum_feature_distribution.png
│   ├── quantum_confusion_matrix.png
│   ├── baseline_vs_quantum_comparison.png
│   └── roc_comparison.png
├── notebooks/
│   ├── 01_setup.ipynb
│   ├── 02_data_graph.ipynb
│   ├── 03_train_gat_baseline.ipynb
│   ├── 04_eval_baseline.ipynb
│   ├── 05_quantum_feature_map.ipynb
│   ├── 06_train_gat_quantum.ipynb
│   ├── 07_eval_quantum.ipynb
│   └── 08_explain_llm.ipynb
└── src/
    ├── __init__.py
    ├── config.py
    ├── models.py
    └── quantum_features.py
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size or use CPU: `device = torch.device('cpu')`

### Dataset Column Names
Edit `02_data_graph.ipynb` Cell 2 if your CSV columns differ

### OpenAI API Issues
Skip Cell 7-9 in `08_explain_llm.ipynb` or use free explanations (attention-based)

### Import Errors
```bash
pip install --upgrade torch torch-geometric
```

## Research Components

### 1. Detector
- **Architecture**: Graph Attention Network (GAT)
- **Layers**: 2 GAT layers with 4 attention heads
- **Input**: Transaction features + graph structure
- **Output**: Binary fraud classification

### 2. Quantum-Inspired Features
- **Method**: Random Fourier Features
- **Transform**: φ(x) = [cos(Wx + b), sin(Wx + b)]
- **Intuition**: Mimics quantum feature maps without hardware
- **Benefit**: Captures non-linear relationships

### 3. Explainability
- **Stage 1**: Extract GAT attention weights
- **Stage 2**: Identify influential neighbors
- **Stage 3**: LLM generates forensic narrative
- **Output**: Human-readable fraud explanations

## Citation

```
@misc{kryptos2026,
  title={Kryptos: Quantum-Inspired and Explainable Graph Neural Networks for Blockchain Fraud Detection},
  author={Your Name},
  year={2026}
}
```

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue.