![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)



# LADDER: Language Agent with Dual-Attention for Enhanced Reasoning

**Implemented by:** Abhijith Utla 
**Paper:** "LADDER: A Model-Agnostic Framework for LLM-Powered Agentic Reasoning" (arXiv:2503.00735)

## Research Motivation and Overview

Large Language Models (LLMs) struggle with complex reasoning tasks that require both broad context understanding and precise detail focus. The LADDER (Language Agent with Dual-attention for enhanced DEductive Reasoning) paper addresses this by introducing a dual-attention mechanism that allows agents to simultaneously maintain global context awareness while performing granular reasoning steps.

The key innovation is a hierarchical attention architecture that:
- Maintains a **global context attention layer** for high-level task understanding
- Employs a **local reasoning attention layer** for step-by-step deductive logic
- Uses cross-attention to bridge macro and micro reasoning processes

This implementation demonstrates LADDER's dual-attention mechanism on a question-answering benchmark with multi-hop reasoning capabilities.


## Implementation Details

This implementation uses **PyTorch** to build a simplified LADDER architecture trained on a synthetic multi-hop QA dataset. The model includes:

- **Global Context Encoder**: Transformer encoder for maintaining question awareness (4 layers)
- **Local Reasoning Decoder**: Transformer decoder for step-by-step reasoning generation (4 layers)
- **Cross-Attention Bridge**: Mechanism connecting global context to local reasoning


### **Training Configuration:**
- **Dataset**: Synthetic multi-hop QA (5000 training examples, 1000 validation)
- **Optimizer**: AdamW (learning rate: 5e-5, weight decay: 0.01)
- **Training Epochs**: 5 epochs
- **Batch Size**: 8

---

### **Results:**
---

### **Files Included:**

├── `ladder_model.py` — Complete LADDER architecture implementation  
├── `train.py` — Training script with synthetic dataset generation  
├── `requirements.txt` — Python dependencies (PyTorch, NumPy, etc.)  
└── `README.md` — Project documentation


---

## How to Run

### **Step 1: Install Dependencies**

pip install -r requirements.txt

### **Step 2: Train the Model**

python train.py


The training script will:
- Generate synthetic multi-hop QA dataset
- Train LADDER model for 5 epochs
- Save checkpoints to `./checkpoints/` directory
- Print training metrics every 100 steps

### **Step 3: View Results**
Model checkpoints are saved to `./checkpoints/ladder_model_epoch_X.pt`

---
## Future Work

-  Extend to real-world datasets (HotpotQA, StrategyQA, GSM8K)
- Implement attention visualization tools for interpretability
-  Experiment with different global-local attention ratios
-  Fine-tune on domain-specific reasoning (math word problems, code reasoning)
-  Test integration with larger base models (LLaMA, GPT-style architectures)

---

## References
1. Chen et al. "LADDER: A Model-Agnostic Framework for LLM-Powered Agentic Reasoning" *arXiv:2503.00735* (2025)
2. Yang et al. "HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering" *EMNLP* (2018)
3. Vaswani et al. "Attention Is All You Need" *NeurIPS* (2017)
