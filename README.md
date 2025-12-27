# Cryptocurrency Fraud Detection with Explainable AI
ACM Research - Spring 2026

## Set-Up Instructions

1. **Install Required Packages**: Open the terminal and run the following command to install all dependencies: pip install -r requirements.txt


2. **Download the Elliptic++ Dataset**: This implementation uses the Elliptic++ Bitcoin transaction dataset. Download it from [Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) or the official Elliptic++ release (sourced Github [both act as the same dataset]). Place all downloaded CSV files in the `data/` directory. Your structure should look like:
   ```
   data/
   ├── txs_edgelist.csv
   ├── txs_features.csv
   ├── txs_classes.csv
   ├── AddrAddr_edgelist.csv
   ├── AddrTx_edgelist.csv
   └── TxAddr_edgelist.csv
   ```

3. **Set Up OpenAI API Key** (for LLM integration): Create a `.env` file in the root directory with your OpenAI API key:........OPENAI_API_KEY=your_api_key_here


4. **Run Notebooks Sequentially**: Execute the notebooks in order:
   - `1_gnn_anomaly_detection.ipynb` - Train GNN model
   - `2_graphlime_explanations.ipynb` - Generate explainability features
   - `3_llm_fraud_classification.ipynb` - LLM-based fraud classification
   - `4_results_analysis_visualization.ipynb` - Analyze and visualize results

---

## 1. Explainable Graph-Based Cryptocurrency Fraud Detection

**Paper**: [Explain First, Trust Later: LLM-Augmented Explanations for Graph-Based Crypto Anomaly Detection](https://arxiv.org/abs/2506.14933)

### Overview

Cryptocurrency fraud detection is a challenging problem due to the complex relationships between wallets and transactions in blockchain networks. This project implements an end-to-end explainable AI system that combines **Graph Neural Networks (GNNs)**, **GraphLIME explanations**, and **Large Language Models (LLMs)** to identify fraudulent Bitcoin transactions and provide human-interpretable explanations for the predictions.

### Motivation

Traditional fraud detection systems often operate as black boxes, making it difficult for investigators to understand why a particular transaction was flagged as suspicious. While Graph Neural Networks excel at learning patterns in transaction networks, they lack interpretability. Existing explainable AI methods provide feature importance scores but fail to translate technical metrics into actionable insights that domain experts can understand. This creates a trust gap between AI systems and human decision-makers, especially in high-stakes financial fraud investigations where false positives can have serious consequences.

### Novelty

This implementation bridges the gap between model performance and human interpretability through a novel three-stage pipeline:

1. **Unsupervised GNN Anomaly Detection**: Uses Graph Convolutional Networks to learn transaction patterns without labeled data, identifying anomalies based on reconstruction error
2. **GraphLIME Feature Attribution**: Generates local explanations by identifying which transaction features contribute most to the anomaly score for each suspicious wallet
3. **LLM-Augmented Interpretation**: Leverages GPT-4's reasoning capabilities to translate technical features and anomaly scores into natural language explanations, using a consensus voting mechanism across multiple LLM samples to ensure reliability

The key innovation is the **consensus-based LLM classification**, where multiple independent LLM inferences vote on fraud classification and explanation quality, significantly improving robustness over single-shot LLM predictions.

### Advantages/Disadvantages

**Advantages**:
- **Explainable predictions**: Every fraud detection comes with a human-readable explanation of why the transaction is suspicious
- **No labeled data required**: The GNN uses unsupervised learning, making it applicable to new cryptocurrency networks without historical fraud labels
- **Robust fraud classification**: Consensus voting across multiple LLM samples (default: 5) reduces hallucination and improves decision reliability
- **Graph-aware**: Considers both node features and network structure, capturing patterns like money laundering chains that isolated feature analysis would miss
- **Low API cost**: Using GPT-4o-mini, the entire pipeline costs approximately $0.0007 per wallet analyzed

**Disadvantages**:
- **Computational overhead**: Running multiple LLM inferences per node increases latency (5x longer than single inference)
- **LLM dependency**: Explanation quality is tied to the capabilities of the underlying language model, and API changes could affect performance
- **Feature interpretability**: The Elliptic++ dataset uses anonymized features (Feature 0, Feature 1, etc.) rather than semantic labels, limiting the depth of LLM explanations
- **Unsupervised limitations**: Without ground truth labels, the GNN may detect anomalies that are unusual but not necessarily fraudulent
- **Scalability**: Full-batch GNN training on CPU takes significant time for large graphs (200K+ nodes); GPU acceleration recommended for production use

### Implementation

I implemented an explainable cryptocurrency fraud detection system using the Elliptic++ Bitcoin transaction dataset, which contains 203,769 wallets with 183 anonymized features and 438,124 transaction edges. The system uses a 3-layer Graph Convolutional Network (GCN) trained as an unsupervised autoencoder to detect anomalies based on reconstruction error. For each suspicious wallet identified by the GNN, GraphLIME generates local explanations by identifying the top-3 most important features contributing to the anomaly score. Finally, GPT-4o-mini translates these technical metrics into human-readable explanations using a consensus mechanism that aggregates predictions from 5 independent LLM samples, improving robustness. The pipeline achieved 88.8% agreement across LLM samples and confirmed 16 fraud cases from 50 analyzed wallets, with all detected fraud classified as money laundering patterns.



## Results Summary

- **Dataset**: 203,769 wallets, 438,124 transactions, 183 features
- **Anomalies Detected**: 10,189 (5.0%)
- **LLM Analysis**: 16 fraud cases confirmed from 50 analyzed (88.8% consensus agreement)
- **Cost**: $0.0007 per wallet

---

## Cost Estimate

- **Computing**: Free (local CPU training, ~20 minutes for 200 epochs)
- **LLM API**: ~$0.0007 per wallet analyzed
  - Full dataset analysis (200K wallets): ~$140
  - Pilot study (50 wallets): ~$0.04
