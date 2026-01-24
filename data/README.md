# Elliptic Bitcoin Dataset

This directory should contain the Elliptic Bitcoin Dataset files.

## Required Files
Download from: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set

The dataset includes:
- `txs_classes.csv` - Transaction classifications
- `txs_features.csv` - Transaction features (166 features)
- `txs_edgelist.csv` - Transaction graph edges
- `wallets_features.csv` - Wallet features
- `wallets_classes.csv` - Wallet classifications
- `wallets_features_classes_combined.csv` - Combined wallet data
- `AddrAddr_edgelist.csv` - Address-to-address edges
- `AddrTx_edgelist.csv` - Address-to-transaction edges
- `TxAddr_edgelist.csv` - Transaction-to-address edges

## Setup
1. Download the dataset from Kaggle
2. Extract all CSV files to this directory
3. Run `01_setup.ipynb` to verify the data

**Note:** Large data files (>100MB) are excluded from git to comply with GitHub file size limits.
