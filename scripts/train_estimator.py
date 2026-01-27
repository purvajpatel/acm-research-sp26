"""
Train the difficulty/utility estimator on collected variant data
Small model - can train on CPU
"""
import argparse
import torch
import torch.nn as nn
import json
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from estimators.difficulty_estimator import DifficultyEstimator
from utils.variant_logger import VariantLogger

def prepare_training_data(logs):
    """Prepare training data from logs"""
    X = []
    y_solve = []
    y_cost = []
    y_value = []
    
    for log in logs:
        # Features: depth, similarity, length
        features = torch.tensor([
            log['tree_depth'],
            log['similarity_to_root'],
            log['variant_length'] / 100.0  # Normalize
        ])
        
        # Labels
        solve_label = 1.0 if log['solve_success'] else 0.0
        cost_label = log['verification_time']
        value_label = log['reward']  # Use reward as curriculum value proxy
        
        X.append(features)
        y_solve.append(solve_label)
        y_cost.append(cost_label)
        y_value.append(value_label)
    
    return torch.stack(X), torch.tensor(y_solve), torch.tensor(y_cost), torch.tensor(y_value)

def main():
    parser = argparse.ArgumentParser(description='Train difficulty estimator')
    parser.add_argument('--data', type=str, default='data/variant_logs.json')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output', type=str, default='checkpoints/estimator.pt')
    
    args = parser.parse_args()
    
    # Load data
    logger = VariantLogger(args.data)
    logs = logger.get_training_data()
    
    if len(logs) < 100:
        print(f"⚠ Warning: Only {len(logs)} examples. Need at least 100 for training.")
        print("Run collect_variant_data.py first to collect more data.")
        return
    
    print(f"Loaded {len(logs)} variant examples")
    
    # Prepare data
    X, y_solve, y_cost, y_value = prepare_training_data(logs)
    
    # Split train/val
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_solve_train, y_solve_val = y_solve[:split], y_solve[split:]
    y_cost_train, y_cost_val = y_cost[:split], y_cost[split:]
    y_value_train, y_value_val = y_value[:split], y_value[split:]
    
    # Initialize model
    model = DifficultyEstimator()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Loss functions
    solve_loss_fn = nn.BCELoss()
    cost_loss_fn = nn.MSELoss()
    value_loss_fn = nn.BCELoss()
    
    print(f"\nTraining estimator for {args.epochs} epochs...")
    print("=" * 60)
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        # Simple batch training
        for i in range(0, len(X_train), args.batch_size):
            batch_X = X_train[i:i+args.batch_size]
            batch_solve = y_solve_train[i:i+args.batch_size]
            batch_cost = y_cost_train[i:i+args.batch_size]
            batch_value = y_value_train[i:i+args.batch_size]
            
            # Forward pass
            variant_embedding = torch.zeros(len(batch_X), model.embedding_dim)
            solve_pred, cost_pred, value_pred = model(variant_embedding, batch_X)
            
            # Compute losses
            solve_loss = solve_loss_fn(solve_pred.squeeze(), batch_solve)
            cost_loss = cost_loss_fn(cost_pred.squeeze(), batch_cost)
            value_loss = value_loss_fn(value_pred.squeeze(), batch_value)
            
            total_loss_batch = solve_loss + cost_loss + value_loss
            
            # Backward
            optimizer.zero_grad()
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item()
        
        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_embedding = torch.zeros(len(X_val), model.embedding_dim)
                solve_pred, cost_pred, value_pred = model(val_embedding, X_val)
                
                val_solve_loss = solve_loss_fn(solve_pred.squeeze(), y_solve_val).item()
                val_cost_loss = cost_loss_fn(cost_pred.squeeze(), y_cost_val).item()
                val_value_loss = value_loss_fn(value_pred.squeeze(), y_value_val).item()
                
                print(f"Epoch {epoch+1}/{args.epochs}:")
                print(f"  Train Loss: {total_loss/len(X_train)*args.batch_size:.4f}")
                print(f"  Val Loss: {val_solve_loss + val_cost_loss + val_value_loss:.4f}")
    
    # Save model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"\n✓ Estimator trained and saved to: {args.output}")

if __name__ == '__main__':
    main()
