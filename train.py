import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
from ladder_model import LADDERModel, count_parameters

# Create checkpoint directory
os.makedirs('./checkpoints', exist_ok=True)


class SyntheticMultiHopQA(Dataset):
    """
    Synthetic dataset for multi-hop question answering
    Generates simple reasoning chains
    """
    def __init__(self, num_samples=5000, vocab_size=10000, max_src_len=64, max_tgt_len=32):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        
        # Generate dataset
        self.data = []
        for _ in range(num_samples):
            # Source: question tokens
            src_len = np.random.randint(20, max_src_len)
            src = np.random.randint(3, vocab_size, size=src_len)
            src = np.concatenate([[1], src])  # Add BOS token
            
            # Target: reasoning chain tokens
            tgt_len = np.random.randint(15, max_tgt_len)
            tgt = np.random.randint(3, vocab_size, size=tgt_len)
            tgt = np.concatenate([[1], tgt, [2]])  # Add BOS and EOS tokens
            
            self.data.append((src, tgt))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """Collate function with padding"""
    srcs, tgts = zip(*batch)
    
    # Pad sources
    max_src_len = max(len(src) for src in srcs)
    src_padded = []
    src_masks = []
    for src in srcs:
        pad_len = max_src_len - len(src)
        padded = np.pad(src, (0, pad_len), constant_values=0)
        mask = np.array([False] * len(src) + [True] * pad_len)
        src_padded.append(padded)
        src_masks.append(mask)
    
    # Pad targets
    max_tgt_len = max(len(tgt) for tgt in tgts)
    tgt_padded = []
    for tgt in tgts:
        pad_len = max_tgt_len - len(tgt)
        padded = np.pad(tgt, (0, pad_len), constant_values=0)
        tgt_padded.append(padded)
    
    return (torch.LongTensor(src_padded), 
            torch.BoolTensor(src_masks),
            torch.LongTensor(tgt_padded))


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (src, src_mask, tgt) in enumerate(progress_bar):
        src = src.to(device)
        src_mask = src_mask.to(device)
        tgt = tgt.to(device)
        
        # Prepare decoder input (shift right) and target
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        # Forward pass
        optimizer.zero_grad()
        logits, alignment = model(src, tgt_input, src_mask=src_mask)
        
        # Compute loss (ignore padding tokens)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        if batch_idx % 100 == 0:
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, src_mask, tgt in tqdm(dataloader, desc="Validating"):
            src = src.to(device)
            src_mask = src_mask.to(device)
            tgt = tgt.to(device)
            
            # Prepare decoder input and target
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Forward pass
            logits, alignment = model(src, tgt_input, src_mask=src_mask)
            
            # Compute loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    # Hyperparameters
    VOCAB_SIZE = 10000
    D_MODEL = 256
    NHEAD = 8
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1
    
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 0.01
    NUM_EPOCHS = 5
    
    TRAIN_SAMPLES = 5000
    VAL_SAMPLES = 1000
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("\n" + "="*60)
    print("Initializing LADDER Model")
    print("="*60)
    
    model = LADDERModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(device)
    
    print(f"\nTotal Parameters: {count_parameters(model):,}")
    print(f"Model Size: ~{count_parameters(model) * 4 / 1024 / 1024:.2f} MB")
    
    # Create datasets
    print("\n" + "="*60)
    print("Creating Datasets")
    print("="*60)
    
    train_dataset = SyntheticMultiHopQA(num_samples=TRAIN_SAMPLES, vocab_size=VOCAB_SIZE)
    val_dataset = SyntheticMultiHopQA(num_samples=VAL_SAMPLES, vocab_size=VOCAB_SIZE)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {BATCH_SIZE}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        print(f"\nTrain Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = f'./checkpoints/ladder_model_epoch_{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        print(f"\nCheckpoint saved: {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = './checkpoints/ladder_model_best.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, best_model_path)
            print(f"Best model saved: {best_model_path} (val_loss: {val_loss:.4f})")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"All checkpoints saved in './checkpoints/' directory")


if __name__ == "__main__":
    main()
