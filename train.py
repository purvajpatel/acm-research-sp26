import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
from ladder_model import LADDERModel, count_parameters

checkpoint_directory_path = './checkpoints'
os.makedirs(checkpoint_directory_path, exist_ok=True)


class SyntheticMultiHopQA(Dataset):
    def __init__(self, num_samples=5000, vocab_size=10000, max_src_len=64, max_tgt_len=32):
        self.number_of_samples = num_samples
        self.vocabulary_size = vocab_size
        self.maximum_source_length = max_src_len
        self.maximum_target_length = max_tgt_len
        
        self.dataset_items = []
        for sample_index in range(num_samples):
            minimum_source_length = 20
            source_length = np.random.randint(minimum_source_length, max_src_len)
            source_tokens = np.random.randint(3, vocab_size, size=source_length)
            beginning_of_sequence_token = np.array([1])
            source_with_bos = np.concatenate([beginning_of_sequence_token, source_tokens])
            
            minimum_target_length = 15
            target_length = np.random.randint(minimum_target_length, max_tgt_len)
            target_tokens = np.random.randint(3, vocab_size, size=target_length)
            beginning_of_sequence_token = np.array([1])
            end_of_sequence_token = np.array([2])
            target_with_tokens = np.concatenate([beginning_of_sequence_token, target_tokens, end_of_sequence_token])
            
            self.dataset_items.append((source_with_bos, target_with_tokens))
    
    def __len__(self):
        return self.number_of_samples
    
    def __getitem__(self, idx):
        source_sequence, target_sequence = self.dataset_items[idx]
        return source_sequence, target_sequence


def collate_fn(batch):
    sources_list, targets_list = zip(*batch)
    
    source_lengths = [len(source_seq) for source_seq in sources_list]
    maximum_source_length = max(source_lengths)
    
    padded_sources = []
    source_masks = []
    for source_sequence in sources_list:
        current_source_length = len(source_sequence)
        padding_length_needed = maximum_source_length - current_source_length
        padded_source = np.pad(source_sequence, (0, padding_length_needed), constant_values=0)
        valid_positions = [False] * current_source_length
        padding_positions = [True] * padding_length_needed
        mask_array = np.array(valid_positions + padding_positions)
        padded_sources.append(padded_source)
        source_masks.append(mask_array)
    
    target_lengths = [len(target_seq) for target_seq in targets_list]
    maximum_target_length = max(target_lengths)
    
    padded_targets = []
    for target_sequence in targets_list:
        current_target_length = len(target_sequence)
        padding_length_needed = maximum_target_length - current_target_length
        padded_target = np.pad(target_sequence, (0, padding_length_needed), constant_values=0)
        padded_targets.append(padded_target)
    
    sources_tensor = torch.LongTensor(padded_sources)
    masks_tensor = torch.BoolTensor(source_masks)
    targets_tensor = torch.LongTensor(padded_targets)
    
    return sources_tensor, masks_tensor, targets_tensor


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    accumulated_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_index, batch_data in enumerate(progress_bar):
        source_tokens, source_mask, target_tokens = batch_data
        
        source_tokens = source_tokens.to(device)
        source_mask = source_mask.to(device)
        target_tokens = target_tokens.to(device)
        
        target_input_sequence = target_tokens[:, :-1]
        target_output_sequence = target_tokens[:, 1:]
        
        optimizer.zero_grad()
        model_output_logits, model_alignment_scores = model(
            source_tokens, 
            target_input_sequence, 
            src_mask=source_mask
        )
        
        batch_size_value = model_output_logits.size(0)
        sequence_length_value = model_output_logits.size(1)
        vocabulary_size_value = model_output_logits.size(2)
        
        flattened_logits = model_output_logits.reshape(-1, vocabulary_size_value)
        flattened_targets = target_output_sequence.reshape(-1)
        
        batch_loss = criterion(flattened_logits, flattened_targets)
        
        batch_loss.backward()
        max_gradient_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_gradient_norm)
        optimizer.step()
        
        loss_value = batch_loss.item()
        accumulated_loss = accumulated_loss + loss_value
        
        update_frequency = 100
        if batch_index % update_frequency == 0:
            current_average_loss = accumulated_loss / (batch_index + 1)
            progress_bar.set_postfix({'loss': f'{current_average_loss:.4f}'})
    
    total_batches = len(dataloader)
    average_loss = accumulated_loss / total_batches
    return average_loss


def validate(model, dataloader, criterion, device):
    model.eval()
    accumulated_validation_loss = 0.0
    
    with torch.no_grad():
        for validation_batch in tqdm(dataloader, desc="Validating"):
            source_tokens, source_mask, target_tokens = validation_batch
            
            source_tokens = source_tokens.to(device)
            source_mask = source_mask.to(device)
            target_tokens = target_tokens.to(device)
            
            target_input_sequence = target_tokens[:, :-1]
            target_output_sequence = target_tokens[:, 1:]
            
            model_output_logits, model_alignment_scores = model(
                source_tokens, 
                target_input_sequence, 
                src_mask=source_mask
            )
            
            vocabulary_size_value = model_output_logits.size(-1)
            flattened_logits = model_output_logits.reshape(-1, vocabulary_size_value)
            flattened_targets = target_output_sequence.reshape(-1)
            
            validation_batch_loss = criterion(flattened_logits, flattened_targets)
            validation_loss_value = validation_batch_loss.item()
            accumulated_validation_loss = accumulated_validation_loss + validation_loss_value
    
    total_validation_batches = len(dataloader)
    average_validation_loss = accumulated_validation_loss / total_validation_batches
    return average_validation_loss


def main():
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
    
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        computation_device = torch.device('cuda')
    else:
        computation_device = torch.device('cpu')
    print(f"Using device: {computation_device}")
    
    separator_line = "=" * 60
    print("\n" + separator_line)
    print("Initializing LADDER Model")
    print(separator_line)
    
    ladder_model = LADDERModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    )
    ladder_model = ladder_model.to(computation_device)
    
    total_model_parameters = count_parameters(ladder_model)
    print(f"\nTotal Parameters: {total_model_parameters:,}")
    
    bytes_per_parameter = 4
    total_model_bytes = total_model_parameters * bytes_per_parameter
    kilobytes_in_model = total_model_bytes / 1024
    megabytes_in_model = kilobytes_in_model / 1024
    print(f"Model Size: ~{megabytes_in_model:.2f} MB")
    
    print("\n" + separator_line)
    print("Creating Datasets")
    print(separator_line)
    
    training_dataset = SyntheticMultiHopQA(
        num_samples=TRAIN_SAMPLES, 
        vocab_size=VOCAB_SIZE
    )
    validation_dataset = SyntheticMultiHopQA(
        num_samples=VAL_SAMPLES, 
        vocab_size=VOCAB_SIZE
    )
    
    training_data_loader = DataLoader(
        training_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    validation_data_loader = DataLoader(
        validation_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    training_dataset_size = len(training_dataset)
    validation_dataset_size = len(validation_dataset)
    print(f"Training samples: {training_dataset_size}")
    print(f"Validation samples: {validation_dataset_size}")
    print(f"Batch size: {BATCH_SIZE}")
    
    padding_token_index = 0
    loss_function = nn.CrossEntropyLoss(ignore_index=padding_token_index)
    model_optimizer = optim.AdamW(
        ladder_model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    
    print("\n" + separator_line)
    print("Starting Training")
    print(separator_line)
    
    best_validation_loss_so_far = float('inf')
    
    for current_epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {current_epoch}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        
        epoch_training_loss = train_epoch(
            ladder_model, 
            training_data_loader, 
            model_optimizer, 
            loss_function, 
            computation_device, 
            current_epoch
        )
        print(f"\nTrain Loss: {epoch_training_loss:.4f}")
        
        epoch_validation_loss = validate(
            ladder_model, 
            validation_data_loader, 
            loss_function, 
            computation_device
        )
        print(f"Validation Loss: {epoch_validation_loss:.4f}")
        
        epoch_checkpoint_path = f'./checkpoints/ladder_model_epoch_{current_epoch}.pt'
        checkpoint_dictionary = {
            'epoch': current_epoch,
            'model_state_dict': ladder_model.state_dict(),
            'optimizer_state_dict': model_optimizer.state_dict(),
            'train_loss': epoch_training_loss,
            'val_loss': epoch_validation_loss,
        }
        torch.save(checkpoint_dictionary, epoch_checkpoint_path)
        print(f"\nCheckpoint saved: {epoch_checkpoint_path}")
        
        if epoch_validation_loss < best_validation_loss_so_far:
            best_validation_loss_so_far = epoch_validation_loss
            best_model_checkpoint_path = './checkpoints/ladder_model_best.pt'
            best_checkpoint_dictionary = {
                'epoch': current_epoch,
                'model_state_dict': ladder_model.state_dict(),
                'optimizer_state_dict': model_optimizer.state_dict(),
                'train_loss': epoch_training_loss,
                'val_loss': epoch_validation_loss,
            }
            torch.save(best_checkpoint_dictionary, best_model_checkpoint_path)
            print(f"Best model saved: {best_model_checkpoint_path} (val_loss: {epoch_validation_loss:.4f})")
    
    print("\n" + separator_line)
    print("Training Complete!")
    print(separator_line)
    print(f"Best Validation Loss: {best_validation_loss_so_far:.4f}")
    print(f"All checkpoints saved in './checkpoints/' directory")


if __name__ == "__main__":
    main()
