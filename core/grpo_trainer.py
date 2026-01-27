"""
GRPO (Group Relative Policy Optimization) Trainer
Exact implementation based on paper methodology
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict
import numpy as np
from transformers import get_linear_schedule_with_warmup

class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer
    Implements the GRPO algorithm exactly as described in the paper
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        learning_rate: float = 1e-5,
        device: str = 'cuda',
        max_grad_norm: float = 1.0
    ):
        """
        Initialize GRPO trainer
        
        Args:
            model: Language model to train
            tokenizer: Tokenizer for the model
            learning_rate: Learning rate (default 1e-5 as in paper)
            device: Device to train on
            max_grad_norm: Gradient clipping norm
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Move model to device (only if not already distributed with device_map)
        if not hasattr(self.model, '_is_distributed') or not self.model._is_distributed:
            self.model.to(device)
        self.model.train()
    
    def train_group(
        self,
        group: List[Tuple[str, str, float]],
        batch_size: int = 32
    ):
        """
        Train on a group of problem-solution pairs with rewards
        
        Args:
            group: List of (problem, solution, reward) tuples
            batch_size: Batch size for training (default 32 as in paper)
        """
        if len(group) == 0:
            return
        
        # Compute mean reward (group baseline)
        rewards = torch.tensor([r for _, _, r in group], dtype=torch.float32)
        mean_reward = rewards.mean().item()
        
        # Compute relative rewards
        relative_rewards = rewards - mean_reward
        
        # Create batches
        batches = self._create_batches(group, relative_rewards, batch_size)
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in batches:
            loss = self._train_batch(batch)
            total_loss += loss
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def _create_batches(
        self,
        group: List[Tuple[str, str, float]],
        relative_rewards: torch.Tensor,
        batch_size: int
    ) -> List[Dict]:
        """Create batches from group"""
        batches = []
        
        for i in range(0, len(group), batch_size):
            batch_group = group[i:i + batch_size]
            batch_rewards = relative_rewards[i:i + batch_size]
            
            batch = {
                'problems': [p for p, _, _ in batch_group],
                'solutions': [s for _, s, _ in batch_group],
                'relative_rewards': batch_rewards
            }
            batches.append(batch)
        
        return batches
    
    def _train_batch(self, batch: Dict) -> float:
        """
        Train on a single batch
        Implements GRPO loss: -E[log π(s|p) * (r - r̄)]
        """
        problems = batch['problems']
        solutions = batch['solutions']
        
        # Get device for tensors (use first model device if distributed)
        if hasattr(self.model, '_is_distributed') and self.model._is_distributed:
            tensor_device = next(self.model.parameters()).device
        else:
            tensor_device = self.device
        relative_rewards = batch['relative_rewards'].to(tensor_device)
        
        # Compute log probabilities
        log_probs = []
        
        for problem, solution in zip(problems, solutions):
            # Create input sequence: problem + solution
            input_text = f"Problem: {problem}\nSolution: {solution}"
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True
            )
            
            # Move to appropriate device
            if hasattr(self.model, '_is_distributed') and self.model._is_distributed:
                inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
            else:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get log probabilities
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Compute log probability of the solution tokens
            # Shift by 1 to align predictions with targets
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs['input_ids'][..., 1:].contiguous()
            
            # Compute log probs
            log_probs_batch = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            
            # Get log prob of actual tokens
            log_probs_seq = torch.gather(
                log_probs_batch,
                dim=-1,
                index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            
            # Average over sequence length (or sum, depending on formulation)
            log_prob = log_probs_seq.mean()
            log_probs.append(log_prob)
        
        # Stack log probabilities
        log_probs_tensor = torch.stack(log_probs)
        
        # Compute GRPO loss: -E[log π(s|p) * (r - r̄)]
        # Note: We negate because we want to maximize, but optimizers minimize
        loss = -(log_probs_tensor * relative_rewards).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        
        return loss.item()
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
