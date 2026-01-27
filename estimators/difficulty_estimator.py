import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Tuple
import json
import os

class DifficultyEstimator(nn.Module):
    """
    Neural estimator for variant difficulty, verification cost, and curriculum value
    Small model that can train on CPU
    """
    
    def __init__(self, hidden_dim=128, num_layers=2):
        super().__init__()
        
        # Small embedding layer (can use pre-trained small model)
        self.embedding_dim = 384  # Small BERT-like
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.embedding_dim + 3, hidden_dim),  # +3 for depth, similarity, etc.
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Three prediction heads
        self.solve_prob_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Probability [0, 1]
        )
        
        self.verifier_cost_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU()  # Cost in seconds [0, inf]
        )
        
        self.curriculum_value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Value [0, 1]
        )
    
    def forward(self, variant_embedding: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict solve probability, verifier cost, and curriculum value
        
        Args:
            variant_embedding: Text embedding of variant (batch_size, embedding_dim)
            features: Additional features [depth, similarity, length] (batch_size, 3)
        
        Returns:
            Tuple of (solve_prob, verifier_cost, curriculum_value)
        """
        # Concatenate embedding and features
        x = torch.cat([variant_embedding, features], dim=1)
        
        # Extract features
        x = self.feature_extractor(x)
        
        # Predict
        solve_prob = self.solve_prob_head(x)
        verifier_cost = self.verifier_cost_head(x)
        curriculum_value = self.curriculum_value_head(x)
        
        return solve_prob, verifier_cost, curriculum_value
    
    def predict(self, variant_text: str, depth: int, similarity: float, variant_length: int) -> Dict[str, float]:
        """
        Predict for a single variant (convenience method)
        
        Returns:
            Dict with 'solve_prob', 'verifier_cost', 'curriculum_value'
        """
        self.eval()
        with torch.no_grad():
            # Simple embedding (can use sentence transformer for better results)
            # For now, use simple feature-based approach
            variant_embedding = torch.zeros(1, self.embedding_dim)  # Placeholder
            
            features = torch.tensor([[depth, similarity, variant_length]], dtype=torch.float32)
            
            solve_prob, verifier_cost, curriculum_value = self.forward(variant_embedding, features)
            
            return {
                'solve_prob': solve_prob.item(),
                'verifier_cost': verifier_cost.item(),
                'curriculum_value': curriculum_value.item()
            }
