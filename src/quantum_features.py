"""
Quantum-inspired feature mapping.

This module implements quantum-inspired feature transformations using Random Fourier Features
to expand the feature space and capture non-linear patterns in the data.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class LearnableQuantumFeatureMap(nn.Module):
    """
    Learnable quantum-inspired feature mapping using trainable Fourier expansion.
    
    This class implements a trainable feature transformation inspired by quantum computing
    principles, using learnable Random Fourier Features to map input features to a higher
    dimensional space. Unlike the fixed version, this allows gradients to flow through
    the feature transformation during training.
    
    Args:
        input_dim: Dimensionality of input features
        expansion_factor: Factor by which to expand feature dimension (default: 2)
        use_fourier: Whether to use Fourier features (True) or simple projection (False)
        random_seed: Random seed for initialization
        
    Attributes:
        input_dim: Input feature dimension
        expansion_factor: Feature expansion factor
        use_fourier: Whether using Fourier transformation
        output_dim: Output feature dimension (input_dim * expansion_factor)
        W: Learnable projection matrix
        b: Learnable bias term (only for Fourier features)
        layer_norm: Layer normalization for stable training
    """
    
    def __init__(
        self, 
        input_dim: int, 
        expansion_factor: int = 2, 
        use_fourier: bool = True, 
        random_seed: int = 42
    ):
        super().__init__()
        
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if expansion_factor <= 0:
            raise ValueError("expansion_factor must be positive")
        
        self.input_dim = input_dim
        self.expansion_factor = expansion_factor
        self.use_fourier = use_fourier
        self.output_dim = input_dim * expansion_factor
        
        # Set seed for initialization
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        if use_fourier:
            # Learnable Random Fourier Features: z = [cos(Wx + b), sin(Wx + b)]
            self.W = nn.Parameter(
                torch.randn(input_dim, self.output_dim // 2) / np.sqrt(input_dim)
            )
            self.b = nn.Parameter(torch.rand(self.output_dim // 2) * 2 * np.pi)
        else:
            # Simple learnable projection: z = tanh(Wx)
            self.W = nn.Parameter(
                torch.randn(input_dim, self.output_dim) / np.sqrt(input_dim)
            )
            self.b = None
        
        # Layer normalization for stable training (replaces aggressive sqrt normalization)
        self.layer_norm = nn.LayerNorm(self.output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply learnable quantum-inspired feature mapping.
        
        Args:
            x: Input features of shape [num_nodes, input_dim]
            
        Returns:
            Transformed features of shape [num_nodes, output_dim]
        """
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch: expected {self.input_dim}, got {x.shape[1]}"
            )
        
        if self.use_fourier:
            # Compute Learnable Random Fourier Features
            projection = torch.matmul(x, self.W) + self.b
            cos_features = torch.cos(projection)
            sin_features = torch.sin(projection)
            
            # Concatenate and normalize with LayerNorm (not aggressive sqrt)
            transformed = torch.cat([cos_features, sin_features], dim=1)
            transformed = self.layer_norm(transformed)
        else:
            # Simple projection with tanh activation
            transformed = torch.matmul(x, self.W)
            transformed = torch.tanh(transformed)
            transformed = self.layer_norm(transformed)
        
        return transformed
    
    def get_config(self) -> dict:
        """
        Get configuration dictionary.
        
        Returns:
            Dictionary containing feature mapper configuration
        """
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "expansion_factor": self.expansion_factor,
            "use_fourier": self.use_fourier,
            "learnable": True,
        }


class QuantumFeatureMap:
    """
    Quantum-inspired feature mapping using Fourier expansion.
    
    This class implements a feature transformation inspired by quantum computing
    principles, using Random Fourier Features to map input features to a higher
    dimensional space. This can help capture non-linear patterns in the data.
    
    Args:
        input_dim: Dimensionality of input features
        expansion_factor: Factor by which to expand feature dimension (default: 2)
        use_fourier: Whether to use Fourier features (True) or simple projection (False)
        random_seed: Random seed for reproducibility
        
    Attributes:
        input_dim: Input feature dimension
        expansion_factor: Feature expansion factor
        use_fourier: Whether using Fourier transformation
        output_dim: Output feature dimension (input_dim * expansion_factor)
        W: Random projection matrix
        b: Random bias term (only for Fourier features)
    """
    
    def __init__(
        self, 
        input_dim: int, 
        expansion_factor: int = 2, 
        use_fourier: bool = True, 
        random_seed: int = 42
    ):
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if expansion_factor <= 0:
            raise ValueError("expansion_factor must be positive")
        
        self.input_dim = input_dim
        self.expansion_factor = expansion_factor
        self.use_fourier = use_fourier
        self.output_dim = input_dim * expansion_factor
        
        # Initialize random projection
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        if use_fourier:
            # Random Fourier Features: z = [cos(Wx + b), sin(Wx + b)]
            self.W = torch.randn(input_dim, self.output_dim // 2)
            self.b = torch.rand(self.output_dim // 2) * 2 * np.pi
        else:
            # Simple random projection: z = tanh(Wx)
            self.W = torch.randn(input_dim, self.output_dim) / np.sqrt(input_dim)
            self.b = None
    
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply quantum-inspired feature mapping.
        
        Args:
            x: Input features of shape [num_nodes, input_dim]
            
        Returns:
            Transformed features of shape [num_nodes, output_dim]
        """
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch: expected {self.input_dim}, got {x.shape[1]}"
            )
        
        if self.use_fourier:
            # Move parameters to same device as input
            W = self.W.to(x.device)
            b = self.b.to(x.device)
            
            # Compute Random Fourier Features
            projection = torch.matmul(x, W) + b
            cos_features = torch.cos(projection)
            sin_features = torch.sin(projection)
            
            # Concatenate and normalize
            transformed = torch.cat([cos_features, sin_features], dim=1)
            transformed = transformed / np.sqrt(self.output_dim)
        else:
            # Simple projection with tanh activation
            W = self.W.to(x.device)
            transformed = torch.matmul(x, W)
            transformed = torch.tanh(transformed)
        
        return transformed
    
    def get_config(self) -> dict:
        """
        Get configuration dictionary.
        
        Returns:
            Dictionary containing feature mapper configuration
        """
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "expansion_factor": self.expansion_factor,
            "use_fourier": self.use_fourier,
        }
