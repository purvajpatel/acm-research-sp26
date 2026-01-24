"""
Utility functions for Aegis project.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union
import json


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(verbose: bool = True) -> torch.device:
    """
    Get the appropriate device (CUDA or CPU).
    
    Args:
        verbose: Whether to print device info
        
    Returns:
        torch.device object
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        if device.type == 'cuda':
            print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
        else:
            print(f"Device: {device}")
    return device


def save_metrics(metrics: Dict[str, Any], filepath: Union[Path, str]):
    """
    Save metrics dictionary to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        filepath: Path to save file
    """
    # Convert numpy types to Python types
    metrics_serializable = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            metrics_serializable[key] = value.item()
        elif isinstance(value, np.ndarray):
            metrics_serializable[key] = value.tolist()
        else:
            metrics_serializable[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    print(f"Metrics saved to {filepath}")


def load_metrics(filepath: Union[Path, str]) -> Dict[str, Any]:
    """
    Load metrics from JSON file.
    
    Args:
        filepath: Path to metrics file
        
    Returns:
        Dictionary of metrics
    """
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    return metrics


def validate_data(data, required_attributes: list = None):
    """
    Validate PyTorch Geometric Data object.
    
    Args:
        data: PyG Data object
        required_attributes: List of required attribute names
        
    Raises:
        ValueError: If validation fails
    """
    if required_attributes is None:
        required_attributes = ['x', 'edge_index', 'y']
    
    for attr in required_attributes:
        if not hasattr(data, attr):
            raise ValueError(f"Data object missing required attribute: {attr}")
    
    # Check for NaN values
    if torch.isnan(data.x).any():
        raise ValueError("Features contain NaN values")
    
    # Check edge_index validity
    if data.edge_index.min() < 0 or data.edge_index.max() >= data.num_nodes:
        raise ValueError("Invalid edge indices")
    
    print("✓ Data validation passed")


def print_model_summary(model, input_shape=None):
    """
    Print model architecture summary.
    
    Args:
        model: PyTorch model
        input_shape: Optional input shape description
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Model Architecture: {model.__class__.__name__}")
    if input_shape:
        print(f"Input Shape: {input_shape}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {total_params - trainable_params:,}")
    print("=" * 60)


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"
