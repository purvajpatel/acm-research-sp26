"""
Visualization utilities for plotting results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc
from pathlib import Path
from typing import Dict, List, Optional


def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = "Confusion Matrix",
    labels: List[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True
):
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix
        title: Plot title
        labels: Class labels
        save_path: Path to save figure
        show: Whether to display the plot
    """
    if labels is None:
        labels = ['Licit', 'Illicit']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'},
        ax=ax
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "ROC Curve",
    label: str = "Model",
    save_path: Optional[Path] = None,
    show: bool = True,
    ax=None
):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        title: Plot title
        label: Curve label
        save_path: Path to save figure
        show: Whether to display the plot
        ax: Matplotlib axis (if None, creates new figure)
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        is_new_fig = True
    else:
        is_new_fig = False
    
    ax.plot(
        fpr, tpr, 
        linewidth=2, 
        label=f'{label} (AUC = {roc_auc:.3f})'
    )
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    if is_new_fig:
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


def plot_training_history(
    history: Dict[str, List],
    save_path: Optional[Path] = None,
    show: bool = True
):
    """
    Plot training history (loss and metrics over epochs).
    
    Args:
        history: Dictionary with 'train_loss' and 'val_metrics' lists
        save_path: Path to save figure
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training loss
    axes[0].plot(history['train_loss'], linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Validation metrics
    if 'val_metrics' in history and len(history['val_metrics']) > 0:
        epochs = range(1, len(history['val_metrics']) + 1)
        
        f1_scores = [m['f1'] for m in history['val_metrics']]
        auc_scores = [m['roc_auc'] for m in history['val_metrics']]
        
        axes[1].plot(epochs, f1_scores, linewidth=2, label='F1 Score')
        axes[1].plot(epochs, auc_scores, linewidth=2, label='ROC-AUC')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Score', fontsize=12)
        axes[1].set_title('Validation Metrics', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_metrics_comparison(
    baseline_metrics: Dict[str, float],
    quantum_metrics: Dict[str, float],
    save_path: Optional[Path] = None,
    show: bool = True
):
    """
    Plot bar chart comparing baseline vs quantum metrics.
    
    Args:
        baseline_metrics: Dictionary of baseline model metrics
        quantum_metrics: Dictionary of quantum model metrics
        save_path: Path to save figure
        show: Whether to display the plot
    """
    metrics_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    baseline_values = [baseline_metrics.get(m, 0) for m in metrics_names]
    quantum_values = [quantum_metrics.get(m, 0) for m in metrics_names]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline GAT', alpha=0.8)
    bars2 = ax.bar(x + width/2, quantum_values, width, label='Quantum GAT', alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Baseline vs Quantum GAT Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_names])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9
            )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_feature_distribution(
    features: np.ndarray,
    title: str = "Feature Distribution",
    save_path: Optional[Path] = None,
    show: bool = True
):
    """
    Plot histogram of feature values.
    
    Args:
        features: Feature array
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(features.flatten(), bins=100, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Feature Value', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
