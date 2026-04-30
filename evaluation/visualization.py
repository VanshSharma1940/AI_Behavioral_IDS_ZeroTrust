"""
Visualization utilities for IDS evaluation results.
Generates publication-ready plots for model performance analysis.
"""
import os
import logging
from typing import Optional, Dict, List

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix
)

logger = logging.getLogger(__name__)

# Set publication style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         class_names: Optional[List[str]] = None,
                         model_name: str = "Model",
                         save_path: Optional[str] = None,
                         figsize: tuple = (8, 6)) -> plt.Figure:
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names for labels
        model_name: Model name for title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Count'})
    
    ax.set_title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    return fig


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray,
                  model_name: str = "Model",
                  save_path: Optional[str] = None,
                  figsize: tuple = (8, 6)) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_scores: Prediction scores
        model_name: Model name
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2,
           label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'{model_name} - ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")
    
    return fig


def plot_precision_recall_curve(y_true: np.ndarray, y_scores: np.ndarray,
                                model_name: str = "Model",
                                save_path: Optional[str] = None,
                                figsize: tuple = (8, 6)) -> plt.Figure:
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_scores: Prediction scores
        model_name: Model name
        save_path: Path to save
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(recall, precision, color='blue', lw=2)
    ax.fill_between(recall, precision, alpha=0.2, color='blue')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'{model_name} - Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Precision-Recall curve saved to {save_path}")
    
    return fig


def plot_score_distribution(y_true: np.ndarray, y_scores: np.ndarray,
                           threshold: Optional[float] = None,
                           model_name: str = "Model",
                           save_path: Optional[str] = None,
                           figsize: tuple = (10, 6)) -> plt.Figure:
    """
    Plot anomaly score distribution.
    
    Args:
        y_true: True labels
        y_scores: Anomaly scores
        threshold: Detection threshold
        model_name: Model name
        save_path: Path to save
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram
    axes[0].hist(y_scores[y_true == 0], bins=50, alpha=0.6, label='Normal', color='blue', density=True)
    axes[0].hist(y_scores[y_true == 1], bins=50, alpha=0.6, label='Attack', color='red', density=True)
    
    if threshold:
        axes[0].axvline(threshold, color='black', linestyle='--', linewidth=2,
                       label=f'Threshold: {threshold:.4f}')
    
    axes[0].set_xlabel('Anomaly Score', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title('Score Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    data_to_plot = [y_scores[y_true == 0], y_scores[y_true == 1]]
    bp = axes[1].boxplot(data_to_plot, labels=['Normal', 'Attack'], patch_artist=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][1].set_facecolor('red')
    
    if threshold:
        axes[1].axhline(threshold, color='black', linestyle='--', linewidth=2,
                       label=f'Threshold: {threshold:.4f}')
    
    axes[1].set_ylabel('Anomaly Score', fontsize=11)
    axes[1].set_title('Score Comparison', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(f'{model_name} - Anomaly Score Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Score distribution saved to {save_path}")
    
    return fig


def plot_training_history(history: Dict, model_name: str = "Model",
                         save_path: Optional[str] = None,
                         figsize: tuple = (12, 4)) -> plt.Figure:
    """
    Plot training history (loss and metrics).
    
    Args:
        history: Training history dictionary from Keras
        model_name: Model name
        save_path: Path to save
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Determine available metrics
    metrics = [k for k in history.keys() if not k.startswith('val_')]
    
    n_plots = len(metrics)
    fig, axes = plt.subplots(1, min(n_plots, 3), figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics[:3]):
        ax = axes[i] if n_plots > 1 else axes[0]
        
        ax.plot(history[metric], label=f'Training {metric}', linewidth=2)
        
        val_metric = f'val_{metric}'
        if val_metric in history:
            ax.plot(history[val_metric], label=f'Validation {metric}', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
        ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{model_name} - Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")
    
    return fig


def plot_evaluation_results(y_true: np.ndarray, y_pred: np.ndarray,
                           y_scores: np.ndarray,
                           model_name: str = "Model",
                           threshold: Optional[float] = None,
                           save_dir: str = "outputs/plots") -> Dict[str, str]:
    """
    Generate all evaluation plots.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Prediction scores
        model_name: Model name
        threshold: Detection threshold
        save_dir: Directory to save plots
        
    Returns:
        Dictionary of saved plot paths
    """
    os.makedirs(save_dir, exist_ok=True)
    
    safe_name = model_name.lower().replace(' ', '_')
    paths = {}
    
    # Confusion matrix
    path = os.path.join(save_dir, f'{safe_name}_confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, model_name=model_name, save_path=path)
    paths['confusion_matrix'] = path
    plt.close()
    
    # ROC curve
    path = os.path.join(save_dir, f'{safe_name}_roc_curve.png')
    plot_roc_curve(y_true, y_scores, model_name=model_name, save_path=path)
    paths['roc_curve'] = path
    plt.close()
    
    # Precision-Recall curve
    path = os.path.join(save_dir, f'{safe_name}_precision_recall.png')
    plot_precision_recall_curve(y_true, y_scores, model_name=model_name, save_path=path)
    paths['precision_recall'] = path
    plt.close()
    
    # Score distribution
    path = os.path.join(save_dir, f'{safe_name}_score_distribution.png')
    plot_score_distribution(y_true, y_scores, threshold=threshold,
                           model_name=model_name, save_path=path)
    paths['score_distribution'] = path
    plt.close()
    
    logger.info(f"All evaluation plots saved to {save_dir}")
    
    return paths


def plot_model_comparison(results: Dict[str, Dict], metric: str = 'f1_score',
                         save_path: Optional[str] = None,
                         figsize: tuple = (10, 6)) -> plt.Figure:
    """
    Compare multiple models on a specific metric.
    
    Args:
        results: Dictionary of model_name -> metrics_dict
        metric: Metric to compare
        save_path: Path to save
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    models = list(results.keys())
    values = [results[m].get(metric, 0) for m in models]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models)))
    bars = ax.bar(models, values, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Model Comparison - {metric.replace("_", " ").title()}', 
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison saved to {save_path}")
    
    return fig
