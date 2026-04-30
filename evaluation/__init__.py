"""
Evaluation module for IDS performance assessment.
Provides metrics calculation, visualization, and ablation studies.
"""
from .metrics import evaluate_model, compute_all_metrics
from .visualization import plot_evaluation_results, plot_training_history, plot_confusion_matrix
from .ablation_study import AblationStudy

__all__ = ['evaluate_model', 'compute_all_metrics', 'plot_evaluation_results', 
           'plot_training_history', 'plot_confusion_matrix', 'AblationStudy']
