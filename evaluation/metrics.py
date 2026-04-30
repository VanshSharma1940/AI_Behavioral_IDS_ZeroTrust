"""
Comprehensive evaluation metrics for IDS performance assessment.
Includes standard classification metrics and IDS-specific metrics.
"""
import logging
import numpy as np
from typing import Dict, Optional, Tuple

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, precision_recall_curve,
    matthews_corrcoef, cohen_kappa_score
)

logger = logging.getLogger(__name__)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute confusion matrix and derived metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with TN, FP, FN, TP and derived rates
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle edge cases
        tn = fp = fn = tp = 0
        if cm.shape == (1, 1):
            if y_true[0] == 0:
                tn = cm[0, 0]
            else:
                tp = cm[0, 0]
    
    # Compute rates
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Recall)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate (Specificity)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate
    
    return {
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'tpr': float(tpr), 'tnr': float(tnr),
        'fpr': float(fpr), 'fnr': float(fnr),
        'confusion_matrix': cm
    }


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                  y_scores: Optional[np.ndarray] = None,
                                  average: str = 'weighted') -> Dict:
    """
    Compute standard classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Prediction scores/probabilities (for AUC)
        average: Averaging method for multi-class
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        'matthews_corrcoef': float(matthews_corrcoef(y_true, y_pred)),
        'cohen_kappa': float(cohen_kappa_score(y_true, y_pred))
    }
    
    # AUC-ROC (requires scores)
    if y_scores is not None:
        try:
            if len(np.unique(y_true)) == 2:
                metrics['auc_roc'] = float(roc_auc_score(y_true, y_scores))
            else:
                # Multi-class AUC
                metrics['auc_roc'] = float(roc_auc_score(y_true, y_scores, multi_class='ovr', average=average))
        except Exception as e:
            logger.warning(f"Could not compute AUC: {e}")
            metrics['auc_roc'] = None
    
    return metrics


def compute_ids_specific_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute IDS-specific metrics.
    
    These metrics are particularly important for intrusion detection:
    - Detection Rate (Recall for attacks)
    - False Alarm Rate (FPR)
    - Attack Detection Rate per class
    - Balanced Accuracy
    
    Args:
        y_true: True labels (0=normal, 1=attack)
        y_pred: Predicted labels
        
    Returns:
        IDS-specific metrics dictionary
    """
    cm_info = compute_confusion_matrix(y_true, y_pred)
    
    # Detection Rate (DR) = TP / (TP + FN) - ability to detect attacks
    detection_rate = cm_info['tpr']
    
    # False Alarm Rate (FAR) = FP / (FP + TN) - false positives
    false_alarm_rate = cm_info['fpr']
    
    # Balanced Accuracy = (TPR + TNR) / 2
    balanced_accuracy = (cm_info['tpr'] + cm_info['tnr']) / 2
    
    # Positive Predictive Value (Precision)
    ppv = cm_info['tp'] / (cm_info['tp'] + cm_info['fp']) if (cm_info['tp'] + cm_info['fp']) > 0 else 0
    
    # Negative Predictive Value
    npv = cm_info['tn'] / (cm_info['tn'] + cm_info['fn']) if (cm_info['tn'] + cm_info['fn']) > 0 else 0
    
    # F-measure (F1)
    f_measure = 2 * (ppv * detection_rate) / (ppv + detection_rate) if (ppv + detection_rate) > 0 else 0
    
    return {
        'detection_rate': float(detection_rate),
        'false_alarm_rate': float(false_alarm_rate),
        'balanced_accuracy': float(balanced_accuracy),
        'positive_predictive_value': float(ppv),
        'negative_predictive_value': float(npv),
        'f_measure': float(f_measure),
        'specificity': float(cm_info['tnr']),
        'miss_rate': float(cm_info['fnr'])
    }


def compute_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                              class_names: Optional[list] = None) -> Dict:
    """
    Compute per-class metrics for multi-class classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional list of class names
        
    Returns:
        Per-class metrics dictionary
    """
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    
    if class_names is None:
        class_names = [str(i) for i in labels]
    
    per_class = {}
    
    for i, label in enumerate(labels):
        # Binary metrics for this class vs rest
        y_true_binary = (y_true == label).astype(int)
        y_pred_binary = (y_pred == label).astype(int)
        
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        class_name = class_names[i] if i < len(class_names) else str(label)
        per_class[class_name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'support': int(np.sum(y_true == label))
        }
    
    return per_class


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                  y_scores: Optional[np.ndarray] = None,
                  model_name: str = "Model") -> Dict:
    """
    Comprehensive model evaluation.
    
    Computes all relevant metrics for IDS evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Prediction scores/probabilities
        model_name: Name of the model for logging
        
    Returns:
        Complete evaluation results dictionary
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation Results: {model_name}")
    logger.info(f"{'='*60}")
    
    # Confusion matrix
    cm_info = compute_confusion_matrix(y_true, y_pred)
    
    # Classification metrics
    cls_metrics = compute_classification_metrics(y_true, y_pred, y_scores)
    
    # IDS-specific metrics
    ids_metrics = compute_ids_specific_metrics(y_true, y_pred)
    
    # Combine all metrics
    results = {
        'model_name': model_name,
        'confusion_matrix': cm_info,
        'classification': cls_metrics,
        'ids_specific': ids_metrics
    }
    
    # Print summary
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  TN={cm_info['tn']}, FP={cm_info['fp']}, FN={cm_info['fn']}, TP={cm_info['tp']}")
    
    logger.info(f"\nClassification Metrics:")
    for k, v in cls_metrics.items():
        if v is not None:
            logger.info(f"  {k}: {v:.4f}")
    
    logger.info(f"\nIDS-Specific Metrics:")
    for k, v in ids_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    logger.info(f"{'='*60}\n")
    
    return results


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       y_scores: Optional[np.ndarray] = None) -> Dict:
    """
    Compute all available metrics in one call.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Prediction scores
        
    Returns:
        Dictionary with all metrics
    """
    return {
        'confusion_matrix': compute_confusion_matrix(y_true, y_pred),
        'classification': compute_classification_metrics(y_true, y_pred, y_scores),
        'ids_specific': compute_ids_specific_metrics(y_true, y_pred)
    }
