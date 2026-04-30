"""
SMOTE-ENN class balancing for imbalanced intrusion detection datasets.
Handles minority class oversampling and noisy sample cleaning.
"""
import logging
import numpy as np
from collections import Counter
from typing import Tuple, Optional
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logger = logging.getLogger(__name__)


def apply_smote_enn(X_train: np.ndarray, y_train: np.ndarray,
                   sampling_strategy: str = "auto",
                   random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE-ENN for class balancing.
    
    SMOTE (Synthetic Minority Over-sampling Technique) generates
    synthetic samples for minority classes. ENN (Edited Nearest Neighbors)
    removes ambiguous samples that may be misclassified.
    
    This is crucial for IDS datasets where attack samples are often
    significantly outnumbered by normal traffic (80-87% benign).
    
    Args:
        X_train: Training features (2D array)
        y_train: Training labels
        sampling_strategy: Sampling strategy ('auto', 'minority', 'not minority', 'all')
        random_state: Random seed
        
    Returns:
        Balanced X_train, y_train
    """
    logger.info("Applying SMOTE-ENN class balancing...")
    
    # Check original distribution
    original_dist = Counter(y_train)
    logger.info(f"  Original class distribution: {dict(original_dist)}")
    
    # For sequence data (3D), reshape to 2D
    original_shape = X_train.shape
    is_sequence = len(original_shape) == 3
    
    if is_sequence:
        n_samples, seq_len, n_features = original_shape
        X_reshaped = X_train.reshape(n_samples, seq_len * n_features)
        logger.info(f"  Reshaped sequences: {original_shape} -> {X_reshaped.shape}")
    else:
        X_reshaped = X_train
    
    # Apply SMOTE-ENN
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    smote_enn = SMOTEENN(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        smote=smote
    )
    
    try:
        X_resampled, y_resampled = smote_enn.fit_resample(X_reshaped, y_train)
        
        resampled_dist = Counter(y_resampled)
        logger.info(f"  Resampled class distribution: {dict(resampled_dist)}")
        logger.info(f"  Total samples: {len(y_train)} -> {len(y_resampled)}")
        
        # Reshape back to 3D if needed
        if is_sequence:
            # Note: SMOTE changes sample count, so we can't preserve exact sequence structure
            # For LSTM, we use the flat data and the LSTM autoencoder handles sequences separately
            pass
        
        return X_resampled, y_resampled
        
    except Exception as e:
        logger.error(f"SMOTE-ENN failed: {e}")
        logger.info("Falling back to original data (class imbalance may affect performance)")
        return X_train, y_train


def apply_smote_only(X_train: np.ndarray, y_train: np.ndarray,
                    sampling_strategy: str = "auto",
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE only (without ENN cleaning).
    
    Use this when you want to preserve all original samples
    and only add synthetic minority samples.
    
    Args:
        X_train: Training features
        y_train: Training labels
        sampling_strategy: Sampling strategy
        random_state: Random seed
        
    Returns:
        Balanced X_train, y_train
    """
    logger.info("Applying SMOTE oversampling...")
    
    original_dist = Counter(y_train)
    logger.info(f"  Original class distribution: {dict(original_dist)}")
    
    # Reshape if sequence data
    is_sequence = len(X_train.shape) == 3
    if is_sequence:
        n_samples, seq_len, n_features = X_train.shape
        X_reshaped = X_train.reshape(n_samples, seq_len * n_features)
    else:
        X_reshaped = X_train
    
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    
    try:
        X_resampled, y_resampled = smote.fit_resample(X_reshaped, y_train)
        
        resampled_dist = Counter(y_resampled)
        logger.info(f"  Resampled class distribution: {dict(resampled_dist)}")
        
        return X_resampled, y_resampled
        
    except Exception as e:
        logger.error(f"SMOTE failed: {e}")
        return X_train, y_train


def check_class_imbalance(y: np.ndarray, threshold: float = 0.2) -> Tuple[bool, Dict]:
    """
    Check if dataset has class imbalance problem.
    
    Args:
        y: Labels array
        threshold: Imbalance ratio threshold (minority/majority)
        
    Returns:
        (is_imbalanced, info_dict)
    """
    counts = Counter(y)
    total = sum(counts.values())
    
    majority_class = max(counts, key=counts.get)
    minority_class = min(counts, key=counts.get)
    
    majority_count = counts[majority_class]
    minority_count = counts[minority_class]
    
    imbalance_ratio = minority_count / majority_count
    is_imbalanced = imbalance_ratio < threshold
    
    info = {
        'is_imbalanced': is_imbalanced,
        'imbalance_ratio': imbalance_ratio,
        'majority_class': int(majority_class),
        'majority_count': majority_count,
        'majority_percentage': majority_count / total * 100,
        'minority_class': int(minority_class),
        'minority_count': minority_count,
        'minority_percentage': minority_count / total * 100,
        'class_distribution': dict(counts)
    }
    
    return is_imbalanced, info


# Type hint for Dict
from typing import Dict
