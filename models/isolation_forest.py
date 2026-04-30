"""
Isolation Forest for unsupervised anomaly detection in network traffic.
Efficiently isolates anomalies by random feature partitioning.
"""
import os
import logging
import numpy as np
import joblib
from typing import Optional, Tuple, Dict
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class IsolationForestIDS:
    """
    Isolation Forest for network intrusion detection.
    
    Key advantage: Unsupervised learning - no labeled attack data needed.
    Anomalies are isolated faster (shorter path length) than normal instances.
    
    Optimized Parameters (from grid search):
    - n_estimators: 200 (number of isolation trees)
    - max_samples: 256 (samples per tree)
    - contamination: 0.1 (expected anomaly proportion)
    - max_features: 1.0 (use all features)
    """
    
    def __init__(self, n_estimators: int = 200, max_samples: int = 256,
                 contamination: float = 0.1, max_features: float = 1.0,
                 bootstrap: bool = False, random_state: int = 42,
                 n_jobs: int = -1, model_name: str = "isolation_forest"):
        """
        Initialize Isolation Forest.
        
        Args:
            n_estimators: Number of isolation trees
            max_samples: Samples to train each tree
            contamination: Expected proportion of anomalies in data
            max_features: Fraction of features to use per tree
            bootstrap: Whether to use bootstrap sampling
            random_state: Random seed
            n_jobs: Parallel jobs (-1 = all cores)
            model_name: Name for saved model files
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model_name = model_name
        
        self.model: Optional[IsolationForest] = None
        self.is_fitted = False
        
    def build_model(self) -> IsolationForest:
        """Build Isolation Forest model."""
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=0
        )
        logger.info(f"Isolation Forest built:")
        logger.info(f"  n_estimators={self.n_estimators}, max_samples={self.max_samples}")
        logger.info(f"  contamination={self.contamination}, max_features={self.max_features}")
        return self.model
    
    def train(self, X_train: np.ndarray):
        """
        Train Isolation Forest.
        
        Args:
            X_train: Training features (normal + some anomaly mixed)
        """
        if self.model is None:
            self.build_model()
        
        logger.info(f"Training Isolation Forest on {len(X_train)} samples...")
        
        self.model.fit(X_train)
        self.is_fitted = True
        
        logger.info("Isolation Forest training complete")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies.
        
        Args:
            X: Input features (2D array)
            
        Returns:
            (predictions, anomaly_scores)
            predictions: 1 = anomaly, 0 = normal
            anomaly_scores: Higher = more anomalous
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        # sklearn IF: -1 = anomaly, 1 = normal
        raw_predictions = self.model.predict(X)
        
        # anomaly_score: negative = more anomalous, positive = more normal
        anomaly_scores = self.model.decision_function(X)
        
        # Convert to binary: 1 = anomaly, 0 = normal
        binary_predictions = (raw_predictions == -1).astype(int)
        
        # Convert scores so higher = more anomalous
        # (flip sign and normalize to [0, 1] range)
        normalized_scores = -anomaly_scores  # Now higher = more anomalous
        
        return binary_predictions, normalized_scores
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: True labels (0 = normal, 1 = anomaly)
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions, scores = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        logger.info(f"\nIsolation Forest Evaluation:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")
        logger.info(f"  FPR:       {fpr:.4f}")
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fpr': fpr,
            'fnr': fnr,
            'confusion_matrix': cm,
            'predictions': predictions,
            'scores': scores
        }
    
    def save(self, path: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Isolation Forest saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        self.model = joblib.load(path)
        self.is_fitted = True
        logger.info(f"Isolation Forest loaded from {path}")
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Estimate feature importance from isolation trees.
        
        Returns:
            Array of feature importance scores
        """
        if not self.is_fitted:
            return None
        
        # Compute average depth per feature across all trees
        feature_importance = np.zeros(self.model.n_features_in_)
        
        for tree in self.model.estimators_:
            # Get tree structure
            tree_features = tree.tree_.feature
            # Count feature usage (excluding leaf nodes marked as -2)
            for f in tree_features:
                if f >= 0:
                    feature_importance[f] += 1
        
        # Normalize
        feature_importance = feature_importance / np.sum(feature_importance)
        
        return feature_importance
