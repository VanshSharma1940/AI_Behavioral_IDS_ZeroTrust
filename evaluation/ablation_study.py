"""
Ablation study to quantify individual component contributions.
Systematically removes each model component to measure its impact.
"""
import logging
import numpy as np
from typing import Dict, Optional, Tuple

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from models.hybrid_ids import HybridIDS
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

logger = logging.getLogger(__name__)


class AblationStudy:
    """
    Ablation study for the Hybrid IDS.
    
    Measures the contribution of each component by:
    1. Full model (LSTM + IF + DNN) - baseline
    2. Without LSTM (IF + DNN only)
    3. Without Isolation Forest (LSTM + DNN only)
    4. Without DNN (LSTM + IF only)
    5. LSTM only
    6. Isolation Forest only
    7. DNN only
    
    This validates that each component contributes uniquely to the ensemble.
    """
    
    def __init__(self, hybrid_model: Optional['HybridIDS'] = None):
        """
        Initialize ablation study.
        
        Args:
            hybrid_model: Trained HybridIDS model
        """
        self.hybrid_model = hybrid_model
        self.results: Dict[str, Dict] = {}
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Compute standard metrics."""
        return {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred, zero_division=0))
        }
    
    def run_ablation(self, X_test_seq: np.ndarray, X_test_flat: np.ndarray,
                    y_test: np.ndarray) -> Dict[str, Dict]:
        """
        Run complete ablation study.
        
        Args:
            X_test_seq: Test sequences for LSTM
            X_test_flat: Test flat features for IF and DNN
            y_test: True labels
            
        Returns:
            Dictionary of ablation results
        """
        if self.hybrid_model is None or not self.hybrid_model.is_trained:
            raise ValueError("Trained hybrid model required for ablation study")
        
        logger.info("\n" + "="*60)
        logger.info("Starting Ablation Study")
        logger.info("="*60)
        
        # 1. Full Hybrid Model (baseline)
        logger.info("\n[1/7] Evaluating full Hybrid model...")
        results = self.hybrid_model.predict(X_test_seq, X_test_flat)
        baseline_metrics = self._compute_metrics(y_test[:len(results['ensemble'])], 
                                                 results['ensemble'])
        self.results['full_hybrid'] = baseline_metrics
        
        # 2. LSTM only
        logger.info("[2/7] Evaluating LSTM Autoencoder only...")
        lstm_metrics = self._compute_metrics(y_test[:len(results['lstm'])], 
                                            results['lstm'])
        self.results['lstm_only'] = lstm_metrics
        
        # 3. Isolation Forest only
        logger.info("[3/7] Evaluating Isolation Forest only...")
        if_metrics = self._compute_metrics(y_test[:len(results['isolation_forest'])], 
                                          results['isolation_forest'])
        self.results['isolation_forest_only'] = if_metrics
        
        # 4. DNN only
        logger.info("[4/7] Evaluating DNN Classifier only...")
        dnn_metrics = self._compute_metrics(y_test[:len(results['dnn'])], 
                                           results['dnn'])
        self.results['dnn_only'] = dnn_metrics
        
        # 5. Without LSTM (IF + DNN)
        logger.info("[5/7] Evaluating without LSTM (IF + DNN)...")
        ensemble_no_lstm = self._weighted_ensemble(
            if_pred=results['isolation_forest'],
            dnn_pred=results['dnn'],
            if_weight=self.hybrid_model.if_weight / (self.hybrid_model.if_weight + self.hybrid_model.dnn_weight),
            dnn_weight=self.hybrid_model.dnn_weight / (self.hybrid_model.if_weight + self.hybrid_model.dnn_weight)
        )
        no_lstm_metrics = self._compute_metrics(y_test[:len(ensemble_no_lstm)], 
                                               ensemble_no_lstm)
        self.results['without_lstm'] = no_lstm_metrics
        
        # 6. Without Isolation Forest (LSTM + DNN)
        logger.info("[6/7] Evaluating without Isolation Forest (LSTM + DNN)...")
        ensemble_no_if = self._weighted_ensemble(
            lstm_pred=results['lstm'],
            dnn_pred=results['dnn'],
            lstm_weight=self.hybrid_model.lstm_weight / (self.hybrid_model.lstm_weight + self.hybrid_model.dnn_weight),
            dnn_weight=self.hybrid_model.dnn_weight / (self.hybrid_model.lstm_weight + self.hybrid_model.dnn_weight)
        )
        no_if_metrics = self._compute_metrics(y_test[:len(ensemble_no_if)], 
                                             ensemble_no_if)
        self.results['without_isolation_forest'] = no_if_metrics
        
        # 7. Without DNN (LSTM + IF)
        logger.info("[7/7] Evaluating without DNN (LSTM + IF)...")
        ensemble_no_dnn = self._weighted_ensemble(
            lstm_pred=results['lstm'],
            if_pred=results['isolation_forest'],
            lstm_weight=self.hybrid_model.lstm_weight / (self.hybrid_model.lstm_weight + self.hybrid_model.if_weight),
            if_weight=self.hybrid_model.if_weight / (self.hybrid_model.lstm_weight + self.hybrid_model.if_weight)
        )
        no_dnn_metrics = self._compute_metrics(y_test[:len(ensemble_no_dnn)], 
                                              ensemble_no_dnn)
        self.results['without_dnn'] = no_dnn_metrics
        
        logger.info("\n" + "="*60)
        logger.info("Ablation Study Complete")
        logger.info("="*60)
        
        return self.results
    
    def _weighted_ensemble(self, lstm_pred: Optional[np.ndarray] = None,
                          if_pred: Optional[np.ndarray] = None,
                          dnn_pred: Optional[np.ndarray] = None,
                          lstm_weight: float = 0.0,
                          if_weight: float = 0.0,
                          dnn_weight: float = 0.0) -> np.ndarray:
        """
        Create weighted ensemble from available predictions.
        
        Args:
            lstm_pred: LSTM predictions
            if_pred: Isolation Forest predictions
            dnn_pred: DNN predictions
            lstm_weight: LSTM weight
            if_weight: IF weight
            dnn_weight: DNN weight
            
        Returns:
            Ensemble predictions
        """
        scores = np.zeros(len(lstm_pred or if_pred or dnn_pred))
        
        if lstm_pred is not None and lstm_weight > 0:
            scores += lstm_weight * lstm_pred.astype(float)
        if if_pred is not None and if_weight > 0:
            scores += if_weight * if_pred.astype(float)
        if dnn_pred is not None and dnn_weight > 0:
            scores += dnn_weight * dnn_pred.astype(float)
        
        return (scores > 0.5).astype(int)
    
    def print_results(self):
        """Print ablation study results table."""
        if not self.results:
            logger.info("No ablation results available. Run run_ablation() first.")
            return
        
        print("\n" + "="*80)
        print("ABLATION STUDY RESULTS")
        print("="*80)
        print(f"{'Configuration':<35} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-"*80)
        
        baseline_f1 = self.results.get('full_hybrid', {}).get('f1_score', 0)
        
        for config, metrics in self.results.items():
            acc = metrics.get('accuracy', 0)
            prec = metrics.get('precision', 0)
            rec = metrics.get('recall', 0)
            f1 = metrics.get('f1_score', 0)
            
            # Calculate F1 drop from baseline
            f1_drop = baseline_f1 - f1 if baseline_f1 > 0 else 0
            drop_str = f" (-{f1_drop:.4f})" if f1_drop > 0.001 and config != 'full_hybrid' else ""
            
            print(f"{config:<35} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}{drop_str}")
        
        print("="*80)
        
        # Print contribution analysis
        print("\nComponent Contribution Analysis:")
        print("-"*80)
        
        if 'without_lstm' in self.results and 'full_hybrid' in self.results:
            drop = baseline_f1 - self.results['without_lstm']['f1_score']
            print(f"  LSTM Autoencoder contribution:        +{drop:.4f} F1")
        
        if 'without_isolation_forest' in self.results and 'full_hybrid' in self.results:
            drop = baseline_f1 - self.results['without_isolation_forest']['f1_score']
            print(f"  Isolation Forest contribution:        +{drop:.4f} F1")
        
        if 'without_dnn' in self.results and 'full_hybrid' in self.results:
            drop = baseline_f1 - self.results['without_dnn']['f1_score']
            print(f"  DNN Classifier contribution:          +{drop:.4f} F1")
        
        print("="*80 + "\n")
    
    def get_component_contributions(self) -> Dict[str, float]:
        """
        Calculate each component's contribution to the ensemble.
        
        Returns:
            Dictionary of component -> F1 score contribution
        """
        if not self.results:
            return {}
        
        baseline_f1 = self.results.get('full_hybrid', {}).get('f1_score', 0)
        
        contributions = {}
        
        component_map = {
            'LSTM Autoencoder': 'without_lstm',
            'Isolation Forest': 'without_isolation_forest',
            'DNN Classifier': 'without_dnn'
        }
        
        for name, key in component_map.items():
            if key in self.results:
                drop = baseline_f1 - self.results[key]['f1_score']
                contributions[name] = float(drop)
        
        return contributions
    
    def plot_results(self, save_path: Optional[str] = None):
        """
        Plot ablation study results.
        
        Args:
            save_path: Path to save figure
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        if not self.results:
            logger.warning("No results to plot")
            return
        
        configs = list(self.results.keys())
        accuracies = [self.results[c].get('accuracy', 0) for c in configs]
        f1_scores = [self.results[c].get('f1_score', 0) for c in configs]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        colors = ['green' if c == 'full_hybrid' else 'steelblue' for c in configs]
        axes[0].bar(range(len(configs)), accuracies, color=colors, edgecolor='black')
        axes[0].set_xticks(range(len(configs)))
        axes[0].set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Ablation Study - Accuracy', fontweight='bold')
        axes[0].set_ylim(0, 1.1)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # F1 Score
        axes[1].bar(range(len(configs)), f1_scores, color=colors, edgecolor='black')
        axes[1].set_xticks(range(len(configs)))
        axes[1].set_xticklabels(configs, rotation=45, ha='right', fontsize=9)
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('Ablation Study - F1 Score', fontweight='bold')
        axes[1].set_ylim(0, 1.1)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            import os
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Ablation study plot saved to {save_path}")
        
        return fig
