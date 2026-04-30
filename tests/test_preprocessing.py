"""
Unit tests for data preprocessing module.
"""
import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessor import DataPreprocessor
from data.smote_balancing import apply_smote_enn, check_class_imbalance


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor."""
    
    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.n_samples = 1000
        self.n_features = 20
        
        # Generate synthetic data
        X = np.random.randn(self.n_samples, self.n_features)
        y = np.random.randint(0, 2, self.n_samples)
        
        self.df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(self.n_features)])
        self.df['label'] = y
        self.df['attack_cat'] = np.where(y == 1, 'DoS', 'Normal')
    
    def test_preprocessor_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = DataPreprocessor(
            normalization_method='minmax',
            sequence_length=10,
            use_pca=True,
            n_components=5
        )
        
        self.assertEqual(preprocessor.sequence_length, 10)
        self.assertEqual(preprocessor.n_components, 5)
        self.assertTrue(preprocessor.use_pca)
        self.assertFalse(preprocessor.is_fitted)
    
    def test_column_identification(self):
        """Test column type identification."""
        preprocessor = DataPreprocessor()
        
        # Add categorical column
        self.df['proto'] = np.random.choice(['tcp', 'udp', 'icmp'], self.n_samples)
        
        cols = preprocessor.identify_columns(self.df)
        
        self.assertIn('categorical', cols)
        self.assertIn('numerical', cols)
        self.assertIn('proto', cols['categorical'])
    
    def test_missing_value_handling(self):
        """Test missing value handling."""
        preprocessor = DataPreprocessor()
        preprocessor.numerical_columns = [f'feature_{i}' for i in range(self.n_features)]
        preprocessor.categorical_columns = []
        
        # Introduce NaN values
        df_with_nan = self.df.copy()
        df_with_nan.iloc[0, 0] = np.nan
        df_with_nan.iloc[1, 1] = np.inf
        
        cleaned = preprocessor.handle_missing_values(df_with_nan)
        
        self.assertFalse(cleaned.isnull().any().any())
        self.assertFalse(np.isinf(cleaned.select_dtypes(include=[np.number])).any().any())
    
    def test_fit_transform(self):
        """Test complete preprocessing pipeline."""
        preprocessor = DataPreprocessor(
            sequence_length=10,
            use_pca=False
        )
        
        results = preprocessor.fit_transform(self.df, test_size=0.2)
        
        # Check results structure
        self.assertIn('X_train_seq', results)
        self.assertIn('X_test_seq', results)
        self.assertIn('y_train_seq', results)
        self.assertIn('y_test_seq', results)
        
        # Check shapes
        self.assertEqual(results['X_train_seq'].ndim, 3)  # (samples, seq_len, features)
        self.assertEqual(results['X_test_seq'].ndim, 3)
        
        # Check preprocessor is fitted
        self.assertTrue(preprocessor.is_fitted)


class TestSMOTEBalancing(unittest.TestCase):
    """Test cases for SMOTE balancing."""
    
    def test_class_imbalance_detection(self):
        """Test class imbalance detection."""
        # Highly imbalanced data
        y = np.array([0] * 900 + [1] * 100)
        
        is_imbalanced, info = check_class_imbalance(y)
        
        self.assertTrue(is_imbalanced)
        self.assertIn('imbalance_ratio', info)
        self.assertIn('majority_count', info)
        self.assertIn('minority_count', info)
    
    def test_smote_enn(self):
        """Test SMOTE-ENN application."""
        np.random.seed(42)
        X = np.random.randn(1000, 20)
        y = np.array([0] * 900 + [1] * 100)
        
        X_balanced, y_balanced = apply_smote_enn(X, y)
        
        # Should have more balanced classes
        unique, counts = np.unique(y_balanced, return_counts=True)
        self.assertGreater(len(unique), 0)


if __name__ == '__main__':
    unittest.main()
