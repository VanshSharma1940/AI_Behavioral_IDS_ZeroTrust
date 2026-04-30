"""
Comprehensive data preprocessing pipeline for IDS datasets.
Handles normalization, encoding, sequence preparation, and PCA.
"""
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict, Any
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import joblib

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for IDS datasets.
    
    Features:
    - Automatic column type identification
    - Missing value handling
    - Categorical encoding (one-hot)
    - Feature normalization (Min-Max or Standard)
    - Sequence preparation for LSTM
    - PCA dimensionality reduction
    - Train/test splitting with stratification
    """
    
    def __init__(self, normalization_method: str = "minmax", 
                 sequence_length: int = 100,
                 use_pca: bool = True, 
                 n_components: int = 10,
                 random_state: int = 42):
        """
        Initialize preprocessor.
        
        Args:
            normalization_method: 'minmax' or 'standard'
            sequence_length: Time steps for LSTM sequences
            use_pca: Whether to apply PCA
            n_components: Number of principal components
            random_state: Random seed
        """
        self.normalization_method = normalization_method
        self.sequence_length = sequence_length
        self.use_pca = use_pca
        self.n_components = n_components
        self.random_state = random_state
        
        # Fitted transformers
        self.scaler = MinMaxScaler() if normalization_method == "minmax" else StandardScaler()
        self.label_encoder = LabelEncoder()
        self.pca = PCA(n_components=n_components) if use_pca else None
        
        # Column tracking
        self.categorical_columns: List[str] = []
        self.numerical_columns: List[str] = []
        self.feature_columns: List[str] = []
        
        # Metadata
        self.is_fitted = False
        self.n_features_original = 0
        self.n_features_processed = 0
        
    def identify_columns(self, df: pd.DataFrame, target_col: str = 'label',
                        attack_cat_col: str = 'attack_cat') -> Dict[str, List[str]]:
        """
        Identify categorical and numerical columns.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            attack_cat_col: Name of attack category column
            
        Returns:
            Dictionary with 'categorical' and 'numerical' column lists
        """
        exclude_cols = [target_col, attack_cat_col, 'dataset_source', 'source_file']
        
        self.categorical_columns = [
            col for col in df.select_dtypes(include=['object', 'category']).columns
            if col not in exclude_cols
        ]
        
        self.numerical_columns = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in exclude_cols and col not in self.categorical_columns
        ]
        
        logger.info(f"Column identification:")
        logger.info(f"  Categorical: {len(self.categorical_columns)} ({self.categorical_columns})")
        logger.info(f"  Numerical: {len(self.numerical_columns)}")
        
        return {
            'categorical': self.categorical_columns,
            'numerical': self.numerical_columns
        }
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing and infinite values.
        
        Replaces infinite values with NaN, then fills NaN with
        column mean for numerical and mode for categorical.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Replace infinite values with NaN
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        
        # Track missing values
        missing_before = df_clean.isnull().sum().sum()
        
        # Fill numerical NaN with mean
        for col in self.numerical_columns:
            if col in df_clean.columns and df_clean[col].isnull().any():
                mean_val = df_clean[col].mean()
                df_clean[col] = df_clean[col].fillna(mean_val)
        
        # Fill categorical NaN with mode
        for col in self.categorical_columns:
            if col in df_clean.columns and df_clean[col].isnull().any():
                mode_val = df_clean[col].mode()
                if len(mode_val) > 0:
                    df_clean[col] = df_clean[col].fillna(mode_val[0])
                else:
                    df_clean[col] = df_clean[col].fillna('unknown')
        
        missing_after = df_clean.isnull().sum().sum()
        if missing_before > 0:
            logger.info(f"Missing values handled: {missing_before} -> {missing_after}")
        
        return df_clean
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode categorical features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Encoded DataFrame
        """
        if not self.categorical_columns:
            return df
        
        # Only encode columns that exist in the dataframe
        cols_to_encode = [col for col in self.categorical_columns if col in df.columns]
        
        if not cols_to_encode:
            return df
        
        df_encoded = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)
        
        logger.info(f"Categorical encoding: {len(cols_to_encode)} columns -> {df_encoded.shape[1]} total features")
        
        return df_encoded
    
    def prepare_sequences(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """
        Prepare sequences for LSTM input.
        
        Creates sliding window sequences from flattened data.
        
        Args:
            data: Normalized feature array (n_samples, n_features)
            sequence_length: Number of time steps per sequence
            
        Returns:
            Sequences array (n_samples - seq_len + 1, seq_len, n_features)
        """
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            seq = data[i:i + sequence_length]
            sequences.append(seq)
        return np.array(sequences)
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'label',
                     attack_cat_col: str = 'attack_cat',
                     test_size: float = 0.2) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline - fit and transform.
        
        This method:
        1. Identifies column types
        2. Handles missing values
        3. Encodes categorical features
        4. Splits train/test
        5. Normalizes features
        6. Applies PCA (if enabled)
        7. Prepares LSTM sequences
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            attack_cat_col: Name of attack category column
            test_size: Fraction for test set
            
        Returns:
            Dictionary containing all processed data splits
        """
        logger.info("Starting data preprocessing pipeline...")
        
        # Step 1: Identify columns
        self.identify_columns(df, target_col, attack_cat_col)
        
        # Step 2: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 3: Separate features and labels
        y_binary = df[target_col].copy()
        
        if attack_cat_col in df.columns:
            y_multi = df[attack_cat_col].copy()
        else:
            y_multi = None
        
        # Drop target columns
        drop_cols = [col for col in [target_col, attack_cat_col, 'dataset_source', 'source_file'] 
                     if col in df.columns]
        X = df.drop(columns=drop_cols)
        
        # Step 4: Encode categorical features
        X = self.encode_categorical(X)
        self.feature_columns = X.columns.tolist()
        self.n_features_original = X.shape[1]
        
        # Step 5: Split data (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=test_size, 
            random_state=self.random_state, stratify=y_binary
        )
        
        logger.info(f"Train/Test split: {len(X_train)}/{len(X_test)}")
        logger.info(f"  Train class distribution: {np.bincount(y_train.astype(int))}")
        logger.info(f"  Test class distribution: {np.bincount(y_test.astype(int))}")
        
        # Step 6: Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Feature normalization: {self.normalization_method}")
        
        # Step 7: Apply PCA if enabled
        if self.use_pca and self.n_components:
            X_train_scaled = self.pca.fit_transform(X_train_scaled)
            X_test_scaled = self.pca.transform(X_test_scaled)
            
            explained_var = np.sum(self.pca.explained_variance_ratio_)
            reduction_ratio = (1 - self.n_components / self.n_features_original) * 100
            
            logger.info(f"PCA applied: {self.n_features_original} -> {self.n_components} features")
            logger.info(f"  Explained variance: {explained_var:.4f}")
            logger.info(f"  Reduction ratio: {reduction_ratio:.2f}%")
        
        self.n_features_processed = X_train_scaled.shape[1]
        
        # Step 8: Prepare sequences for LSTM
        X_train_seq = self.prepare_sequences(X_train_scaled, self.sequence_length)
        X_test_seq = self.prepare_sequences(X_test_scaled, self.sequence_length)
        
        # Adjust labels for sequences (label at end of sequence)
        y_train_seq = y_train.iloc[self.sequence_length - 1:].values if hasattr(y_train, 'iloc') else y_train[self.sequence_length - 1:]
        y_test_seq = y_test.iloc[self.sequence_length - 1:].values if hasattr(y_test, 'iloc') else y_test[self.sequence_length - 1:]
        
        self.is_fitted = True
        
        logger.info(f"Sequence preparation: length={self.sequence_length}")
        logger.info(f"  Train sequences: {X_train_seq.shape}")
        logger.info(f"  Test sequences: {X_test_seq.shape}")
        logger.info(f"Preprocessing complete!")
        
        result = {
            # Sequence data (for LSTM)
            'X_train_seq': X_train_seq,
            'X_test_seq': X_test_seq,
            'y_train_seq': y_train_seq,
            'y_test_seq': y_test_seq,
            # Flat data (for IF and DNN)
            'X_train_flat': X_train_scaled,
            'X_test_flat': X_test_scaled,
            'y_train_flat': y_train.values if hasattr(y_train, 'values') else y_train,
            'y_test_flat': y_test.values if hasattr(y_test, 'values') else y_test,
            # Metadata
            'n_features': self.n_features_processed,
            'feature_columns': self.feature_columns,
            'class_distribution': {
                'train': np.bincount(y_train.astype(int)).tolist(),
                'test': np.bincount(y_test.astype(int)).tolist()
            }
        }
        
        return result
    
    def transform(self, df: pd.DataFrame, target_col: str = 'label') -> np.ndarray:
        """
        Transform new data using fitted preprocessor.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Processed features array
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform. Call fit_transform first.")
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Drop target if present
        if target_col in df.columns:
            X = df.drop(columns=[target_col])
        else:
            X = df.copy()
        
        # Encode categorical
        X = self.encode_categorical(X)
        
        # Ensure same columns as training
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_columns]
        
        # Normalize
        X_scaled = self.scaler.transform(X)
        
        # PCA
        if self.use_pca and self.pca:
            X_scaled = self.pca.transform(X_scaled)
        
        return X_scaled
    
    def save(self, path: str):
        """Save preprocessor state."""
        joblib.dump({
            'scaler': self.scaler,
            'pca': self.pca,
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted,
            'n_features_original': self.n_features_original,
            'n_features_processed': self.n_features_processed,
            'normalization_method': self.normalization_method,
            'sequence_length': self.sequence_length,
            'use_pca': self.use_pca,
            'n_components': self.n_components
        }, path)
        logger.info(f"Preprocessor saved to {path}")
    
    def load(self, path: str):
        """Load preprocessor state."""
        state = joblib.load(path)
        self.scaler = state['scaler']
        self.pca = state['pca']
        self.categorical_columns = state['categorical_columns']
        self.numerical_columns = state['numerical_columns']
        self.feature_columns = state['feature_columns']
        self.is_fitted = state['is_fitted']
        self.n_features_original = state['n_features_original']
        self.n_features_processed = state['n_features_processed']
        logger.info(f"Preprocessor loaded from {path}")
