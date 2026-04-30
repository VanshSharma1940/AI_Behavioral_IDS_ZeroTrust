"""
Dataset loader for UNSW-NB15 and CICIDS2017 datasets.
Handles downloading, loading, and initial exploration.
"""
import os
import glob
import logging
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Unified data loader for IDS benchmark datasets.
    Supports UNSW-NB15 and CICIDS2017.
    """
    
    # Dataset metadata
    DATASET_INFO = {
        'unsw-nb15': {
            'n_features': 47,
            'n_attack_categories': 9,
            'attack_types': [
                'Fuzzers', 'Analysis', 'Backdoors', 'DoS', 
                'Exploits', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms'
            ],
            'benign_ratio': 0.87,
            'collection_period': '2 days (2015)'
        },
        'cicids2017': {
            'n_features': 80,
            'n_attack_categories': 8,
            'attack_types': [
                'Brute Force FTP', 'Brute Force SSH', 'DoS', 'Heartbleed',
                'Web Attack', 'Infiltration', 'Botnet', 'DDoS'
            ],
            'benign_ratio': 0.80,
            'collection_period': '5 days (July 2017)'
        }
    }
    
    def __init__(self, data_dir: str = "datasets"):
        """
        Initialize data loader.
        
        Args:
            data_dir: Root directory containing datasets
        """
        self.data_dir = Path(data_dir)
        self.datasets = {}
        
    def load_unsw_nb15(self, train_path: Optional[str] = None, 
                       test_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load UNSW-NB15 dataset.
        
        Args:
            train_path: Path to training CSV
            test_path: Path to testing CSV
            
        Returns:
            Combined DataFrame
        """
        train_path = train_path or self.data_dir / "UNSW_NB15_training.csv"
        test_path = test_path or self.data_dir / "UNSW_NB15_testing.csv"
        
        logger.info(f"Loading UNSW-NB15 dataset...")
        logger.info(f"  Training: {train_path}")
        logger.info(f"  Testing: {test_path}")
        
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Add dataset source
            train_df['dataset_source'] = 'train'
            test_df['dataset_source'] = 'test'
            
            combined = pd.concat([train_df, test_df], ignore_index=True)
            
            logger.info(f"  Loaded {len(combined)} records ({len(train_df)} train + {len(test_df)} test)")
            logger.info(f"  Features: {len(combined.columns)} columns")
            logger.info(f"  Attack categories: {combined['attack_cat'].nunique() if 'attack_cat' in combined.columns else 'N/A'}")
            
            self.datasets['unsw-nb15'] = combined
            return combined
            
        except FileNotFoundError as e:
            logger.error(f"Dataset file not found: {e}")
            logger.info("Please download UNSW-NB15 from: https://www.unb.ca/cic/datasets/cic-unsw-nb15.html")
            raise
    
    def load_cicids2017(self, data_dir: Optional[str] = None) -> pd.DataFrame:
        """
        Load CICIDS2017 dataset from directory of CSV files.
        
        Args:
            data_dir: Directory containing CICIDS2017 CSV files
            
        Returns:
            Combined DataFrame
        """
        data_dir = Path(data_dir or self.data_dir / "CICIDS2017")
        
        logger.info(f"Loading CICIDS2017 dataset from {data_dir}...")
        
        csv_files = sorted(glob.glob(str(data_dir / "*.csv")))
        
        if not csv_files:
            logger.error(f"No CSV files found in {data_dir}")
            logger.info("Please download CICIDS2017 from: https://www.unb.ca/cic/datasets/ids-2017.html")
            raise FileNotFoundError(f"No CICIDS2017 data files found in {data_dir}")
        
        logger.info(f"  Found {len(csv_files)} CSV files")
        
        dfs = []
        for file in csv_files:
            logger.debug(f"  Loading: {Path(file).name}")
            try:
                df = pd.read_csv(file)
                df['source_file'] = Path(file).name
                dfs.append(df)
            except Exception as e:
                logger.warning(f"  Error loading {file}: {e}")
        
        combined = pd.concat(dfs, ignore_index=True)
        
        logger.info(f"  Loaded {len(combined)} records from {len(dfs)} files")
        logger.info(f"  Features: {len(combined.columns)} columns")
        
        self.datasets['cicids2017'] = combined
        return combined
    
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """Get metadata about a dataset."""
        return self.DATASET_INFO.get(dataset_name.lower(), {})
    
    def print_dataset_summary(self, df: pd.DataFrame, dataset_name: str = "Dataset"):
        """Print comprehensive dataset summary."""
        print(f"\n{'='*60}")
        print(f"Dataset Summary: {dataset_name}")
        print(f"{'='*60}")
        print(f"Shape: {df.shape}")
        print(f"\nColumn Names:")
        print(f"  {', '.join(df.columns.tolist())}")
        print(f"\nData Types:")
        print(df.dtypes.value_counts())
        print(f"\nMissing Values:")
        missing = df.isnull().sum()
        print(missing[missing > 0] if missing.sum() > 0 else "  None")
        print(f"\nLabel Distribution:")
        if 'label' in df.columns:
            print(df['label'].value_counts())
        if 'attack_cat' in df.columns:
            print(f"\nAttack Category Distribution:")
            print(df['attack_cat'].value_counts())
        print(f"{'='*60}\n")
    
    @staticmethod
    def generate_synthetic_data(n_samples: int = 10000, n_features: int = 47,
                                n_attack_types: int = 5, random_state: int = 42) -> pd.DataFrame:
        """
        Generate synthetic network traffic data for testing.
        
        Args:
            n_samples: Total number of samples
            n_features: Number of features
            n_attack_types: Number of attack categories
            random_state: Random seed
            
        Returns:
            Synthetic DataFrame
        """
        np.random.seed(random_state)
        
        # Generate normal traffic (87% benign)
        n_normal = int(n_samples * 0.87)
        n_attack = n_samples - n_normal
        
        # Normal traffic features
        normal_data = np.random.normal(loc=0.5, scale=0.1, size=(n_normal, n_features))
        normal_data = np.clip(normal_data, 0, 1)
        
        # Attack traffic features (more varied)
        attack_data = np.random.normal(loc=0.7, scale=0.2, size=(n_attack, n_features))
        attack_data = np.clip(attack_data, 0, 1)
        
        # Combine
        X = np.vstack([normal_data, attack_data])
        y = np.concatenate([np.zeros(n_normal), np.ones(n_attack)])
        
        # Create attack categories
        attack_cats = ['Normal'] * n_normal
        attack_types = ['DoS', 'Exploits', 'Reconnaissance', 'Fuzzers', 'Generic'][:n_attack_types]
        
        for i in range(n_attack):
            attack_cats.append(np.random.choice(attack_types))
        
        # Shuffle
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]
        attack_cats = [attack_cats[i] for i in indices]
        
        # Create DataFrame
        feature_cols = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_cols)
        df['label'] = y.astype(int)
        df['attack_cat'] = attack_cats
        
        return df
