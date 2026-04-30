#!/usr/bin/env python3
"""
Training script for the Hybrid IDS model.
Usage: python scripts/train_model.py --dataset unsw-nb15 --epochs 50
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import config, Config
from data.data_loader import DataLoader
from data.preprocessor import DataPreprocessor
from data.smote_balancing import apply_smote_enn
from models.hybrid_ids import HybridIDS
from utils.logger import setup_logger
from utils.helpers import timer, save_json

logger = setup_logger("IDS-Training", log_file="outputs/logs/training.log")


def parse_args():
    parser = argparse.ArgumentParser(description='Train Hybrid IDS Model')
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['unsw-nb15', 'cicids2017', 'synthetic'],
                       help='Dataset to use')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--sequence-length', type=int, default=100, help='LSTM sequence length')
    parser.add_argument('--use-pca', action='store_true', help='Apply PCA')
    parser.add_argument('--n-components', type=int, default=10, help='PCA components')
    parser.add_argument('--use-smote', action='store_true', help='Apply SMOTE-ENN')
    parser.add_argument('--output-dir', type=str, default='outputs/models', help='Output directory')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    return parser.parse_args()


@timer
def main():
    args = parse_args()
    
    logger.info("="*60)
    logger.info("Hybrid IDS Training Pipeline")
    logger.info("="*60)
    
    # Load config if provided
    if args.config:
        cfg = Config.from_yaml(args.config)
    else:
        cfg = config
    
    # Ensure output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('outputs/logs', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)
    
    # Step 1: Load dataset
    logger.info("\n[Step 1/5] Loading dataset...")
    loader = DataLoader()
    
    if args.dataset == 'synthetic':
        df = DataLoader.generate_synthetic_data(n_samples=10000, n_features=47)
    elif args.dataset == 'unsw-nb15':
        df = loader.load_unsw_nb15()
    else:
        df = loader.load_cicids2017()
    
    loader.print_dataset_summary(df, args.dataset)
    
    # Step 2: Preprocess
    logger.info("\n[Step 2/5] Preprocessing data...")
    preprocessor = DataPreprocessor(
        normalization_method='minmax',
        sequence_length=args.sequence_length,
        use_pca=args.use_pca,
        n_components=args.n_components
    )
    
    results = preprocessor.fit_transform(df)
    
    X_train_seq = results['X_train_seq']
    X_test_seq = results['X_test_seq']
    y_train_seq = results['y_train_seq']
    y_test_seq = results['y_test_seq']
    X_train_flat = results['X_train_flat']
    X_test_flat = results['X_test_flat']
    y_train_flat = results['y_train_flat']
    y_test_flat = results['y_test_flat']
    n_features = results['n_features']
    
    # Save preprocessor
    preprocessor.save(os.path.join(args.output_dir, 'preprocessor.pkl'))
    
    # Step 3: Apply SMOTE-ENN if requested
    if args.use_smote:
        logger.info("\n[Step 3/5] Applying SMOTE-ENN...")
        X_train_flat, y_train_flat = apply_smote_enn(X_train_flat, y_train_flat)
    else:
        logger.info("\n[Step 3/5] Skipping SMOTE-ENN...")
    
    # Step 4: Train Hybrid Model
    logger.info("\n[Step 4/5] Training Hybrid IDS Model...")
    
    hybrid = HybridIDS(
        sequence_length=args.sequence_length,
        n_features=n_features,
        n_classes=2,
        model_dir=args.output_dir
    )
    
    histories = hybrid.train(
        X_train_seq=X_train_seq,
        X_train_flat=X_train_flat,
        y_train=y_train_flat,
        X_val_seq=X_test_seq,
        X_val_flat=X_test_flat,
        y_val=y_test_flat,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Step 5: Evaluate
    logger.info("\n[Step 5/5] Evaluating model...")
    metrics = hybrid.evaluate(X_test_seq, X_test_flat, y_test_flat)
    
    # Save model
    logger.info("\nSaving model...")
    hybrid.save(args.output_dir)
    
    # Save metrics
    metrics_dict = {
        'dataset': args.dataset,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'sequence_length': args.sequence_length,
        'use_pca': args.use_pca,
        'n_components': args.n_components,
        'use_smote': args.use_smote,
        'metrics': {k: {mk: float(mv) if isinstance(mv, (int, float, np.integer, np.floating)) else str(mv) 
                       for mk, mv in v.items()} 
                   for k, v in metrics.items()}
    }
    save_json(metrics_dict, os.path.join(args.output_dir, 'training_metrics.json'))
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info("="*60)


if __name__ == '__main__':
    import numpy as np
    main()
