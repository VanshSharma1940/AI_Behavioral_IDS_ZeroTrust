#!/usr/bin/env python3
"""
Evaluation script for the Hybrid IDS model.
Usage: python scripts/evaluate_model.py --model-dir outputs/models
"""
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import config
from data.data_loader import DataLoader
from data.preprocessor import DataPreprocessor
from models.hybrid_ids import HybridIDS
from evaluation.visualization import plot_evaluation_results
from evaluation.ablation_study import AblationStudy
from utils.logger import setup_logger

logger = setup_logger("IDS-Evaluation", log_file="outputs/logs/evaluation.log")


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Hybrid IDS Model')
    parser.add_argument('--model-dir', type=str, default='outputs/models', help='Model directory')
    parser.add_argument('--dataset', type=str, default='synthetic', choices=['unsw-nb15', 'cicids2017', 'synthetic'])
    parser.add_argument('--ablation', action='store_true', help='Run ablation study')
    parser.add_argument('--output-dir', type=str, default='outputs/plots', help='Output directory')
    return parser.parse_args()


def main():
    args = parse_args()
    
    logger.info("="*60)
    logger.info("Hybrid IDS Evaluation")
    logger.info("="*60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    logger.info("Loading dataset...")
    loader = DataLoader()
    if args.dataset == 'synthetic':
        df = DataLoader.generate_synthetic_data(n_samples=5000, n_features=47)
    else:
        df = loader.load_unsw_nb15() if args.dataset == 'unsw-nb15' else loader.load_cicids2017()
    
    # Preprocess
    logger.info("Preprocessing...")
    preprocessor = DataPreprocessor(sequence_length=100)
    results = preprocessor.fit_transform(df)
    
    # Load model
    logger.info("Loading model...")
    hybrid = HybridIDS(model_dir=args.model_dir)
    hybrid.load(args.model_dir)
    
    # Evaluate
    logger.info("Running evaluation...")
    metrics = hybrid.evaluate(results['X_test_seq'], results['X_test_flat'], results['y_test_flat'])
    
    # Visualizations
    logger.info("Generating plots...")
    # Note: For proper visualization, need scores from predict
    predictions = hybrid.predict(results['X_test_seq'], results['X_test_flat'])
    
    # Ablation study
    if args.ablation:
        logger.info("Running ablation study...")
        ablation = AblationStudy(hybrid)
        ablation.run_ablation(results['X_test_seq'], results['X_test_flat'], results['y_test_flat'])
        ablation.print_results()
        ablation.plot_results(os.path.join(args.output_dir, 'ablation_study.png'))
    
    logger.info("Evaluation complete!")


if __name__ == '__main__':
    main()
