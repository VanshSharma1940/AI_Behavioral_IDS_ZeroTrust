#!/usr/bin/env python3
"""
Run the IDS system in real-time or simulation mode.
Usage: python scripts/run_ids.py --mode simulation
"""
import os
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import config
from models.hybrid_ids import HybridIDS
from realtime.ids_engine import RealtimeIDSEngine
from simulation.traffic_generator import create_simulated_dataset
from simulation.zero_trust_simulator import ZeroTrustSimulator
from api.alert_manager import AlertManager
from utils.logger import setup_logger

logger = setup_logger("IDS-Runtime", log_file="outputs/logs/runtime.log")


def parse_args():
    parser = argparse.ArgumentParser(description='Run IDS System')
    parser.add_argument('--mode', type=str, default='simulation', 
                       choices=['realtime', 'simulation', 'zerotrust'],
                       help='Run mode')
    parser.add_argument('--model-dir', type=str, default='outputs/models', help='Model directory')
    parser.add_argument('--interface', type=str, help='Network interface for capture')
    parser.add_argument('--duration', type=int, default=60, help='Duration in seconds')
    return parser.parse_args()


def run_realtime(args):
    """Run IDS in real-time mode."""
    logger.info("Starting real-time IDS...")
    
    hybrid = HybridIDS(model_dir=args.model_dir)
    if os.path.exists(args.model_dir):
        hybrid.load(args.model_dir)
    
    alert_manager = AlertManager()
    engine = RealtimeIDSEngine(model=hybrid, alert_manager=alert_manager)
    
    try:
        engine.start(interface=args.interface)
        logger.info(f"Running for {args.duration} seconds...")
        time.sleep(args.duration)
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        engine.stop()
    
    stats = engine.get_statistics()
    logger.info(f"Statistics: {stats}")


def run_simulation(args):
    """Run IDS in simulation mode."""
    logger.info("Starting IDS simulation...")
    
    # Generate synthetic traffic
    X, y = create_simulated_dataset(n_normal=5000, n_attack=500, n_features=47)
    
    logger.info(f"Generated dataset: {X.shape[0]} samples")
    logger.info(f"  Normal: {sum(y==0)}, Attack: {sum(y==1)}")
    
    # Load model if available
    hybrid = HybridIDS(model_dir=args.model_dir)
    if os.path.exists(args.model_dir) and any(os.listdir(args.model_dir)):
        hybrid.load(args.model_dir)
        logger.info("Model loaded")
    else:
        logger.info("No trained model found - running in simulation mode without detection")
    
    # Simulate detection
    alert_manager = AlertManager()
    
    logger.info("Simulating detection on synthetic traffic...")
    for i in range(min(100, len(X))):
        # Simulate detection
        is_anomaly = y[i] == 1
        confidence = 0.9 if is_anomaly else 0.1
        
        if is_anomaly and confidence > 0.7:
            alert = alert_manager.generate_alert(
                confidence=confidence,
                attack_type="Simulated Attack",
                description=f"Simulated anomaly detected in flow {i}"
            )
    
    stats = alert_manager.get_statistics()
    logger.info(f"Alert statistics: {stats}")


def run_zerotrust(args):
    """Run Zero-Trust simulation."""
    logger.info("Starting Zero-Trust simulation...")
    
    simulator = ZeroTrustSimulator(n_segments=5, n_entities=50)
    
    # Simulate normal traffic
    logger.info("Simulating normal access patterns...")
    for _ in range(50):
        simulator.simulate_access_request(anomaly_score=random.uniform(0, 0.2))
    
    # Simulate attack scenarios
    logger.info("Simulating attack scenarios...")
    for scenario in ['lateral_movement', 'privilege_escalation', 'data_exfiltration']:
        results = simulator.simulate_attack_scenario(scenario)
        denied = sum(1 for r in results if r['decision'] == 'deny')
        logger.info(f"  {scenario}: {len(results)} requests, {denied} denied")
    
    stats = simulator.get_access_statistics()
    logger.info(f"Zero-Trust statistics: {stats}")


def main():
    args = parse_args()
    
    logger.info("="*60)
    logger.info(f"IDS System - Mode: {args.mode.upper()}")
    logger.info("="*60)
    
    if args.mode == 'realtime':
        run_realtime(args)
    elif args.mode == 'simulation':
        run_simulation(args)
    elif args.mode == 'zerotrust':
        import random
        run_zerotrust(args)
    
    logger.info("IDS run complete!")


if __name__ == '__main__':
    main()
