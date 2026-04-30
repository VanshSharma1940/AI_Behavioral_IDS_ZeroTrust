#!/usr/bin/env python3
"""
Simulation runner for IDS testing.
Usage: python simulation/run_simulation.py --scenario all
"""
import os
import sys
import time
import argparse
import random
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulation.traffic_generator import TrafficGenerator, create_simulated_dataset
from simulation.zero_trust_simulator import ZeroTrustSimulator
from utils.logger import setup_logger

logger = setup_logger("IDS-Simulation", log_file="outputs/logs/simulation.log")


def parse_args():
    parser = argparse.ArgumentParser(description='Run IDS Simulation')
    parser.add_argument('--scenario', type=str, default='all',
                       choices=['all', 'traffic', 'zerotrust', 'attacks'],
                       help='Simulation scenario')
    parser.add_argument('--n-normal', type=int, default=5000, help='Normal samples')
    parser.add_argument('--n-attack', type=int, default=500, help='Attack samples')
    parser.add_argument('--duration', type=int, default=60, help='Simulation duration')
    return parser.parse_args()


def run_traffic_simulation(args):
    """Run traffic generation simulation."""
    logger.info("="*60)
    logger.info("Traffic Generation Simulation")
    logger.info("="*60)
    
    # Generate dataset
    logger.info(f"Generating {args.n_normal} normal + {args.n_attack} attack samples...")
    X, y = create_simulated_dataset(
        n_normal=args.n_normal,
        n_attack=args.n_attack,
        output_file='outputs/simulated_dataset'
    )
    
    # Analyze generated traffic
    logger.info("\nGenerated Traffic Analysis:")
    logger.info(f"  Total samples: {len(y)}")
    logger.info(f"  Normal: {np.sum(y==0)} ({np.sum(y==0)/len(y)*100:.1f}%)")
    logger.info(f"  Attack: {np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.1f}%)")
    
    # Per-attack-type statistics
    attack_types = ['DoS', 'Probe', 'R2L', 'U2R', 'Fuzzers', 'Exploits', 'Recon', 'Worms']
    logger.info("\nAttack type distribution:")
    for i, atype in enumerate(attack_types, 1):
        count = np.sum(y == i)
        if count > 0:
            logger.info(f"  {atype}: {count}")
    
    # Feature statistics
    logger.info("\nFeature Statistics:")
    logger.info(f"  Mean: {np.mean(X):.4f}")
    logger.info(f"  Std:  {np.std(X):.4f}")
    logger.info(f"  Min:  {np.min(X):.4f}")
    logger.info(f"  Max:  {np.max(X):.4f}")


def run_zerotrust_simulation(args):
    """Run Zero-Trust simulation."""
    logger.info("="*60)
    logger.info("Zero-Trust Architecture Simulation")
    logger.info("="*60)
    
    simulator = ZeroTrustSimulator(n_segments=5, n_entities=50)
    
    # Phase 1: Normal operations
    logger.info("\n[Phase 1] Normal access patterns...")
    for _ in range(50):
        result = simulator.simulate_access_request(anomaly_score=random.uniform(0, 0.2))
    
    # Phase 2: Attack scenarios
    logger.info("\n[Phase 2] Attack scenario simulations...")
    
    attack_scenarios = ['lateral_movement', 'privilege_escalation', 'data_exfiltration', 'compromised_credentials']
    
    for scenario in attack_scenarios:
        logger.info(f"\n  Scenario: {scenario}")
        results = simulator.simulate_attack_scenario(scenario)
        
        decisions = {}
        for r in results:
            d = r['decision']
            decisions[d] = decisions.get(d, 0) + 1
        
        logger.info(f"    Total requests: {len(results)}")
        for decision, count in decisions.items():
            logger.info(f"    {decision}: {count}")
    
    # Phase 3: Statistics
    logger.info("\n[Phase 3] Access Statistics")
    stats = simulator.get_access_statistics()
    for key, value in stats.items():
        logger.info(f"  {key}: {value}")


def run_attack_simulation(args):
    """Run comprehensive attack simulation."""
    logger.info("="*60)
    logger.info("Attack Pattern Simulation")
    logger.info("="*60)
    
    generator = TrafficGenerator(n_features=47)
    
    # Generate different attack types
    attack_types = ['dos', 'probe', 'r2l', 'u2r', 'fuzzers', 'exploits', 'reconnaissance', 'worms']
    
    for attack in attack_types:
        X, y = generator.generate_attack_traffic(n_samples=100, attack_types=[attack])
        
        logger.info(f"\n{attack.upper()} Attack Profile:")
        logger.info(f"  Samples: {len(X)}")
        logger.info(f"  Mean packet rate: {np.mean(X[:, 3]):.2f}")
        logger.info(f"  Mean packet size: {np.mean(X[:, 4]):.2f}")
        logger.info(f"  Mean IAT: {np.mean(X[:, 5]):.6f}")
        logger.info(f"  SYN ratio: {np.mean(X[:, 6]):.4f}")


def main():
    args = parse_args()
    
    random.seed(42)
    np.random.seed(42)
    
    os.makedirs('outputs/logs', exist_ok=True)
    os.makedirs('outputs/plots', exist_ok=True)
    
    if args.scenario in ('all', 'traffic'):
        run_traffic_simulation(args)
    
    if args.scenario in ('all', 'zerotrust'):
        run_zerotrust_simulation(args)
    
    if args.scenario in ('all', 'attacks'):
        run_attack_simulation(args)
    
    logger.info("\n" + "="*60)
    logger.info("Simulation complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
