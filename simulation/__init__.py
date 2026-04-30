"""
Simulation module for testing IDS without live network data.
Provides traffic generation and Zero-Trust environment simulation.
"""
from .traffic_generator import TrafficGenerator, create_simulated_dataset
from .zero_trust_simulator import ZeroTrustSimulator

__all__ = ['TrafficGenerator', 'create_simulated_dataset', 'ZeroTrustSimulator']
