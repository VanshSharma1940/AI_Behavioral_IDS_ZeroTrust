"""
Network traffic generator for IDS testing and simulation.
Generates realistic synthetic traffic patterns with configurable attack scenarios.
"""
import random
import logging
import numpy as np
from typing import Tuple, Dict, List, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class TrafficGenerator:
    """
    Synthetic network traffic generator.
    
    Generates realistic flow-level features that mimic real network traffic
    patterns including various attack scenarios.
    
    Attack types supported:
    - DoS: High packet rate, large packet sizes
    - Probe: Many connections to different ports, short duration
    - R2L: Failed login attempts, unauthorized access patterns
    - U2R: Privilege escalation patterns
    - Fuzzers: Random payload patterns
    - Exploits: Specific vulnerability exploitation patterns
    - Reconnaissance: Scanning behavior
    """
    
    ATTACK_PROFILES = {
        'normal': {
            'packet_count': (5, 50),
            'byte_count': (500, 5000),
            'duration': (1.0, 60.0),
            'packets_per_sec': (0.5, 5.0),
            'mean_packet_length': (200, 800),
            'inter_arrival_mean': (0.1, 2.0),
            'syn_ratio': (0.1, 0.3),
            'connection_count': (1, 5),
            'failed_logins': (0, 0)
        },
        'dos': {
            'packet_count': (500, 5000),
            'byte_count': (50000, 500000),
            'duration': (0.5, 10.0),
            'packets_per_sec': (100.0, 1000.0),
            'mean_packet_length': (800, 1500),
            'inter_arrival_mean': (0.001, 0.01),
            'syn_ratio': (0.8, 1.0),
            'connection_count': (1, 2),
            'failed_logins': (0, 0)
        },
        'probe': {
            'packet_count': (10, 100),
            'byte_count': (500, 5000),
            'duration': (0.1, 5.0),
            'packets_per_sec': (5.0, 50.0),
            'mean_packet_length': (40, 200),
            'inter_arrival_mean': (0.01, 0.5),
            'syn_ratio': (0.6, 0.9),
            'connection_count': (20, 100),
            'failed_logins': (0, 0)
        },
        'r2l': {
            'packet_count': (10, 200),
            'byte_count': (1000, 20000),
            'duration': (10.0, 300.0),
            'packets_per_sec': (0.5, 2.0),
            'mean_packet_length': (100, 500),
            'inter_arrival_mean': (0.5, 5.0),
            'syn_ratio': (0.1, 0.3),
            'connection_count': (1, 3),
            'failed_logins': (3, 20)
        },
        'u2r': {
            'packet_count': (20, 300),
            'byte_count': (5000, 50000),
            'duration': (30.0, 600.0),
            'packets_per_sec': (0.2, 1.0),
            'mean_packet_length': (300, 1000),
            'inter_arrival_mean': (1.0, 10.0),
            'syn_ratio': (0.1, 0.2),
            'connection_count': (1, 2),
            'failed_logins': (0, 0)
        },
        'fuzzers': {
            'packet_count': (50, 500),
            'byte_count': (1000, 50000),
            'duration': (1.0, 60.0),
            'packets_per_sec': (1.0, 20.0),
            'mean_packet_length': (100, 1400),
            'inter_arrival_mean': (0.05, 1.0),
            'syn_ratio': (0.3, 0.6),
            'connection_count': (5, 20),
            'failed_logins': (0, 2)
        },
        'exploits': {
            'packet_count': (20, 200),
            'byte_count': (2000, 50000),
            'duration': (0.5, 30.0),
            'packets_per_sec': (2.0, 50.0),
            'mean_packet_length': (500, 1500),
            'inter_arrival_mean': (0.01, 0.5),
            'syn_ratio': (0.4, 0.8),
            'connection_count': (1, 10),
            'failed_logins': (0, 5)
        },
        'reconnaissance': {
            'packet_count': (30, 300),
            'byte_count': (1500, 15000),
            'duration': (5.0, 120.0),
            'packets_per_sec': (1.0, 10.0),
            'mean_packet_length': (50, 300),
            'inter_arrival_mean': (0.1, 1.0),
            'syn_ratio': (0.5, 0.9),
            'connection_count': (50, 500),
            'failed_logins': (0, 0)
        },
        'worms': {
            'packet_count': (100, 1000),
            'byte_count': (10000, 100000),
            'duration': (1.0, 30.0),
            'packets_per_sec': (10.0, 200.0),
            'mean_packet_length': (200, 1000),
            'inter_arrival_mean': (0.005, 0.1),
            'syn_ratio': (0.3, 0.7),
            'connection_count': (10, 100),
            'failed_logins': (0, 0)
        }
    }
    
    def __init__(self, n_features: int = 47, random_state: int = 42):
        """
        Initialize traffic generator.
        
        Args:
            n_features: Number of features to generate
            random_state: Random seed
        """
        self.n_features = n_features
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)
    
    def _generate_flow_from_profile(self, profile: Dict) -> np.ndarray:
        """
        Generate a single flow feature vector from a traffic profile.
        
        Args:
            profile: Dictionary of feature ranges
            
        Returns:
            Feature vector
        """
        features = np.zeros(self.n_features)
        
        # Map profile parameters to feature indices
        features[0] = random.uniform(*profile['packet_count'])
        features[1] = random.uniform(*profile['byte_count'])
        features[2] = random.uniform(*profile['duration'])
        features[3] = random.uniform(*profile['packets_per_sec'])
        features[4] = random.uniform(*profile['mean_packet_length'])
        features[5] = random.uniform(*profile['inter_arrival_mean'])
        features[6] = random.uniform(*profile['syn_ratio'])
        features[7] = random.uniform(*profile['connection_count'])
        features[8] = random.uniform(*profile['failed_logins'])
        
        # Add derived features
        features[9] = features[1] / max(features[0], 1)  # bytes per packet
        features[10] = features[0] / max(features[2], 0.001)  # packet rate
        features[11] = features[3] * features[4]  # throughput estimate
        
        # Add noise to remaining features
        for i in range(12, self.n_features):
            features[i] = np.random.normal(0.5, 0.1)
        
        # Clip to valid range
        features = np.clip(features, 0, None)
        
        return features
    
    def generate_normal_traffic(self, n_samples: int = 1000) -> np.ndarray:
        """
        Generate normal (benign) traffic samples.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Feature array (n_samples, n_features)
        """
        logger.debug(f"Generating {n_samples} normal traffic samples")
        
        samples = []
        for _ in range(n_samples):
            features = self._generate_flow_from_profile(self.ATTACK_PROFILES['normal'])
            samples.append(features)
        
        return np.array(samples)
    
    def generate_attack_traffic(self, n_samples: int = 100, 
                                attack_types: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate attack traffic samples.
        
        Args:
            n_samples: Number of attack samples
            attack_types: List of attack types to generate (default: all)
            
        Returns:
            (features, labels) where labels are attack type indices
        """
        if attack_types is None:
            attack_types = ['dos', 'probe', 'r2l', 'u2r', 'fuzzers', 'exploits', 'reconnaissance', 'worms']
        
        logger.debug(f"Generating {n_samples} attack samples: {attack_types}")
        
        samples = []
        labels = []
        
        n_per_type = n_samples // len(attack_types)
        remainder = n_samples % len(attack_types)
        
        for i, attack_type in enumerate(attack_types):
            profile = self.ATTACK_PROFILES.get(attack_type, self.ATTACK_PROFILES['dos'])
            count = n_per_type + (1 if i < remainder else 0)
            
            for _ in range(count):
                features = self._generate_flow_from_profile(profile)
                samples.append(features)
                labels.append(i + 1)  # Attack types start from 1 (0 = normal)
        
        return np.array(samples), np.array(labels)
    
    def generate_mixed_traffic(self, n_normal: int = 5000, 
                               n_attack: int = 500,
                               attack_types: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate mixed normal and attack traffic.
        
        Args:
            n_normal: Number of normal samples
            n_attack: Number of attack samples
            attack_types: Attack types to include
            
        Returns:
            (features, labels)
        """
        # Generate normal traffic
        X_normal = self.generate_normal_traffic(n_normal)
        y_normal = np.zeros(n_normal, dtype=int)
        
        # Generate attack traffic
        X_attack, y_attack = self.generate_attack_traffic(n_attack, attack_types)
        
        # Combine
        X = np.vstack([X_normal, X_attack])
        y = np.concatenate([y_normal, y_attack])
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        logger.info(f"Generated mixed traffic: {len(X)} samples")
        logger.info(f"  Normal: {n_normal}, Attack: {n_attack}")
        
        return X, y
    
    def generate_time_series(self, n_timesteps: int = 1000, 
                            anomaly_points: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate time series traffic data with optional anomaly points.
        
        Args:
            n_timesteps: Number of time steps
            anomaly_points: Indices where anomalies occur
            
        Returns:
            (features_sequence, labels)
        """
        features = []
        labels = []
        
        if anomaly_points is None:
            # Random anomaly points (10% anomaly rate)
            anomaly_points = set(random.sample(range(n_timesteps), n_timesteps // 10))
        else:
            anomaly_points = set(anomaly_points)
        
        for t in range(n_timesteps):
            if t in anomaly_points:
                # Generate attack traffic
                attack_type = random.choice(list(self.ATTACK_PROFILES.keys())[1:])
                profile = self.ATTACK_PROFILES[attack_type]
                feat = self._generate_flow_from_profile(profile)
                label = 1
            else:
                # Normal traffic
                feat = self._generate_flow_from_profile(self.ATTACK_PROFILES['normal'])
                label = 0
            
            features.append(feat)
            labels.append(label)
        
        return np.array(features), np.array(labels)


def create_simulated_dataset(n_normal: int = 5000, n_attack: int = 500,
                             n_features: int = 47, random_state: int = 42,
                             output_file: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a complete simulated dataset for IDS testing.
    
    Args:
        n_normal: Number of normal samples
        n_attack: Number of attack samples
        n_features: Number of features
        random_state: Random seed
        output_file: Optional path to save as .npy files
        
    Returns:
        (X, y) feature and label arrays
    """
    generator = TrafficGenerator(n_features=n_features, random_state=random_state)
    X, y = generator.generate_mixed_traffic(n_normal, n_attack)
    
    logger.info(f"Simulated dataset created: X.shape={X.shape}, y.shape={y.shape}")
    logger.info(f"  Class distribution: normal={np.sum(y==0)}, attack={np.sum(y==1)}")
    
    if output_file:
        np.save(output_file + '_X.npy', X)
        np.save(output_file + '_y.npy', y)
        logger.info(f"Dataset saved to {output_file}_X.npy and {output_file}_y.npy")
    
    return X, y
