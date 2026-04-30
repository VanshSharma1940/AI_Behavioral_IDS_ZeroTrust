"""
Real-time traffic analysis module.
Provides live network traffic capture and feature extraction using Scapy.
"""
from .feature_extractor import RealTimeFeatureExtractor
from .ids_engine import RealtimeIDSEngine

__all__ = ['RealTimeFeatureExtractor', 'RealtimeIDSEngine']
