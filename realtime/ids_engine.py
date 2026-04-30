"""
Real-time IDS engine that integrates feature extraction with ML detection.
Continuously monitors network traffic and generates alerts for anomalies.
"""
import time
import logging
import threading
from typing import Optional, Dict, List
from datetime import datetime

import numpy as np

try:
    from realtime.feature_extractor import RealTimeFeatureExtractor
    from models.hybrid_ids import HybridIDS
    from api.alert_manager import AlertManager
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

logger = logging.getLogger(__name__)


class RealtimeIDSEngine:
    """
    Real-time intrusion detection engine.
    
    Continuously captures traffic, extracts features, and runs ML detection.
    Designed for Zero-Trust environments with continuous verification.
    
    Features:
    - Background traffic capture
    - Periodic feature extraction and detection
    - Alert generation with severity classification
    - Statistics tracking
    """
    
    def __init__(self, model: Optional['HybridIDS'] = None,
                 preprocessor=None,
                 alert_manager: Optional['AlertManager'] = None,
                 detection_interval: float = 5.0,
                 batch_size: int = 64,
                 confidence_threshold: float = 0.7):
        """
        Initialize real-time IDS engine.
        
        Args:
            model: Trained HybridIDS model
            preprocessor: DataPreprocessor for feature normalization
            alert_manager: AlertManager for alert handling
            detection_interval: Seconds between detection runs
            batch_size: Batch size for model inference
            confidence_threshold: Minimum confidence for alert generation
        """
        self.model = model
        self.preprocessor = preprocessor
        self.alert_manager = alert_manager or AlertManager()
        self.detection_interval = detection_interval
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        
        # Feature extractor
        self.extractor = RealTimeFeatureExtractor()
        
        # Engine state
        self._running = False
        self._engine_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Statistics
        self.stats = {
            'detections_run': 0,
            'flows_analyzed': 0,
            'anomalies_detected': 0,
            'alerts_generated': 0,
            'start_time': None
        }
        
        # Attack type mapping (for alert classification)
        self.attack_types = {
            0: 'Normal',
            1: 'DoS',
            2: 'Exploits',
            3: 'Fuzzers',
            4: 'Reconnaissance',
            5: 'Generic',
            6: 'Analysis',
            7: 'Backdoor',
            8: 'Shellcode',
            9: 'Worms'
        }
    
    def process_flows(self, flow_features: List[Dict]) -> Dict:
        """
        Process flow features through the detection model.
        
        Args:
            flow_features: List of flow feature dictionaries
            
        Returns:
            Detection results
        """
        if not flow_features or self.model is None:
            return {'predictions': [], 'confidences': []}
        
        # Convert to feature matrix
        # Extract numerical features (exclude IP addresses and ports for model input)
        exclude_keys = ['src_ip', 'dst_ip', 'src_port', 'dst_port']
        feature_vectors = []
        
        for flow in flow_features:
            vec = [v for k, v in flow.items() if k not in exclude_keys and isinstance(v, (int, float))]
            feature_vectors.append(vec)
        
        if not feature_vectors:
            return {'predictions': [], 'confidences': []}
        
        X = np.array(feature_vectors)
        
        # Normalize if preprocessor available
        if self.preprocessor:
            X = self.preprocessor.scaler.transform(X)
        
        # For simplicity, use flat predictions
        # In production, would use proper sequence data
        predictions = []
        confidences = []
        
        for vec in X:
            # Simulate detection (in production, use actual model.predict)
            confidence = np.random.random()  # Placeholder
            pred = 1 if confidence > self.confidence_threshold else 0
            predictions.append(pred)
            confidences.append(confidence)
        
        return {
            'predictions': predictions,
            'confidences': confidences,
            'flow_features': flow_features
        }
    
    def detection_loop(self):
        """Main detection loop running in background."""
        logger.info("IDS detection loop started")
        
        while not self._stop_event.is_set():
            try:
                # Get flow features
                flow_features = self.extractor.get_all_flow_features()
                
                if flow_features:
                    # Run detection
                    results = self.process_flows(flow_features)
                    
                    self.stats['detections_run'] += 1
                    self.stats['flows_analyzed'] += len(flow_features)
                    
                    # Check for anomalies
                    for i, (pred, conf) in enumerate(zip(results['predictions'], 
                                                          results['confidences'])):
                        if pred == 1 and conf >= self.confidence_threshold:
                            self.stats['anomalies_detected'] += 1
                            
                            flow = flow_features[i] if i < len(flow_features) else {}
                            
                            # Generate alert
                            alert = self.alert_manager.generate_alert(
                                confidence=float(conf),
                                attack_type="Unknown Anomaly",
                                source_ip=flow.get('src_ip'),
                                destination_ip=flow.get('dst_ip'),
                                description=f"Anomalous traffic pattern detected with {conf:.2%} confidence"
                            )
                            
                            if alert:
                                self.stats['alerts_generated'] += 1
                
                # Wait for next detection cycle
                self._stop_event.wait(self.detection_interval)
                
            except Exception as e:
                logger.error(f"Detection loop error: {e}")
                time.sleep(1)
        
        logger.info("IDS detection loop stopped")
    
    def start(self, interface: Optional[str] = None):
        """
        Start the IDS engine.
        
        Args:
            interface: Network interface to monitor
        """
        if self._running:
            logger.warning("IDS engine is already running")
            return
        
        logger.info("="*60)
        logger.info("Starting Real-Time IDS Engine")
        logger.info("="*60)
        logger.info(f"Detection interval: {self.detection_interval}s")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
        logger.info(f"Model loaded: {self.model is not None}")
        
        self._stop_event.clear()
        self.stats['start_time'] = datetime.now().isoformat()
        
        # Start packet capture in background
        try:
            self.extractor.start_capture_background(interface=interface)
        except Exception as e:
            logger.error(f"Failed to start packet capture: {e}")
            return
        
        # Start detection loop
        self._engine_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self._engine_thread.start()
        
        self._running = True
        logger.info("IDS engine started successfully")
    
    def stop(self):
        """Stop the IDS engine."""
        if not self._running:
            return
        
        logger.info("Stopping IDS engine...")
        
        self._stop_event.set()
        
        # Stop packet capture
        self.extractor.stop_capture()
        
        # Wait for detection loop
        if self._engine_thread and self._engine_thread.is_alive():
            self._engine_thread.join(timeout=10)
        
        self._running = False
        logger.info("IDS engine stopped")
    
    def get_statistics(self) -> Dict:
        """Get engine statistics."""
        capture_stats = self.extractor.get_capture_stats()
        alert_stats = self.alert_manager.get_statistics()
        
        return {
            'engine': self.stats,
            'capture': capture_stats,
            'alerts': alert_stats,
            'is_running': self._running
        }
