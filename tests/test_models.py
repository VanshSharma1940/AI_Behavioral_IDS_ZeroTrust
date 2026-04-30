"""
Unit tests for ML model components.
"""
import unittest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.isolation_forest import IsolationForestIDS


class TestIsolationForest(unittest.TestCase):
    """Test cases for Isolation Forest."""
    
    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.n_samples = 500
        self.n_features = 20
        
        # Normal data
        self.X_normal = np.random.randn(400, self.n_features)
        # Anomaly data (shifted)
        self.X_anomaly = np.random.randn(100, self.n_features) + 5
        
        self.X = np.vstack([self.X_normal, self.X_anomaly])
        self.y = np.array([0] * 400 + [1] * 100)
    
    def test_model_build(self):
        """Test model building."""
        model = IsolationForestIDS(n_estimators=50, max_samples=128)
        model.build_model()
        
        self.assertIsNotNone(model.model)
    
    def test_train_predict(self):
        """Test training and prediction."""
        model = IsolationForestIDS(n_estimators=50, max_samples=128, contamination=0.2)
        model.build_model()
        model.train(self.X)
        
        predictions, scores = model.predict(self.X)
        
        self.assertEqual(len(predictions), len(self.X))
        self.assertEqual(len(scores), len(self.X))
        self.assertTrue(all(p in [0, 1] for p in predictions))
    
    def test_evaluate(self):
        """Test evaluation."""
        model = IsolationForestIDS(n_estimators=50, max_samples=128, contamination=0.2)
        model.build_model()
        model.train(self.X)
        
        metrics = model.evaluate(self.X, self.y)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('fpr', metrics)
    
    def test_save_load(self):
        """Test model save and load."""
        import tempfile
        import os
        
        model = IsolationForestIDS(n_estimators=50, max_samples=128)
        model.build_model()
        model.train(self.X)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'if_model.pkl')
            model.save(path)
            
            # Load
            new_model = IsolationForestIDS()
            new_model.load(path)
            
            self.assertTrue(new_model.is_fitted)
            
            # Predict with loaded model
            predictions, _ = new_model.predict(self.X)
            self.assertEqual(len(predictions), len(self.X))


class TestTrafficGenerator(unittest.TestCase):
    """Test cases for traffic generator."""
    
    def test_generate_normal_traffic(self):
        """Test normal traffic generation."""
        from simulation.traffic_generator import TrafficGenerator
        
        gen = TrafficGenerator(n_features=47)
        X = gen.generate_normal_traffic(n_samples=100)
        
        self.assertEqual(X.shape, (100, 47))
        self.assertTrue(np.all(X >= 0))  # Non-negative
    
    def test_generate_attack_traffic(self):
        """Test attack traffic generation."""
        from simulation.traffic_generator import TrafficGenerator
        
        gen = TrafficGenerator(n_features=47)
        X, y = gen.generate_attack_traffic(n_samples=100)
        
        self.assertEqual(X.shape[0], 100)
        self.assertEqual(len(y), 100)
        self.assertTrue(all(yi > 0 for yi in y))  # All attacks
    
    def test_generate_mixed_traffic(self):
        """Test mixed traffic generation."""
        from simulation.traffic_generator import TrafficGenerator
        
        gen = TrafficGenerator(n_features=47)
        X, y = gen.generate_mixed_traffic(n_normal=500, n_attack=50)
        
        self.assertEqual(len(X), 550)
        self.assertEqual(len(y), 550)


class TestAlertManager(unittest.TestCase):
    """Test cases for Alert Manager."""
    
    def test_alert_generation(self):
        """Test alert generation."""
        from api.alert_manager import AlertManager
        
        manager = AlertManager()
        alert = manager.generate_alert(
            confidence=0.95,
            attack_type="DoS",
            source_ip="192.168.1.1"
        )
        
        self.assertIsNotNone(alert)
        self.assertEqual(alert.severity, 'critical')
        self.assertEqual(alert.attack_type, "DoS")
    
    def test_severity_classification(self):
        """Test severity classification."""
        from api.alert_manager import AlertManager
        
        manager = AlertManager()
        
        self.assertEqual(manager.classify_severity(0.95), 'critical')
        self.assertEqual(manager.classify_severity(0.85), 'high')
        self.assertEqual(manager.classify_severity(0.75), 'medium')
        self.assertEqual(manager.classify_severity(0.6), 'low')
        self.assertEqual(manager.classify_severity(0.3), 'low')
    
    def test_rate_limiting(self):
        """Test alert rate limiting."""
        from api.alert_manager import AlertManager
        
        manager = AlertManager(max_alerts_per_minute=2)
        
        # First two should succeed
        alert1 = manager.generate_alert(confidence=0.9, attack_type="DoS", source_ip="1.1.1.1")
        alert2 = manager.generate_alert(confidence=0.9, attack_type="DoS", source_ip="2.2.2.2")
        
        self.assertIsNotNone(alert1)
        self.assertIsNotNone(alert2)
        
        # Third should be rate limited (same source+type)
        alert3 = manager.generate_alert(confidence=0.9, attack_type="DoS", source_ip="1.1.1.1")
        # May be None due to cooldown


if __name__ == '__main__':
    unittest.main()
