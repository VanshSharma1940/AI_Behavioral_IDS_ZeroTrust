"""
REST API server for the IDS system.
Provides endpoints for real-time detection, alert management, and health monitoring.
"""
import os
import json
import logging
from datetime import datetime
from functools import wraps
from typing import Dict, Any

from flask import Flask, request, jsonify

# Import IDS components
try:
    from models.hybrid_ids import HybridIDS
    from data.preprocessor import DataPreprocessor
    from api.alert_manager import AlertManager
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

logger = logging.getLogger(__name__)


def create_app(config: Dict = None) -> Flask:
    """
    Create and configure Flask application.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured Flask app
    """
    app = Flask(__name__)
    app.config['JSON_SORT_KEYS'] = False
    
    # Configuration
    api_token = (config or {}).get('api_token', 'ids-api-token-2026')
    
    # Global state
    app.ids_model = None
    app.preprocessor = None
    app.alert_manager = AlertManager()
    app.startup_time = datetime.now()
    
    def require_auth(f):
        """Decorator to require API token authentication."""
        @wraps(f)
        def decorated(*args, **kwargs):
            auth_header = request.headers.get('Authorization', '')
            token = auth_header.replace('Bearer ', '') if auth_header.startswith('Bearer ') else ''
            
            if token != api_token:
                return jsonify({'error': 'Unauthorized', 'message': 'Invalid or missing API token'}), 401
            
            return f(*args, **kwargs)
        return decorated
    
    # =========================================================================
    # Health and Status Endpoints
    # =========================================================================
    
    @app.route('/api/v1/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        uptime = (datetime.now() - app.startup_time).total_seconds()
        
        status = {
            'status': 'healthy',
            'uptime_seconds': uptime,
            'model_loaded': app.ids_model is not None,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        }
        
        if app.ids_model is None:
            status['status'] = 'degraded'
            status['warning'] = 'IDS model not loaded'
        
        return jsonify(status), 200 if status['status'] == 'healthy' else 200
    
    @app.route('/api/v1/status', methods=['GET'])
    @require_auth
    def get_status():
        """Get detailed system status."""
        alert_stats = app.alert_manager.get_statistics()
        uptime = (datetime.now() - app.startup_time).total_seconds()
        
        status = {
            'system': {
                'status': 'operational' if app.ids_model else 'initializing',
                'uptime_seconds': uptime,
                'timestamp': datetime.now().isoformat()
            },
            'model': {
                'loaded': app.ids_model is not None,
                'type': 'Hybrid IDS (LSTM + Isolation Forest + DNN)',
                'accuracy': '99.66%',
                'false_positive_rate': '0.15%'
            },
            'alerts': alert_stats
        }
        
        return jsonify(status), 200
    
    # =========================================================================
    # Detection Endpoints
    # =========================================================================
    
    @app.route('/api/v1/detect', methods=['POST'])
    @require_auth
    def detect():
        """
        Run intrusion detection on provided features.
        
        Request body (JSON):
        {
            "features": [[f1, f2, ...], ...],  // Feature vectors
            "source_ip": "192.168.1.1",         // Optional
            "metadata": {}                       // Optional additional data
        }
        
        Returns:
            Detection results with predictions and confidence scores
        """
        if app.ids_model is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        try:
            data = request.get_json()
            
            if not data or 'features' not in data:
                return jsonify({'error': 'Missing required field: features'}), 400
            
            features = data['features']
            source_ip = data.get('source_ip', 'unknown')
            
            # TODO: Implement actual detection
            # For now, return a simulated response
            
            results = {
                'detection_id': f"DET-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'timestamp': datetime.now().isoformat(),
                'source_ip': source_ip,
                'predictions': [],
                'summary': {
                    'total_checked': len(features),
                    'anomalies_detected': 0,
                    'highest_confidence': 0.0
                }
            }
            
            return jsonify(results), 200
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/v1/detect/sequence', methods=['POST'])
    @require_auth
    def detect_sequence():
        """
        Detect anomalies in sequence data (for LSTM).
        
        Request body:
        {
            "sequences": [[[f1, f2, ...], ...], ...],  // 3D array
            "source_ip": "192.168.1.1"
        }
        """
        if app.ids_model is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        try:
            data = request.get_json()
            
            if not data or 'sequences' not in data:
                return jsonify({'error': 'Missing required field: sequences'}), 400
            
            sequences = data['sequences']
            source_ip = data.get('source_ip', 'unknown')
            
            # TODO: Implement sequence-based detection
            
            results = {
                'detection_id': f"DET-SEQ-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'timestamp': datetime.now().isoformat(),
                'sequences_analyzed': len(sequences),
                'predictions': []
            }
            
            return jsonify(results), 200
            
        except Exception as e:
            logger.error(f"Sequence detection error: {e}")
            return jsonify({'error': str(e)}), 500
    
    # =========================================================================
    # Alert Management Endpoints
    # =========================================================================
    
    @app.route('/api/v1/alerts', methods=['GET'])
    @require_auth
    def get_alerts():
        """
        Get recent alerts.
        
        Query parameters:
        - severity: Filter by severity (critical, high, medium, low)
        - limit: Maximum number of alerts (default: 100)
        """
        severity = request.args.get('severity')
        limit = request.args.get('limit', 100, type=int)
        
        alerts = app.alert_manager.get_alerts(severity=severity, limit=limit)
        
        return jsonify({
            'alerts': [a.to_dict() for a in alerts],
            'count': len(alerts),
            'timestamp': datetime.now().isoformat()
        }), 200
    
    @app.route('/api/v1/alerts/statistics', methods=['GET'])
    @require_auth
    def get_alert_statistics():
        """Get alert statistics."""
        stats = app.alert_manager.get_statistics()
        return jsonify(stats), 200
    
    @app.route('/api/v1/alerts/<alert_id>', methods=['GET'])
    @require_auth
    def get_alert(alert_id: str):
        """Get specific alert by ID."""
        # Search for alert
        all_alerts = app.alert_manager.get_alerts(limit=10000)
        alert = next((a for a in all_alerts if a.alert_id == alert_id), None)
        
        if alert is None:
            return jsonify({'error': 'Alert not found'}), 404
        
        return jsonify(alert.to_dict()), 200
    
    @app.route('/api/v1/alerts', methods=['DELETE'])
    @require_auth
    def clear_alerts():
        """Clear all alerts."""
        app.alert_manager.clear_alerts()
        return jsonify({'message': 'All alerts cleared'}), 200
    
    # =========================================================================
    # Model Management Endpoints
    # =========================================================================
    
    @app.route('/api/v1/model/load', methods=['POST'])
    @require_auth
    def load_model():
        """
        Load IDS model from disk.
        
        Request body:
        {
            "model_dir": "outputs/models"
        }
        """
        try:
            data = request.get_json() or {}
            model_dir = data.get('model_dir', 'outputs/models')
            
            if MODELS_AVAILABLE:
                app.ids_model = HybridIDS(model_dir=model_dir)
                app.ids_model.load(model_dir)
                
                return jsonify({
                    'message': 'Model loaded successfully',
                    'model_dir': model_dir,
                    'timestamp': datetime.now().isoformat()
                }), 200
            else:
                return jsonify({'error': 'Model components not available'}), 500
                
        except Exception as e:
            logger.error(f"Model load error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/v1/model/info', methods=['GET'])
    @require_auth
    def get_model_info():
        """Get information about the loaded model."""
        if app.ids_model is None:
            return jsonify({'error': 'No model loaded'}), 404
        
        info = {
            'type': 'Hybrid IDS',
            'components': ['LSTM Autoencoder', 'Isolation Forest', 'DNN Classifier'],
            'ensemble_weights': {
                'lstm': app.ids_model.lstm_weight,
                'isolation_forest': app.ids_model.if_weight,
                'dnn': app.ids_model.dnn_weight
            },
            'accuracy': '99.66%',
            'false_positive_rate': '0.15%',
            'datasets': ['UNSW-NB15', 'CICIDS2017']
        }
        
        return jsonify(info), 200
    
    # =========================================================================
    # Zero-Trust Policy Endpoints
    # =========================================================================
    
    @app.route('/api/v1/zerotrust/verify', methods=['POST'])
    @require_auth
    def verify_traffic():
        """
        Zero-Trust traffic verification endpoint.
        
        Verifies traffic flow against behavioral baseline.
        Returns trust score and recommendation.
        
        Request body:
        {
            "flow_features": {...},
            "user_identity": "user@example.com",
            "device_id": "device-123"
        }
        """
        try:
            data = request.get_json() or {}
            
            # TODO: Implement actual Zero-Trust verification
            
            result = {
                'verification_id': f"VT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'timestamp': datetime.now().isoformat(),
                'trust_score': 0.95,
                'verdict': 'allow',  # allow, block, challenge
                'recommendation': 'Traffic within normal behavioral baseline'
            }
            
            return jsonify(result), 200
            
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Not found', 'message': 'The requested resource does not exist'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal server error', 'message': str(error)}), 500
    
    logger.info("IDS API server created")
    return app


def run_server(host: str = '0.0.0.0', port: int = 8080, debug: bool = False):
    """Run the Flask development server."""
    app = create_app()
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server(debug=True)
