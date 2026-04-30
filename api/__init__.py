"""
API module for IDS REST API and SIEM integration.
Provides REST endpoints for threat detection and alert management.
"""
from .alert_manager import AlertManager
from .server import create_app

__all__ = ['AlertManager', 'create_app']
