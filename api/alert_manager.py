"""
Alert management system for IDS.
Handles alert generation, formatting, throttling, and SIEM integration.
"""
import json
import time
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from collections import deque

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """Represents a security alert."""
    alert_id: str
    timestamp: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    confidence: float
    attack_type: str
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    description: str = ""
    recommended_action: str = ""
    raw_scores: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert alert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_syslog(self) -> str:
        """Convert alert to syslog format."""
        return (f"<{self.severity.upper()}> {self.timestamp} IDS: "
                f"Alert {self.alert_id}: {self.attack_type} detected "
                f"(confidence: {self.confidence:.4f}) - {self.description}")
    
    def to_cef(self) -> str:
        """Convert alert to CEF (Common Event Format) for SIEM."""
        severity_map = {'critical': 10, 'high': 8, 'medium': 5, 'low': 2}
        sev = severity_map.get(self.severity.lower(), 5)
        
        return (f"CEF:0|AI_IDS|ZeroTrustIDS|1.0|{self.alert_id}|{self.attack_type}|{sev}|"
                f"confidence={self.confidence:.4f} "
                f"src={self.source_ip or 'unknown'} "
                f"dst={self.destination_ip or 'unknown'} "
                f"msg={self.description}")


class AlertManager:
    """
    Manages alert lifecycle: generation, storage, throttling, and forwarding.
    
    Features:
    - Alert generation with severity classification
    - Rate limiting to prevent alert storms
    - Multiple output formats (JSON, Syslog, CEF)
    - SIEM webhook integration
    - Alert history and statistics
    """
    
    SEVERITY_THRESHOLDS = {
        'critical': 0.9,
        'high': 0.8,
        'medium': 0.7,
        'low': 0.5
    }
    
    def __init__(self, max_alerts_per_minute: int = 100,
                 cooldown_seconds: int = 60,
                 siem_webhook_url: Optional[str] = None,
                 siem_api_key: Optional[str] = None,
                 alert_format: str = "json"):
        """
        Initialize alert manager.
        
        Args:
            max_alerts_per_minute: Maximum alerts per minute
            cooldown_seconds: Cooldown period for repeated alerts
            siem_webhook_url: SIEM webhook URL for forwarding
            siem_api_key: API key for SIEM authentication
            alert_format: Output format ('json', 'syslog', 'cef')
        """
        self.max_alerts_per_minute = max_alerts_per_minute
        self.cooldown_seconds = cooldown_seconds
        self.siem_webhook_url = siem_webhook_url
        self.siem_api_key = siem_api_key
        self.alert_format = alert_format
        
        # Alert storage
        self.alerts: deque = deque(maxlen=10000)
        self.alert_count = 0
        self.recent_alerts: deque = deque(maxlen=1000)  # For rate limiting
        
        # Cooldown tracking
        self.cooldown_map: Dict[str, float] = {}
        
        # Statistics
        self.stats = {
            'total_alerts': 0,
            'alerts_by_severity': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0},
            'alerts_by_type': {},
            'dropped_due_rate_limit': 0
        }
        
        # Callbacks for alert handlers
        self.handlers: List[Callable] = []
        
        self._lock = threading.Lock()
    
    def classify_severity(self, confidence: float) -> str:
        """
        Classify alert severity based on confidence score.
        
        Args:
            confidence: Detection confidence (0-1)
            
        Returns:
            Severity level string
        """
        for severity, threshold in sorted(self.SEVERITY_THRESHOLDS.items(), 
                                          key=lambda x: x[1], reverse=True):
            if confidence >= threshold:
                return severity
        return 'low'
    
    def _check_rate_limit(self, alert_key: str) -> bool:
        """
        Check if alert should be rate limited.
        
        Args:
            alert_key: Key to identify alert type (e.g., source_ip + attack_type)
            
        Returns:
            True if alert should be processed, False if rate limited
        """
        now = time.time()
        
        # Check per-type cooldown
        if alert_key in self.cooldown_map:
            if now - self.cooldown_map[alert_key] < self.cooldown_seconds:
                return False
        
        # Check global rate limit (alerts per minute)
        minute_ago = now - 60
        recent_count = sum(1 for t in self.recent_alerts if t > minute_ago)
        if recent_count >= self.max_alerts_per_minute:
            return False
        
        self.cooldown_map[alert_key] = now
        self.recent_alerts.append(now)
        
        return True
    
    def generate_alert(self, confidence: float, attack_type: str = "Unknown",
                      source_ip: Optional[str] = None,
                      destination_ip: Optional[str] = None,
                      raw_scores: Optional[Dict] = None,
                      description: Optional[str] = None,
                      recommended_action: Optional[str] = None) -> Optional[Alert]:
        """
        Generate a new alert if not rate limited.
        
        Args:
            confidence: Detection confidence
            attack_type: Type of detected attack
            source_ip: Source IP address
            destination_ip: Destination IP address
            raw_scores: Raw model scores
            description: Alert description
            recommended_action: Recommended response action
            
        Returns:
            Alert object or None if rate limited
        """
        severity = self.classify_severity(confidence)
        alert_key = f"{source_ip or 'unknown'}:{attack_type}"
        
        # Check rate limiting
        if not self._check_rate_limit(alert_key):
            with self._lock:
                self.stats['dropped_due_rate_limit'] += 1
            return None
        
        # Generate alert ID
        self.alert_count += 1
        alert_id = f"IDS-{datetime.now().strftime('%Y%m%d')}-{self.alert_count:06d}"
        
        # Default description and action
        if description is None:
            description = f"{attack_type} attack detected with {confidence:.2%} confidence"
        
        if recommended_action is None:
            if severity in ['critical', 'high']:
                recommended_action = "Immediate containment recommended: block source IP and isolate affected segment"
            elif severity == 'medium':
                recommended_action = "Investigate and monitor: enable enhanced logging"
            else:
                recommended_action = "Monitor: standard procedure"
        
        alert = Alert(
            alert_id=alert_id,
            timestamp=datetime.now().isoformat(),
            severity=severity,
            confidence=confidence,
            attack_type=attack_type,
            source_ip=source_ip,
            destination_ip=destination_ip,
            description=description,
            recommended_action=recommended_action,
            raw_scores=raw_scores or {}
        )
        
        # Store alert
        with self._lock:
            self.alerts.append(alert)
            self.stats['total_alerts'] += 1
            self.stats['alerts_by_severity'][severity] += 1
            self.stats['alerts_by_type'][attack_type] = \
                self.stats['alerts_by_type'].get(attack_type, 0) + 1
        
        # Notify handlers
        for handler in self.handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
        
        # Forward to SIEM
        if self.siem_webhook_url:
            self._forward_to_siem(alert)
        
        logger.info(f"Alert generated: {alert_id} - {severity.upper()} - {attack_type} "
                   f"(confidence: {confidence:.4f})")
        
        return alert
    
    def _forward_to_siem(self, alert: Alert):
        """Forward alert to SIEM via webhook."""
        if not REQUESTS_AVAILABLE:
            logger.warning("requests library not available, cannot forward to SIEM")
            return
        
        try:
            headers = {'Content-Type': 'application/json'}
            if self.siem_api_key:
                headers['Authorization'] = f'Bearer {self.siem_api_key}'
            
            response = requests.post(
                self.siem_webhook_url,
                json=alert.to_dict(),
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                logger.debug(f"Alert {alert.alert_id} forwarded to SIEM")
            else:
                logger.warning(f"SIEM forwarding failed: HTTP {response.status_code}")
                
        except Exception as e:
            logger.error(f"SIEM forwarding error: {e}")
    
    def format_alert(self, alert: Alert, format_type: Optional[str] = None) -> str:
        """
        Format alert in specified format.
        
        Args:
            alert: Alert object
            format_type: 'json', 'syslog', 'cef' (default: self.alert_format)
            
        Returns:
            Formatted alert string
        """
        fmt = format_type or self.alert_format
        
        if fmt == 'json':
            return alert.to_json()
        elif fmt == 'syslog':
            return alert.to_syslog()
        elif fmt == 'cef':
            return alert.to_cef()
        else:
            return alert.to_json()
    
    def get_alerts(self, severity: Optional[str] = None,
                  limit: int = 100,
                  since: Optional[float] = None) -> List[Alert]:
        """
        Get recent alerts with optional filtering.
        
        Args:
            severity: Filter by severity
            limit: Maximum number of alerts
            since: Unix timestamp to get alerts after
            
        Returns:
            List of Alert objects
        """
        with self._lock:
            filtered = list(self.alerts)
        
        if severity:
            filtered = [a for a in filtered if a.severity == severity]
        
        if since:
            filtered = [a for a in filtered 
                       if datetime.fromisoformat(a.timestamp).timestamp() > since]
        
        return filtered[-limit:]
    
    def get_statistics(self) -> Dict:
        """Get alert statistics."""
        with self._lock:
            return self.stats.copy()
    
    def register_handler(self, handler: Callable):
        """Register a callback for new alerts."""
        self.handlers.append(handler)
    
    def clear_alerts(self):
        """Clear all stored alerts."""
        with self._lock:
            self.alerts.clear()
            self.alert_count = 0
            self.cooldown_map.clear()
            self.stats = {
                'total_alerts': 0,
                'alerts_by_severity': {'critical': 0, 'high': 0, 'medium': 0, 'low': 0},
                'alerts_by_type': {},
                'dropped_due_rate_limit': 0
            }
