"""
Zero-Trust Architecture simulator for IDS testing.
Simulates network segments, user behaviors, and access control policies.
"""
import random
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class TrustLevel(Enum):
    """Trust levels in Zero-Trust architecture."""
    FULL = "full_trust"
    PARTIAL = "partial_trust"
    MINIMAL = "minimal_trust"
    NONE = "no_trust"
    BLOCKED = "blocked"


class AccessDecision(Enum):
    """Access decisions in Zero-Trust."""
    ALLOW = "allow"
    DENY = "deny"
    CHALLENGE_MFA = "challenge_mfa"
    ISOLATE = "isolate"
    MONITOR = "monitor"


@dataclass
class Entity:
    """Represents a network entity (user, device, or service)."""
    entity_id: str
    entity_type: str  # 'user', 'device', 'service'
    trust_score: float = 1.0
    last_verified: Optional[str] = None
    location: str = "internal"
    attributes: Dict = field(default_factory=dict)


@dataclass
class NetworkSegment:
    """Represents a microsegmented network zone."""
    segment_id: str
    name: str
    sensitivity: str = "medium"  # low, medium, high, critical
    allowed_entities: List[str] = field(default_factory=list)
    policies: List[Dict] = field(default_factory=list)


@dataclass
class AccessRequest:
    """Represents an access request in Zero-Trust."""
    request_id: str
    entity: Entity
    target_segment: NetworkSegment
    timestamp: str
    context: Dict = field(default_factory=dict)


@dataclass
class AccessLog:
    """Log entry for access decisions."""
    timestamp: str
    entity_id: str
    segment_id: str
    decision: str
    trust_score: float
    anomaly_score: float
    reason: str


class ZeroTrustSimulator:
    """
    Simulates Zero-Trust Architecture for IDS evaluation.
    
    Simulates:
    - Network microsegments with different sensitivity levels
    - User/device entities with trust scores
    - Continuous verification with behavioral monitoring
    - Dynamic access control based on anomaly scores
    - Attack scenarios (compromised credentials, lateral movement)
    
    This allows testing the IDS within a realistic Zero-Trust context
    without requiring a full enterprise deployment.
    """
    
    def __init__(self, n_segments: int = 5, n_entities: int = 50, random_state: int = 42):
        """
        Initialize Zero-Trust simulator.
        
        Args:
            n_segments: Number of network segments
            n_entities: Number of entities (users + devices)
            random_state: Random seed
        """
        self.n_segments = n_segments
        self.n_entities = n_entities
        self.random_state = random_state
        
        random.seed(random_state)
        np.random.seed(random_state)
        
        # Network topology
        self.segments: Dict[str, NetworkSegment] = {}
        self.entities: Dict[str, Entity] = {}
        
        # Access logs
        self.access_logs: List[AccessLog] = []
        
        # Attack simulation state
        self.compromised_entities: set = set()
        self.attack_stage = 0
        
        # Initialize topology
        self._initialize_topology()
    
    def _initialize_topology(self):
        """Create simulated network topology."""
        logger.info("Initializing Zero-Trust network topology...")
        
        # Create segments (microzones)
        segment_configs = [
            ('seg-ext', 'External/DMZ', 'low'),
            ('seg-web', 'Web Services', 'medium'),
            ('seg-app', 'Application Tier', 'medium'),
            ('seg-db', 'Database Tier', 'high'),
            ('seg-admin', 'Admin/Control', 'critical'),
            ('seg-iot', 'IoT Devices', 'medium'),
            ('seg-dev', 'Development', 'low'),
        ]
        
        for seg_id, name, sensitivity in segment_configs[:self.n_segments]:
            self.segments[seg_id] = NetworkSegment(
                segment_id=seg_id,
                name=name,
                sensitivity=sensitivity
            )
        
        # Create entities
        entity_types = ['user'] * (self.n_entities // 2) + ['device'] * (self.n_entities // 2)
        
        for i in range(self.n_entities):
            entity_type = entity_types[i] if i < len(entity_types) else 'user'
            entity_id = f"{entity_type}-{i:03d}"
            
            self.entities[entity_id] = Entity(
                entity_id=entity_id,
                entity_type=entity_type,
                trust_score=random.uniform(0.7, 1.0),
                last_verified=datetime.now().isoformat(),
                location=random.choice(['internal', 'remote', 'partner']),
                attributes={
                    'role': random.choice(['employee', 'admin', 'contractor', 'service']),
                    'department': random.choice(['IT', 'HR', 'Finance', 'Engineering', 'Sales']),
                    'mfa_enabled': random.choice([True, True, True, False]),
                    'device_compliant': random.choice([True, True, False])
                }
            )
        
        logger.info(f"Created {len(self.segments)} segments and {len(self.entities)} entities")
    
    def evaluate_trust(self, entity: Entity, anomaly_score: float = 0.0) -> TrustLevel:
        """
        Evaluate trust level of an entity.
        
        Args:
            entity: Entity to evaluate
            anomaly_score: Anomaly score from IDS (0-1, higher = more anomalous)
            
        Returns:
            TrustLevel enum
        """
        # Base trust from entity attributes
        base_trust = entity.trust_score
        
        # Reduce trust based on anomaly score
        adjusted_trust = base_trust * (1 - anomaly_score)
        
        # MFA bonus
        if entity.attributes.get('mfa_enabled', False):
            adjusted_trust = min(1.0, adjusted_trust * 1.1)
        
        # Device compliance
        if not entity.attributes.get('device_compliant', True):
            adjusted_trust *= 0.8
        
        # Compromised entities have zero trust
        if entity.entity_id in self.compromised_entities:
            adjusted_trust = 0.0
        
        # Map to trust level
        if adjusted_trust >= 0.9:
            return TrustLevel.FULL
        elif adjusted_trust >= 0.7:
            return TrustLevel.PARTIAL
        elif adjusted_trust >= 0.5:
            return TrustLevel.MINIMAL
        elif adjusted_trust > 0:
            return TrustLevel.NONE
        else:
            return TrustLevel.BLOCKED
    
    def make_access_decision(self, entity: Entity, 
                            target_segment: NetworkSegment,
                            anomaly_score: float = 0.0) -> Tuple[AccessDecision, str]:
        """
        Make access decision based on Zero-Trust policy.
        
        Args:
            entity: Requesting entity
            target_segment: Target network segment
            anomaly_score: Anomaly score from IDS
            
        Returns:
            (decision, reason)
        """
        trust_level = self.evaluate_trust(entity, anomaly_score)
        
        # Segment sensitivity requirements
        sensitivity_requirements = {
            'low': TrustLevel.MINIMAL,
            'medium': TrustLevel.PARTIAL,
            'high': TrustLevel.FULL,
            'critical': TrustLevel.FULL
        }
        
        required_trust = sensitivity_requirements.get(target_segment.sensitivity, TrustLevel.PARTIAL)
        
        # Decision logic
        if trust_level == TrustLevel.BLOCKED:
            return AccessDecision.DENY, f"Entity blocked - trust level: {trust_level.value}"
        
        if anomaly_score > 0.8:
            return AccessDecision.ISOLATE, f"Critical anomaly detected (score: {anomaly_score:.4f})"
        
        if anomaly_score > 0.6:
            return AccessDecision.CHALLENGE_MFA, f"Suspicious activity (score: {anomaly_score:.4f})"
        
        # Check trust level against requirement
        trust_order = [TrustLevel.FULL, TrustLevel.PARTIAL, TrustLevel.MINIMAL, 
                       TrustLevel.NONE, TrustLevel.BLOCKED]
        
        entity_idx = trust_order.index(trust_level)
        required_idx = trust_order.index(required_trust)
        
        if entity_idx <= required_idx:
            if anomaly_score > 0.3:
                return AccessDecision.MONITOR, f"Access granted with monitoring (score: {anomaly_score:.4f})"
            return AccessDecision.ALLOW, f"Access granted - trust: {trust_level.value}"
        else:
            return AccessDecision.DENY, f"Insufficient trust level: {trust_level.value} < {required_trust.value}"
    
    def simulate_access_request(self, entity_id: Optional[str] = None,
                               segment_id: Optional[str] = None,
                               anomaly_score: float = 0.0) -> Dict:
        """
        Simulate a single access request.
        
        Args:
            entity_id: Entity making request (random if None)
            segment_id: Target segment (random if None)
            anomaly_score: Simulated anomaly score
            
        Returns:
            Access request result
        """
        # Select random entity and segment if not specified
        if entity_id is None:
            entity_id = random.choice(list(self.entities.keys()))
        if segment_id is None:
            segment_id = random.choice(list(self.segments.keys()))
        
        entity = self.entities[entity_id]
        segment = self.segments[segment_id]
        
        # Make access decision
        decision, reason = self.make_access_decision(entity, segment, anomaly_score)
        
        # Log the access
        log_entry = AccessLog(
            timestamp=datetime.now().isoformat(),
            entity_id=entity_id,
            segment_id=segment_id,
            decision=decision.value,
            trust_score=entity.trust_score,
            anomaly_score=anomaly_score,
            reason=reason
        )
        self.access_logs.append(log_entry)
        
        return {
            'entity_id': entity_id,
            'entity_type': entity.entity_type,
            'segment': segment.name,
            'segment_sensitivity': segment.sensitivity,
            'trust_score': entity.trust_score,
            'anomaly_score': anomaly_score,
            'decision': decision.value,
            'reason': reason
        }
    
    def simulate_attack_scenario(self, scenario: str = 'lateral_movement') -> List[Dict]:
        """
        Simulate an attack scenario in Zero-Trust environment.
        
        Scenarios:
        - lateral_movement: Compromised user moves between segments
        - privilege_escalation: Attacker gains higher privileges
        - data_exfiltration: Unusual data access patterns
        - compromised_credentials: Stolen credentials used
        
        Args:
            scenario: Attack scenario name
            
        Returns:
            List of access request results
        """
        logger.info(f"Simulating attack scenario: {scenario}")
        
        results = []
        
        if scenario == 'lateral_movement':
            # Compromise a user
            compromised = random.choice(list(self.entities.keys()))
            self.compromised_entities.add(compromised)
            
            # Try to access multiple segments
            for segment_id in self.segments.keys():
                # Increasing anomaly scores as attacker moves
                anomaly = random.uniform(0.3, 0.9)
                result = self.simulate_access_request(compromised, segment_id, anomaly)
                results.append(result)
        
        elif scenario == 'privilege_escalation':
            # Normal user tries to access critical segment
            normal_users = [eid for eid, e in self.entities.items() 
                          if e.attributes.get('role') == 'employee']
            attacker = random.choice(normal_users)
            
            # Multiple attempts with increasing anomaly
            for i in range(5):
                anomaly = 0.3 + (i * 0.15)
                result = self.simulate_access_request(
                    attacker, 'seg-admin', min(anomaly, 0.95)
                )
                results.append(result)
        
        elif scenario == 'data_exfiltration':
            # Large data access from database tier
            db_users = [eid for eid, e in self.entities.items()
                       if e.attributes.get('department') == 'IT']
            attacker = random.choice(db_users)
            
            for i in range(3):
                anomaly = random.uniform(0.5, 0.9)
                result = self.simulate_access_request(attacker, 'seg-db', anomaly)
                results.append(result)
        
        elif scenario == 'compromised_credentials':
            # Login from unusual location with high anomaly
            for eid, entity in self.entities.items():
                if entity.attributes.get('role') == 'admin':
                    anomaly = random.uniform(0.7, 0.95)
                    result = self.simulate_access_request(eid, 'seg-admin', anomaly)
                    results.append(result)
                    break
        
        # Reset compromised entities
        self.compromised_entities.clear()
        
        return results
    
    def get_access_statistics(self) -> Dict:
        """Get statistics from access logs."""
        if not self.access_logs:
            return {}
        
        total = len(self.access_logs)
        decisions = {}
        for log in self.access_logs:
            decisions[log.decision] = decisions.get(log.decision, 0) + 1
        
        avg_trust = np.mean([log.trust_score for log in self.access_logs])
        avg_anomaly = np.mean([log.anomaly_score for log in self.access_logs])
        
        return {
            'total_access_requests': total,
            'decisions': decisions,
            'average_trust_score': float(avg_trust),
            'average_anomaly_score': float(avg_anomaly),
            'deny_rate': decisions.get('deny', 0) / total if total > 0 else 0
        }
    
    def reset(self):
        """Reset simulator state."""
        self.access_logs.clear()
        self.compromised_entities.clear()
        self.attack_stage = 0
        
        # Reset entity trust scores
        for entity in self.entities.values():
            entity.trust_score = random.uniform(0.7, 1.0)
