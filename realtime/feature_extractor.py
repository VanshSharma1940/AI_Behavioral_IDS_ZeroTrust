"""
Real-time network feature extraction using Scapy.
Captures live traffic and computes flow-level statistical features.
"""
import time
import logging
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional, Callable

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    logging.warning("Scapy not available. Real-time capture will not function.")

import numpy as np

logger = logging.getLogger(__name__)


class RealTimeFeatureExtractor:
    """
    Real-time network feature extractor for IDS.
    
    Captures live packets and computes flow-level features similar to CICFlowMeter:
    - Basic features: duration, protocol, packet count, byte count
    - Time-based: inter-arrival times, active/inactive periods
    - Packet length statistics: mean, std, min, max, IQR
    - Flag counts: SYN, ACK, FIN, RST, PSH
    - Flow rates: packets/sec, bytes/sec
    
    Designed for integration with Zero-Trust continuous verification.
    """
    
    # Protocol number mapping
    PROTOCOL_MAP = {6: 'TCP', 17: 'UDP', 1: 'ICMP', 2: 'IGMP'}
    
    def __init__(self, window_size: int = 100, flow_timeout: float = 60.0):
        """
        Initialize feature extractor.
        
        Args:
            window_size: Number of packets to buffer
            flow_timeout: Flow expiration timeout in seconds
        """
        if not SCAPY_AVAILABLE:
            raise ImportError("Scapy is required for real-time feature extraction")
        
        self.window_size = window_size
        self.flow_timeout = flow_timeout
        
        # Packet buffer for recent packets
        self.packet_buffer: deque = deque(maxlen=window_size)
        
        # Flow statistics: flow_key -> flow_stats
        self.flow_stats: Dict = defaultdict(lambda: {
            'packet_count': 0,
            'byte_count': 0,
            'start_time': None,
            'last_time': None,
            'packet_lengths': [],
            'inter_arrival_times': [],
            'protocol': None,
            'src_ip': None,
            'dst_ip': None,
            'src_port': None,
            'dst_port': None,
            'tcp_flags': {'SYN': 0, 'ACK': 0, 'FIN': 0, 'RST': 0, 'PSH': 0, 'URG': 0},
            'forward_packets': 0,
            'backward_packets': 0,
            'forward_bytes': 0,
            'backward_bytes': 0
        })
        
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # Callback for new features
        self.feature_callback: Optional[Callable] = None
    
    def extract_packet_features(self, packet) -> Optional[Dict]:
        """
        Extract features from a single packet.
        
        Args:
            packet: Scapy packet object
            
        Returns:
            Dictionary of packet features or None if not IP
        """
        if IP not in packet:
            return None
        
        timestamp = time.time()
        features = {
            'timestamp': timestamp,
            'src_ip': packet[IP].src,
            'dst_ip': packet[IP].dst,
            'protocol': packet[IP].proto,
            'ip_length': len(packet[IP]),
            'ttl': packet[IP].ttl,
            'header_length': packet[IP].ihl * 4
        }
        
        # TCP features
        if TCP in packet:
            tcp = packet[TCP]
            features.update({
                'src_port': tcp.sport,
                'dst_port': tcp.dport,
                'tcp_flags': int(tcp.flags),
                'tcp_window': tcp.window,
                'payload_length': len(bytes(tcp.payload)),
                'syn': 1 if tcp.flags.S else 0,
                'ack': 1 if tcp.flags.A else 0,
                'fin': 1 if tcp.flags.F else 0,
                'rst': 1 if tcp.flags.R else 0,
                'psh': 1 if tcp.flags.P else 0
            })
        
        # UDP features
        elif UDP in packet:
            udp = packet[UDP]
            features.update({
                'src_port': udp.sport,
                'dst_port': udp.dport,
                'payload_length': len(bytes(udp.payload)),
                'syn': 0, 'ack': 0, 'fin': 0, 'rst': 0, 'psh': 0
            })
        
        # ICMP features
        elif ICMP in packet:
            features.update({
                'src_port': 0,
                'dst_port': 0,
                'payload_length': len(bytes(packet[ICMP].payload)),
                'icmp_type': packet[ICMP].type,
                'icmp_code': packet[ICMP].code,
                'syn': 0, 'ack': 0, 'fin': 0, 'rst': 0, 'psh': 0
            })
        
        return features
    
    def update_flow_stats(self, features: Dict):
        """
        Update flow statistics with new packet features.
        
        Args:
            features: Extracted packet features
        """
        if 'src_ip' not in features:
            return
        
        # Create flow key (5-tuple)
        flow_key = (
            features['src_ip'],
            features['dst_ip'],
            features.get('src_port', 0),
            features.get('dst_port', 0),
            features['protocol']
        )
        
        stats = self.flow_stats[flow_key]
        
        # Initialize flow
        if stats['start_time'] is None:
            stats['start_time'] = features['timestamp']
            stats['protocol'] = features['protocol']
            stats['src_ip'] = features['src_ip']
            stats['dst_ip'] = features['dst_ip']
            stats['src_port'] = features.get('src_port', 0)
            stats['dst_port'] = features.get('dst_port', 0)
        
        # Update packet count and bytes
        stats['packet_count'] += 1
        packet_len = features.get('ip_length', 0)
        stats['byte_count'] += packet_len
        stats['packet_lengths'].append(packet_len)
        
        # Forward/backward direction (first seen direction is forward)
        if features['src_ip'] == flow_key[0]:
            stats['forward_packets'] += 1
            stats['forward_bytes'] += packet_len
        else:
            stats['backward_packets'] += 1
            stats['backward_bytes'] += packet_len
        
        # Inter-arrival time
        if stats['last_time'] is not None:
            iat = features['timestamp'] - stats['last_time']
            stats['inter_arrival_times'].append(iat)
        
        stats['last_time'] = features['timestamp']
        
        # TCP flags
        if features.get('syn'):
            stats['tcp_flags']['SYN'] += 1
        if features.get('ack'):
            stats['tcp_flags']['ACK'] += 1
        if features.get('fin'):
            stats['tcp_flags']['FIN'] += 1
        if features.get('rst'):
            stats['tcp_flags']['RST'] += 1
        if features.get('psh'):
            stats['tcp_flags']['PSH'] += 1
    
    def compute_flow_features(self, flow_key: tuple) -> Optional[Dict]:
        """
        Compute aggregated features for a flow.
        
        Produces features similar to CICFlowMeter output.
        
        Args:
            flow_key: 5-tuple flow identifier
            
        Returns:
            Dictionary of computed flow features
        """
        stats = self.flow_stats[flow_key]
        
        if stats['packet_count'] == 0:
            return None
        
        # Duration
        duration = stats['last_time'] - stats['start_time'] if stats['last_time'] else 0
        duration = max(duration, 1e-6)  # Avoid division by zero
        
        # Packet length statistics
        lengths = stats['packet_lengths']
        
        # IAT statistics
        iats = stats['inter_arrival_times']
        
        features = {
            # Basic features
            'flow_duration': duration,
            'total_packets': stats['packet_count'],
            'total_bytes': stats['byte_count'],
            'protocol': stats['protocol'],
            
            # Rate features
            'packets_per_second': stats['packet_count'] / duration,
            'bytes_per_second': stats['byte_count'] / duration,
            
            # Packet length statistics
            'mean_packet_length': np.mean(lengths),
            'std_packet_length': np.std(lengths),
            'min_packet_length': min(lengths),
            'max_packet_length': max(lengths),
            
            # IAT statistics
            'mean_iat': np.mean(iats) if iats else 0,
            'std_iat': np.std(iats) if len(iats) > 1 else 0,
            
            # Direction statistics
            'fwd_packets': stats['forward_packets'],
            'bwd_packets': stats['backward_packets'],
            'fwd_bytes': stats['forward_bytes'],
            'bwd_bytes': stats['backward_bytes'],
            'fwd_bwd_ratio': stats['forward_packets'] / max(stats['backward_packets'], 1),
            
            # TCP flags
            'syn_count': stats['tcp_flags']['SYN'],
            'ack_count': stats['tcp_flags']['ACK'],
            'fin_count': stats['tcp_flags']['FIN'],
            'rst_count': stats['tcp_flags']['RST'],
            'psh_count': stats['tcp_flags']['PSH'],
            'flag_sum': sum(stats['tcp_flags'].values()),
            
            # Flow identifiers
            'src_ip': stats['src_ip'],
            'dst_ip': stats['dst_ip'],
            'src_port': stats['src_port'],
            'dst_port': stats['dst_port']
        }
        
        return features
    
    def get_all_flow_features(self) -> List[Dict]:
        """
        Get features for all active flows.
        
        Returns:
            List of flow feature dictionaries
        """
        with self._lock:
            flow_features = []
            expired_flows = []
            now = time.time()
            
            for flow_key, stats in list(self.flow_stats.items()):
                # Check flow timeout
                if stats['last_time'] and (now - stats['last_time']) > self.flow_timeout:
                    expired_flows.append(flow_key)
                    continue
                
                # Only include flows with enough packets
                if stats['packet_count'] >= 2:
                    features = self.compute_flow_features(flow_key)
                    if features:
                        flow_features.append(features)
            
            # Remove expired flows
            for key in expired_flows:
                del self.flow_stats[key]
        
        return flow_features
    
    def packet_callback(self, packet):
        """Callback function for each captured packet."""
        features = self.extract_packet_features(packet)
        if features:
            with self._lock:
                self.packet_buffer.append(features)
            self.update_flow_stats(features)
    
    def start_capture(self, interface: Optional[str] = None, 
                     filter_expr: str = "ip",
                     timeout: Optional[int] = None):
        """
        Start packet capture in blocking mode.
        
        Args:
            interface: Network interface to capture on
            filter_expr: BPF filter expression
            timeout: Capture timeout in seconds
        """
        logger.info(f"Starting packet capture on {interface or 'default interface'}")
        logger.info(f"Filter: {filter_expr}")
        
        self._stop_event.clear()
        
        try:
            sniff(
                iface=interface,
                filter=filter_expr,
                prn=self.packet_callback,
                store=0,
                timeout=timeout,
                stop_filter=lambda x: self._stop_event.is_set()
            )
        except PermissionError:
            logger.error("Permission denied. Run with sudo for packet capture.")
            raise
    
    def start_capture_background(self, interface: Optional[str] = None,
                                 filter_expr: str = "ip") -> threading.Thread:
        """
        Start packet capture in background thread.
        
        Args:
            interface: Network interface
            filter_expr: BPF filter
            
        Returns:
            Capture thread
        """
        self._capture_thread = threading.Thread(
            target=self.start_capture,
            kwargs={'interface': interface, 'filter_expr': filter_expr},
            daemon=True
        )
        self._capture_thread.start()
        logger.info("Packet capture started in background thread")
        return self._capture_thread
    
    def stop_capture(self):
        """Stop packet capture."""
        self._stop_event.set()
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=5)
        logger.info("Packet capture stopped")
    
    def get_capture_stats(self) -> Dict:
        """Get capture statistics."""
        with self._lock:
            return {
                'packets_captured': len(self.packet_buffer),
                'active_flows': len(self.flow_stats),
                'is_capturing': self._capture_thread is not None and self._capture_thread.is_alive()
            }
    
    def clear_stats(self):
        """Clear all flow statistics."""
        with self._lock:
            self.flow_stats.clear()
            self.packet_buffer.clear()
