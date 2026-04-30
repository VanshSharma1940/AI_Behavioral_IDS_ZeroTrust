"""
Central configuration management for the IDS system.
All hyperparameters and paths are defined here for easy tuning.
"""
import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    """Dataset and preprocessing configuration."""
    # Paths
    data_dir: str = "datasets"
    unsw_train_path: str = "datasets/UNSW_NB15_training.csv"
    unsw_test_path: str = "datasets/UNSW_NB15_testing.csv"
    cicids_dir: str = "datasets/CICIDS2017"
    output_dir: str = "outputs"
    model_dir: str = "outputs/models"
    log_dir: str = "outputs/logs"
    plot_dir: str = "outputs/plots"
    
    # Preprocessing
    test_size: float = 0.2
    random_state: int = 42
    sequence_length: int = 100
    
    # Normalization
    normalization_method: str = "minmax"  # minmax, standard
    
    # PCA
    use_pca: bool = True
    n_components: int = 10  # 22.22% reduction for UNSW-NB15
    
    # SMOTE-ENN
    use_smote_enn: bool = True
    smote_sampling_strategy: str = "auto"
    
    # Features
    categorical_features: List[str] = field(default_factory=lambda: [
        'proto', 'service', 'state'
    ])
    target_column: str = "label"
    attack_category_column: str = "attack_cat"


@dataclass
class ModelConfig:
    """Model architecture and training hyperparameters."""
    # LSTM Autoencoder
    lstm_units: List[int] = field(default_factory=lambda: [64, 32])
    latent_dim: int = 16
    lstm_dropout: float = 0.2
    lstm_learning_rate: float = 0.001
    lstm_batch_size: int = 64
    lstm_epochs: int = 50
    lstm_sequence_length: int = 100
    lstm_threshold_method: str = "percentile"  # percentile, statistical, max
    lstm_threshold_percentile: float = 95
    lstm_threshold_value: float = 0.07
    
    # Isolation Forest
    if_n_estimators: int = 200
    if_max_samples: int = 256
    if_contamination: float = 0.1
    if_max_features: float = 1.0
    if_bootstrap: bool = False
    if_n_jobs: int = -1
    
    # DNN Classifier
    dnn_hidden_layers: List[int] = field(default_factory=lambda: [128, 64, 32])
    dnn_dropout_rates: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.2])
    dnn_learning_rate: float = 0.001
    dnn_batch_size: int = 64
    dnn_epochs: int = 50
    dnn_activation: str = "relu"
    dnn_output_activation: str = "softmax"
    
    # Hybrid Ensemble Weights
    lstm_weight: float = 0.4
    if_weight: float = 0.3
    dnn_weight: float = 0.3
    ensemble_threshold: float = 0.5
    
    # Training
    early_stopping_patience: int = 10
    early_stopping_restore_best: bool = True
    model_checkpoint_dir: str = "outputs/models/checkpoints"
    
    # Alert threshold
    alert_confidence_threshold: float = 0.7


@dataclass
class APIConfig:
    """REST API and SIEM integration configuration."""
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    
    # Alert formats
    alert_format: str = "json"  # json, syslog, cef
    siem_webhook_url: Optional[str] = None
    siem_api_key: Optional[str] = None
    
    # Rate limiting
    max_alerts_per_minute: int = 100
    alert_cooldown_seconds: int = 60
    
    # Authentication
    api_auth_enabled: bool = True
    api_token: str = "ids-api-token-2026"


@dataclass
class RealtimeConfig:
    """Real-time traffic analysis configuration."""
    # Packet capture
    capture_interface: Optional[str] = None  # None = default
    capture_filter: str = "ip"
    capture_timeout: int = 60
    
    # Feature extraction window
    window_size: int = 100
    flow_timeout: float = 60.0  # seconds
    
    # Detection
    detection_interval: float = 5.0  # seconds
    batch_size: int = 64
    
    # Performance
    max_flows_tracked: int = 10000
    cleanup_interval: float = 30.0  # seconds


@dataclass
class Config:
    """Master configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    api: APIConfig = field(default_factory=APIConfig)
    realtime: RealtimeConfig = field(default_factory=RealtimeConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(
            data=DataConfig(**config_dict.get('data', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            api=APIConfig(**config_dict.get('api', {})),
            realtime=RealtimeConfig(**config_dict.get('realtime', {}))
        )
    
    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'api': self.api.__dict__,
            'realtime': self.realtime.__dict__
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def ensure_directories(self):
        """Create all necessary output directories."""
        dirs = [
            self.data.output_dir,
            self.data.model_dir,
            self.data.log_dir,
            self.data.plot_dir,
            self.model.model_checkpoint_dir,
            self.data.data_dir
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)


# Global configuration instance
config = Config()
