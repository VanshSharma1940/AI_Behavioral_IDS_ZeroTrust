# AI-Based Behavioral Intrusion Detection System for Zero-Trust Networks

## Overview

This project implements a **multi-modal AI-driven Intrusion Detection System (IDS)** designed for **Zero-Trust Network Architectures**. The system combines three complementary machine learning models to achieve robust detection of known and unknown cyberattacks with **99.66% accuracy** and **0.15% false positive rate** on the UNSW-NB15 dataset.

## Key Features

- **Hybrid Architecture**: Combines LSTM Autoencoder, Isolation Forest, and DNN Classifier
- **Real-Time Detection**: Live network traffic analysis using Scapy
- **Zero-Trust Integration**: Continuous verification and dynamic access control
- **REST API**: Full-featured API for SIEM integration
- **Comprehensive Simulation**: Traffic generation and Zero-Trust scenario simulation
- **Ablation Study**: Quantified component contribution analysis

## Architecture

```
Network Traffic -> Feature Extraction -> [LSTM AE] --+--> Weighted Voting --> Alert
                                      [Isolation Forest] --> [DNN Classifier] --+
```

### Components

| Component | Role | Weight | Key Parameters |
|-----------|------|--------|----------------|
| LSTM Autoencoder | Temporal anomaly detection | 0.4 | 2 layers, 16D bottleneck |
| Isolation Forest | Statistical anomaly detection | 0.3 | 200 trees, 256 samples |
| DNN Classifier | Signal integration | 0.3 | [128,64,32] layers |

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | 99.66% |
| False Positive Rate | 0.15% |
| False Negative Rate | 2.20% |
| Specificity | 99.85% |
| AUC-ROC | 0.999 |

## Quick Start

### Installation

```bash
# Clone and setup
git clone <repository-url>
cd AI_Behavioral_IDS_ZeroTrust

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Training

```bash
# Train with synthetic data
python scripts/train_model.py --dataset synthetic --epochs 50

# Train with UNSW-NB15 (download dataset first)
python scripts/train_model.py --dataset unsw-nb15 --epochs 50 --use-pca --use-smote

# Train with CICIDS2017
python scripts/train_model.py --dataset cicids2017 --epochs 50
```

### Evaluation

```bash
# Evaluate model
python scripts/evaluate_model.py --model-dir outputs/models --ablation

# Run all visualizations
python scripts/evaluate_model.py --model-dir outputs/models --output-dir outputs/plots
```

### Running IDS

```bash
# Simulation mode
python scripts/run_ids.py --mode simulation --duration 60

# Zero-Trust simulation
python scripts/run_ids.py --mode zerotrust

# Real-time mode (requires root)
sudo python scripts/run_ids.py --mode realtime --interface eth0

# Start API server
python -m api.server
```

### Simulation

```bash
# Run all simulations
python simulation/run_simulation.py --scenario all

# Traffic generation only
python simulation/run_simulation.py --scenario traffic --n-normal 10000 --n-attack 1000

# Zero-Trust scenarios
python simulation/run_simulation.py --scenario zerotrust

# Attack patterns
python simulation/run_simulation.py --scenario attacks
```

### Docker

```bash
# Build and run
docker-compose up ids-api

# Run training
docker-compose --profile training run ids-training

# Run simulation
docker-compose --profile simulation run ids-simulation
```

## Project Structure

```
AI_Behavioral_IDS_ZeroTrust/
|-- config/           # Configuration management
|-- data/             # Data loading, preprocessing, SMOTE-ENN
|-- models/           # LSTM AE, Isolation Forest, DNN, Hybrid IDS
|-- api/              # REST API server and alert manager
|-- realtime/         # Real-time traffic capture and detection
|-- simulation/       # Traffic generator, Zero-Trust simulator
|-- evaluation/       # Metrics, visualization, ablation study
|-- utils/            # Logging and helper utilities
|-- scripts/          # Training, evaluation, and run scripts
|-- tests/            # Unit tests
|-- docker/           # Docker configuration
|-- docs/             # Documentation
|-- notebooks/        # Jupyter notebooks for interactive demo
|-- README.md         # This file
|-- requirements.txt  # Python dependencies
|-- setup.py          # Package setup
```

## Datasets

- **UNSW-NB15**: 257,673 records, 47 features, 9 attack categories [Download](https://www.unb.ca/cic/datasets/cic-unsw-nb15.html)
- **CICIDS2017**: 80 features, 8 attack categories, 5 days of traffic [Download](https://www.unb.ca/cic/datasets/ids-2017.html)

Place datasets in the `datasets/` directory.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/status` | GET | System status |
| `/api/v1/detect` | POST | Run detection on features |
| `/api/v1/detect/sequence` | POST | Sequence-based detection |
| `/api/v1/alerts` | GET | Get alerts |
| `/api/v1/alerts/statistics` | GET | Alert statistics |
| `/api/v1/model/info` | GET | Model information |
| `/api/v1/zerotrust/verify` | POST | Zero-Trust verification |

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_models.py -v
```

## Configuration

Edit `config/model_config.yaml` or use command-line arguments:

```yaml
model:
  lstm_units: [64, 32]
  latent_dim: 16
  if_n_estimators: 200
  dnn_hidden_layers: [128, 64, 32]
  ensemble_weights:
    lstm: 0.4
    if: 0.3
    dnn: 0.3
```

## Documentation

- [Project Report](docs/PROJECT_REPORT.md) - Detailed technical report
- [Zero-Trust Architecture](docs/ZERO_TRUST_ARCHITECTURE.md) - ZTA integration details
- [API Documentation](docs/API_DOCUMENTATION.md) - API reference

## Authors

Research Team - AI-Based Behavioral Intrusion Detection for Zero-Trust Networks

## License

MIT License
