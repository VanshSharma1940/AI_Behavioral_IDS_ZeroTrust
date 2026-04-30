# API Documentation

## Overview

The IDS REST API provides endpoints for real-time threat detection, alert management, and Zero-Trust verification. All endpoints use JSON format and require authentication.

## Authentication

### Bearer Token

Include the API token in the Authorization header:

```
Authorization: Bearer ids-api-token-2026
```

### Token Configuration

Default token can be changed in `config/model_config.yaml`:

```yaml
api:
  api_token: "your-secure-token"
```

## Base URL

```
http://localhost:8080/api/v1
```

## Endpoints

### Health Check

Check API service health status.

```
GET /health
```

**Response (200 OK)**:
```json
{
  "status": "healthy",
  "uptime_seconds": 3600,
  "model_loaded": true,
  "timestamp": "2026-04-29T10:00:00",
  "version": "1.0.0"
}
```

### System Status

Get detailed system status including alert statistics.

```
GET /status
Authorization: Bearer <token>
```

**Response (200 OK)**:
```json
{
  "system": {
    "status": "operational",
    "uptime_seconds": 3600,
    "timestamp": "2026-04-29T10:00:00"
  },
  "model": {
    "loaded": true,
    "type": "Hybrid IDS (LSTM + Isolation Forest + DNN)",
    "accuracy": "99.66%",
    "false_positive_rate": "0.15%"
  },
  "alerts": {
    "total_alerts": 150,
    "alerts_by_severity": {
      "critical": 5,
      "high": 20,
      "medium": 45,
      "low": 80
    }
  }
}
```

### Run Detection

Run intrusion detection on provided feature vectors.

```
POST /detect
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body**:
```json
{
  "features": [
    [0.5, 0.3, 0.8, 0.1, 0.9, ...],
    [0.2, 0.1, 0.4, 0.7, 0.3, ...]
  ],
  "source_ip": "192.168.1.100",
  "metadata": {
    "capture_time": "2026-04-29T10:00:00",
    "interface": "eth0"
  }
}
```

**Response (200 OK)**:
```json
{
  "detection_id": "DET-20260429100000",
  "timestamp": "2026-04-29T10:00:00",
  "source_ip": "192.168.1.100",
  "predictions": [
    {
      "is_anomaly": true,
      "confidence": 0.92,
      "attack_type": "DoS",
      "severity": "critical"
    }
  ],
  "summary": {
    "total_checked": 2,
    "anomalies_detected": 1,
    "highest_confidence": 0.92
  }
}
```

### Sequence Detection

Run detection on sequential data (for LSTM model).

```
POST /detect/sequence
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body**:
```json
{
  "sequences": [
    [
      [0.1, 0.2, 0.3, ...],
      [0.2, 0.3, 0.4, ...],
      ...
    ]
  ],
  "source_ip": "192.168.1.100"
}
```

**Response (200 OK)**:
```json
{
  "detection_id": "DET-SEQ-20260429100000",
  "timestamp": "2026-04-29T10:00:00",
  "sequences_analyzed": 1,
  "predictions": [
    {
      "is_anomaly": false,
      "reconstruction_error": 0.05,
      "threshold": 0.07
    }
  ]
}
```

### Get Alerts

Retrieve recent alerts with optional filtering.

```
GET /alerts?severity=high&limit=50
Authorization: Bearer <token>
```

**Query Parameters**:
- `severity` (optional): Filter by severity (critical, high, medium, low)
- `limit` (optional): Maximum number of alerts (default: 100)

**Response (200 OK)**:
```json
{
  "alerts": [
    {
      "alert_id": "IDS-20260429-000001",
      "timestamp": "2026-04-29T10:00:00",
      "severity": "critical",
      "confidence": 0.95,
      "attack_type": "DoS",
      "source_ip": "10.0.0.1",
      "destination_ip": "192.168.1.50",
      "description": "DoS attack detected with 95% confidence",
      "recommended_action": "Immediate containment recommended: block source IP"
    }
  ],
  "count": 1,
  "timestamp": "2026-04-29T10:00:00"
}
```

### Get Alert Statistics

Get aggregated alert statistics.

```
GET /alerts/statistics
Authorization: Bearer <token>
```

**Response (200 OK)**:
```json
{
  "total_alerts": 150,
  "alerts_by_severity": {
    "critical": 5,
    "high": 20,
    "medium": 45,
    "low": 80
  },
  "alerts_by_type": {
    "DoS": 30,
    "Exploits": 25,
    "Reconnaissance": 20
  },
  "dropped_due_rate_limit": 10
}
```

### Get Specific Alert

```
GET /alerts/IDS-20260429-000001
Authorization: Bearer <token>
```

**Response (200 OK)**:
```json
{
  "alert_id": "IDS-20260429-000001",
  "timestamp": "2026-04-29T10:00:00",
  "severity": "critical",
  "confidence": 0.95,
  "attack_type": "DoS",
  "source_ip": "10.0.0.1",
  "description": "DoS attack detected",
  "recommended_action": "Block source IP",
  "raw_scores": {
    "lstm_score": 0.92,
    "if_score": 0.88,
    "dnn_score": 0.95
  }
}
```

### Clear Alerts

```
DELETE /alerts
Authorization: Bearer <token>
```

**Response (200 OK)**:
```json
{
  "message": "All alerts cleared"
}
```

### Load Model

Load a trained model from disk.

```
POST /model/load
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body**:
```json
{
  "model_dir": "outputs/models"
}
```

**Response (200 OK)**:
```json
{
  "message": "Model loaded successfully",
  "model_dir": "outputs/models",
  "timestamp": "2026-04-29T10:00:00"
}
```

### Get Model Info

```
GET /model/info
Authorization: Bearer <token>
```

**Response (200 OK)**:
```json
{
  "type": "Hybrid IDS",
  "components": ["LSTM Autoencoder", "Isolation Forest", "DNN Classifier"],
  "ensemble_weights": {
    "lstm": 0.4,
    "isolation_forest": 0.3,
    "dnn": 0.3
  },
  "accuracy": "99.66%",
  "false_positive_rate": "0.15%",
  "datasets": ["UNSW-NB15", "CICIDS2017"]
}
```

### Zero-Trust Verification

Verify traffic against Zero-Trust policies.

```
POST /zerotrust/verify
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body**:
```json
{
  "flow_features": {
    "packet_rate": 100.5,
    "byte_count": 50000,
    "duration": 5.0,
    "protocol": "TCP"
  },
  "user_identity": "user@example.com",
  "device_id": "device-123",
  "source_segment": "seg-web",
  "target_segment": "seg-db"
}
```

**Response (200 OK)**:
```json
{
  "verification_id": "VT-20260429100000",
  "timestamp": "2026-04-29T10:00:00",
  "trust_score": 0.85,
  "verdict": "allow",
  "reason": "Traffic within normal behavioral baseline",
  "anomaly_score": 0.15,
  "recommended_action": "Continue monitoring"
}
```

## Error Responses

### 400 Bad Request
```json
{
  "error": "Missing required field: features"
}
```

### 401 Unauthorized
```json
{
  "error": "Unauthorized",
  "message": "Invalid or missing API token"
}
```

### 404 Not Found
```json
{
  "error": "Not found",
  "message": "The requested resource does not exist"
}
```

### 500 Internal Server Error
```json
{
  "error": "Internal server error",
  "message": "Detailed error message"
}
```

## Rate Limiting

- Maximum 100 alerts per minute
- 60-second cooldown for repeated alerts from same source
- Excess alerts are logged but not forwarded

## Alert Formats

The API supports multiple alert output formats:

### JSON (default)
```json
{
  "alert_id": "IDS-20260429-000001",
  "severity": "critical",
  "timestamp": "2026-04-29T10:00:00"
}
```

### Syslog
```
<CRITICAL> 2026-04-29T10:00:00 IDS: Alert IDS-20260429-000001: DoS detected (confidence: 0.9500)
```

### CEF (Common Event Format)
```
CEF:0|AI_IDS|ZeroTrustIDS|1.0|IDS-20260429-000001|DoS|10|confidence=0.9500 src=10.0.0.1
```

## SIEM Integration

Configure SIEM webhook in `config/model_config.yaml`:

```yaml
api:
  siem_webhook_url: "https://your-siem.com/webhook"
  siem_api_key: "your-api-key"
  alert_format: "cef"
```

## Python Client Example

```python
import requests

BASE_URL = "http://localhost:8080/api/v1"
HEADERS = {
    "Authorization": "Bearer ids-api-token-2026",
    "Content-Type": "application/json"
}

# Health check
response = requests.get(f"{BASE_URL}/health", headers=HEADERS)
print(response.json())

# Run detection
features = {"features": [[0.5, 0.3, 0.8, 0.1, 0.9]]}
response = requests.post(f"{BASE_URL}/detect", json=features, headers=HEADERS)
print(response.json())

# Get alerts
response = requests.get(f"{BASE_URL}/alerts?severity=critical&limit=10", headers=HEADERS)
print(response.json())
```

## Postman Collection

A Postman collection is available at `docs/postman_collection.json` for testing all endpoints.

## Changelog

### v1.0.0 (2026-04-29)
- Initial API release
- Detection, alert management, and Zero-Trust endpoints
- Bearer token authentication
- CEF/Syslog/JSON alert formats
