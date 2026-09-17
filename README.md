# Wearable Anomaly Detection

End-to-end **Machine Learning + MLOps** system for detecting unusual patterns in wearable sensor time-series data.

The project uses an **LSTM Autoencoder** trained on baseline wearable data. Reconstruction error is used as the anomaly score, with a **"What Changed?"** explainability layer for detected anomalies.

The system is served through **FastAPI**, containerized with **Docker**, published to **GitHub Container Registry (GHCR)**, deployed through **FastAPI Cloud**, monitored with **Prometheus**, and extended with **distribution drift detection** and **drift-triggered retraining with a model evaluation gate**.

> **Important:** This is an anomaly detection research and engineering project. An anomaly represents a deviation from learned wearable behavior and is **not a medical diagnosis**.

---

## Highlights

* **41 engineered wearable features**
* **12-step temporal sequences**
* LSTM Autoencoder for unsupervised anomaly detection
* Participant-level train/validation/test split
* Leakage-safe preprocessing
* Reconstruction-based anomaly scoring
* **"What Changed?" feature-level explainability**
* FastAPI inference service
* Dockerized deployment
* GitHub Actions CI/CD
* GHCR image publishing
* Public HTTPS API
* Prometheus monitoring
* Wasserstein-based drift detection
* Drift-triggered candidate retraining
* Automated candidate evaluation gate
* Model archiving before promotion
* Automated rejection of inferior candidates

---

## Architecture

```text
Raw Wearable Data
        │
        ▼
Data Validation
        │
        ▼
Preprocessing
        │
        ▼
10-Second Feature Engineering
        │
        ▼
Protocol Labeling
        │
        ▼
Participant-Level Split
        │
        ▼
Sequence Preparation
        │
        ▼
LSTM Autoencoder
        │
        ▼
Reconstruction Error
        │
        ▼
Anomaly Detection
        │
        ▼
"What Changed?" Explainability
        │
        ▼
FastAPI
        │
        ├──────────────► Prometheus Metrics
        │
        ▼
Docker
        │
        ▼
GitHub Actions
        │
        ▼
GHCR
        │
        ▼
FastAPI Cloud
        │
        ▼
Public HTTPS API

Monitoring Loop
────────────────────────────────────
Production / Current Batch
        │
        ▼
Drift Detection
        │
        ▼
Drift Detected?
        │
       Yes
        ▼
Candidate Retraining
        │
        ▼
Evaluation Gate
        │
   ┌────┴────┐
   ▼         ▼
Promote    Reject
```

---

## Dataset

The project uses the **PhysioNet Wearable Device Dataset from Induced Stress and Structured Exercise Sessions v1.0.1**.

The dataset contains structured sessions including:

* Stress
* Aerobic exercise
* Anaerobic exercise

### Signals

* Heart Rate (`HR`)
* Blood Volume Pulse (`BVP`)
* Electrodermal Activity (`EDA`)
* Temperature (`TEMP`)
* Accelerometer (`ACC_X`, `ACC_Y`, `ACC_Z`)
* Accelerometer Magnitude (`ACC_MAG`)
* Inter-Beat Interval (`IBI`)

The sensor streams are aligned to a **1 Hz representation** before feature engineering.

---

## Data Pipeline

### 1. Validation

The validation stage checks:

* expected dataset structure
* required files
* sampling information
* timestamp consistency
* malformed IBI records
* known dataset-specific constraints

### 2. Preprocessing

The pipeline:

* reconstructs timestamps from session metadata
* processes accelerometer axes
* calculates `ACC_MAG`
* aligns sensor streams to a 1-second timeline
* handles irregular IBI observations
* combines signals into a unified time series

Primary processed output:

```text
data/processed/wearable_timeseries_1hz.csv
```

### 3. Feature Engineering

Features are computed over **10-second windows**.

Final model representation:

```text
41 features
12 timesteps
```

Sequence shape:

```text
(12, 41)
```

### 4. Protocol Labeling

Each feature window is associated with protocol context such as baseline, stress, aerobic, and anaerobic stages.

These labels are used for analysis and evaluation and are **not used as supervised anomaly labels**.

### 5. Participant-Level Split

The dataset is split by participant to reduce leakage between training and evaluation.

```text
41 participants

Train       : 28
Validation  : 6
Test        : 7
```

Participant/session variants are normalized under a shared participant identifier when applicable.

---

## Leakage-Safe Sequence Preparation

The preprocessing pipeline is designed to avoid data leakage:

* missing-value medians are learned from training data
* the scaler is fitted using training baseline windows
* validation and test data use the training preprocessing artifacts
* sequences never cross participant/session boundaries

Saved preprocessing artifact:

```text
models/preprocessor.pkl
```

---

## Model

### LSTM Autoencoder

```text
Input Sequence
      │
      ▼
Encoder LSTM
      │
      ▼
Latent Representation
      │
      ▼
Decoder LSTM
      │
      ▼
Reconstructed Sequence
```

Configuration:

| Parameter        |            Value |
| ---------------- | ---------------: |
| Architecture     | LSTM Autoencoder |
| Input dimension  |               41 |
| Hidden dimension |               64 |
| Latent dimension |               32 |
| LSTM layers      |                2 |
| Dropout          |              0.2 |

The model learns baseline wearable behavior and detects unusual sequences through reconstruction error.

---

## Training

Training uses:

* **PyTorch**
* Adam optimizer
* Mean Squared Error reconstruction loss
* validation monitoring
* early stopping
* best-checkpoint saving

Reference training run:

```text
Best epoch           : 34
Best validation loss : 0.628600
```

Model artifact:

```text
models/lstm_autoencoder.pt
```

---

## Anomaly Detection

For each input sequence, the model reconstructs the original sequence and calculates the mean squared reconstruction error.

### Decision Rule

```text
reconstruction_error > threshold
              │
              ▼
           anomaly
```

Current threshold:

```text
5.322128
```

The threshold is derived from the **99th percentile of validation baseline reconstruction errors**.

Threshold artifact:

```text
models/threshold.json
```

---

## Explainability

The project includes a **"What Changed?"** analysis to provide context for detected anomalies.

### Reconstruction Contributors

Feature-level reconstruction errors identify features contributing most strongly to the anomaly score.

Examples include:

```text
ACC_MAG_mean
ACC_MAG_min
ACC_MAG_max
```

### Baseline-Relative Changes

Detected anomalies can also be compared against participant-specific baseline behavior across signals such as:

```text
EDA
ACC_MAG
TEMP
HR
IBI
```

This provides more context than a binary anomaly flag alone.

---

## Evaluation

This is an **unsupervised anomaly detection system**, so evaluation focuses on:

* reconstruction error
* anomaly rates
* participant-level behavior
* protocol-stage behavior
* robustness analysis
* anomaly investigation
* explainability

No clinical ground-truth anomaly labels are assumed.

---

## FastAPI

The trained model is exposed through a FastAPI inference service.

### Run locally

```powershell
.\.venv\Scripts\activate
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

### API Endpoints

| Method | Endpoint      | Purpose                         |
| ------ | ------------- | ------------------------------- |
| GET    | `/health`     | Service health                  |
| GET    | `/model-info` | Model and threshold information |
| POST   | `/predict`    | Sequence anomaly prediction     |
| GET    | `/metrics`    | Prometheus metrics              |

Swagger / OpenAPI:

```text
http://127.0.0.1:8000/docs
```

### Prediction Input

The model expects:

```text
12 consecutive windows × 41 features
```

Prediction responses include fields such as:

```text
is_anomaly
reconstruction_error
anomaly_threshold
anomaly_score
top_reconstruction_features
participant_id
session_type
protocol_stage
sequence_shape
```

---

## Monitoring

The API exposes Prometheus-compatible metrics through:

```text
GET /metrics
```

Tracked metrics include:

```text
http_requests_total
http_errors_total
http_request_latency_seconds
predictions_total
anomalies_total
```

Monitoring was validated locally, inside Docker, and on the deployed service.

---

## Drift Detection

The project implements feature-distribution drift detection using **normalized Wasserstein distance**.

### Detection Logic

For each numerical model feature:

1. compare the reference distribution with the current distribution
2. calculate normalized Wasserstein distance
3. apply the feature-level threshold
4. calculate the overall drift rate
5. trigger dataset-level drift when enough features exceed the threshold

Configuration:

```text
Feature drift threshold      : 0.20
Overall drift-rate threshold : 20%
Minimum valid values         : 20
```

### Simulated Production Test

A deliberately shifted synthetic production batch produced:

```text
Reference rows     : 15,015
Current rows       : 211
Features checked   : 41
Drifted features   : 37
Overall drift rate : 90.24%
Drift detected     : True
```

This value is a **synthetic stress-test result**, not a claim about real production drift.

### Run Drift Detection

```powershell
python -m src.monitoring.drift
```

Or with a specific current batch:

```powershell
python -m src.monitoring.drift `
  --current "data\processed\production_batch.csv"
```

---

## Automatic Retraining

The project includes **drift-triggered candidate retraining**.

### Retraining Flow

```text
Drift Detected
      │
      ▼
Load Current Feature Batch
      │
      ▼
Apply Existing Preprocessing
      │
      ▼
Build Current Baseline Sequences
      │
      ▼
Combine Original + Current Data
      │
      ▼
Train Candidate Model
      │
      ▼
Evaluate on Untouched Test Baseline
      │
      ▼
Evaluation Gate
      │
   ┌──┴──┐
   ▼     ▼
Promote Reject
```

A drift event alone is **not enough** to replace the active model.

The candidate must achieve at least:

```text
1% improvement on the holdout metric
```

Otherwise:

```text
Candidate rejected
Current model remains active
```

### Safety Validation

Using a simulated production batch:

```text
Drift rate              : 90.24%
Current sequences       : 167
Original training      : 3,219
Combined training      : 3,386
```

Candidate evaluation:

```text
Current holdout loss    : 4.921436
Candidate holdout loss  : 4.987376
Improvement rate        : -1.34%
Evaluation gate         : False
Promotion               : Rejected
```

This confirms that the retraining pipeline can **detect an inferior candidate and keep the current model active**.

### Model Archiving

Before a successful promotion, the active model artifacts are archived under:

```text
models/retraining/archive/
```

Archived artifacts may include:

```text
lstm_autoencoder.pt
threshold.json
training_history.csv
```

---

## Testing

Dedicated drift tests cover both:

```text
Reference = Current
        ↓
No drift detected

Synthetic distribution shift
        ↓
Drift detected
```

Current full test suite:

```text
5 passed
```

Run:

```powershell
python -m pytest -q
```

---

## CI/CD

GitHub Actions provides automated testing and container delivery.

### CI Flow

```text
git push
   │
   ▼
Automated Tests
   │
   ▼
Docker Build
   │
   ▼
GHCR Publish
```

The project is integrated with:

* GitHub
* GitHub Container Registry
* FastAPI Cloud

---

## Retraining Workflow

A separate workflow is defined in:

```text
.github/workflows/retraining.yml
```

It supports:

* manual execution
* scheduled execution
* test execution
* artifact availability checks
* drift detection
* conditional retraining
* candidate evaluation
* safe model promotion

The workflow intentionally avoids retraining when the required model/data artifacts are unavailable in the repository.

This keeps wearable-derived datasets and large model artifacts outside the public source repository.

> Fully unattended production retraining from live wearable data would require a dedicated artifact/data storage layer.

---

## Docker

The FastAPI service is containerized with Docker.

### Build

```powershell
docker build -t wearable-anomaly-api .
```

### Run

```powershell
docker run --rm -p 8000:8000 wearable-anomaly-api
```

The API uses:

```text
models/lstm_autoencoder.pt
models/preprocessor.pkl
models/threshold.json
```

Health check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

---

## Deployment

Deployment architecture:

```text
Local Project
     │
     ▼
GitHub
     │
     ▼
GitHub Actions
     │
     ▼
Docker Image
     │
     ▼
GHCR
     │
     ▼
FastAPI Cloud
     │
     ▼
Public HTTPS API
```

### Live API

**Swagger / OpenAPI**

[Open API Documentation](https://wearable-anomaly-detection.fastapicloud.dev/docs)

**Base URL**

```text
https://wearable-anomaly-detection.fastapicloud.dev
```

The deployed service has been validated with live health, model-information, prediction, and monitoring requests.

---

## Project Structure

```text
wearable-anomaly-detection/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── lstm_autoencoder.pt
│   ├── preprocessor.pkl
│   ├── threshold.json
│   └── generated evaluation / visualization artifacts
│
├── src/
│   ├── api/
│   │   ├── main.py
│   │   └── make_sample_payload.py
│   │
│   ├── data/
│   │   ├── validate.py
│   │   ├── preprocess.py
│   │   ├── label_protocol.py
│   │   └── split.py
│   │
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   ├── robustness.py
│   │   └── build_report.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── models/
│   │   ├── prepare_sequences.py
│   │   ├── lstm_autoencoder.py
│   │   ├── train.py
│   │   ├── detect_anomalies.py
│   │   ├── analyze_anomalies.py
│   │   ├── investigate_anomalies.py
│   │   ├── explain_anomalies.py
│   │   └── visualize_anomalies.py
│   │
│   └── monitoring/
│       ├── drift.py
│       ├── generate_production_batch.py
│       └── retrain.py
│
├── tests/
│   └── test_drift.py
│   └── test_validate.py
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── retraining.yml
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── requirements-api.txt
└── README.md
```

---

## Main Commands

### Data

```powershell
python -m src.data.validate
python -m src.data.preprocess
python -m src.data.label_protocol
python -m src.data.split
```

### Features & Sequences

```powershell
python -m src.features.build_features
python -m src.models.prepare_sequences
```

### Model

```powershell
python -m src.models.train
python -m src.models.detect_anomalies
python -m src.models.analyze_anomalies
python -m src.models.investigate_anomalies
python -m src.models.explain_anomalies
python -m src.models.visualize_anomalies
```

### Evaluation

```powershell
python -m src.evaluation.evaluate
python -m src.evaluation.robustness
python -m src.evaluation.build_report
```

### Monitoring

```powershell
python -m src.monitoring.drift
python -m src.monitoring.retrain
```

### Tests

```powershell
python -m pytest -q
```

---

## Git Hygiene

Raw datasets and generated artifacts are intentionally excluded from normal source control.

The project uses `.gitignore` and `.dockerignore` rules to keep temporary files, generated artifacts, local environments, and dataset-specific files outside the source repository.

Model delivery can be moved to dedicated artifact/model storage in a future production architecture.

---

## Current Status

| Component                  | Status |
| -------------------------- | :----: |
| Data validation            |    ✅   |
| Preprocessing              |    ✅   |
| Feature engineering        |    ✅   |
| Protocol labeling          |    ✅   |
| Participant-level split    |    ✅   |
| Sequence preparation       |    ✅   |
| LSTM Autoencoder           |    ✅   |
| Training                   |    ✅   |
| Anomaly detection          |    ✅   |
| Explainability             |    ✅   |
| Evaluation                 |    ✅   |
| FastAPI                    |    ✅   |
| Docker                     |    ✅   |
| GHCR                       |    ✅   |
| Cloud deployment           |    ✅   |
| Public HTTPS API           |    ✅   |
| Prometheus monitoring      |    ✅   |
| CI/CD                      |    ✅   |
| Drift detection            |    ✅   |
| Drift testing              |    ✅   |
| Automatic retraining       |    ✅   |
| Retraining evaluation gate |    ✅   |
| Model archiving            |    ✅   |
| GitHub retraining workflow |    ✅   |

### Production Automation Note

The drift detection and retraining pipeline has been **implemented and locally validated using simulated production batches**.

The GitHub Actions retraining workflow includes artifact-availability safeguards. Fully unattended retraining from live wearable data would additionally require a dedicated external data/artifact source.

---

## Limitations

* This is an unsupervised anomaly detection system.
* An anomaly represents a deviation from learned wearable behavior.
* The system is not a medical diagnostic tool.
* No clinical ground-truth anomaly labels are used.
* Results may vary across datasets, wearable devices, populations, and protocols.
* Synthetic production batches are used to validate drift and retraining behavior.
* Fully unattended retraining from live wearable data requires dedicated artifact/data infrastructure.

---

## Author

**Shahd Fayez**
AI / Machine Learning Engineer

[GitHub](https://github.com/ShahdFayezNegm) · [LinkedIn](https://linkedin.com/in/shahd-fayez-70b9a331b)
