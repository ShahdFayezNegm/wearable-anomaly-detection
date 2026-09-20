# Wearable Anomaly Detection

An end-to-end **Machine Learning + MLOps system** for detecting anomalies in wearable sensor time-series data.

The project uses an **LSTM Autoencoder** trained on baseline wearable data. Anomaly detection is based on **reconstruction error**, with an explainability layer that identifies the wearable features contributing most to an anomalous observation.

The system is exposed through a **FastAPI REST API**, containerized with **Docker**, published through **GitHub Container Registry (GHCR)**, deployed to **FastAPI Cloud**, monitored with **Prometheus-compatible metrics**, and extended with **data drift detection and guarded automatic retraining**.

---

## Highlights

* **41 engineered wearable features**
* **12-step temporal sequences**
* Participant-level train/validation/test splitting
* Leakage-safe preprocessing
* LSTM Autoencoder anomaly detection
* Reconstruction-error based anomaly scoring
* Feature-level anomaly explanation
* FastAPI REST API
* Interactive Swagger/OpenAPI documentation
* Dockerized deployment
* GitHub Container Registry (GHCR)
* GitHub Actions CI/CD
* Public HTTPS deployment on FastAPI Cloud
* Prometheus-compatible monitoring metrics
* Wasserstein-based data drift detection
* Drift-triggered candidate retraining
* Evaluation gate before model promotion
* Automatic model archiving before promotion
* Safe rejection of inferior candidate models

---

# Architecture

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
Leakage-Safe Scaling
        │
        ▼
12-Step Sequences
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


Monitoring / Retraining Loop
        │
        ▼
Data Drift Detection
        │
        ▼
Drift Detected?
        │
        ▼
Candidate Retraining
        │
        ▼
Evaluation Gate
      /     \
   Promote  Reject
      │
      ▼
Archive Previous Model
```

---

# Dataset

The project uses the:

**PhysioNet Wearable Device Dataset from Induced Stress and Structured Exercise Sessions v1.0.1**

The dataset contains wearable recordings collected during:

* Stress sessions
* Aerobic exercise
* Anaerobic exercise

Available physiological signals include:

* Heart Rate (HR)
* Blood Volume Pulse (BVP)
* Electrodermal Activity (EDA)
* Temperature (TEMP)
* Accelerometer axes
* Accelerometer magnitude (ACC_MAG)
* Inter-Beat Interval (IBI)

The signals were aligned to a **1 Hz representation** before feature engineering.

---

# Data Pipeline

The preprocessing pipeline consists of:

1. Raw wearable data loading
2. Data validation
3. Signal alignment
4. Missing-value handling
5. 10-second window feature extraction
6. Protocol labeling
7. Participant-level dataset splitting
8. Train-only preprocessing fitting
9. Sequence generation
10. LSTM Autoencoder training

The final dataset contains:

```text
Participants: 41

Train: 28 participants
Validation: 6 participants
Test: 7 participants
```

The resulting feature representation contains:

```text
41 features
12 timesteps per sequence
```

---

# Leakage-Safe Data Preparation

The project uses participant-level splitting to reduce information leakage between train, validation, and test sets.

Preprocessing artifacts are learned only from the training data:

* Missing-value medians are learned from training data.
* The scaler is fitted using training baseline data.
* Validation and test data use the already-fitted training artifacts.
* Sequences do not cross participant boundaries.
* Sequences do not cross session boundaries.

The preprocessing pipeline is stored as:

```text
preprocessor.pkl
```

This allows the same preprocessing logic to be reused during inference and retraining.

---

# Model

The anomaly detection model is an **LSTM Autoencoder** implemented with PyTorch.

### Architecture

```text
Input
41 features × 12 timesteps
        │
        ▼
LSTM Encoder
        │
Hidden Size: 64
        │
        ▼
Latent Representation
Size: 32
        │
        ▼
LSTM Decoder
        │
        ▼
Reconstructed Sequence
41 features × 12 timesteps
```

### Configuration

| Parameter       |   Value |
| --------------- | ------: |
| Input features  |      41 |
| Sequence length |      12 |
| Hidden size     |      64 |
| Latent size     |      32 |
| LSTM layers     |       2 |
| Dropout         |     0.2 |
| Framework       | PyTorch |

Total model parameters:

```text
134,089
```

---

# Training

The model is trained using:

* **PyTorch**
* **Adam optimizer**
* **Mean Squared Error (MSE)**
* Validation monitoring
* Early stopping
* Best-model checkpointing

Training result:

```text
Best Epoch: 34
Best Validation Loss: 0.628600
```

The trained model is stored as:

```text
lstm_autoencoder.pt
```

---

# Anomaly Detection

The system uses **reconstruction error** as the anomaly score.

The autoencoder learns to reconstruct normal baseline wearable sequences.

When a new sequence produces a significantly higher reconstruction error, it can be flagged as anomalous.

The anomaly threshold was determined using the **99th percentile of validation baseline reconstruction errors**.

```text
Anomaly Threshold:
5.3221282958984375
```

Conceptually:

```text
Reconstruction Error
        │
        ├── below threshold ──► Normal
        │
        └── above threshold ──► Anomaly
```

The threshold is stored in:

```text
threshold.json
```

---

# Explainability — "What Changed?"

The API does not only return whether a sequence is anomalous.

It also identifies features that contribute strongly to the reconstruction error.

Example high-impact features include:

* `EDA_range`
* `EDA_std`
* `HR_min`
* `ACC_MAG_mean`
* `ACC_MAG_min`
* `ACC_MAG_max`

The system can also compare incoming observations against participant-level baseline statistics for features such as:

* EDA
* ACC_MAG
* TEMP
* HR
* IBI

This provides a more interpretable answer to:

> **What changed compared with the expected wearable pattern?**

---

# Evaluation

This project uses **unsupervised anomaly detection**.

The dataset does not provide clinical ground-truth anomaly labels for evaluating the model as a supervised anomaly classifier.

Therefore, evaluation focuses on:

* Reconstruction performance
* Validation baseline behavior
* Threshold calibration
* Drift detection
* Candidate-vs-current model comparison
* Safe model promotion

This distinction is important because a high anomaly score indicates deviation from the learned baseline; it does **not** by itself represent a medical diagnosis.

---

# FastAPI

The trained model is exposed through a REST API using **FastAPI**.

Application:

```text
src.api.main:app
```

Application title:

```text
Wearable Anomaly Detection API
```

Version:

```text
1.0.0
```

Run locally:

```powershell
uvicorn src.api.main:app --reload
```

The local API provides interactive Swagger/OpenAPI documentation at:

```text
http://127.0.0.1:8000/docs
```

OpenAPI specification:

```text
http://127.0.0.1:8000/openapi.json
```

### Prediction Input

The `/predict` endpoint expects a temporal sequence containing:

```text
12 timesteps
×
41 features
```

Conceptually:

```json
{
  "sequence": [
    [feature_1, feature_2, "...", feature_41"],
    "...",
    "12 timesteps total"
  ]
}
```

### Prediction Response

The response includes information such as:

```text
is_anomaly
reconstruction_error
anomaly_score
top_features
```

Example response:

```text
is_anomaly: false
reconstruction_error: 0.429891
anomaly_score: 0.080774
```

Example top contributing features:

```text
EDA_range
EDA_std
HR_min
```

---

# Live API Demo

The deployed API is publicly accessible through FastAPI Cloud.

### Swagger UI

```text
https://wearable-anomaly-detection.fastapicloud.dev/docs
```

### Base URL

```text
https://wearable-anomaly-detection.fastapicloud.dev
```

### Available Endpoints

| Endpoint      | Method | Purpose                                  |
| ------------- | ------ | ---------------------------------------- |
| `/health`     | GET    | API and model health status              |
| `/model-info` | GET    | Model configuration and metadata         |
| `/predict`    | POST   | Run anomaly detection                    |
| `/metrics`    | GET    | Prometheus-compatible monitoring metrics |

The deployed `/health` and `/model-info` endpoints were validated successfully.

---

# Monitoring

The API exposes Prometheus-compatible metrics through:

```text
/metrics
```

Tracked metrics include:

```text
http_requests_total
http_errors_total
http_request_latency_seconds
predictions_total
anomalies_total
```

These metrics make it possible to monitor:

* API traffic
* Request errors
* Latency
* Number of predictions
* Number of detected anomalies

Monitoring was validated locally, inside the Docker deployment, and on the deployed service.

---

# Data Drift Detection

The project includes feature-level drift detection using the **Wasserstein distance**.

The detector compares a reference distribution against a current incoming batch.

Configuration:

```text
Feature drift threshold: 0.20
Overall drift-rate threshold: 20%
Minimum valid values: 20
```

A feature is considered drifted when its normalized Wasserstein distance exceeds the configured threshold.

### Synthetic Stress Test

A synthetic shifted batch was used to validate the drift detection system.

Result:

```text
Reference samples: 15,015
Current samples: 211
Features checked: 41
Drifted features: 37
Drift rate: 90.24%
Drift detected: True
```

This was intentionally a **synthetic stress test** to validate the monitoring mechanism.

It should not be interpreted as evidence of real-world production drift.

---

# Automatic Retraining

The project includes a guarded retraining workflow.

The intended flow is:

```text
Incoming Data
      │
      ▼
Drift Detection
      │
      ├── No Drift ──► Continue Using Current Model
      │
      └── Drift
           │
           ▼
     Candidate Retraining
           │
           ▼
     Candidate Evaluation
           │
           ▼
       Evaluation Gate
          /       \
      Promote    Reject
```

When drift is detected:

1. The current data batch is processed.
2. Existing preprocessing artifacts are reused.
3. Current baseline sequences are generated.
4. Original training data and current data are combined.
5. A candidate model is trained.
6. The candidate is evaluated on an untouched test baseline.
7. The candidate is compared with the currently deployed model.
8. The evaluation gate determines whether promotion is allowed.

---

# Retraining Safety Validation

The retraining pipeline was tested using a synthetic drifted batch.

Test configuration:

```text
Detected drift: 90.24%
Current sequences: 167
Original training sequences: 3,219
Combined training sequences: 3,386
```

Candidate evaluation:

```text
Current model holdout loss: 4.921436
Candidate model holdout loss: 4.987376
Improvement: -1.34%
```

The promotion requirement is:

```text
Minimum improvement: 1%
```

Because the candidate did not improve the required amount:

```text
Gate result: False
Promotion: Rejected
```

This demonstrates that the retraining pipeline can detect a candidate model that does not meet the promotion requirement and prevent automatic replacement of the current model.

---

# Model Archiving

Before a successful model promotion, the existing model artifacts can be archived.

Archived artifacts include:

```text
lstm_autoencoder.pt
threshold.json
training_history.csv
```

Archive location:

```text
models/retraining/archive/
```

This provides a recovery path and preserves previous model versions.

---

# Testing

The project includes automated tests covering the main monitoring and retraining functionality.

Validated scenarios include:

* No drift when reference and current distributions are identical.
* Drift detection under a synthetic distribution shift.
* Retraining workflow behavior.
* Evaluation gate behavior.
* Artifact validation.

Current test result:

```text
5 passed
```

---

# CI/CD

The project uses **GitHub Actions** to automate the application delivery pipeline.

The CI/CD flow includes:

```text
Git Push
   │
   ▼
GitHub Actions
   │
   ├── Run Tests
   │
   ├── Build Docker Image
   │
   └── Publish Image
           │
           ▼
          GHCR
```

The project is integrated with:

* GitHub
* GitHub Actions
* GitHub Container Registry (GHCR)
* FastAPI Cloud

This provides an automated path from source-code changes to container image publication.

---

# Retraining Workflow

The repository contains:

```text
.github/workflows/retraining.yml
```

The workflow supports:

* Manual execution
* Scheduled execution
* Automated testing
* Artifact checks
* Drift detection
* Conditional retraining
* Candidate evaluation
* Safe model promotion

The workflow is designed to avoid retraining when required data or model artifacts are unavailable.

Large datasets and model artifacts are intentionally excluded from the Git repository.

For a completely unattended production retraining system, an external data and artifact storage layer would be required.

---

# Docker

The API is containerized using Docker.

The Docker image is based on:

```text
python:3.12-slim
```

The container exposes:

```text
8000
```

Build:

```powershell
docker build -t wearable-anomaly-api .
```

Run:

```powershell
docker run --rm -p 8000:8000 wearable-anomaly-api
```

Health check:

```text
http://127.0.0.1:8000/health
```

The Dockerized API was tested locally before deployment.

---

# Deployment

The deployment architecture is:

```text
Developer
    │
    ▼
GitHub Repository
    │
    ▼
GitHub Actions
    │
    ├── Tests
    ├── Docker Build
    └── GHCR Publish
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

The application is deployed using **FastAPI Cloud**.

The live service and interactive Swagger documentation are available in the **Live API Demo** section above.

---

# Project Structure

```text
wearable-anomaly-detection/
│
├── src/
│   ├── api/
│   │   └── main.py
│   │
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── monitoring/
│   └── retraining/
│
├── tests/
│
├── models/
│   └── retraining/
│       └── archive/
│
├── .github/
│   └── workflows/
│       └── retraining.yml
│
├── Dockerfile
├── requirements.txt
├── preprocessor.pkl
├── threshold.json
├── lstm_autoencoder.pt
└── README.md
```

Large datasets and generated artifacts are excluded from version control where appropriate.

---

# Main Commands

### Create virtual environment

```powershell
python -m venv .venv
```

### Activate environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

### Run FastAPI locally

```powershell
uvicorn src.api.main:app --reload
```

### Build Docker image

```powershell
docker build -t wearable-anomaly-api .
```

### Run Docker container

```powershell
docker run --rm -p 8000:8000 wearable-anomaly-api
```

---

# Git Hygiene

The repository does not store large raw datasets or unnecessary generated artifacts.

Examples of files that should remain outside Git when appropriate:

```text
Raw datasets
Large model checkpoints
Temporary outputs
Local virtual environments
Generated caches
```

Sensitive credentials and deployment secrets should never be committed to the repository.

---

# Current Status

| Component                  | Status |
| -------------------------- | ------ |
| Data preprocessing         | ✅      |
| Feature engineering        | ✅      |
| Participant-level split    | ✅      |
| Leakage-safe preprocessing | ✅      |
| LSTM Autoencoder           | ✅      |
| Anomaly thresholding       | ✅      |
| Explainability             | ✅      |
| FastAPI API                | ✅      |
| Swagger/OpenAPI            | ✅      |
| Docker                     | ✅      |
| GHCR integration           | ✅      |
| FastAPI Cloud deployment   | ✅      |
| Prometheus metrics         | ✅      |
| Drift detection            | ✅      |
| Retraining pipeline        | ✅      |
| Evaluation gate            | ✅      |
| Model archiving            | ✅      |
| Automated tests            | ✅      |
| GitHub Actions CI/CD       | ✅      |

---

# Limitations

### 1. No clinical anomaly ground truth

The project is an unsupervised anomaly detection system.

The anomaly score represents deviation from the learned baseline and should not be interpreted as a clinical diagnosis.

### 2. Retraining validation uses synthetic drift

The drift and retraining mechanisms have been locally validated using simulated distribution shifts.

This demonstrates that the system behaves correctly under controlled drift scenarios, but it does not establish performance under real production drift.

### 3. Fully unattended live retraining requires external storage

The repository intentionally does not contain large production datasets or model artifacts.

A fully automated production retraining system would require an external artifact/data storage layer for:

* Incoming production batches
* Historical training data
* Model artifacts
* Versioned preprocessing artifacts
* Retraining outputs

### 4. Deployment environment

The current deployed model runs on CPU.

The system is designed as a portfolio/research implementation demonstrating the complete ML deployment and monitoring workflow rather than a clinically validated medical device.

---

# Future Improvements

Potential next steps include:

* Real production wearable data ingestion
* External artifact storage
* Model version registry
* Grafana dashboards
* Alerting for sustained drift
* More advanced temporal architectures
* Online/streaming anomaly detection
* Additional explainability methods
* More robust drift baselines
* Automated rollback
* Production authentication and rate limiting
* Cloud-based experiment tracking
* Real-world anomaly ground-truth collection

---

# Author

**Shahd Fayez**

AI Engineer | Machine Learning | Computer Vision | RAG/LLM | ML Deployment

GitHub:

https://github.com/ShahdFayezNegm

LinkedIn:

https://linkedin.com/in/shahd-fayez-70b9a331b
