from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import json
import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.lstm_autoencoder import LSTMAutoencoder


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "lstm_autoencoder.pt"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
THRESHOLD_PATH = MODELS_DIR / "threshold.json"


# ============================================================
# Configuration
# ============================================================

SEQUENCE_LENGTH = 12

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Global artifacts
# ============================================================

model = None
preprocessor = None
threshold = None
feature_columns = []


# ============================================================
# Request schema
# ============================================================

class PredictionRequest(BaseModel):

    sequence: list[dict[str, float | None]] = Field(
        ...,
        min_length=SEQUENCE_LENGTH,
        max_length=SEQUENCE_LENGTH,
    )

    participant_id: str | None = None

    session_type: str | None = None

    protocol_stage: str | None = None

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )


# ============================================================
# Load model
# ============================================================

def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    loaded_model = LSTMAutoencoder(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        latent_dim=checkpoint["latent_dim"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    ).to(DEVICE)

    loaded_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    loaded_model.eval()

    return loaded_model


# ============================================================
# Load preprocessor
# ============================================================

def load_preprocessor():

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessor not found:\n{PREPROCESSOR_PATH}"
        )

    loaded = joblib.load(
        PREPROCESSOR_PATH
    )

    if not isinstance(loaded, dict):
        raise ValueError(
            "preprocessor.pkl must contain a dictionary."
        )

    return loaded


# ============================================================
# Load threshold
# ============================================================

def load_threshold():

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Threshold not found:\n{THRESHOLD_PATH}"
        )

    with open(
        THRESHOLD_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    return float(
        data["threshold"]
    )


# ============================================================
# Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app_instance: FastAPI):

    global model
    global preprocessor
    global threshold
    global feature_columns

    print("=" * 70)
    print("Loading Wearable Anomaly Detection API")
    print("=" * 70)

    model = load_model()

    preprocessor = load_preprocessor()

    threshold = load_threshold()

    feature_columns = list(
        preprocessor["feature_columns"]
    )

    print(
        f"Device          : {DEVICE}"
    )

    print(
        f"Features        : {len(feature_columns)}"
    )

    print(
        f"Sequence length : {SEQUENCE_LENGTH}"
    )

    print(
        f"Threshold       : {threshold:.6f}"
    )

    print("Model loaded successfully.")

    yield


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(
    title="Wearable Anomaly Detection API",
    description=(
        "LSTM Autoencoder based wearable anomaly detection service."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# Health endpoint
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(DEVICE),
        "feature_count": len(feature_columns),
        "sequence_length": SEQUENCE_LENGTH,
        "anomaly_threshold": threshold,
    }


# ============================================================
# Model info
# ============================================================

@app.get("/model-info")
def model_info():

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    return {
        "architecture": "LSTM Autoencoder",
        "input_dim": len(feature_columns),
        "sequence_length": SEQUENCE_LENGTH,
        "feature_count": len(feature_columns),
        "parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
        "device": str(DEVICE),
        "anomaly_threshold": threshold,
    }


# ============================================================
# Preprocess sequence
# ============================================================

def preprocess_sequence(
    sequence: list[dict[str, float | None]],
) -> np.ndarray:

    frame = pd.DataFrame(
        sequence
    )

    missing_columns = [
        column
        for column in feature_columns
        if column not in frame.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required features: {missing_columns}"
        )

    frame = frame[
        feature_columns
    ].copy()

    # --------------------------------------------------------
    # Imputation
    # --------------------------------------------------------

    imputer = preprocessor.get(
        "imputer"
    )

    if (
        imputer is not None
        and hasattr(imputer, "transform")
    ):

        matrix = imputer.transform(
            frame
        )

    else:

        medians = preprocessor.get(
            "medians"
        )

        if medians is None:

            medians = preprocessor.get(
                "train_medians"
            )

        if medians is not None:

            if isinstance(
                medians,
                dict,
            ):

                medians = pd.Series(
                    medians
                )

            elif not isinstance(
                medians,
                pd.Series,
            ):

                medians = pd.Series(
                    medians,
                    index=feature_columns,
                )

            frame = frame.fillna(
                medians.reindex(
                    feature_columns
                )
            )

        elif frame.isna().any().any():

            raise ValueError(
                "Input contains missing values and "
                "no compatible imputer was found."
            )

        matrix = frame.to_numpy(
            dtype=np.float32
        )

    # --------------------------------------------------------
    # Scaling
    # --------------------------------------------------------

    scaler = preprocessor.get(
        "scaler"
    )

    if scaler is None:
        raise ValueError(
            "No scaler found in preprocessor.pkl."
        )

    matrix = scaler.transform(
        matrix
    )

    matrix = np.asarray(
        matrix,
        dtype=np.float32,
    )

    if matrix.shape != (
        SEQUENCE_LENGTH,
        len(feature_columns),
    ):

        raise ValueError(
            f"Unexpected processed shape: {matrix.shape}"
        )

    if not np.isfinite(matrix).all():

        raise ValueError(
            "Processed input contains non-finite values."
        )

    return matrix


# ============================================================
# Prediction
# ============================================================

def run_prediction(
    sequence: np.ndarray,
    top_k: int,
):

    tensor = (
        torch.from_numpy(
            sequence
        )
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        reconstruction = model(
            tensor
        )

    # Overall reconstruction error
    error = torch.mean(
        (tensor - reconstruction) ** 2
    ).item()

    error = float(error)

    is_anomaly = (
        error > threshold
    )

    anomaly_score = (
        error / threshold
    )

    # --------------------------------------------------------
    # Feature-level contributions
    # --------------------------------------------------------

    contributions = torch.mean(
        (tensor - reconstruction) ** 2,
        dim=1,
    )[0]

    contributions = (
        contributions
        .detach()
        .cpu()
        .numpy()
    )

    ranked_indices = np.argsort(
        contributions
    )[::-1]

    top_features = []

    for index in ranked_indices[:top_k]:

        top_features.append(
            {
                "feature": feature_columns[index],
                "reconstruction_contribution": round(
                    float(
                        contributions[index]
                    ),
                    6,
                ),
            }
        )

    return {
        "is_anomaly": bool(
            is_anomaly
        ),
        "reconstruction_error": round(
            error,
            6,
        ),
        "anomaly_threshold": round(
            float(threshold),
            6,
        ),
        "anomaly_score": round(
            float(anomaly_score),
            6,
        ),
        "top_reconstruction_features": top_features,
    }


# ============================================================
# Predict endpoint
# ============================================================

@app.post("/predict")
def predict_endpoint(
    request: PredictionRequest,
):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    try:

        processed = preprocess_sequence(
            request.sequence
        )

        result = run_prediction(
            sequence=processed,
            top_k=request.top_k,
        )

        result.update(
            {
                "participant_id":
                    request.participant_id,

                "session_type":
                    request.session_type,

                "protocol_stage":
                    request.protocol_stage,

                "sequence_shape":
                    list(processed.shape),

                "note": (
                    "Anomaly indicates deviation from "
                    "the learned baseline pattern; it "
                    "is not a medical diagnosis."
                ),
            }
        )

        return result

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc