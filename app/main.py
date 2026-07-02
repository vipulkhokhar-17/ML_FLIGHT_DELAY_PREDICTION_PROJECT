from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import joblib
import os
import numpy as np
import pandas as pd
from .schemas import FlightInput, PredictionOutput, HealthCheck, ModelInfo
from .utils import ModelMetadata, get_risk_level, get_top_risk_factors, estimate_delay_minutes

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "flight_delay_pipeline.pkl")

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model = None
    model_loaded = False
    print(f"Warning: Could not load model from {MODEL_PATH}: {e}")

app = FastAPI(
    title="Flight Delay Prediction API",
    description="Predict flight delays using an XGBoost model trained on 2015 US DOT data",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

start_time = datetime.now()


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Flight Delay Prediction API",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model-info"
    }


@app.get("/health", response_model=HealthCheck, tags=["Health"])
def health():
    uptime = (datetime.now() - start_time).total_seconds()
    return HealthCheck(
        status="healthy",
        model_loaded=model_loaded,
        model_version="1.0.0",
        uptime_seconds=uptime
    )


@app.get("/model-info", response_model=ModelInfo, tags=["Model"])
def model_info():
    return ModelInfo(
        model_type=ModelMetadata.model_type,
        validation_auc=ModelMetadata.validation_auc,
        test_auc=ModelMetadata.test_auc,
        features_used=ModelMetadata.features_used,
        training_records=ModelMetadata.training_records,
        description=ModelMetadata.description
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(flight: FlightInput):
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build a DataFrame with the exact column names the pipeline expects
    features = pd.DataFrame([{
        "MONTH": flight.MONTH,
        "DAY": flight.DAY,
        "DAY_OF_WEEK": flight.DAY_OF_WEEK,
        "DISTANCE": flight.DISTANCE,
        "SCHED_DEP_HOUR": flight.SCHED_DEP_HOUR,
        "IS_PEAK_SEASON": flight.IS_PEAK_SEASON,
        "IS_EARLY_MORNING": flight.IS_EARLY_MORNING,
        "CARRIER_DELAY_RATE": flight.CARRIER_DELAY_RATE,
        "ORIGIN_MONTHLY_DELAY_RATE": flight.ORIGIN_MONTHLY_DELAY_RATE,
        "ROUTE_DELAY_RATE": flight.ROUTE_DELAY_RATE,
        "AIRLINE": flight.AIRLINE,
        "ORIGIN_AIRPORT": flight.ORIGIN_AIRPORT,
        "DESTINATION_AIRPORT": flight.DESTINATION_AIRPORT,
    }])

    try:
        probability = float(model.predict_proba(features)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    risk_level = get_risk_level(probability)
    top_factors = get_top_risk_factors(flight.model_dump())
    expected_delay = estimate_delay_minutes(probability)

    return PredictionOutput(
        delay_probability=round(probability, 4),
        risk_level=risk_level,
        expected_delay_minutes=expected_delay,
        top_risk_factors=top_factors,
        model_version="1.0.0",
        prediction_time=datetime.now()
    )

@app.get("/sample-input", tags=["Utility"])
def sample_input():
    return {
        "description": "Example input for testing the /predict endpoint",
        "sample": {
            "MONTH": 7,
            "DAY": 15,
            "DAY_OF_WEEK": 4,
            "DISTANCE": 1500.0,
            "SCHED_DEP_HOUR": 18,
            "IS_PEAK_SEASON": 1,
            "IS_EARLY_MORNING": 0,
            "CARRIER_DELAY_RATE": 0.22,
            "ORIGIN_MONTHLY_DELAY_RATE": 0.18,
            "ROUTE_DELAY_RATE": 0.25,
            "AIRLINE": "AA",
            "ORIGIN_AIRPORT": "JFK",
            "DESTINATION_AIRPORT": "LAX"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)