# Flight Delay Prediction API

Live API: https://mlflightdelaypredictionproject-production.up.railway.app/docs

## What it does
Predicts US flight delays (&gt;15 min) using 2015 DOT data with 72.5% accuracy on high-risk flights.

## Business Impact
- **$168M potential savings** for a 150-pax fleet
- **17.5% reduction** in delay-related operational costs

## Tech Stack
- **ML:** XGBoost, scikit-learn, Optuna (hyperparameter tuning)
- **API:** FastAPI, Pydantic
- **Deployment:** Railway, GitHub auto-deploy

## Key Features
- Time-based train/validation/test split (no leakage)
- Historical delay rates computed on training data only
- 13 engineered features including carrier, route, and seasonal risk

## Endpoints
| Endpoint | Description |
|----------|-------------|
| `POST /predict` | Single flight delay prediction |
| `GET /health` | Model health check |
| `GET /model-info` | Model metadata & performance |
