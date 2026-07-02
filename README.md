# Flight Delay Prediction

Predicting if a US flight will be delayed (>15 mins) using 2015 BTS data.

## Setup

```bash
pip install -r requirements.txt
```

## Data

Get it from Kaggle: https://www.kaggle.com/datasets/usdot/flight-delays
Put `flights.csv`, `airports.csv`, `airlines.csv` in `data/raw/`.

## Run

1. `notebooks/01_eda.ipynb` - explore the data, make some plots
2. `notebooks/02_modeling.ipynb` - train models, compare, tune

## Notes

- Historical features computed on training data only (no leakage)
- Time-based split (60/20/20)
- Target: DEPARTURE_DELAY > 15, excluding cancelled/diverted flights
