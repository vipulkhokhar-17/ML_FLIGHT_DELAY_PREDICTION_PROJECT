from typing import List, Dict, Any
from pydantic import BaseModel


class ModelMetadata(BaseModel):
    model_type: str = "XGBoost"
    validation_auc: float = 0.649
    test_auc: float = 0.638
    features_used: int = 13
    training_records: int = 392788
    description: str = "Flight delay prediction model trained on 2015 US DOT data"


def get_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "low"
    elif probability < 0.6:
        return "medium"
    else:
        return "high"


def get_top_risk_factors(input_data: Dict[str, Any]) -> List[str]:
    factors = []
    if input_data.get("CARRIER_DELAY_RATE", 0) > 0.25:
        factors.append("High historical carrier delay rate")
    if input_data.get("ORIGIN_MONTHLY_DELAY_RATE", 0) > 0.25:
        factors.append("High origin airport delay rate this month")
    if input_data.get("ROUTE_DELAY_RATE", 0) > 0.25:
        factors.append("High route delay rate")
    if input_data.get("IS_PEAK_SEASON", 0) == 1:
        factors.append("Peak travel season")
    if input_data.get("IS_EARLY_MORNING", 0) == 1:
        factors.append("Early morning departure")
    if input_data.get("SCHED_DEP_HOUR", 12) in [17, 18, 19, 20]:
        factors.append("Evening departure (peak hours)")
    if input_data.get("DISTANCE", 0) > 1500:
        factors.append("Long-haul flight")
    if not factors:
        factors.append("No major risk factors identified")
    return factors[:3]


def estimate_delay_minutes(probability: float) -> float:
    if probability < 0.2:
        return 5.0
    elif probability < 0.4:
        return 15.0
    elif probability < 0.6:
        return 30.0
    elif probability < 0.8:
        return 60.0
    else:
        return 90.0