from pydantic import BaseModel, Field
from typing import Literal, Optional
from datetime import datetime


class FlightInput(BaseModel):
    MONTH: int = Field(..., ge=1, le=12, description="Month of departure (1-12)")
    DAY: int = Field(..., ge=1, le=31, description="Day of month")
    DAY_OF_WEEK: int = Field(..., ge=1, le=7, description="Day of week (1=Monday, 7=Sunday)")
    DISTANCE: float = Field(..., gt=0, description="Flight distance in miles")
    SCHED_DEP_HOUR: int = Field(..., ge=0, le=23, description="Scheduled departure hour (0-23)")
    IS_PEAK_SEASON: int = Field(..., ge=0, le=1, description="1 if June/July/December, else 0")
    IS_EARLY_MORNING: int = Field(..., ge=0, le=1, description="1 if 5-7 AM departure, else 0")
    CARRIER_DELAY_RATE: float = Field(..., ge=0, le=1, description="Historical delay rate for this carrier")
    ORIGIN_MONTHLY_DELAY_RATE: float = Field(..., ge=0, le=1, description="Historical delay rate for origin airport this month")
    ROUTE_DELAY_RATE: float = Field(..., ge=0, le=1, description="Historical delay rate for this route")
    AIRLINE: str = Field(..., description="Airline code (e.g., AA, DL, UA)")
    ORIGIN_AIRPORT: str = Field(..., description="Origin airport code (e.g., JFK, LAX)")
    DESTINATION_AIRPORT: str = Field(..., description="Destination airport code (e.g., JFK, LAX)")


class PredictionOutput(BaseModel):
    delay_probability: float
    risk_level: Literal["low", "medium", "high"]
    expected_delay_minutes: float
    top_risk_factors: list[str]
    model_version: str
    prediction_time: datetime


class HealthCheck(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: Optional[float] = None


class ModelInfo(BaseModel):
    model_type: str
    validation_auc: float
    test_auc: float
    features_used: int
    training_records: int
    description: str