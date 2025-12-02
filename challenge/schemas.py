"""
Pydantic models for request/response validation.
"""

from pydantic import BaseModel, field_validator
from typing import List


class FlightData(BaseModel):
    """Single flight prediction request."""

    OPERA: str
    TIPOVUELO: str
    MES: int

    @field_validator("MES")
    @classmethod
    def validate_mes(cls, v):
        """MES must be between 1 and 12."""
        if not isinstance(v, int) or v < 1 or v > 12:
            raise ValueError("MES must be an integer between 1 and 12")
        return v

    @field_validator("TIPOVUELO")
    @classmethod
    def validate_tipovuelo(cls, v):
        """TIPOVUELO must be either 'N' or 'I'."""
        if v not in {"N", "I"}:
            raise ValueError("TIPOVUELO must be either 'N' or 'I'")
        return v


class PredictRequest(BaseModel):
    """Request payload for prediction."""

    flights: List[FlightData]

    @field_validator("flights")
    @classmethod
    def validate_flights_not_empty(cls, v):
        """Flights list must not be empty."""
        if not v:
            raise ValueError("flights list cannot be empty")
        return v


class PredictResponse(BaseModel):
    """Response payload for prediction."""

    predict: List[int]


class HealthResponse(BaseModel):
    """Response payload for health check."""

    status: str
