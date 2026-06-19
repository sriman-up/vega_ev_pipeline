# api/schemas.py
"""Pydantic request/response models for the site-simulator API (see api/main.py)."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    lat: float
    lon: float
    chargers: Dict[str, int] = Field(..., examples=[{"60": 2, "120": 1}])
    prediction_month: Optional[str] = Field(None, description="'YYYY-MM' or 'YYYY-MM-01'. Defaults to next calendar month.")
    contracted_load_kva: Optional[float] = None
    has_attached_restaurant: Optional[bool] = Field(None, description="Override — omit to use live Places data")
    location_type: Optional[str] = None
    direction_side: Optional[str] = Field(None, description="Override — see /predict response's direction_side for the derived value")
    fetch_competition: bool = Field(True, description="Set false to skip the Google Places lookup (saves quota)")
    model_version: str = "latest"


class PredictResponse(BaseModel):
    prediction_month: str
    model_version: str
    predicted_kwh: float
    predicted_kwh_lower: float
    predicted_kwh_upper: float
    peer_avg_kwh: Optional[float] = None
    peer_group_size: int
    peer_group_criteria: str
    zone_band: Optional[str] = None
    nearest_city: Optional[str] = None
    direction_side: Optional[str] = None
    has_attached_restaurant: Optional[bool] = None
    location_type: Optional[str] = None
    total_power_kw: float
    contracted_load_kva: Optional[float] = None
    caveats: List[str]


class ScanRequest(BaseModel):
    lat: float
    lon: float
    charger_grid: List[Dict[str, int]] = Field(..., examples=[[{"30": 4}, {"60": 2, "120": 1}, {"150": 1, "60": 2}]])
    prediction_month: Optional[str] = None
    restaurant_options: Optional[List[Optional[bool]]] = Field(None, description="e.g. [true, false] to compare both. Omit to use live Places data only.")
    location_type_options: Optional[List[Optional[str]]] = None
    contracted_load_options: Optional[List[Optional[float]]] = None
    scan_direction: bool = Field(False, description="Try both highway directions for this site, if it's on a tracked highway pair")
    fetch_competition: bool = True
    model_version: str = "latest"


class ScanRow(BaseModel):
    chargers: str
    total_power_kw: float
    contracted_load_kva: float
    has_attached_restaurant: Optional[bool] = None
    direction_side: Optional[str] = None
    location_type: Optional[str] = None
    predicted_kwh: float
    predicted_kwh_lower: float
    predicted_kwh_upper: float
    peer_avg_kwh: Optional[float] = None
    peer_group_size: int


class HealthResponse(BaseModel):
    status: str
    db_reachable: bool
    model_loaded: bool
