"""API response models."""

from __future__ import annotations

from pydantic import BaseModel

DISCLAIMER = "Research prototype. Not a medical device and not for diagnosis or clinical decisions."


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class PredictionResponse(BaseModel):
    probabilities: dict[str, float]
    top_class: str
    confidence: float
    gradcam_png_base64: str
    disclaimer: str = DISCLAIMER
