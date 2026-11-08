"""FastAPI app: GET /health and POST /predict."""

from __future__ import annotations

import io
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from skin_lesion.api.schemas import HealthResponse, PredictionResponse
from skin_lesion.api.service import Predictor

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "10")) * 1024 * 1024


def create_app(predictor: Predictor | None = None) -> FastAPI:
    """Build the app. Pass a predictor (tests), or one is loaded from SKIN_LESION_CHECKPOINT."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        ckpt = Path(os.environ.get("SKIN_LESION_CHECKPOINT", "checkpoints/best.pth"))
        if app.state.predictor is None and ckpt.is_file():
            app.state.predictor = Predictor.from_checkpoint(ckpt)
        yield

    app = FastAPI(title="Skin Lesion Classifier (research prototype)", lifespan=lifespan)
    app.state.predictor = predictor

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", model_loaded=app.state.predictor is not None)

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(file: UploadFile = File(...)) -> dict:
        if app.state.predictor is None:
            raise HTTPException(503, "No model loaded; set SKIN_LESION_CHECKPOINT")
        data = await file.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "File too large")
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(422, "Upload is not a valid image") from exc
        return app.state.predictor.predict(image)

    return app


app = create_app()
