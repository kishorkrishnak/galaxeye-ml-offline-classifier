"""FastAPI application for offline tile classification."""

from __future__ import annotations

import hashlib
import io
import os
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel, ConfigDict
from PIL import Image, UnidentifiedImageError

from galaxeye_ml.classifier import LocalClassifier
from galaxeye_ml.storage import PredictionStore

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "artifacts" / "landcover_model.joblib"))
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", ROOT / "var" / "predictions.sqlite3"))
TILE_DIRECTORY = Path(os.getenv("TILE_DIRECTORY", ROOT / "var" / "tiles"))
THRESHOLD_OVERRIDE = os.getenv("UNCERTAINTY_THRESHOLD")
UNCERTAINTY_THRESHOLD = float(THRESHOLD_OVERRIDE) if THRESHOLD_OVERRIDE is not None else None
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_IMAGE_SIDE = 4096
ALLOWED_FORMATS = {"PNG", "JPEG"}


class PredictionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    image_sha256: str
    predicted_class: str
    confidence: float
    status: str
    model_version: str
    created_at: str


class PredictionListResponse(BaseModel):
    items: list[PredictionResponse]
    count: int


def _response(row: dict[str, Any]) -> PredictionResponse:
    return PredictionResponse(**row)


def create_app(
    *,
    model_path: Path = MODEL_PATH,
    database_path: Path = DATABASE_PATH,
    tile_directory: Path = TILE_DIRECTORY,
    uncertainty_threshold: float | None = UNCERTAINTY_THRESHOLD,
) -> FastAPI:
    if uncertainty_threshold is not None and not 0.0 <= uncertainty_threshold <= 1.0:
        raise ValueError("uncertainty_threshold must be between 0 and 1")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.classifier = LocalClassifier(model_path)
        app.state.store = PredictionStore(database_path)
        tile_directory.mkdir(parents=True, exist_ok=True)
        yield

    app = FastAPI(
        title="GalaxEye Offline Tile Classifier",
        version="0.1.0",
        description="Classify satellite tiles with a local CPU model and save results in SQLite.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": "loaded"}

    @app.post("/tiles", response_model=PredictionResponse, status_code=201)
    async def classify_tile(
        file: Annotated[UploadFile, File(...)], response: Response
    ) -> PredictionResponse:
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds the 8 MiB upload limit")

        try:
            with Image.open(io.BytesIO(contents)) as image:
                if image.format not in ALLOWED_FORMATS:
                    raise HTTPException(status_code=415, detail="Only PNG and JPEG images are supported")
                if image.width > MAX_IMAGE_SIDE or image.height > MAX_IMAGE_SIDE:
                    raise HTTPException(status_code=413, detail="Image dimensions may not exceed 4096x4096")
                image.verify()
        except HTTPException:
            raise
        except Image.DecompressionBombError:
            raise HTTPException(status_code=413, detail="Image dimensions are too large")
        except (UnidentifiedImageError, OSError, ValueError):
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid PNG or JPEG image")

        digest = hashlib.sha256(contents).hexdigest()
        store: PredictionStore = app.state.store
        classifier: LocalClassifier = app.state.classifier
        existing = store.get_by_hash_and_model(digest, classifier.version)
        if existing:
            response.status_code = 200
            return _response(existing)

        review_threshold = (
            classifier.uncertainty_threshold
            if uncertainty_threshold is None else uncertainty_threshold
        )
        try:
            predicted_class, confidence = classifier.predict(contents)
        except (UnidentifiedImageError, OSError, ValueError):
            raise HTTPException(status_code=400, detail="Image could not be decoded for classification")

        tile_directory.mkdir(parents=True, exist_ok=True)
        image_path = tile_directory / f"{digest}.png"
        temporary_path = tile_directory / f".{digest}.{uuid.uuid4().hex}.tmp"
        created_image = False
        try:
            if not image_path.exists():
                with Image.open(io.BytesIO(contents)) as image:
                    image.convert("RGB").save(temporary_path, format="PNG")
                try:
                    os.link(temporary_path, image_path)
                    created_image = True
                except FileExistsError:
                    pass
        finally:
            temporary_path.unlink(missing_ok=True)

        row = {
            "id": str(uuid.uuid4()),
            "image_sha256": digest,
            "image_path": str(image_path),
            "predicted_class": predicted_class,
            "confidence": confidence,
            "status": "classified" if confidence >= review_threshold else "uncertain",
            "model_version": classifier.version,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            saved = store.create(row)
        except sqlite3.IntegrityError:
            existing = store.get_by_hash_and_model(digest, classifier.version)
            if existing:
                response.status_code = 200
                return _response(existing)
            if created_image:
                image_path.unlink(missing_ok=True)
            raise
        except Exception:
            if created_image:
                image_path.unlink(missing_ok=True)
            raise
        return _response(saved)

    @app.get("/predictions", response_model=PredictionListResponse)
    def list_predictions(
        class_name: Annotated[str | None, Query(alias="class")] = None,
        status: Annotated[str | None, Query(pattern="^(classified|uncertain)$")] = None,
        image_sha256: Annotated[str | None, Query(pattern="^[0-9a-f]{64}$")] = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
    ) -> PredictionListResponse:
        store: PredictionStore = app.state.store
        items = [_response(row) for row in store.list(
            class_name=class_name, status=status, image_sha256=image_sha256, limit=limit
        )]
        return PredictionListResponse(items=items, count=len(items))

    @app.get("/predictions/{prediction_id}", response_model=PredictionResponse)
    def get_prediction(prediction_id: str) -> PredictionResponse:
        store: PredictionStore = app.state.store
        row = store.get(prediction_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Prediction not found")
        return _response(row)

    return app


app = create_app()
