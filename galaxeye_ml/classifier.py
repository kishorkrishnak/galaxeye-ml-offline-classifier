"""Loading and calling the locally stored land-cover classifier."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from galaxeye_ml.features import FEATURE_VERSION, extract_features


def artifact_version(artifact_path: Path) -> str:
    """Identify the exact serialized artifact used for inference."""
    digest = hashlib.sha256()
    with artifact_path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"artifact-sha256-{digest.hexdigest()}"


class LocalClassifier:
    def __init__(self, artifact_path: Path) -> None:
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Model artifact not found at {artifact_path}. "
                "Run: python -m galaxeye_ml.train --dataset-zip <path-to-tiles.zip>"
            )
        artifact: dict[str, Any] = joblib.load(artifact_path)
        if artifact.get("feature_version") != FEATURE_VERSION:
            raise ValueError("Model artifact uses an unsupported feature version")
        self.model = artifact["model"]
        self.classes = list(artifact["classes"])
        self.version = artifact_version(artifact_path)
        self.uncertainty_threshold = float(artifact["uncertainty_threshold"])
        self.validation = artifact["validation"]

    def predict(self, image_bytes: bytes) -> tuple[str, float]:
        features = extract_features(image_bytes).reshape(1, -1)
        probabilities = self.model.predict_proba(features)[0]
        best_index = int(np.argmax(probabilities))
        return self.classes[best_index], float(probabilities[best_index])
