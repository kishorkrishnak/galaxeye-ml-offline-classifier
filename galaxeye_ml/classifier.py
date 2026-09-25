"""Loading and calling the locally stored land-cover classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from galaxeye_ml.features import FEATURE_VERSION, extract_features


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
        self.version = str(artifact["model_version"])

    def predict(self, image_bytes: bytes) -> tuple[str, float]:
        features = extract_features(image_bytes).reshape(1, -1)
        probabilities = self.model.predict_proba(features)[0]
        best_index = int(np.argmax(probabilities))
        return self.classes[best_index], float(probabilities[best_index])
