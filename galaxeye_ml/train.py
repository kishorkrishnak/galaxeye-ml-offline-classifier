"""Train the small CPU baseline from labelled candidate tiles in the ZIP."""

from __future__ import annotations

import argparse
import hashlib
import io
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.ensemble import RandomForestClassifier

from galaxeye_ml.features import FEATURE_VERSION, extract_features

EXPECTED_CLASSES = {
    "AnnualCrop",
    "Forest",
    "Highway",
    "Industrial",
    "Residential",
    "River",
    "SeaLake",
}


def load_candidate_tiles(dataset_zip: Path) -> tuple[np.ndarray, np.ndarray, list[str], Counter[str]]:
    features: list[np.ndarray] = []
    labels: list[str] = []
    class_counts: Counter[str] = Counter()
    with zipfile.ZipFile(dataset_zip) as archive:
        candidates = sorted(
            name
            for name in archive.namelist()
            if "/candidate_tiles/" in name and name.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if not candidates:
            raise ValueError("The ZIP contains no images under candidate_tiles/")
        for name in candidates:
            parts = Path(name).parts
            try:
                candidate_index = parts.index("candidate_tiles")
                class_name = parts[candidate_index + 1]
            except (ValueError, IndexError) as error:
                raise ValueError(f"Unexpected candidate image path: {name}") from error
            if class_name not in EXPECTED_CLASSES:
                raise ValueError(f"Unexpected candidate class folder {class_name!r} in {name}")
            try:
                image_bytes = archive.read(name)
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image.verify()
                features.append(extract_features(image_bytes))
            except (UnidentifiedImageError, OSError, ValueError) as error:
                raise ValueError(f"Could not read candidate image {name}: {error}") from error
            labels.append(class_name)
            class_counts[class_name] += 1

    missing = EXPECTED_CLASSES.difference(class_counts)
    if missing:
        raise ValueError(f"Dataset is missing candidate classes: {', '.join(sorted(missing))}")
    return np.vstack(features), np.asarray(labels), sorted(class_counts), class_counts


def train(dataset_zip: Path, output_path: Path) -> dict[str, object]:
    features, labels, classes, class_counts = load_candidate_tiles(dataset_zip)
    model = RandomForestClassifier(
        n_estimators=180,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(features, labels)
    dataset_sha256 = hashlib.sha256(dataset_zip.read_bytes()).hexdigest()
    artifact = {
        "model": model,
        "classes": classes,
        "model_version": f"rf-rgb-grid-v1-{dataset_sha256[:12]}",
        "feature_version": FEATURE_VERSION,
        "training_examples": int(len(labels)),
        "class_counts": dict(sorted(class_counts.items())),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path, compress=3)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-zip", type=Path, required=True, help="Path to the provided tile ZIP")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/landcover_model.joblib"),
        help="Where to save the local model artifact",
    )
    args = parser.parse_args()
    artifact = train(args.dataset_zip, args.output)
    print(f"Saved model: {args.output}")
    print(f"Training examples: {artifact['training_examples']}")
    print(f"Class counts: {artifact['class_counts']}")
    print(f"Model version: {artifact['model_version']}")


if __name__ == "__main__":
    main()
