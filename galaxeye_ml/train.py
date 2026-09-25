"""Train the small CPU baseline from labelled candidate tiles in the ZIP."""

from __future__ import annotations

import argparse
import io
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from galaxeye_ml.classifier import artifact_version
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
TARGET_ACCEPTED_ACCURACY = 0.80
MINIMUM_COVERAGE = 0.50
THRESHOLD_CANDIDATES = [round(step * 0.05, 2) for step in range(20)]


def new_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=180,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )


def choose_review_threshold(
    features: np.ndarray, labels: np.ndarray, classes: list[str], class_counts: Counter[str]
) -> tuple[float, dict[str, object]]:
    """Use candidate-only out-of-fold predictions to set an illustrative policy."""
    smallest_class = min(class_counts.values())
    if smallest_class < 2:
        raise ValueError("At least two candidate images per class are needed for validation")
    folds = min(5, smallest_class)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    probabilities = cross_val_predict(
        new_model(), features, labels, cv=splitter, method="predict_proba"
    )
    predictions = np.asarray(classes)[np.argmax(probabilities, axis=1)]
    scores = np.max(probabilities, axis=1)
    correct = predictions == labels
    rows: list[dict[str, float | int]] = []
    for threshold in THRESHOLD_CANDIDATES:
        accepted = scores >= threshold
        accepted_count = int(np.sum(accepted))
        rows.append(
            {
                "threshold": threshold,
                "accepted": accepted_count,
                "coverage": accepted_count / len(labels),
                "accepted_accuracy": float(np.mean(correct[accepted])) if accepted_count else 0.0,
            }
        )

    eligible = [
        row for row in rows
        if row["coverage"] >= MINIMUM_COVERAGE
        and row["accepted_accuracy"] >= TARGET_ACCEPTED_ACCURACY
    ]
    if eligible:
        selected = eligible[0]  # Lowest threshold gives the most accepted tiles.
        target_met = True
    else:
        feasible = [row for row in rows if row["coverage"] >= MINIMUM_COVERAGE]
        selected = max(feasible, key=lambda row: (row["accepted_accuracy"], row["coverage"]))
        target_met = False

    validation = {
        "method": f"{folds}-fold stratified out-of-fold candidate predictions",
        "examples": int(len(labels)),
        "overall_accuracy": float(np.mean(correct)),
        "target_accepted_accuracy": TARGET_ACCEPTED_ACCURACY,
        "minimum_coverage": MINIMUM_COVERAGE,
        "target_met": target_met,
        "selected": selected,
        "threshold_rows": rows,
    }
    return float(selected["threshold"]), validation


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
    threshold, validation = choose_review_threshold(features, labels, classes, class_counts)
    model = new_model()
    model.fit(features, labels)
    artifact = {
        "model": model,
        "classes": classes,
        "feature_version": FEATURE_VERSION,
        "uncertainty_threshold": threshold,
        "validation": validation,
        "training_examples": int(len(labels)),
        "class_counts": dict(sorted(class_counts.items())),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path, compress=3)
    artifact["model_version"] = artifact_version(output_path)
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
    validation = artifact["validation"]
    selected = validation["selected"]
    print(f"Validation: {validation['method']}, overall accuracy {validation['overall_accuracy']:.1%}")
    print(
        f"Review threshold: {artifact['uncertainty_threshold']:.2f}; "
        f"accepted {selected['accepted']}/{validation['examples']} "
        f"({selected['coverage']:.1%}) with "
        f"{selected['accepted_accuracy']:.1%} accuracy"
    )


if __name__ == "__main__":
    main()
