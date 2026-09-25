"""Measure the trained model on the supplied eval_set and labels."""

from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path

from galaxeye_ml.classifier import LocalClassifier


def evaluate(dataset_zip: Path, model_path: Path) -> tuple[float, int, int]:
    classifier = LocalClassifier(model_path)
    with zipfile.ZipFile(dataset_zip) as archive:
        labels_path = next(
            (name for name in archive.namelist() if name.endswith("/eval_labels.csv")), None
        )
        if labels_path is None:
            raise ValueError("The ZIP does not contain eval_labels.csv")
        labels = {
            row["filename"]: row["true_label"]
            for row in csv.DictReader(archive.read(labels_path).decode("utf-8").splitlines())
        }
        base = labels_path.rsplit("/", 1)[0]
        evaluated = 0
        correct = 0
        per_class: dict[str, list[int]] = {}
        for filename, expected in labels.items():
            image_name = f"{base}/eval_set/{filename}"
            predicted, _confidence = classifier.predict(archive.read(image_name))
            evaluated += 1
            correct += predicted == expected
            counts = per_class.setdefault(expected, [0, 0])
            counts[0] += predicted == expected
            counts[1] += 1

    accuracy = correct / evaluated if evaluated else 0.0
    for class_name, (class_correct, class_total) in sorted(per_class.items()):
        print(f"{class_name}: {class_correct}/{class_total} ({class_correct / class_total:.1%})")
    return accuracy, correct, evaluated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-zip", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/landcover_model.joblib"))
    args = parser.parse_args()
    accuracy, correct, evaluated = evaluate(args.dataset_zip, args.model)
    print(f"Overall accuracy: {correct}/{evaluated} ({accuracy:.1%})")


if __name__ == "__main__":
    main()
