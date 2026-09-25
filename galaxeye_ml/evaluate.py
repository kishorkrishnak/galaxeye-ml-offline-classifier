"""Report model quality on the supplied eval set."""

from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from galaxeye_ml.classifier import LocalClassifier


def evaluate(dataset_zip: Path, model_path: Path) -> dict[str, object]:
    classifier = LocalClassifier(model_path)
    true_labels: list[str] = []
    predicted_labels: list[str] = []
    scores: list[float] = []
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
        for filename, expected in labels.items():
            image_name = f"{base}/eval_set/{filename}"
            predicted, score = classifier.predict(archive.read(image_name))
            true_labels.append(expected)
            predicted_labels.append(predicted)
            scores.append(score)

    if not true_labels:
        raise ValueError("The evaluation set is empty")
    classes = classifier.classes
    matrix = confusion_matrix(true_labels, predicted_labels, labels=classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=classes, zero_division=0
    )
    correct = np.asarray(true_labels) == np.asarray(predicted_labels)
    accepted = np.asarray(scores) >= classifier.uncertainty_threshold
    accepted_count = int(np.sum(accepted))
    return {
        "model_version": classifier.version,
        "classes": classes,
        "matrix": matrix.tolist(),
        "per_class": [
            {
                "class_name": class_name,
                "precision": float(class_precision),
                "recall": float(class_recall),
                "f1": float(class_f1),
                "support": int(class_support),
            }
            for class_name, class_precision, class_recall, class_f1, class_support in zip(
                classes, precision, recall, f1, support, strict=True
            )
        ],
        "examples": len(true_labels),
        "correct": int(np.sum(correct)),
        "accuracy": float(np.mean(correct)),
        "threshold": classifier.uncertainty_threshold,
        "accepted": accepted_count,
        "coverage": accepted_count / len(true_labels),
        "accepted_correct": int(np.sum(correct[accepted])),
        "accepted_accuracy": float(np.mean(correct[accepted])) if accepted_count else 0.0,
        "validation": classifier.validation,
    }


def render_markdown(result: dict[str, object]) -> str:
    classes = result["classes"]
    validation = result["validation"]
    selected = validation["selected"]
    lines = [
        "# Model evaluation",
        "",
        f"Model: `{result['model_version']}`. Training and threshold selection used only "
        f"the {validation['examples']} labelled `candidate_tiles`. The "
        f"{result['examples']} labelled `eval_set` tiles were used for reporting, "
        "not model fitting or threshold selection.",
        "",
        "## Review threshold from candidate tiles",
        "",
        f"The training script used {validation['method']} ({validation['examples']} examples). "
        f"Its overall accuracy was {validation['overall_accuracy']:.1%}. "
        "The illustrative policy selects the lowest threshold that reaches at least "
        f"{validation['target_accepted_accuracy']:.0%} accuracy among automatically classified tiles "
        f"while keeping at least {validation['minimum_coverage']:.0%} of tiles automatic. "
        "If no threshold meets both targets, the script selects the highest accepted accuracy "
        "at the required minimum coverage and reports that the target was missed.",
        "",
        "| Score threshold | Automatic tiles | Coverage | Accuracy of automatic tiles |",
        "|---:|---:|---:|---:|",
    ]
    shown_thresholds = {0.0, 0.35, 0.4, 0.45, 0.5, 0.6, result["threshold"]}
    for row in validation["threshold_rows"]:
        if row["threshold"] in shown_thresholds:
            mark = " **selected**" if row["threshold"] == result["threshold"] else ""
            lines.append(
                f"| {row['threshold']:.2f}{mark} | {row['accepted']}/{validation['examples']} "
                f"| {row['coverage']:.1%} | {row['accepted_accuracy']:.1%} |"
            )
    lines.extend(
        [
            "",
            f"Chosen threshold: **{result['threshold']:.2f}**, with "
            f"{selected['accepted']}/{validation['examples']} ({selected['coverage']:.1%}) "
            f"candidate validation tiles classified automatically at "
            f"{selected['accepted_accuracy']:.1%} accuracy. "
            f"Target met: **{'yes' if validation['target_met'] else 'no'}**. "
            "This is a review policy chosen from observed scores, not a calibrated probability or a business requirement supplied by GalaxEye.",
            "",
            "## Final evaluation set",
            "",
            f"Overall: **{result['correct']}/{result['examples']} correct "
            f"({result['accuracy']:.1%})**. At the chosen threshold, "
            f"{result['accepted']}/{result['examples']} tiles ({result['coverage']:.1%}) "
            f"were classified automatically; {result['accepted_correct']}/{result['accepted']} "
            f"of those were correct ({result['accepted_accuracy']:.1%}).",
            "",
            "| True class | Precision | Recall | F1 | Tiles |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in result["per_class"]:
        lines.append(
            f"| {row['class_name']} | {row['precision']:.1%} | {row['recall']:.1%} "
            f"| {row['f1']:.1%} | {row['support']} |"
        )
    lines.extend(
        [
            "",
            "### Confusion matrix",
            "",
            "Rows are true classes; columns are predicted classes. The diagonal contains correct predictions.",
            "",
            "| True \\ Predicted | " + " | ".join(classes) + " |",
            "|---|" + "---:|" * len(classes),
        ]
    )
    for class_name, row in zip(classes, result["matrix"], strict=True):
        lines.append("| " + class_name + " | " + " | ".join(map(str, row)) + " |")
    mistakes = sorted(
        (
            (int(count), actual, predicted)
            for actual, row in zip(classes, result["matrix"], strict=True)
            for predicted, count in zip(classes, row, strict=True)
            if actual != predicted and count
        ),
        reverse=True,
    )
    if mistakes:
        lines.append("")
        lines.append(
            "Most common mistakes: "
            + "; ".join(
                f"{actual} predicted as {predicted} ({count})"
                for count, actual, predicted in mistakes[:3]
            )
            + "."
        )
    lines.extend(
        [
            "",
            "This evaluation split was inspected during v1 development, so it is not a blind "
            "external test. The tiles lack location metadata, and the result cannot establish "
            "performance in new geographic regions or seasons. Reconsider the threshold once "
            "the costs of wrong labels and analyst review capacity are known.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-zip", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/landcover_model.joblib"))
    parser.add_argument("--report", type=Path, help="Write a Markdown evaluation report")
    args = parser.parse_args()
    report = render_markdown(evaluate(args.dataset_zip, args.model))
    if args.report is not None:
        args.report.write_text(report)
        print(f"Saved evaluation report: {args.report}")
    else:
        print(report)


if __name__ == "__main__":
    main()
