"""Compute metrics on the dataset."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics


def compute_metrics(dataset_file: str, label_col: str, prediction_col: str) -> None:
    """Compute metrics on the dataset."""
    dataset_file = Path(dataset_file)

    # Read dataset
    ds = pd.read_csv(dataset_file)
    labels = ds[label_col]
    predictions = ds[prediction_col]

    # Compute metrics
    accuracy = metrics.accuracy_score(labels, predictions)

    macro_precision = metrics.precision_score(labels, predictions, average="macro")
    weighted_precision = metrics.precision_score(
        labels, predictions, average="weighted"
    )

    macro_recall = metrics.recall_score(labels, predictions, average="macro")
    weighted_recall = metrics.recall_score(labels, predictions, average="weighted")

    macro_f1 = metrics.f1_score(labels, predictions, average="macro")
    weighted_f1 = metrics.f1_score(labels, predictions, average="weighted")

    confusion_matrix = metrics.confusion_matrix(labels, predictions)

    report = metrics.classification_report(labels, predictions, output_dict=True)
    print(report)

    # Save results
    save_metrics = dataset_file.parent / f"{dataset_file.stem}.json"
    save_plot = dataset_file.parent / f"{dataset_file.stem}.png"
    metrics.ConfusionMatrixDisplay(confusion_matrix).plot()
    plt.savefig(save_plot)
    with save_metrics.open("w") as f:
        json.dump(report, f, indent=4)


if __name__ == "__main__":
    import fire

    fire.Fire(compute_metrics)
