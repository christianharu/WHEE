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
    raw_ds = pd.read_csv(dataset_file)
    ds = raw_ds.dropna(axis=0, subset=[label_col, prediction_col])
    if len(raw_ds) > len(ds):
        print(
            f"Computing metrics without {len(raw_ds) - len(ds)} samples because they are null"
        )

    labels = ds[label_col]
    predictions = ds[prediction_col]

    # Compute metrics
    report = metrics.classification_report(labels, predictions, output_dict=True)
    confusion_matrix = metrics.confusion_matrix(labels, predictions, normalize="true")

    # Save results
    save_metrics = dataset_file.parent / f"{dataset_file.stem}.json"
    with save_metrics.open("w") as f:
        json.dump(report, f, indent=4)

    save_plot = dataset_file.parent / f"{dataset_file.stem}.png"
    metrics.ConfusionMatrixDisplay(confusion_matrix).plot()
    plt.savefig(save_plot)


if __name__ == "__main__":
    import fire

    fire.Fire(compute_metrics)
