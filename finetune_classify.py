"""Classify the dataset on the finetuned model."""

from transformers import pipeline
import pandas as pd
from tqdm import tqdm
from pathlib import Path


def classify(
    model_dir: str,
    dataset_file: str,
    inputs_col: str,
    save_file: str,
) -> None:
    """Classify the dataset on the finetuned model."""
    dataset_file = Path(dataset_file)

    # Read dataset
    ds = pd.read_csv(dataset_file)

    # Load model
    classifier = pipeline(task="text-classification", model=model_dir)
    model_id = model_dir.split('/')[-1]

    # Classify function
    def classify_fn(row: pd.Series) -> str:
        text = row.to_dict()[inputs_col]
        predictions = classifier(text)
        return predictions[0]['label']

    # Classification
    tqdm.pandas(desc=f"Classify with {model_id}")
    ds["classification_label"] = ds.progress_apply(classify_fn, axis=1)

    # Save results
    save_file = Path(f"results/{dataset_file.stem}/{save_file.format(model=model_id)}")
    save_file.parent.mkdir(parents=True, exist_ok=True)
    ds.to_csv(save_file, index=False)

if __name__ == "__main__":
    import fire

    fire.Fire(classify)
