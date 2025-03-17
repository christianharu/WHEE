"""Finetune the model on the dataset."""

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def finetune(
    model_name: str,
    dataset_files: dict[str, str],
    inputs_col: str,
    labels_col: str,
    output_dir: str,
    *,
    n: int | None = None,
) -> None:
    """Finetune the model on the dataset."""
    # Read dataset
    ds = load_dataset("csv", data_files=dataset_files)
    num_labels = len(ds.unique(labels_col))
    print(num_labels)

    # Prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples[inputs_col], padding="max_length", truncation=True)

    tokenized_ds = ds.map(tokenize_function, batched=True)

    # Define model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    # Prepare trainer
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    train_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_ds["train"].select(range(n)),
        eval_dataset=tokenized_ds["validation"].select(range(n)),
        test_dataset=tokenized_ds["test"].select(range(n)),
        compute_metrics=compute_metrics,
    )

    # Finetune
    trainer.train()


if __name__ == "__main__":
    import fire

    fire.Fire(finetune)
