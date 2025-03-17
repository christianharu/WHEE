"""Finetune the model on the dataset."""

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
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

    # Prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples[inputs_col], padding="max_length", truncation=True)

    tokenized_ds = ds.map(tokenize_function, batched=True)
    train_dataset = tokenized_ds["train"].select(range(n)) if n is not None else tokenized_ds["train"]
    eval_dataset = tokenized_ds["validation"].select(range(n)) if n is not None else tokenized_ds["validation"]
    test_dataset = tokenized_ds["test"].select(range(n)) if n is not None else tokenized_ds["test"]

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
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        num_train_epochs=10,
        bf16=True,
        optim="adamw_torch_fused",
        learning_rate=5e-5,
        warmup_steps=500,
        logging_strategy="steps",
        logging_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Finetune
    trainer.train()

    # Predict
    predictions = trainer.predict(test_dataset)
    print(predictions.metrics)
    test_dataset = test_dataset.add_column("prediction", predictions.predictions.argmax(-1))
    test_dataset.to_csv(f'{output_dir}/test_results.csv')

if __name__ == "__main__":
    import fire

    fire.Fire(finetune)
