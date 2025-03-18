"""Finetune the model on the dataset."""

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
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
    ds = ds.select(range(n)) if n is not None else ds
    labels = sorted(ds["train"].unique(labels_col))
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for label, i in label2id.items()}

    # Prepare dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorWithPadding(tokenizer, padding=True, pad_to_multiple_of=8, return_tensors="pt")

    def tokenize_function(examples):
        return tokenizer(examples[inputs_col], truncation=True)

    tokenized_ds = ds.map(tokenize_function, batched=True)
    train_dataset = tokenized_ds["train"]
    eval_dataset = tokenized_ds["validation"]
    test_dataset = tokenized_ds["test"]

    # Define model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
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
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Finetune
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    import fire

    fire.Fire(finetune)
