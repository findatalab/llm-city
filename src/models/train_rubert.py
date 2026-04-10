from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.data.preprocess import preprocess_dataframe


LABEL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RuBERT on RuReviews.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/interim/cleaned_reviews.csv"),
        help="Путь к очищенному датасету.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/rubert"),
        help="Папка для сохранения модели.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="DeepPavlov/rubert-base-cased",
        help="Базовая модель для fine-tuning.",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=100)
    args = parser.parse_args()

    if not args.data_path.exists():
        raise FileNotFoundError(f"Файл не найден: {args.data_path}")

    df = pd.read_csv(args.data_path)
    df = preprocess_dataframe(df, text_col="review", label_col="sentiment")
    df = df[df["sentiment"].isin(LABEL2ID.keys())].copy()

    if df.empty:
        raise ValueError("После фильтрации не осталось данных для обучения.")

    df["label"] = df["sentiment"].map(LABEL2ID)

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["label"],
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_dataset = Dataset.from_pandas(train_df[["review", "label"]], preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df[["review", "label"]], preserve_index=False)

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(
            batch["review"],
            truncation=True,
            max_length=args.max_length,
        )

    train_dataset = train_dataset.map(tokenize_batch, batched=True)
    test_dataset = test_dataset.map(tokenize_batch, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    checkpoints_dir = args.output_dir / "checkpoints"

    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    eval_metrics = trainer.evaluate()
    print("\nEvaluation metrics:")
    print(eval_metrics)

    preds_output = trainer.predict(test_dataset)
    preds = np.argmax(preds_output.predictions, axis=1)

    print("\nClassification report:")
    print(
        classification_report(
            test_df["label"],
            preds,
            target_names=["negative", "neutral", "positive"],
            digits=4,
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    label_mapping_path = args.output_dir / "label_mapping.json"
    with open(label_mapping_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label2id": LABEL2ID,
                "id2label": {str(k): v for k, v in ID2LABEL.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nRuBERT модель сохранена в: {args.output_dir}")
    print(f"Label mapping сохранен в: {label_mapping_path}")


if __name__ == "__main__":
    main()