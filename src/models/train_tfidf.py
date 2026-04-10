from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data.preprocess import preprocess_dataframe


def build_model(model_name: str) -> Pipeline:
    if model_name == "logreg":
        clf = LogisticRegression(max_iter=3000, n_jobs=-1)
    elif model_name == "svm":
        clf = LinearSVC()
    else:
        raise ValueError("model_name должен быть 'svm' или 'logreg'")

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            ("clf", clf),
        ]
    )
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TF-IDF sentiment classifier.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/interim/cleaned_reviews.csv"),
        help="Путь к очищенному датасету.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/tfidf"),
        help="Папка для сохранения модели.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="svm",
        choices=["svm", "logreg"],
        help="Какую модель обучать.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if not args.data_path.exists():
        raise FileNotFoundError(f"Файл не найден: {args.data_path}")

    df = pd.read_csv(args.data_path)

    if "review" not in df.columns or "sentiment" not in df.columns:
        df = preprocess_dataframe(df, text_col="review", label_col="sentiment")

    X_train, X_test, y_train, y_test = train_test_split(
        df["review"],
        df["sentiment"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["sentiment"],
    )

    model = build_model(args.model_name)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, preds, digits=4))

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / f"{args.model_name}_pipeline.joblib"
    joblib.dump(model, model_path)

    print(f"\nМодель сохранена в: {model_path}")


if __name__ == "__main__":
    main()