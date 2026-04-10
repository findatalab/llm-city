from __future__ import annotations

import argparse
from pathlib import Path

import joblib


def predict_text(model_path: Path, text: str) -> str:
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")

    model = joblib.load(model_path)
    prediction = model.predict([text])[0]
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict sentiment with TF-IDF model.")
    parser.add_argument("--text", type=str, required=True, help="Текст для классификации.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/tfidf/svm_pipeline.joblib"),
        help="Путь к сохраненной TF-IDF модели.",
    )
    args = parser.parse_args()

    pred = predict_text(args.model_path, args.text)

    print(f"Текст: {args.text}")
    print(f"Предсказанный класс: {pred}")


if __name__ == "__main__":
    main()