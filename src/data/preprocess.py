from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd


LABEL_NORMALIZATION = {
    "negative": "negative",
    "positive": "positive",
    "neutral": "neutral",
    "neautral": "neutral", 
}


def normalize_text(text: str) -> str:
    text = str(text).strip()
    text = text.replace("ё", "е")
    text = re.sub(r"http\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label(label: str) -> str:
    value = str(label).strip().lower()
    return LABEL_NORMALIZATION.get(value, value)


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "review",
    label_col: Optional[str] = "sentiment",
) -> pd.DataFrame:
    if text_col not in df.columns:
        raise ValueError(f"Колонка '{text_col}' не найдена в датасете.")
    if label_col is not None and label_col not in df.columns:
        raise ValueError(f"Колонка '{label_col}' не найдена в датасете.")

    if label_col is not None:
        data = df[[text_col, label_col]].copy()
    else:
        data = df[[text_col]].copy()

    data = data.dropna(subset=[text_col])
    data[text_col] = data[text_col].astype(str).map(normalize_text)
    data = data[data[text_col].str.len() > 0]

    if label_col is not None:
        data = data.dropna(subset=[label_col])
        data[label_col] = data[label_col].astype(str).map(normalize_label)

    data = data.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    return data


def load_dataset(data_path: Path, sep: str = "\t") -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Файл не найден: {data_path}")
    return pd.read_csv(data_path, sep=sep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess RuReviews dataset.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/rureviews.csv"),
        help="Путь к сырому датасету.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/interim/cleaned_reviews.csv"),
        help="Куда сохранить очищенный датасет.",
    )
    parser.add_argument(
        "--sep",
        type=str,
        default="\t",
        help="Разделитель в исходном файле.",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="review",
        help="Название текстовой колонки.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="sentiment",
        help="Название колонки с метками.",
    )
    args = parser.parse_args()

    df = load_dataset(args.data_path, sep=args.sep)
    processed = preprocess_dataframe(df, text_col=args.text_col, label_col=args.label_col)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(args.output_path, index=False)

    print(f"Очищенный датасет сохранен в: {args.output_path}")
    print(f"Количество строк: {len(processed)}")
    print("Распределение классов:")
    print(processed[args.label_col].value_counts(dropna=False))


if __name__ == "__main__":
    main()