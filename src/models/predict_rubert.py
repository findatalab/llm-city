from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_label_mapping(model_dir: Path) -> dict[int, str]:
    mapping_path = model_dir / "label_mapping.json"

    if mapping_path.exists():
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        return {int(k): v for k, v in mapping["id2label"].items()}

    return {
        0: "negative",
        1: "neutral",
        2: "positive",
    }


def predict_text(model_dir: Path, text: str, max_length: int = 256) -> tuple[str, list[float]]:
    if not model_dir.exists():
        raise FileNotFoundError(f"Папка модели не найдена: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    pred_id = int(probs.argmax())
    id2label = load_label_mapping(model_dir)
    pred_label = id2label[pred_id]

    return pred_label, probs.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict sentiment with trained RuBERT.")
    parser.add_argument("--text", type=str, required=True, help="Текст для классификации.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/rubert"),
        help="Папка с сохраненной RuBERT моделью.",
    )
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    pred_label, probs = predict_text(args.model_dir, args.text, max_length=args.max_length)

    print(f"Текст: {args.text}")
    print(f"Предсказанный класс: {pred_label}")
    print(f"Вероятности: {probs}")


if __name__ == "__main__":
    main()