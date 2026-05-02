from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
import joblib
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
TFIDF_MODEL_PATH = BASE_DIR / "models" / "tfidf" / "svm_pipeline.joblib"
BERT_MODEL_DIR = BASE_DIR / "models" / "rubert"

TFIDF_MODEL: Any | None = None
TFIDF_LOAD_ERROR: str | None = None
BERT_TOKENIZER: Any | None = None
BERT_MODEL: Any | None = None
BERT_DEVICE: torch.device | None = None
BERT_LOAD_ERROR: str | None = None


def normalize_label(label: Any) -> str:
    label_map = {
        "negative": "Негативный",
        "neutral": "Нейтральный",
        "positive": "Позитивный",
        "NEGATIVE": "Негативный",
        "NEUTRAL": "Нейтральный",
        "POSITIVE": "Позитивный",
        "LABEL_0": "Негативный",
        "LABEL_1": "Нейтральный",
        "LABEL_2": "Позитивный",
        0: "Негативный",
        1: "Нейтральный",
        2: "Позитивный",
        "0": "Негативный",
        "1": "Нейтральный",
        "2": "Позитивный",
    }
    return label_map.get(label, str(label))


def load_tfidf_model() -> Any | None:
    global TFIDF_MODEL, TFIDF_LOAD_ERROR

    if TFIDF_MODEL is not None:
        return TFIDF_MODEL

    if not TFIDF_MODEL_PATH.exists():
        TFIDF_LOAD_ERROR = f"Файл не найден: {TFIDF_MODEL_PATH}"
        return None

    try:
        TFIDF_MODEL = joblib.load(TFIDF_MODEL_PATH)
        TFIDF_LOAD_ERROR = None
        return TFIDF_MODEL
    except Exception as exc:
        TFIDF_LOAD_ERROR = str(exc)
        return None


def load_bert_model() -> tuple[Any | None, Any | None]:
    global BERT_TOKENIZER, BERT_MODEL, BERT_DEVICE, BERT_LOAD_ERROR

    if BERT_TOKENIZER is not None and BERT_MODEL is not None:
        return BERT_TOKENIZER, BERT_MODEL

    if not BERT_MODEL_DIR.exists():
        BERT_LOAD_ERROR = f"Папка не найдена: {BERT_MODEL_DIR}"
        return None, None

    try:
        BERT_TOKENIZER = AutoTokenizer.from_pretrained(BERT_MODEL_DIR)
        BERT_MODEL = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
        BERT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        BERT_MODEL.to(BERT_DEVICE)
        BERT_MODEL.eval()
        BERT_LOAD_ERROR = None
        return BERT_TOKENIZER, BERT_MODEL
    except Exception as exc:
        BERT_LOAD_ERROR = str(exc)
        return None, None


def predict_tfidf(text: str) -> tuple[str, str]:
    model = load_tfidf_model()
    if model is None:
        details = TFIDF_LOAD_ERROR or f"Файл не найден: {TFIDF_MODEL_PATH}"
        return "TF-IDF модель не найдена. Сначала обучите модель.", details

    try:
        raw_label = model.predict([text])[0]
        prediction = normalize_label(raw_label)
        details = f"Модель: TF-IDF\nКласс: {raw_label}"
        return prediction, details
    except Exception as exc:
        return "Ошибка предикта TF-IDF.", f"Модель: TF-IDF\nОшибка: {exc}"


def predict_bert(text: str) -> tuple[str, str]:
    tokenizer, model = load_bert_model()
    if tokenizer is None or model is None:
        details = BERT_LOAD_ERROR or f"Папка не найдена: {BERT_MODEL_DIR}"
        return "BERT модель не найдена. Сначала обучите модель.", details

    try:
        device = BERT_DEVICE or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred_id = int(torch.argmax(logits, dim=-1).item())

        id2label = getattr(model.config, "id2label", None) or {
            0: "negative",
            1: "neutral",
            2: "positive",
        }
        raw_label = id2label.get(pred_id, pred_id)
        prediction = normalize_label(raw_label)
        details = (
            f"Модель: BERT\n"
            f"Класс: {raw_label}"
        )
        return prediction, details
    except Exception as exc:
        return "Ошибка предикта BERT.", f"Модель: BERT\nОшибка: {exc}"


def predict_review(model_name: str, text: str) -> tuple[str, str]:
    if not text or not text.strip():
        return "Введите отзыв", ""

    if model_name == "TF-IDF":
        return predict_tfidf(text.strip())

    if model_name == "BERT":
        return predict_bert(text.strip())

    return "Выберите модель", f"Неизвестная модель: {model_name}"


with gr.Blocks(title="Предикт тональности отзыва") as demo:
    gr.Markdown("# Предикт тональности отзыва")
    gr.Markdown("Введите отзыв, выберите модель, получите класс тональности.")

    model_dropdown = gr.Dropdown(
        choices=["TF-IDF", "BERT"],
        value="TF-IDF",
        label="Модель",
    )
    review_text = gr.Textbox(
        label="Напишите отзыв",
        placeholder="Например: товар понравился, всё отлично",
        lines=5,
    )
    predict_button = gr.Button("Получить предикт")
    prediction_output = gr.Textbox(label="Предикт", interactive=False)
    details_output = gr.Textbox(label="Детали", interactive=False, lines=6)

    predict_button.click(
        fn=predict_review,
        inputs=[model_dropdown, review_text],
        outputs=[prediction_output, details_output],
    )


if __name__ == "__main__":
    demo.launch()
