# llm-city

MVP-проект для анализа тональности русскоязычных отзывов с использованием TF-IDF baseline и модели RuBERT.

Проект поддерживает классификацию отзывов по трём классам:
- `negative`
- `neutral`
- `positive`

## О проекте

Реализованы:
- предобработка датасета
- TF-IDF + Linear SVM
- TF-IDF + Logistic Regression
- RuBERT (`DeepPavlov/rubert-base-cased`) с fine-tuning на датасете RuReviews
- отдельные скрипты для инференса и обучения

Проект приведён к состоянию MVP: обученные модели уже сохранены, поэтому классификацию можно запускать без повторного обучения.

## Структура проекта

```text
llm-city/
├─ data/
│  ├─ raw/
│  │  └─ rureviews.csv
│  └─ interim/
│     └─ cleaned_reviews.csv
├─ models/
│  ├─ tfidf/
│  │  └─ svm_pipeline.joblib
│  └─ rubert/
├─ src/
│  ├─ __init__.py
│  ├─ data/
│  │  ├─ __init__.py
│  │  └─ preprocess.py
│  └─ models/
│     ├─ __init__.py
│     ├─ train_tfidf.py
│     ├─ predict_tfidf.py
│     ├─ train_rubert.py
│     └─ predict_rubert.py
├─ .gitignore
├─ LICENSE
├─ README.md
└─ requirements.txt
```

## Датасет

Используется датасет **RuReviews**:

- https://github.com/sismetanin/rureviews

Ожидаемые колонки:
- `review`
- `sentiment`

В датасете может встречаться метка `neautral` с опечаткой. В проекте она автоматически преобразуется в `neutral`.

## Требования

- Python 3.11+
- зависимости из `requirements.txt`

Установка:

```bash
pip install -r requirements.txt
```

## Быстрый запуск 

Если модели уже находятся в `models/tfidf/` и `models/rubert/`

### Инференс TF-IDF

```bash
python -m src.models.predict_tfidf --text "Очень хороший товар, мне понравилось" --model-path models/tfidf/svm_pipeline.joblib
```

### Инференс RuBERT

```bash
python -m src.models.predict_rubert --text "Очень хороший товар, рекомендую" --model-dir models/rubert
```

## Результаты

### TF-IDF + Linear SVM
    Accuracy: 0.7273
    Macro F1: 0.7267

### RuBERT
    Accuracy: 0.7750
    Macro F1: 0.7768

RuBERT показал более высокое качество по сравнению с baseline-моделью на TF-IDF.

## Переобучение моделей при необходимости
### Предобработка датасета

```bash
python -m src.data.preprocess --data-path data/raw/rureviews.csv --output-path data/interim/cleaned_reviews.csv --sep "\t"
```

### Обучение TF-IDF

```bash
python -m src.models.train_tfidf --data-path data/interim/cleaned_reviews.csv --model-dir models/tfidf --model-name svm
```

### Обучение RuBERT

```bash
python -m src.models.train_rubert --data-path data/interim/cleaned_reviews.csv --output-dir models/rubert --model-name DeepPavlov/rubert-base-cased --epochs 2 --train-batch-size 8 --eval-batch-size 8
```
