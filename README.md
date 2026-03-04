# llm-city
Virtual model of a city in which residents are represented using large language models (LLM).

Baseline модели для анализа тональности русскоязычных отзывов  
(negative / neutral / positive).

Dataset: RuReviews  
https://github.com/sismetanin/rureviews

# Структура проекта
.
├── baseline_tfidf_models.ipynb   # Baseline модели: TF-IDF + Logistic Regression и Linear SVM
├── rubert_finetuning.ipynb       # RuBERT для анализа тональности
├── data/                         # Директория с датасетом
│   └── data.csv                  # Файл с отзывами
├── requirements.txt              # Зависимости проекта
└── README.md                     # Описание проекта

## Требования
Python 3.11+

Установленные библиотеки из `requirements.txt`.

# Модели
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- RuBERT (DeepPavlov/rubert-base-cased)

# Результаты

TF-IDF + Linear SVM: accuracy ≈ 0.70,  
TF-IDF + Logistic Regression: accuracy ≈ 0.73,  
RuBERT: accuracy ≈ 0.78.


```bash
pip install -r requirements.txt