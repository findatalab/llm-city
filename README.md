# llm-city
Virtual model of a city in which residents are represented using large language models (LLM).

Baseline модели для анализа тональности русскоязычных отзывов  
(negative / neutral / positive).

Dataset: RuReviews  
https://github.com/sismetanin/rureviews

# Модели
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- RuBERT (DeepPavlov/rubert-base-cased)

# Результаты

TF-IDF + Linear SVM: accuracy ≈ 0.70,  
TF-IDF + Logistic Regression: accuracy ≈ 0.73,  
RuBERT: accuracy ≈ 0.78.

# Запуск

```bash
pip install -r requirements.txt