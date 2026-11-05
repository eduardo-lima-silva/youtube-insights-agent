---
tags: [projeto]
---

# Projeto 1: YouTube Views Prediction

Repositório: https://github.com/eduardo-lima-silva/youtube-views-prediction
- 9,4 M registros (19,5 GB) da [[Base YouTube Trending]] → 956 mil vídeos, 37 features.
- [[LightGBM]] com [[Validação temporal]]: R² 0,85 em log(views), 84% das previsões dentro de 2×.
- [[TF-IDF + Ridge]] isolado: R² 0,38, sem ganho no LightGBM.
- Explicado com [[SHAP]]. Limitações: [[Viés de seleção]], [[Vazamento de dados]] leve (snapshot do canal).
- Fornece dados e modelo para o [[Projeto 2 - YouTube Insights Agent]].
