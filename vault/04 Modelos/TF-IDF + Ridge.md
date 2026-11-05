---
tags: [modelo]
---

# TF-IDF + Ridge

Modelo de NLP sobre título e tags (R² 0,38 isolado). Gera o `text_score` usado pelo [[LightGBM]] e pelo [[Predictor de títulos]].
Achado: não acrescentou ganho ao LightGBM, porque é redundante com o histórico do canal.
