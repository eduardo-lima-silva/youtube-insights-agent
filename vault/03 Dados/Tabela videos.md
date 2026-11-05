---
tags: [dados, tabela]
---

# Tabela `videos`

1 linha por vídeo (último snapshot). Colunas principais: title, category, published_at, pub_hour, pub_dow, duration_s, is_short,
views, likes, comments, age_days, channel_subs, first_trending_country, hours_to_trend, n_trending_countries.
- `views` é medido em idades diferentes (`age_days`) → cuidado ao comparar.
- Junta com [[Tabela video_country]] por `video_id`. Parte do [[Warehouse DuckDB]].
