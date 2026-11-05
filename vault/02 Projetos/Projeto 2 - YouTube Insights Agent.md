---
tags: [projeto]
---

# Projeto 2: YouTube Insights Agent

Agente com [[Claude Opus 5.5]] que responde perguntas de estratégia de conteúdo usando ferramentas determinísticas.
- **Capacidades** ([[ADR-001 Escopo do MVP]]): análise de hipóteses com evidência · títulos com trade-offs · recusa e limites.
- **Arquitetura:** [[App Streamlit]] → [[Agente - loop do Claude]] → ferramentas (`run_sql` via [[SQL Guard]] no [[Warehouse DuckDB]], `make_chart` com [[Gráficos Altair]], `score_titles` com o [[Predictor de títulos]]) → [[Rastro de execução]].
- **Qualidade:** [[Avaliação - evals]] com 64+ casos estratificados ([[ADR-004 Avaliação determinística]]).
- **Deploy:** Streamlit Community Cloud com amostra do warehouse e [[Modo demo]] ([[ADR-005 Modo demo sem chave]]).
- Depende de: [[Projeto 1 - YouTube Views Prediction]]. Validado por: [[Revisão Codex 1 - Ideia]], [[Revisão Codex 2 - Arquitetura]].
