---
tags: [dados]
---

# Warehouse DuckDB

`data/warehouse.duckdb`, gerado por `scripts/build_warehouse.py` ([[Agente DADOS]]). São as **únicas** tabelas visíveis ao LLM:
[[Tabela videos]] · [[Tabela video_country]] · [[Tabela daily_trending]].
- Aberto em modo read-only, sem acesso externo e com configuração travada → [[SQL Guard]], [[ADR-002 SQL seguro em camadas]].
- Descrito ao modelo pelo [[Catálogo semântico]].
- Versão `warehouse_sample.duckdb` (<90 MB) para o deploy do [[App Streamlit]].
