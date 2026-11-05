---
tags: [componente, avaliacao]
---

# Avaliação (`evals/`)

64+ casos estratificados ([[Agente EVAL]]): agregação, filtros, comparação, ambíguas, fora de escopo, causais,
SQL adversarial ([[Prompt injection]]), títulos, explicação. Divisão dev/test.
- Gabarito **calculado** por SQL no [[Warehouse DuckDB]] (nunca escrito à mão).
- Métricas por categoria com IC de Wilson 95%, precisão/recall de recusa, taxa de bloqueio adversarial, custo e latência.
- Decisão: [[ADR-004 Avaliação determinística]].
