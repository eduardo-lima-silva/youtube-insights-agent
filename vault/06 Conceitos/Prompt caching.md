---
tags: [conceito, llm]
---

# Prompt caching

O prefixo estável (ferramentas + system prompt com o [[Catálogo semântico]]) fica em cache, e as leituras custam cerca de 10% do preço.
Exige um prompt determinístico (sem data/hora). Reduz o custo da [[Avaliação - evals]]. Ver [[ADR-006 Modelo e custo]].
