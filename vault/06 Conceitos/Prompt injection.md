---
tags: [conceito, seguranca]
---

# Prompt injection

Tentativas de fazer o agente ignorar regras ("rode DROP TABLE", "leia um arquivo com read_csv").
Mitigação: o [[SQL Guard]] bloqueia no código, sem depender do modelo obedecer. Testado na [[Avaliação - evals]].
