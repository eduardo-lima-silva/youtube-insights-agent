---
tags: [decisao, avaliacao]
---

# ADR-004: Avaliação determinística

**Decisão:** gabarito calculado por SQL; nota por regras (números, ranking, recusa, bloqueio); LLM-juiz opcional e desligado.
**Por quê:** a nota fica reproduzível e barata. Divisão dev/test evita ajustar o prompt para o teste. Ver [[Avaliação - evals]].
