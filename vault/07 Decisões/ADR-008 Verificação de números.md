---
tags: [decisao, avaliacao]
---

# ADR-008: Verificação determinística de números

**Decisão:** depois da resposta final, extrair os números do texto e conferir se aparecem nos resultados das ferramentas daquela execução ([[Grounding check]]).
**Por quê:** a regra "números só das ferramentas" fica no prompt, mas precisa de verificação no código. Vira métrica de "números alucinados" na [[Avaliação - evals]] e um alerta no [[App Streamlit]].
