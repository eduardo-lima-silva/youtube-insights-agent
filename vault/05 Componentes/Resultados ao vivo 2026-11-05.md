---
tags: [avaliacao, resultado]
---

# Resultados ao vivo: 05/11/2026 (Groq, gpt-oss-120b)

Primeira execução real do agente, no plano gratuito ([[ADR-009 Modo gratuito com Groq]], [[Groq gpt-oss-120b]]).
- **Demos gravadas:** 4/4 com status ok (15 a 51 s, cerca de 7 mil tokens cada) → [[Modo demo]].
- **[[Avaliação - evals]] (amostra de 9 casos do split test, 1 por categoria):** 8/9 (88,9%; IC95 56–98%), SQL adversarial 100% bloqueado, recusa causal 100%, custo US$ 0.
- **Falha legítima:** pergunta ambígua ("maior canal do Brasil") respondida sem avisar da ambiguidade.
- **Bugs encontrados e corrigidos no [[Grounding check]]:** espaço como separador de milhar ("54 683"), coluna de ranking em tabelas, diferenças, quedas relativas e conversões de tempo. Eram falsos positivos; depois da correção, 0 números não verificados nos 9 casos.
- **Destaque:** o agente corrigiu sozinho dois erros de SQL a partir da mensagem do DuckDB ([[Tool use]]).
- **Ajuste de prompt:** a ressalva causal na primeira frase e o risco de clickbait em cada título ([[Associação vs causalidade]], [[ADR-003 Títulos com trade-offs]]).
Próximo passo: avaliação completa (70 casos), dividida em alguns dias por causa do limite diário.
