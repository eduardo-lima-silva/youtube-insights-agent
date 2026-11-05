---
tags: [revisao]
---

# Revisão Codex 1: Ideia (nota 8/10)

Texto completo: `docs/01_codex_review.md`.
- **Forte**, porque une API de LLM, ferramentas, SQL, ML e avaliação → [[Vaga - AI Engineer]].
- **Riscos:** [[Viés de seleção]], [[Vazamento de dados]], títulos frágeis ([[ADR-003 Títulos com trade-offs]]), escopo grande ([[ADR-001 Escopo do MVP]]), segurança do SQL ([[ADR-002 SQL seguro em camadas]]).
- **Alternativas:** (1) agente de hipóteses com evidência (adotada como núcleo), (2) títulos multicritério (versão reduzida), (3) copiloto de monitoramento de tendências (futuro).
- **Avaliação:** 60–100 casos estratificados, sem "acurácia genérica" → [[Avaliação - evals]].
- **Demo "uau":** resposta com evidência, recusa causal e títulos com trade-offs → [[Roteiro da demo]], [[Rastro de execução]].
