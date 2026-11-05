---
tags: [revisao]
---

# Revisão Codex 2: Arquitetura

Texto completo: [[Codex 2 - texto completo]]. Veredito: a arquitetura incorporou bem a 1ª revisão ([[Revisão Codex 1 - Ideia]]).
**Lacunas apontadas → o que fizemos:**
- Negação de serviço é o risco residual do SQL → limites de complexidade no [[SQL Guard]] ([[Negação de serviço]]).
- Resultados de ferramentas podem carregar [[Prompt injection]] (títulos de vídeos) → marcados como dados não confiáveis no [[Agente - loop do Claude]].
- Números da resposta podem não vir das ferramentas → [[Grounding check]] ([[ADR-008 Verificação de números]]).
- Proveniência dos dados → manifesto do [[Warehouse DuckDB]].
- Rótulos do [[Predictor de títulos]] sugeriam causalidade → renomeados ("contribuição do texto", "faixa heurística").
- Avaliação sem critérios de reprovação → gates de regressão na [[Avaliação - evals]].
**Visões diferentes do Codex:** (1) trocar SQL livre por intenções estruturadas → ver [[ADR-007 SQL livre com guard]]; (2) tirar os títulos do centro do produto → já adotado no [[ADR-003 Títulos com trade-offs]].
Também gerou as [[Perguntas de entrevista]] mais difíceis.
