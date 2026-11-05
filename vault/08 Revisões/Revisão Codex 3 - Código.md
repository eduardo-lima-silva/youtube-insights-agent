---
tags: [revisao]
---

# Revisão Codex 3: Código

Texto completo: [[Codex 3 - texto completo]]. Revisão do código entregue pelos agentes ([[Agente DADOS]], [[Agente ML]], [[Agente LLM]], [[Agente EVAL]]).
**Bugs confirmados e corrigidos (com testes de regressão):**
1. `range()`/`generate_series()`/`repeat()` na forma escalar escapavam do [[SQL Guard]] → risco de [[Negação de serviço]]. Agora são bloqueados.
2. `memory_limit` e `threads` podiam falhar silenciosamente no [[Warehouse DuckDB]] → agora são verificados ao conectar.
3. O [[Agente - loop do Claude]] ignorava `tool_use` quando o `stop_reason` vinha inconsistente → agora processa.
4. A [[Avaliação - evals]] contava qualquer erro de SQL como "bloqueio do guard" → agora exige rejeição real.
5. A versão mínima do Streamlit era incompatível com o [[App Streamlit]] → `streamlit>=1.60`.
**Pontos fortes mantidos:** AST em vez de regex, conexão read-only + travas, estado isolado por execução, schemas estritos, dados marcados como não confiáveis, gates de avaliação.
**Pendências conhecidas:** allowlist de funções em vez de denylist ([[ADR-007 SQL livre com guard]]); o dry-run "oráculo" não mede o agente real; o [[Grounding check]] é só um alerta.
