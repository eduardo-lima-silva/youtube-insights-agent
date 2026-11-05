---
tags: [decisao, seguranca]
---

# ADR-007: SQL livre com guard (e não só templates)

**Contexto:** a [[Revisão Codex 2 - Arquitetura]] sugeriu intenções estruturadas + templates de SQL.
**Decisão (MVP):** manter o SQL gerado pelo LLM como caminho principal, protegido pelo [[SQL Guard]] (AST, allowlist, limites de complexidade, read-only, timeout) e auditado no [[Rastro de execução]].
**Por quê:** é exatamente a habilidade que a [[Vaga - AI Engineer]] quer ver (LLM + ferramentas + segurança), e a [[Avaliação - evals]] mede o risco.
**Futuro:** uma ferramenta `query_metric` com intenções estruturadas para as perguntas mais comuns, deixando o SQL livre como modo avançado.
