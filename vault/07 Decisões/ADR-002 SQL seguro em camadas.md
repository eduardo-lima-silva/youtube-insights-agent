---
tags: [decisao, seguranca]
---

# ADR-002: SQL seguro em camadas

**Decisão:** AST do sqlglot + allowlist + conexão read-only + sem acesso externo + configuração travada + LIMIT + timeout.
**Por quê:** "somente leitura" não basta, porque o DuckDB pode ler arquivos locais (`read_csv`). Ver [[SQL Guard]], [[Prompt injection]].
