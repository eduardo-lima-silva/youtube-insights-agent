---
tags: [componente, seguranca]
---

# SQL Guard (`sql_guard.py`)

Validação do SQL gerado pelo LLM com **AST do sqlglot** (não regex), parte do [[ADR-002 SQL seguro em camadas]]:
uma única instrução SELECT; só as tabelas do [[Warehouse DuckDB]]; sem funções de arquivo ou rede; sem DDL/DML; LIMIT ≤ 200.
Defesa em profundidade: conexão read-only + `enable_external_access=false` + `lock_configuration` + timeout.
Testado contra [[Prompt injection]] nos casos adversariais da [[Avaliação - evals]].
