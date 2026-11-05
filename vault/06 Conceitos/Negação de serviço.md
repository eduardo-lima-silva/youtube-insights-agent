---
tags: [conceito, seguranca]
---

# Negação de serviço (DoS)

Mesmo um SELECT somente leitura pode travar o sistema (produto cartesiano, agregações gigantes).
Mitigação: limites de complexidade no [[SQL Guard]], timeout e limite de memória no [[Warehouse DuckDB]]. Apontado na [[Revisão Codex 2 - Arquitetura]].
