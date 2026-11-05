---
tags: [entrevista]
---

# Perguntas de entrevista

Preparação baseada na [[Revisão Codex 1 - Ideia]] e na [[Revisão Codex 2 - Arquitetura]].
- Por que agente e não um pipeline fixo? → [[Tool use]], [[ADR-001 Escopo do MVP]]
- Como garante que o SQL é seguro? → [[SQL Guard]], [[ADR-002 SQL seguro em camadas]], [[Negação de serviço]], [[ADR-007 SQL livre com guard]]
- E se o modelo inventar um número? → [[Grounding check]], [[ADR-008 Verificação de números]]
- Como sabe que o agente acerta? → [[Avaliação - evals]], [[ADR-004 Avaliação determinística]]
- O modelo prevê qualquer vídeo? → [[Viés de seleção]]
- O título "causa" mais views? → [[Associação vs causalidade]]
- Há vazamento no modelo? → [[Vazamento de dados]], [[Validação temporal]]
- Quanto custa por pergunta? → [[ADR-006 Modelo e custo]], [[Prompt caching]]
