---
tags: [componente]
---

# Agente: loop do Claude (`agent.py`)

Loop manual de [[Tool use]] com o [[Claude Opus 5.5]] ([[Agente LLM]]):
1. a pergunta e o system prompt (com o [[Catálogo semântico]]) vão para o modelo;
2. o modelo escolhe uma ferramenta (`run_sql`, `make_chart`, `score_titles`);
3. a ferramenta roda localmente, passando pelo [[SQL Guard]] ou pelo [[Predictor de títulos]];
4. o resultado volta ao modelo, até no máximo 8 chamadas;
5. a resposta final segue o formato direta → evidência → ressalvas.
Tudo fica registrado no [[Rastro de execução]]. Recusa perguntas causais ([[Associação vs causalidade]]) e [[Prompt injection]].
