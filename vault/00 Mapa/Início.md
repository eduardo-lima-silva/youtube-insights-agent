---
tags: [mapa]
---

# 🧭 Início: YouTube Insights Agent

Vault de contexto do **segundo projeto de portfólio** de [[Eduardo Lima da Silva]] para o programa de estágio
([[Vaga - AI Engineer]] e [[Vaga - Marketing AI Builder]]).

> *"O LLM interpreta a intenção e orquestra ferramentas; os números vêm de consultas e modelos determinísticos;
> o sistema expõe suas limitações e é avaliado por casos reproduzíveis."* (posicionamento, ver [[Revisão Codex 1 - Ideia]])

## Mapa
- **Objetivo:** [[Vaga - AI Engineer]] · [[Vaga - Marketing AI Builder]] · [[Currículo]] · [[Formulário da vaga]]
- **Projetos:** [[Projeto 1 - YouTube Views Prediction]] → [[Projeto 2 - YouTube Insights Agent]]
- **Dados:** [[Base YouTube Trending]] · [[Warehouse DuckDB]] · [[Tabela videos]] · [[Tabela video_country]] · [[Tabela daily_trending]]
- **Modelos:** [[LightGBM]] · [[TF-IDF + Ridge]] · [[SHAP]] · [[Claude Opus 5.5]] · [[Claude Haiku 5.5]] · [[Groq gpt-oss-120b]]
- **Componentes:** [[Agente - loop do Claude]] · [[SQL Guard]] · [[Catálogo semântico]] · [[Predictor de títulos]] · [[Gráficos Altair]] · [[App Streamlit]] · [[Avaliação - evals]] · [[Modo demo]] · [[Rastro de execução]] · [[Grounding check]]
- **Conceitos:** [[Viés de seleção]] · [[Associação vs causalidade]] · [[Vazamento de dados]] · [[Validação temporal]] · [[Tool use]] · [[Prompt caching]] · [[Prompt injection]] · [[Negação de serviço]]
- **Decisões:** [[ADR-001 Escopo do MVP]] · [[ADR-002 SQL seguro em camadas]] · [[ADR-003 Títulos com trade-offs]] · [[ADR-004 Avaliação determinística]] · [[ADR-005 Modo demo sem chave]] · [[ADR-006 Modelo e custo]] · [[ADR-007 SQL livre com guard]] · [[ADR-008 Verificação de números]] · [[ADR-009 Modo gratuito com Groq]]
- **Revisões externas:** [[Revisão Codex 1 - Ideia]] · [[Revisão Codex 2 - Arquitetura]] · [[Revisão Codex 3 - Código]]
- **Desenvolvimento:** [[Agente DADOS]] · [[Agente ML]] · [[Agente LLM]] · [[Agente EVAL]] · [[Linha do tempo]]
- **Entrevista:** [[Perguntas de entrevista]] · [[Roteiro da demo]]

## Como usar
Abra o **Graph View** (Ctrl+G). As cores separam as pastas: dados, modelos, componentes, conceitos e decisões.
