---
tags: [decisao, llm]
---

# ADR-009: Modo gratuito com Groq

**Contexto:** o usuário pediu uma opção de agente mais barata ou gratuita. Opções pesquisadas: [[Claude Haiku 5.5]] (US$ 0,002/pergunta), Groq e Gemini (planos gratuitos) e Ollama local (modelos pequenos demais para 7,7 GB de RAM).
**Decisão:** dois caminhos baratos.
1. **Haiku 5.5:** basta trocar `INSIGHTS_MODEL`; o fallback no servidor fica desligado para ele.
2. **Groq (`openai/gpt-oss-120b`):** um provedor alternativo em `groq_agent.py`, que reaproveita ferramentas, [[SQL Guard]], [[Rastro de execução]] e [[Grounding check]].
**Por quê o prompt compacto:** o plano gratuito permite ~8 mil tokens/min e ~200 mil/dia. O prompt caiu de ~6 mil para ~3,5 mil tokens (países como texto em vez de enum, regras condensadas, resultados de ferramenta de no máximo 2.500 caracteres) e há espera automática em 429.
**Trade-off:** custo zero, mas latência de minutos e poucas dezenas de perguntas por dia. Bom para a demo e para testes pontuais; para a [[Avaliação - evals]] completa, o Haiku é mais prático.
