# 🤖 YouTube Insights Agent

> **O LLM interpreta a intenção e orquestra ferramentas; os números vêm de consultas e modelos determinísticos; o sistema expõe suas limitações e é avaliado por casos reproduzíveis.**

Agente de IA (Claude, API da Anthropic, *tool use*) que responde perguntas de estratégia de conteúdo sobre **9,4 milhões de registros de vídeos em alta do YouTube em 112 países**. Ele escreve e executa SQL protegido, gera gráficos, pontua títulos com um modelo de ML e mostra o rastro completo de cada resposta.

Continuação do projeto [youtube-views-prediction](https://github.com/eduardo-lima-silva/youtube-views-prediction), que fornece os dados e o modelo LightGBM.

**Persona:** analista de conteúdo de um canal brasileiro que quer comparar padrões de vídeos em alta e explorar hipóteses para novos vídeos.

## O que ele faz

| Capacidade | Exemplo | Como |
|---|---|---|
| **Análise de hipóteses com evidência** | "Compare a mediana de views de vídeos de 8–12 min e 4–8 min no Brasil" | SQL validado → resposta com **n da amostra**, gráfico e ressalvas |
| **Recusa e limites** | "Qual duração **causa** mais views?" | Explica que dados observacionais não permitem afirmar causalidade e oferece a análise associativa |
| **Títulos com trade-offs** | "Sugira 3 títulos para um vídeo sobre Python para entrevistas" | O LLM gera; o LightGBM pontua; o SHAP explica; empates dentro da margem do modelo são sinalizados. **Nunca "o melhor título"** |

## Arquitetura

```
Pergunta ──► Streamlit ──► Agente (Claude Opus 5.5, loop manual de tool use)
                              │   system prompt estável + catálogo semântico (prompt caching)
                              ├─► run_sql ──► SQL Guard (AST sqlglot) ──► DuckDB read-only (warehouse)
                              ├─► make_chart ──► Altair
                              └─► score_titles ──► LightGBM + TF-IDF + SHAP
                              ▼
                       Resposta ──► Grounding check (números conferidos com as ferramentas)
                              ▼
                       Rastro: passos, SQL, linhas, tokens, custo (US$), latência
```

### Segurança do SQL gerado pelo LLM (defesa em camadas)
1. **AST com sqlglot** (não regex): uma única instrução SELECT; só as tabelas `videos`, `video_country` e `daily_trending` ou CTEs; funções de arquivo, rede e sistema bloqueadas (`read_csv`, `glob`, `getenv`…); sem DDL/DML/`ATTACH`/`COPY`/`PRAGMA`.
2. **Limites de complexidade contra negação de serviço:** ≤ 4.000 caracteres, ≤ 4 JOINs, ≤ 4 CTEs, sem CROSS JOIN nem join sem condição, sem `SELECT *` nas tabelas grandes, `LIMIT` ≤ 200.
3. **Conexão DuckDB** `read_only`, `enable_external_access=false` e `lock_configuration=true` (verificados antes de qualquer consulta), limite de memória e timeout com interrupção.
4. Resultados de ferramentas são marcados como **dados não confiáveis** (títulos de vídeos podem conter *prompt injection*).

Testado com **90 consultas maliciosas** em `tests/test_sql_guard.py`.

## Avaliação

`evals/`: **70 casos** escritos como perguntas reais, estratificados em 9 categorias (agregação, filtros, comparação, ambíguas, fora de escopo, causais, SQL adversarial, títulos, explicação) e divididos em `dev` (ajuste do prompt) e `test` (só para relatar).

- O gabarito é **calculado por SQL** no warehouse, nunca escrito à mão.
- A nota é determinística: números em formato brasileiro, ranking, direção da conclusão, recusa, bloqueio de SQL perigoso e ferramentas obrigatórias.
- Métrica de **números alucinados**: números da resposta sem origem nos resultados das ferramentas.
- **Gates de regressão:** segurança de SQL = 100%; recusa causal ≥ 80%; recusa indevida ≤ 10%; números alucinados ≤ 10%; latência p95 ≤ 90 s.
- Intervalos de confiança de Wilson 95% e estabilidade entre repetições (`--repeat`).

> **Resultados ao vivo:** ainda não executados (exigem chave da API). Custo estimado: cerca de US$ 2,8 no split `dev` e US$ 2,7 no `test` com Claude Opus 5.5.

```bash
python -m evals.run_eval --split dev --dry-run      # valida o pipeline sem API
python -m evals.run_eval --split test --yes         # avaliação real (mostra o custo antes)
```

## Como executar

O repositório já inclui a **amostra do warehouse** (`data/warehouse_sample.duckdb`, 12 países, 325 mil vídeos) e os **modelos** em `models/`, então o app roda direto após o `pip install`. Para o warehouse completo, gere os parquets do projeto 1.

```bash
pip install -r requirements.txt
python scripts/build_warehouse.py            # warehouse completo (requer os parquets do projeto 1)
python scripts/build_warehouse.py --sample   # amostra de 57 MB para o deploy
python -m pytest -q                          # 297 testes offline
streamlit run app/streamlit_app.py           # sem chave = modo demo
```

Para o modo ao vivo, defina `ANTHROPIC_API_KEY` no ambiente. Modelo e esforço são configuráveis com `INSIGHTS_MODEL` e `INSIGHTS_EFFORT`.

### Custo por modelo

| Modelo (`INSIGHTS_MODEL`) | US$ por milhão de tokens (entrada / saída) | Avaliação `dev` (36 casos, estimativa) |
|---|---|---|
| `claude-opus-5-5` (padrão) | 4,00 / 20,00 | ~US$ 2,80 |
| `claude-sonnet-5-5` | 2,00 / 10,00 | ~US$ 1,40 |
| `claude-haiku-5-5` | 0,10 / 0,50 | ~US$ 0,07 |

Para desenvolver e iterar, o Haiku 5.5 sai por cerca de US$ 0,002 por pergunta: `INSIGHTS_MODEL=claude-haiku-5-5`. A avaliação mostra quanto de qualidade se perde em relação ao Opus.

### Modo gratuito (Groq)

O mesmo agente roda no plano gratuito do [Groq](https://console.groq.com) com o modelo aberto `openai/gpt-oss-120b`, sem custo:

```bash
set GROQ_API_KEY=...            # Windows (no Linux/macOS: export GROQ_API_KEY=...)
set INSIGHTS_PROVIDER=groq
streamlit run app/streamlit_app.py   # ou escolha "Groq (grátis)" na barra lateral
python -m evals.run_eval --provider groq --split dev --limit 5 --yes
```

`insights_agent/groq_agent.py` reaproveita as ferramentas, o SQL Guard, o rastro e a checagem de números; só muda o formato da API (OpenAI) e o tamanho do prompt.
- **Prompt compacto:** cerca de 3,5 mil tokens, contra cerca de 6 mil no Claude, porque o plano gratuito permite **~8 mil tokens/min e ~200 mil/dia**.
- **Espera automática** ao receber 429 (limite por minuto). Por isso uma pergunta pode levar alguns minutos.
- **Capacidade diária:** cerca de 40 a 50 chamadas ao modelo, ou seja, poucas dezenas de perguntas por dia. Dá para demo e testes pontuais, não para rodar a avaliação inteira de uma vez.
- Os limites do plano gratuito mudam com frequência. Confira em console.groq.com.

## Estrutura

| Caminho | Papel |
|---|---|
| `insights_agent/agent.py` | loop do agente, limites, custo, tratamento de recusa/erros |
| `insights_agent/groq_agent.py` | o mesmo agente no Groq (plano gratuito), com prompt compacto e espera em 429 |
| `insights_agent/tools.py`, `prompts.py` | ferramentas (JSON Schema estrito) e system prompt |
| `insights_agent/sql_guard.py`, `db.py` | validação AST e execução segura |
| `insights_agent/catalog.py` | catálogo semântico das tabelas (com ressalvas) |
| `insights_agent/predictor.py` | score de títulos + SHAP |
| `insights_agent/grounding.py` | verificação dos números da resposta |
| `app/streamlit_app.py` | interface com aba **Rastro** e modo demo |
| `evals/` | casos, gabaritos, corretor e runner |
| `docs/` | briefing, revisões externas (Codex) e arquitetura |
| `vault/` | vault Obsidian com o contexto do projeto (abra no Obsidian e use o Graph View) |

## Limitações (honestas)

- **Viés de seleção:** a base só tem vídeos que **entraram em alta**; nada aqui vale para "todo o YouTube".
- **Associação ≠ causalidade:** valores SHAP explicam o modelo, não o mundo.
- **Títulos:** só trocar o título altera a previsão entre 1,2× e 1,8×, dentro da margem de erro do modelo (×2). Por isso o agente trata diferenças pequenas como empate e prioriza o critério editorial.
- **Vazamento temporal leve** no modelo do projeto 1: as estatísticas do canal vêm do último snapshot.
- O corretor da avaliação usa heurísticas para detectar recusas; falhas devem ser revisadas manualmente.

## Processo de desenvolvimento

- A ideia e a arquitetura foram revisadas por um segundo modelo (OpenAI Codex): veja `docs/01_codex_review.md`, `docs/03_codex_review_arquitetura.md` e `docs/04_codex_code_review.md` (os bugs encontrados na revisão de código foram corrigidos e ganharam testes de regressão).
- A implementação foi feita por agentes especializados do Claude Code em paralelo (dados, ML, LLM e avaliação), seguindo contratos em `docs/02_arquitetura.md`.
- As decisões estão registradas como ADRs em `vault/07 Decisões/`.

---
Autor: **Eduardo Lima da Silva** · Ciência de Dados e IA, PUC-Campinas · [LinkedIn](https://www.linkedin.com/in/eduardo-lima-da-silva-958582382/) · [GitHub](https://github.com/eduardo-lima-silva)
