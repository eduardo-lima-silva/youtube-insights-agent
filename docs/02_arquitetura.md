# Arquitetura e contratos: YouTube Insights Agent

> **Posicionamento** (frase sugerida pela revisão do Codex): *"O LLM interpreta a intenção e orquestra ferramentas; os números vêm de consultas e modelos determinísticos; o sistema expõe suas limitações e é avaliado por casos reproduzíveis."*

**Persona:** analista de conteúdo de um canal brasileiro que quer comparar padrões de vídeos em alta e explorar hipóteses para novos vídeos.

**Escopo do MVP (3 capacidades):**
1. **Análise de hipóteses com evidência:** responde perguntas analíticas sobre a base com SQL validado, tamanho da amostra, gráfico e ressalvas.
2. **Títulos com trade-offs:** o LLM sugere até 3 títulos; o LightGBM pontua cada um (associação); a resposta mostra o score, os fatores SHAP e o risco de clickbait. Nunca diz "melhor título".
3. **Recusa e limites:** recusa perguntas causais ("o que *causa* views?"), fora de escopo e pedidos de SQL perigoso, oferecendo a análise associativa possível.

Fora do MVP: memória longa, upload de arquivos, multiagentes, ingestão automática, dados do canal do usuário.

---

## Estrutura de pastas

```
youtube-insights-agent/
├─ insights_agent/
│  ├─ config.py            # caminhos, modelo, limites (pronto)
│  ├─ trace.py             # tipos do rastro (pronto)
│  ├─ catalog.py           # catálogo semântico                [Agente DADOS]
│  ├─ sql_guard.py         # validação AST do SQL             [Agente DADOS]
│  ├─ db.py                # conexão read-only + run_query     [Agente DADOS]
│  ├─ charts.py            # spec de gráfico -> Altair         [Agente DADOS]
│  ├─ predictor.py         # score de títulos + SHAP           [Agente ML]
│  ├─ tools.py             # schemas das ferramentas + dispatch [Agente LLM]
│  └─ agent.py             # loop do Claude, prompt, custo     [Agente LLM]
├─ scripts/build_warehouse.py  # parquet -> data/warehouse.duckdb [Agente DADOS]
├─ app/streamlit_app.py    # UI com painel de rastro           [Agente LLM]
├─ evals/                  # casos, runner, avaliação          [Agente EVAL]
├─ tests/                  # pytest
├─ demo_traces/            # execuções gravadas (modo demo sem chave)
├─ docs/  vault/           # documentação e vault Obsidian
```

Os dados de origem ficam no projeto 1 (caminhos em `config.py`):
- `../youtube-views-prediction/data/trending_slim.parquet` (9,4 M linhas: vídeo × país × dia)
- `../youtube-views-prediction/data/videos_features.parquet` (956 mil vídeos, 55 colunas)
- `../youtube-views-prediction/models/lgbm_views.txt`, `text_tfidf_ridge.joblib`, `metadata.json`

---

## Warehouse (`data/warehouse.duckdb`): as únicas tabelas visíveis ao LLM

Construído por `scripts/build_warehouse.py`. Todos os nomes estão em inglês e em snake_case. Datas em UTC.

### `videos`: 1 linha por vídeo (último snapshot observado)
| coluna | tipo | descrição |
|---|---|---|
| video_id | VARCHAR | id do vídeo |
| title | VARCHAR | título |
| channel_title | VARCHAR | nome do canal |
| category | VARCHAR | categoria do YouTube (ex.: Music, Gaming, Sports) |
| published_at | TIMESTAMP | data/hora de publicação (UTC) |
| pub_hour | INTEGER | hora de publicação (0-23, UTC) |
| pub_dow | INTEGER | dia da semana (0 = segunda … 6 = domingo) |
| duration_s | INTEGER | duração em segundos |
| is_short | BOOLEAN | duração ≤ 180 s |
| views | BIGINT | views no último snapshot |
| likes | BIGINT | likes (pode ser NULL) |
| comments | BIGINT | comentários (pode ser NULL) |
| age_days | DOUBLE | idade do vídeo no último snapshot, em dias |
| channel_subs | BIGINT | inscritos do canal (no snapshot) |
| channel_country | VARCHAR | país do canal (pode ser 'Unknown') |
| first_trending_country | VARCHAR | primeiro país em que entrou em alta |
| first_trending_date | DATE | primeiro dia em alta |
| hours_to_trend | DOUBLE | horas entre publicação e 1º dia em alta (aprox.) |
| n_trending_countries | INTEGER | nº de países em que ficou em alta |
| n_trending_days | INTEGER | nº de dias distintos em alta |

### `video_country`: 1 linha por (vídeo, país) em que o vídeo ficou em alta (~2,16 M linhas)
| coluna | tipo | descrição |
|---|---|---|
| video_id | VARCHAR | → videos.video_id |
| country | VARCHAR | país (nome em inglês, ex.: 'Brazil') |
| first_date | DATE | 1º dia em alta nesse país |
| last_date | DATE | último dia em alta nesse país |
| days_trending | INTEGER | nº de dias em alta nesse país |

### `daily_trending`: agregado por (dia, país, categoria) (~572 mil linhas)
| coluna | tipo | descrição |
|---|---|---|
| trending_date | DATE | dia |
| country | VARCHAR | país |
| category | VARCHAR | categoria |
| n_videos | INTEGER | vídeos distintos em alta |
| median_views | DOUBLE | mediana das views registradas nesse dia |

Para o deploy: `build_warehouse.py --sample` gera `data/warehouse_sample.duckdb` com menos de 90 MB (ex.: `videos` de um subconjunto de países, incluindo Brazil, com `video_country` e `daily_trending` filtrados de forma consistente). O app usa o arquivo que existir (`config.WAREHOUSE_PATH`).

---

## Contratos dos módulos (assinaturas obrigatórias)

### `catalog.py` [DADOS]
```python
TABLES: dict[str, dict]          # {"videos": {"description": str, "columns": {col: "tipo — descrição"}}, ...}
def catalog_markdown() -> str     # texto do catálogo para o system prompt (estável, determinístico)
CAVEATS: list[str]                # ressalvas da base (viés de seleção, período, etc.)
```

### `sql_guard.py` [DADOS]
```python
class SQLRejected(Exception): ...           # .reason: str (mensagem em PT para o LLM)
def validate_sql(sql: str) -> str            # devolve o SQL normalizado e seguro, ou levanta SQLRejected
```
Regras (com AST do `sqlglot`, dialeto duckdb, **não regex**):
- exatamente 1 instrução; precisa ser SELECT (WITH/CTE permitido);
- só as tabelas `videos`, `video_country` e `daily_trending` (e CTEs definidas na própria query);
- proibidas: funções de arquivo/rede (`read_csv*`, `read_parquet`, `read_json*`, `glob`, `httpfs`…), `COPY`, `ATTACH`, `PRAGMA`, `SET`, `INSTALL`, `LOAD`, `CREATE`, `INSERT`, `UPDATE`, `DELETE`, `DROP`, `EXPORT`, table functions em geral;
- sem `LIMIT` ou com `LIMIT` > `MAX_ROWS` → aplica `LIMIT MAX_ROWS`;
- colunas desconhecidas → o próprio DuckDB acusa o erro, que volta ao LLM como mensagem.

### `db.py` [DADOS]
```python
def get_connection() -> duckdb.DuckDBPyConnection
    # read_only=True; SET enable_external_access=false; SET lock_configuration=true;
    # memory_limit e threads de config
def run_query(sql: str) -> QueryResult
    # validate_sql -> execução com timeout (config.SQL_TIMEOUT_S, via con.interrupt() numa thread)
@dataclass
class QueryResult:
    sql: str; columns: list[str]; rows: list[list]; row_count: int; truncated: bool; elapsed_ms: float
    def to_markdown(max_rows: int = 30) -> str
```

### `charts.py` [DADOS]
```python
def build_chart(df: pandas.DataFrame, kind: str, x: str, y: str, color: str | None, title: str) -> altair.Chart
    # kind ∈ {"bar","line","scatter"}; valida se x/y existem em df; levanta ValueError com mensagem clara
```

### `predictor.py` [ML]
```python
@dataclass
class VideoContext:
    category: str = "Entertainment"; duration_s: int = 600; channel_subs: int = 100_000
    channel_views: int = 20_000_000; channel_videos: int = 300; channel_age_days: int = 1500
    market: str = "Brazil"; pub_hour_utc: int = 21; pub_dow: int = 4; horizon_days: int = 7
    tags: str = ""; desc_len: int = 500
def score_titles(titles: list[str], ctx: VideoContext, top_k_factors: int = 5) -> list[dict]
    # cada item: {"title", "predicted_views", "interval_2x": [low, high],
    #   "top_factors": [{"feature", "label_pt", "contribution_log", "direction"}],
    #   "title_effect_log": float  # soma das contribuições SHAP das features de título/texto
    # }
```
Reutiliza a lógica de features de `../youtube-views-prediction/src/predict.py` (copiar e adaptar, sem importar de lá). Carrega os modelos uma vez (cache em nível de módulo). O SHAP vem de `Booster.predict(X, pred_contrib=True)`.

### `tools.py` + `agent.py` [LLM]
Ferramentas expostas ao Claude (todas com `strict: true` e `additionalProperties: false`):
| nome | entrada | saída (texto para o LLM) |
|---|---|---|
| `run_sql` | `{sql: str, purpose: str}` | tabela markdown + row_count + truncated + `result_id` |
| `make_chart` | `{result_id: str, kind: "bar"\|"line"\|"scatter", x: str, y: str, color: str\|null, title: str}` | ok ou erro |
| `score_titles` | `{titles: [str] (1-3), context: {...VideoContext}}` | scores + fatores + aviso de associação |

`agent.py`:
```python
def run_agent(question: str, history: list | None = None, *, client=None) -> AgentResult
```
- Loop manual com `anthropic.Anthropic().beta.messages.create(...)`, modelo `config.MODEL` (padrão `claude-opus-5-5`), `output_config={"effort": config.EFFORT}`, `betas=["server-side-fallback-2026-07-01"]`, `fallbacks="default"`, `max_tokens=16000`, `cache_control={"type": "ephemeral"}` no topo (system + tools estáveis).
- Antes de ler o conteúdo, verifica `stop_reason`: `refusal` → AgentResult com status "refused_by_model"; `max_tokens` → erro amigável.
- Limites: `config.MAX_TOOL_CALLS` por pergunta; resultados de ferramenta truncados.
- Todos os `tool_result` de um turno voltam numa única mensagem `user`; erros de ferramenta com `is_error: true`.
- Acumula `usage` (input, output, cache read/write) e calcula o custo em US$ com os preços de `config.PRICES`.
- Formato da resposta final (o system prompt exige): resposta direta → evidência (números, n da amostra) → ressalvas → (opcional) próxima análise sugerida. Português do Brasil.

### `trace.py` (pronto): tipos compartilhados
`AgentResult(answer, status, steps: list[TraceStep], charts, usage, cost_usd, latency_s, model)` e `TraceStep(kind, name, input, output_preview, ok, elapsed_ms)`. É serializável em JSON (`to_dict` / `from_dict`) para gravar em `demo_traces/`.

### `app/streamlit_app.py` [LLM]
- Campo de pergunta + 4 exemplos clicáveis (1 análise, 1 causal → recusa, 1 títulos, 1 por país).
- Resposta, gráficos Altair e aba **"Rastro"** com cada passo (ferramenta, SQL validado, linhas, tempo), tokens, custo e latência.
- Sem chave de API → **modo demo**: lista as execuções gravadas em `demo_traces/*.json` e as reproduz (com aviso visível).
- Faixa fixa de ressalvas: "Base com apenas vídeos que entraram em alta · associação ≠ causalidade · out/2024–mai/2026".

### `evals/` [EVAL]
- `cases.jsonl`: 60+ casos estratificados (ver docs/01_codex_review.md §5). Cada caso:
  `{"id", "category": "agregacao|filtro_grupo|comparacao|ambigua|fora_escopo|causal|sql_adversarial|titulos|explicacao", "question", "expected_behavior": "answer|refuse|clarify", "reference_sql": str|null, "answer_check": {"type": "number|ranking|set|none", "path": ..., "tolerance": ...}, "must_mention": [str], "must_not": [str], "expected_tools": [str]}`
- As respostas de referência são **calculadas executando `reference_sql`** no warehouse (nunca escritas à mão) e salvas em `evals/reference_answers.json`.
- `run_eval.py`: roda o agente em cada caso (com `--limit` e `--repeat`), grava `evals/results/<timestamp>.jsonl` com o trace completo e calcula as métricas por categoria: acerto numérico (tolerância relativa), recusa correta, não-recusa indevida, ferramentas esperadas, SQL bloqueado em casos adversariais, custo e latência (mediana e p95).
- `grade.py`: avaliação determinística (números extraídos do texto da resposta e dos resultados das ferramentas); o LLM-juiz é opcional, desligado por padrão.
- **Divisão:** `dev` (usado para ajustar o prompt) e `test` (só para relatar). Marcar com `"split"`.

### Testes (`tests/`)
- `test_sql_guard.py`: 25+ casos maliciosos/inválidos bloqueados e 10+ válidos aceitos.
- `test_db.py`: read-only de verdade (INSERT falha), timeout, LIMIT aplicado.
- `test_predictor.py`: shapes, determinismo, títulos diferentes → scores possivelmente diferentes, contexto inválido → erro claro.
- `test_agent_offline.py`: loop do agente com cliente **falso** (mock que devolve tool_use roteirizado), sem rede.

---

## Limitações conhecidas (vão para README e UI)
- Viés de seleção: só vídeos que entraram em alta; o modelo é condicional a esse grupo.
- Associação ≠ causalidade.
- `channel_subs` e o histórico do canal vêm do snapshot (possivelmente posterior à publicação): é um vazamento temporal leve no modelo do projeto 1.
- Período de out/2024 a mai/2026; cobertura desigual entre países.
- O score de títulos reflete padrões estatísticos de títulos parecidos, não qualidade editorial.
