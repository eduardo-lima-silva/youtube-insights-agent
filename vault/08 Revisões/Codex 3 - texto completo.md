Ver resumo em [[Revisão Codex 3 - Código]].

# Revisão de código

Revisei os arquivos de `insights_agent/`, `app/`, `evals/`, `scripts/` e `tests/` somente por leitura. Não modifiquei arquivos nem executei comandos que escrevessem em disco.

## 1. Bugs reais

### Alta severidade

#### 1. Limites de memória e threads podem falhar silenciosamente

**Arquivo:** `insights_agent/db.py:75-88`

`_connect()` ignora qualquer erro ao aplicar:

```python
SET memory_limit='1GB'
SET threads=4
```

Depois, `_verify()` confirma apenas `enable_external_access` e `lock_configuration`, não `memory_limit` nem `threads`.

Cenário concreto: se a configuração global do DuckDB já estiver bloqueada ou incompatível, a conexão continua funcionando sem limite de memória. Uma consulta permitida pelo AST, mas com agregação ou join caro, pode consumir muito mais que 1 GB. Isso mantém o risco de DoS apontado em `docs/03_codex_review_arquitetura.md`.

#### 2. O guard permite alocações enormes através de `range()`/`generate_series()` escalar

**Arquivo:** `insights_agent/sql_guard.py:62-67`, `:111-126`

`range` e `generate_series` estão em `_SCALAR_OK`. O bloqueio só ocorre quando aparecem como table function:

```sql
SELECT * FROM range(1000000000)
```

Mas a forma escalar passa pela regra:

```sql
SELECT range(1000000000)
```

O resultado é uma única linha, portanto o `LIMIT` externo não protege a alocação da lista produzida. O mesmo padrão vale para `generate_series`.

Cenário concreto: o agente gera `SELECT range(1000000000)`. A consulta aparenta retornar uma linha, mas pode materializar uma estrutura gigantesca antes do limite de linhas.

#### 3. O loop pode aceitar uma resposta final contendo chamadas de ferramenta sem processá-las

**Arquivo:** `insights_agent/agent.py:275-310`

As chamadas só são processadas quando:

```python
if stop == "tool_use" and tool_uses:
```

Se a API devolver blocos `tool_use` com outro `stop_reason`, o código cai no caminho de resposta final em `:310-319`. Como `_text_of()` ignora blocos `tool_use`, o agente pode retornar:

```text
O modelo não devolveu texto. Tente de novo.
```

sem executar a ferramenta nem registrar o motivo adequadamente.

É um caso dependente do comportamento do provedor, mas a implementação não trata a condição internamente inconsistente “há `tool_use`, mas o motivo não é `tool_use`”.

### Média severidade

#### 4. A faixa `interval_2x` é exibida como se fosse intervalo visual, mas a UI não reforça sua natureza heurística

**Arquivo:** `insights_agent/tools.py:327-355`

O código renderiza a coluna `faixa 2x`, mas depende do texto produzido pelo modelo para explicar que não é intervalo calibrado. Se o modelo omitir essa ressalva, a tabela ainda apresentará uma faixa numérica com aparência de intervalo estatístico.

O predictor documenta corretamente a limitação, mas a ferramenta não a torna estruturalmente impossível de perder.

#### 5. `purpose` não participa de nenhuma validação

**Arquivo:** `insights_agent/tools.py:249`, `insights_agent/agent.py:290`

O campo `purpose` é coletado e gravado no trace, mas não há verificação de coerência entre propósito e SQL. Uma chamada pode declarar “comparar medianas por país” e executar outra consulta completamente diferente.

Isso confirma a lacuna apontada em `docs/03_codex_review_arquitetura.md`: a autorização semântica continua dependente do prompt.

#### 6. Erros de execução podem ser classificados como bloqueios de segurança nos evals

**Arquivo:** `evals/grade.py:409-429`

Em `adversarial_safety()`, qualquer passo `run_sql` com `ok=False` é contado como `guard_blocks`:

```python
if ... run_sql ... and not s.get("ok", True):
    guard_blocks += 1
    continue
```

Não se distingue:

- SQL rejeitado pelo `sql_guard`;
- erro de coluna;
- timeout;
- falha interna;
- erro de DuckDB.

Cenário concreto: um SQL adversarial chega ao dispatcher, falha por um erro genérico e recebe `ok=False`. O avaliador pode reportar o caso como “bloqueado”, embora nunca tenha comprovado que o guard impediu a execução.

#### 7. Os testes não cobrem o bypass escalar de `range`

**Arquivo:** `tests/test_sql_guard.py:71-72`, `:143`

Há teste para:

```sql
SELECT * FROM range(1000000000)
```

e teste válido para:

```sql
SELECT range(3)
```

Mas não há caso adversarial para `SELECT range(1000000000)`. A própria whitelist que permite o caso pequeno permite também o caso potencialmente destrutivo.

## 2. Brechas remanescentes no caminho SQL

O caminho está bem mais forte que uma validação por regex, mas ainda há vetores concretos:

### DoS por funções escalares

O principal é:

```sql
SELECT range(1000000000)
SELECT generate_series(1, 1000000000)
```

A forma escalar é liberada por `_SCALAR_OK`, enquanto apenas a forma tabular é bloqueada.

Também não existe allowlist de funções. Portanto, funções escalares caras, como repetições de strings, regex ou transformações que materializem listas grandes, dependem apenas dos limites gerais de DuckDB.

### Limite de linhas não limita o trabalho intermediário

`LIMIT` é aplicado ao SELECT externo em `sql_guard.py:225-231`, mas não limita:

- scans completos;
- `GROUP BY` de alta cardinalidade;
- `ORDER BY` global;
- joins com muitas duplicações;
- materialização de CTEs;
- funções escalares que produzem valores grandes.

O teste de timeout reduz o impacto, mas não impede consumo elevado de CPU/memória antes da interrupção.

### Configurações de recurso não são verificadas

Como observado acima, `memory_limit` e `threads` podem falhar silenciosamente em `db.py:75-88`. Isso enfraquece a segunda camada de defesa.

### Dependência de denylist

O bloqueio de funções depende de `_DENY_FUNCS` e prefixos em `sql_guard.py:53-87`. Novas funções DuckDB, aliases ou extensões podem não estar cobertos. O `enable_external_access=false` ajuda contra acesso externo, mas não substitui uma allowlist de funções analíticas.

### O que não encontrei

Não encontrei, por leitura, um bypass óbvio que permita:

- `INSERT`, `UPDATE`, `DELETE`, `DROP`;
- `COPY`, `ATTACH`, `INSTALL`, `LOAD`;
- leitura direta de arquivo por table function;
- acesso a tabelas fora da allowlist.

O bloqueio AST, a rejeição de nomes qualificados e a conexão `read_only=True` são camadas efetivas contra esses caminhos.

## 3. Loop do agente e Streamlit

### Problemas no loop

- `max_llm_calls = MAX_TOOL_CALLS + 4` limita chamadas ao modelo, mas não existe um limite equivalente de tamanho acumulado de `messages`. Histórico, SQL e resultados podem crescer até atingir a janela de contexto.
- O `history` é apenas truncado por caracteres em `agent.py:92-108`; não há validação de estrutura ou origem. Uma resposta anterior contendo instruções pode voltar ao modelo como mensagem de assistant e influenciar o novo turno.
- Após `MAX_TOOL_CALLS`, o agente envia uma mensagem textual junto dos `tool_result` em `agent.py:301-307`. Isso é intencional no teste, mas o fluxo depende de o provedor aceitar esse formato híbrido em todos os modos beta.
- A checagem de grounding é apenas uma anotação: números não verificados não alteram o status final. O agente retorna `status="ok"` mesmo quando `grounding` encontra números não verificados em `agent.py:314-319`.

### Problema real no Streamlit

**Arquivo:** `app/streamlit_app.py:145-151`

O código usa:

```python
st.altair_chart(..., width="stretch")
st.vega_lite_chart(..., width="stretch")
```

Mas `requirements.txt` declara apenas:

```text
streamlit>=1.40
```

A API de largura `"stretch"` não é garantida pela versão mínima declarada. Em um ambiente instalado exatamente com uma versão antiga dentro desse requisito, a renderização pode lançar `TypeError`; o fallback repete o mesmo parâmetro e também pode falhar.

### Primeira execução real com API

O código envia ao SDK:

```python
cache_control={"type": "ephemeral"}
betas=["server-side-fallback-2026-07-01"]
fallbacks="default"
```

em `agent.py:213-223`. Esses parâmetros estão acoplados a uma API beta muito específica. Não há teste de integração real — apenas um cliente falso em `tests/test_agent_offline.py`. Portanto, uma chave válida não garante que a primeira chamada seja aceita pelo SDK/endpoint configurado; qualquer incompatibilidade resulta imediatamente no caminho de erro genérico.

Além disso, o projeto não contém arquivos em `demo_traces/` no inventário revisado. Sem chave e sem traces, o modo demo abre corretamente, mas não oferece nenhuma execução para reproduzir.

## 4. Avaliação enganosa

### Dry-run oracle não avalia o agente

**Arquivo:** `evals/run_eval.py:122-194`

O modo `--dry-run --fake oracle` constrói a resposta diretamente a partir de `reference_answers.json`. Isso testa a coerência entre casos, referência e grader, mas não testa:

- seleção de ferramenta;
- geração de SQL;
- correção do loop;
- comportamento do modelo;
- grounding real;
- recusas reais.

O próprio código documenta isso, mas resultados de dry-run podem ser interpretados indevidamente como qualidade do agente.

### Referência e agente usam a mesma fonte

**Arquivo:** `evals/build_references.py:74-110`

As respostas esperadas são calculadas no mesmo warehouse consultado pelo agente. Um erro no warehouse, no período ou na definição das tabelas contamina simultaneamente gabarito e resposta avaliada.

### Checagem numérica aceita qualquer número correto

**Arquivo:** `evals/grade.py:458-465`

`number_matches()` passa se qualquer número da resposta estiver dentro da tolerância. Uma resposta pode conter o valor correto e também afirmar um valor errado, e ainda passar na checagem numérica.

### Ranking é puramente textual

**Arquivo:** `evals/grade.py:475-509`

O ranking verifica apenas a posição dos rótulos no texto. Ele não confirma:

- se o rótulo foi usado como conclusão;
- se o grupo correto foi associado ao valor correto;
- se o modelo confundiu país, categoria ou unidade;
- se o ranking foi apresentado como causalidade.

### Grounding depende do texto truncado das ferramentas

**Arquivo:** `evals/grade.py:586-606`

Os números são comparados contra `result_text`/`output_preview`. Como as saídas podem ser truncadas, um número legítimo que ficou fora do preview pode ser marcado como alucinação.

### Casos de zero usam apenas a primeira linha

**Arquivo:** `evals/build_references.py:101-108`

`expect_zero` e `is_zero_like` olham essencialmente para a primeira linha/primeiro valor. Uma referência ordenada com primeiro valor zero e linhas posteriores não nulas pode ser classificada de forma enganosa.

### Segurança avalia principalmente string e trace

**Arquivo:** `evals/grade.py:409-429`

A segurança usa `sql_danger()` sobre o SQL armazenado no trace. Isso não comprova que:

- o SQL realmente foi rejeitado pelo guard;
- a consulta não causou alto custo;
- não houve exfiltração via conteúdo textual;
- não houve prompt injection nos dados.

## 5. O que eu manteria

- Validação AST com `sqlglot`, em vez de regex.
- Conexão DuckDB `read_only=True`.
- `enable_external_access=false` e `lock_configuration=true`.
- Limite de tamanho do SQL, número de joins, CTEs, profundidade e `SELECT *`.
- Timeout com descarte da conexão após interrupção.
- Estado de resultados isolado por execução do agente.
- `strict: true` e `additionalProperties: false` nos schemas.
- Resultados de ferramentas marcados como não confiáveis para reduzir prompt injection.
- Mensagens de erro de ferramentas devolvidas ao modelo como `is_error: true`.
- Separação clara entre dados determinísticos, predictor e redação do LLM.
- Avisos explícitos de associação versus causalidade.
- Rastro com SQL validado, tempo, linhas, custo e tokens.
- Testes offline do loop, inclusive chamadas paralelas, erros, timeout e limite de ferramentas.
- Divisão `dev`/`test`, referências calculadas e gates de regressão.
- Honestidade conceitual do predictor sobre viés de seleção, faixa `2x` heurística e SHAP não causal.