# 1. A arquitetura incorporou bem a revisão?

Sim. A incorporação foi muito boa, especialmente em:

- definição de persona e escopo do MVP;
- separação entre LLM, SQL, modelo preditivo e apresentação;
- exposição de SQL, evidências, amostra e ressalvas;
- tratamento explícito de associação versus causalidade;
- recusa de perguntas fora do escopo;
- modo demo sem chave de API;
- rastreamento de custo, latência e ferramentas;
- testes offline e divisão `dev`/`test`;
- documentação dos vieses e do vazamento temporal.

O que ficou de fora ou insuficientemente especificado:

1. **Política determinística de autorização de ferramentas.**  
   A arquitetura ainda depende bastante do prompt para decidir quando usar `run_sql`, `score_titles` ou recusar. Essa decisão deveria ser reforçada por um classificador/roteador determinístico ou por intenções estruturadas.

2. **Limites de custo computacional do SQL.**  
   Há limite de linhas e timeout, mas não há limite explícito para complexidade da consulta, número de joins, cardinalidade de `GROUP BY`, `ORDER BY`, produto cartesiano ou uso de memória.

3. **Isolamento por sessão.**  
   `result_id`, resultados SQL, gráficos e rastros precisam ter escopo por usuário/sessão, TTL e limite de armazenamento.

4. **Proveniência e versionamento dos dados.**  
   Falta um contrato para versão do warehouse, data de construção, snapshot utilizado, hash dos arquivos, versão do modelo e compatibilidade entre amostra e dataset completo.

5. **Validação independente do preditor.**  
   A arquitetura reconhece vazamento temporal, mas não define um teste que impeça o modelo de ser usado até que a disponibilidade temporal das features esteja comprovada.

6. **Critérios de regressão.**  
   Os evals calculam métricas, mas não definem limites que façam uma alteração ser rejeitada, por exemplo: segurança SQL não pode cair, recusa causal deve permanecer acima de determinado nível e p95 não pode ultrapassar um teto.

# 2. Revisão dos contratos

## Warehouse

Principais problemas:

- `videos` representa o último snapshot, enquanto `daily_trending` representa observações diárias. Comparar essas tabelas pode misturar períodos e significados diferentes.
- `views`, `likes`, `comments` e `channel_subs` não têm um `as_of_timestamp` explícito. O usuário pode interpretar esses valores como disponíveis no momento da publicação.
- A tabela contém variáveis potencialmente posteriores ao evento previsto, como `views`, `n_trending_days`, `n_trending_countries` e `first_trending_date`.
- O contrato não define claramente se `video_id` é único em `videos`, nem como duplicatas são resolvidas.
- A descrição menciona `videos_features.parquet` com 55 colunas, mas o warehouse documenta apenas um subconjunto. Falta declarar quais features do modelo realmente estão disponíveis.
- `VideoContext` usa `channel_views` e `channel_videos`, mas essas colunas não existem em `videos`. O contrato não explica se serão defaults, calculadas, ignoradas ou obtidas de outro lugar.
- Não há contrato para valores nulos, categorias desconhecidas, timezone, unidades e limites válidos.
- `sample` preserva apenas alguns países, mas isso pode alterar distribuições, rankings e conclusões. A equivalência estatística entre amostra e dataset completo não é garantida.

Recomendação: adicionar manifesto do warehouse com versão, data de corte, origem, chaves, cardinalidade esperada, semântica temporal, regras de nulidade e compatibilidade com o modelo.

## `sql_guard`

O desenho é bom, mas ainda incompleto para segurança:

- A allowlist é de tabelas, não de colunas, expressões ou operações analíticas.
- “Tabela permitida” precisa distinguir corretamente tabelas reais de CTEs, aliases, subconsultas e nomes qualificados.
- Bloquear table functions em geral é adequado, mas deve haver testes explícitos contra todas as formas de invocação suportadas pelo DuckDB.
- Não há limite de tamanho do SQL, profundidade de expressão, quantidade de CTEs, joins, agrupamentos ou ordenações.
- Aplicar `LIMIT` não impede uma consulta que faça um scan completo ou uma agregação gigantesca antes de retornar poucas linhas.
- Não há bloqueio explícito de `SELECT *`, embora ele dificulte controle de dados expostos e aumente custo.
- O contrato deixa a validação de colunas desconhecidas para o DuckDB. Isso valida existência, mas não valida se a coluna é apropriada para aquela intenção.
- Não há menção a parâmetros tipados. Mesmo sendo leitura, interpolação de valores no SQL aumenta risco e dificulta auditoria.
- `purpose` é recebido pela ferramenta, mas não é usado para validar se o SQL corresponde à finalidade declarada.
- A segurança depende também das configurações do DuckDB. O AST sozinho não garante que futuras funções ou extensões não criem novas formas de acesso externo.

O risco principal remanescente é **negação de serviço**, não apenas escrita no banco: joins cartesianos, agregações de alta cardinalidade, `ORDER BY` amplo, expressões caras e consultas que materializam grandes intermediários.

## `db`

Há várias ambiguidades operacionais:

- `run_query` pode executar a consulta inteira e só depois converter o resultado para listas. Nesse caso, `MAX_ROWS` limita a resposta, mas não necessariamente a memória ou o trabalho realizado.
- Interromper a conexão por outra thread pode causar condições de corrida, conexão inutilizável ou interferência entre requisições.
- Não está definido se cada chamada usa uma conexão isolada ou se existe pool compartilhado.
- Não está definido o comportamento depois de timeout: a conexão é descartada, reiniciada ou reutilizada?
- `read_only=True` protege contra escrita no arquivo, mas não substitui AST, isolamento e bloqueio de acesso externo.
- A configuração de `enable_external_access` e `lock_configuration` precisa ser aplicada e testada antes de qualquer consulta; o contrato não especifica a ordem nem o que ocorre se a configuração falhar.
- O fallback “usa o arquivo que existir” é perigoso. Pode selecionar acidentalmente um warehouse antigo, incompleto ou incompatível com o modelo.
- `truncated` é ambíguo: significa que o SQL tinha limite aplicado, que o resultado foi cortado pela aplicação ou que o resultado real excedeu `MAX_ROWS`?

## `predictor`

Pontos críticos:

- `interval_2x` parece ser uma faixa heurística, não um intervalo de predição calibrado. Deve ser rotulado como tal.
- `predicted_views` pode exigir transformação de log para escala original com correção de viés; isso não está especificado.
- `title_effect_log` não é necessariamente “efeito do título”. SHAP distribui a previsão entre features e depende do conjunto de features, baseline e implementação. O nome pode sugerir causalidade.
- A diferença entre títulos pode não alterar as features usadas pelo modelo; portanto, scores diferentes não são garantidos nem necessariamente significativos.
- O contrato não define limites para `duration_s`, `horizon_days`, `pub_hour_utc`, `desc_len`, tamanho do título e quantidade de tags.
- `market` e `category` não têm vocabulário fechado nem regra de desconhecido.
- Os defaults de contexto podem produzir uma previsão artificialmente precisa para um canal hipotético.
- O contrato não informa a versão do modelo, ordem das features, tratamento de nulos e compatibilidade com `metadata.json`.
- Há risco de vazamento ao usar estatísticas de canal e variáveis de desempenho posterior à publicação.

A maior falha conceitual é apresentar contribuições SHAP como justificativa para “qual título performa melhor”, quando o modelo foi treinado em uma população selecionada e pode não representar títulos novos.

## `tools`

`strict: true` e `additionalProperties: false` ajudam, mas não resolvem validação semântica:

- `run_sql` não impõe limite de caracteres, complexidade ou número de tentativas.
- `purpose` pode ser qualquer texto e não cria vínculo verificável com a consulta.
- `make_chart` não valida tipos, cardinalidade, número de categorias ou tamanho do dataframe.
- `result_id` precisa ser vinculado à sessão do usuário, ter expiração e não permitir acesso cruzado.
- Os resultados das tabelas podem conter texto controlado por usuários ou canais. Esse texto pode atuar como prompt injection quando devolvido ao LLM.
- Não há contrato para erro, schema, versão ou tamanho máximo das saídas.
- `score_titles` precisa validar o contexto, tamanho e conteúdo dos títulos, além de tratar títulos que contenham instruções maliciosas.
- O LLM recebe resultados de dados e mensagens de erro como texto confiável. Esses conteúdos deveriam ser explicitamente delimitados como dados não confiáveis.

## `agent`

Riscos e ambiguidades:

- O contrato parece depender de uma combinação específica de parâmetros beta, fallback e cache da API. Esses parâmetros devem ser validados em testes de integração, pois podem mudar ou não ser aceitos simultaneamente pelo SDK.
- `history` não tem schema, limite de tamanho, política de sanitização nem distinção entre instruções do sistema, usuário e dados.
- Não há limite explícito de tempo total, número de turnos, retries ou backoff.
- O agente pode chamar uma ferramenta válida mesmo quando a pergunta deveria ser recusada, caso a decisão fique apenas no prompt.
- Não há validação final para verificar se os números citados na resposta realmente aparecem no resultado da ferramenta.
- `stop_reason == "refusal"` não cobre necessariamente recusas produzidas como texto normal.
- O custo calculado depende de `config.PRICES`; falta versionamento, tratamento de modelos desconhecidos e distinção entre preço estimado e preço efetivo.
- Uma resposta parcial depois de timeout, erro de API ou limite de tokens não tem contrato claro.
- Não está definido se o modelo pode receber SQL, nomes de tabelas ou conteúdo bruto de usuários em prompts futuros.

Seria importante separar: roteamento de intenção, autorização das ferramentas, execução e redação final. O LLM não deveria ser o único componente responsável por todas essas decisões.

## `app`

Problemas principais:

- “Usar o arquivo que existir” pode fazer o deploy rodar com warehouse incorreto.
- O modo demo pode expor dados, rastros ou prompts que não deveriam estar públicos.
- Não há limites claros para tamanho de pergunta, número de requisições, concorrência ou custo por sessão.
- Streamlit mantém estado de sessão; o contrato não define isolamento entre usuários nem limpeza de resultados.
- Exibir SQL e rastros é excelente para transparência, mas pode revelar detalhes internos, caminhos locais, mensagens de erro ou metadados excessivos.
- A faixa fixa de limitações não substitui ressalvas específicas da consulta.
- Não há especificação de comportamento para API indisponível depois que o usuário já iniciou uma sessão.
- O deploy não define uma política de segredo, logging, retenção e remoção de traces.

## `evals`

O desenho é melhor que o original, mas ainda tem riscos de avaliação enganosa:

- A resposta de referência é gerada pelo mesmo warehouse que o agente consulta. Um erro de construção do warehouse pode contaminar o gabarito.
- `reference_sql` precisa ser tratado como código confiável separado do caminho de SQL gerado pelo LLM.
- Extrair números da resposta textual pode aceitar um número correto acompanhado de uma conclusão errada.
- Tolerância relativa é problemática para valores próximos de zero, percentuais, contagens e unidades diferentes.
- `expected_tools` exato pode penalizar soluções equivalentes ou premiar uma chamada desnecessária.
- `must_mention` por busca textual é frágil para paráfrases e pode ser facilmente satisfeito sem entendimento real.
- Não há métrica explícita para alucinação de números, mistura de grupos, unidade errada ou evidência que não sustenta a conclusão.
- Segurança avalia bloqueio de SQL perigoso, mas não necessariamente exfiltração, prompt injection nos dados ou negação de serviço.
- Não há testes claros de multi-turno, histórico contaminado, títulos maliciosos ou resultados com zero linhas.
- 60 casos ainda são poucos para conclusões fortes, principalmente se houver grande variação entre execuções.
- `repeat` deveria produzir intervalos de confiança ou distribuição, não somente médias.
- Não estão definidos limiares de aprovação, critérios de regressão ou comparação obrigatória com baselines.
- Geração de títulos não possui avaliação objetiva de adequação, clickbait, diversidade e qualidade editorial.
- Explicações SHAP não são avaliadas quanto à fidelidade ao modelo.

# 3. Duas decisões que eu faria de outro jeito

## Decisão 1: não usaria SQL livre como caminho principal

Eu transformaria a entrada do agente em uma intenção estruturada, por exemplo:

- métrica;
- população;
- filtros;
- agrupamento;
- período;
- comparação;
- limite;
- tipo de gráfico.

O sistema geraria SQL internamente por templates seguros.

Motivos:

- reduz drasticamente a superfície de ataque do DuckDB;
- torna a avaliação semântica mais simples;
- permite validar unidades, datas e população antes da execução;
- facilita explicar por que uma pergunta foi recusada;
- reduz consultas caras ou ambíguas.

O SQL livre poderia existir apenas em modo experimental ou administrativo, fora do caminho público do MVP.

## Decisão 2: tiraria o ranking de títulos do centro do produto

Eu priorizaria o agente de análise de hipóteses com evidência e deixaria a geração de títulos como demonstração secundária.

Motivos:

- o modelo foi treinado em vídeos que já entraram em alta;
- há risco de vazamento temporal;
- SHAP não transforma associação em recomendação causal;
- scores de títulos podem dar uma aparência de precisão maior que a validade real;
- análise de hipóteses demonstra melhor SQL, avaliação, rastreabilidade e disciplina estatística.

Se mantivesse títulos, apresentaria uma tabela de trade-offs, não uma ordenação simples: score associativo, clareza, adequação ao tema, risco de clickbait e incerteza.

# 4. Cinco perguntas difíceis de entrevista

## 1. “Seu modelo prevê views de qualquer vídeo novo?”

Resposta esperada:

> Não. O modelo é condicional à população observada: vídeos que entraram em alta entre outubro de 2024 e maio de 2026. Ele estima padrões associados a vídeos semelhantes aos da base, não a probabilidade universal de sucesso nem a chance de entrar em alta. Também preciso controlar possíveis vazamentos temporais nas features.

## 2. “Como você garante que o SQL gerado pelo LLM é seguro?”

Resposta esperada:

> Uso defesa em camadas: conexão read-only, AST com `sqlglot`, uma única instrução, allowlist de tabelas, bloqueio de funções externas e comandos perigosos, limites de tempo, memória, linhas e complexidade, além de execução isolada e logs. Ainda assim, não considero “SELECT somente leitura” suficiente contra negação de serviço; por isso, o ideal é usar intenções estruturadas e SQL por templates para os casos principais.

## 3. “Por que usar SHAP para escolher títulos?”

Resposta esperada:

> SHAP ajuda a explicar como o modelo distribuiu sua previsão entre as features, mas não mede o efeito causal do título. O score indica apenas compatibilidade com padrões estatísticos da base. Eu não diria que o sistema encontra o melhor título; apresentaria alternativas, fatores, incerteza e risco de clickbait.

## 4. “Como você sabe que o agente respondeu corretamente?”

Resposta esperada:

> Não avalio apenas se o SQL executou. Tenho casos estratificados com respostas de referência, tolerância numérica, filtros e agrupamentos esperados, ferramentas esperadas, recusa causal, segurança SQL, ausência de números inventados, latência e custo. Também comparo com baselines e mantenho um conjunto `test` separado do conjunto usado para ajustar o prompt.

## 5. “O que acontece quando a API falha ou o modelo gera uma resposta errada?”

Resposta esperada:

> O sistema tem timeout, limite de chamadas, tratamento de erros, modo demo sem API e rastreamento da execução. Se a ferramenta falhar, o agente não deve inventar um resultado: deve informar que não conseguiu concluir. Para reduzir erros, separo dados determinísticos da redação do LLM, valido as consultas antes da execução e verifico se os números apresentados aparecem nos resultados obtidos.