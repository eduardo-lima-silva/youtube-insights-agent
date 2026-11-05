Ver resumo em [[Revisão Codex 1 - Ideia]].

# 1. Veredito

**Nota: 8,0/10.**

A ideia é forte para o objetivo do candidato porque demonstra, em um único projeto:

- integração com API de LLM;
- uso de ferramentas;
- geração e validação de SQL;
- integração com DuckDB;
- uso de modelo de ML já treinado;
- explicabilidade com SHAP;
- geração de visualizações;
- avaliação offline;
- preocupação com custo, latência, escopo e deploy.

Isso cobre exatamente o que um avaliador espera de alguém que afirma conhecer *prompt engineering* e orquestração de agentes.

O ponto forte principal é que o LLM não seria apenas um chatbot: ele precisaria consultar dados, chamar um modelo preditivo e explicar resultados. A conexão com o projeto existente também dá continuidade ao portfólio.

A nota não é maior porque, no formato atual, existe risco de o projeto parecer uma combinação genérica de “chat com dados + gráficos + títulos gerados”. O diferencial não está no agente em si, mas na qualidade da engenharia:

- controle rigoroso do SQL;
- avaliação reproduzível;
- tratamento honesto do viés dos dados;
- separação entre associação e causalidade;
- demonstração de que o agente sabe quando não responder;
- explicação clara de como o sistema falha.

Se esses pontos forem bem executados, o projeto pode ser muito competitivo para estágio. Se forem apenas declarados no README, mas não demonstrados, pode parecer um *wrapper* de API.

# 2. Riscos e pontos fracos

## Técnicos

### O modelo preditivo pode estar mal alinhado com o caso de uso

O LightGBM prevê views futuras para vídeos que já entraram em alta. Isso não equivale a prever o desempenho de qualquer novo vídeo publicado.

Um avaliador pode perguntar:

> “Como o sistema prevê o desempenho de um vídeo que nunca entraria em trending?”

A resposta honesta é: provavelmente não prevê bem. O modelo sofre de viés de seleção e deve ser apresentado como um estimador condicional para vídeos semelhantes aos observados na base, não como um preditor universal de sucesso.

### Possível vazamento temporal

É necessário verificar se todas as features usadas na previsão estavam disponíveis no instante em que a previsão supostamente seria feita. Exemplos perigosos:

- estatísticas do canal coletadas depois da publicação;
- views, likes ou comentários posteriores ao horizonte de previsão;
- agregações que incluem dados futuros;
- títulos ou metadados atualizados depois do lançamento.

Se houver vazamento, o R² de 0,85 perde credibilidade. Um entrevistador técnico experiente provavelmente investigará isso.

### “Sugestão de títulos” pode ser mais marketing do que AI Engineering

Gerar títulos com LLM e ranqueá-los pelo LightGBM é demonstrável, mas tecnicamente frágil:

- pequenas mudanças no título podem não alterar as features relevantes do modelo;
- o modelo pode não ter capacidade real de avaliar semântica;
- o LLM pode gerar títulos artificiais ou pouco adequados ao canal;
- otimizar apenas views pode incentivar clickbait;
- não há métrica de qualidade editorial, clareza, originalidade ou adequação à marca.

O candidato precisa evitar afirmar que o sistema encontra “o melhor título”. O correto seria: **“gera alternativas e as ordena segundo o padrão estatístico aprendido na base.”**

### Muitas ferramentas para um MVP curto

O escopo mistura:

- agente;
- SQL;
- gráficos;
- geração de títulos;
- predição;
- SHAP;
- deploy;
- avaliação;
- controle de custos.

Isso aumenta a superfície de bugs e pode resultar em uma demo superficial. Para poucos dias, o risco maior é entregar seis componentes medianos em vez de dois componentes muito bem construídos.

### Dependência da API da Anthropic

A chave ainda não está configurada, há custo variável e podem existir diferenças de comportamento entre modelos. Também é necessário tratar:

- rate limit;
- indisponibilidade da API;
- respostas malformadas;
- chamadas repetidas;
- timeout;
- controle de gastos.

Uma aplicação que simplesmente quebra quando a API falha transmite pouca maturidade de engenharia.

### Compatibilidade do ambiente

Python 3.13, bibliotecas de ML, SHAP, LightGBM, DuckDB e Streamlit podem exigir atenção especial a versões e binários. Isso deve ser validado cedo, especialmente no deploy.

## Produto

### O usuário-alvo está indefinido

“Estratégia de conteúdo” pode significar coisas diferentes para:

- um criador individual;
- uma agência;
- um time de marketing;
- um canal educacional;
- uma marca.

Sem um público definido, as respostas tendem a ser genéricas. O projeto ficaria mais forte com uma persona explícita, por exemplo:

> “Analista de conteúdo de um canal brasileiro que quer comparar padrões de vídeos em alta e explorar hipóteses para novos vídeos.”

### A base não representa todo o YouTube

A base contém vídeos que entraram em alta, em 112 países e em um período específico. Ela não representa:

- vídeos que nunca entraram em alta;
- canais pequenos fora da lista de trending;
- conteúdo de nichos sub-representados;
- mudanças futuras no algoritmo;
- necessariamente o comportamento do público brasileiro;
- diferenças de idioma e cultura.

O produto não pode apresentar recomendações como regras gerais do YouTube.

### Valor prático limitado sem dados do canal do usuário

Perguntas como “qual duração funciona melhor para meu canal?” não podem ser respondidas de forma personalizada sem dados históricos daquele canal. O sistema deve diferenciar claramente:

- análise descritiva da base;
- previsão condicional;
- recomendação personalizada.

## Credibilidade em entrevista

Os principais pontos que podem prejudicar o candidato são:

- chamar o sistema de “agente autônomo” quando ele é um fluxo com poucas ferramentas;
- afirmar causalidade a partir de correlações;
- usar “acurácia” para previsões contínuas;
- apresentar R² alto sem discutir seleção e vazamento;
- dizer que o modelo “otimiza títulos” sem validar títulos reais;
- medir apenas se o SQL executou, e não se respondeu à pergunta;
- esconder exemplos de falha;
- demonstrar apenas perguntas fáceis e pré-selecionadas.

Um avaliador tende a confiar mais em um candidato que mostra:

1. uma resposta correta;
2. uma resposta recusada corretamente;
3. uma resposta em que o agente detectou baixa confiança;
4. uma limitação conhecida e documentada.

## Custo

As chamadas podem ficar caras se:

- o agente gerar SQL inválido e tentar novamente muitas vezes;
- cada pergunta usar várias chamadas LLM;
- o histórico inteiro da conversa for reenviado;
- o sistema gerar gráficos e explicações em chamadas separadas;
- prompts incluírem schemas extensos;
- o usuário fizer perguntas abertas com consultas grandes.

É necessário medir custo por interação e impor limites de:

- número de chamadas;
- tokens;
- tempo;
- tamanho do resultado;
- custo máximo por sessão.

## Segurança do SQL gerado por LLM

“SQL somente leitura” não é suficiente como controle de segurança.

Riscos específicos:

- `DROP`, `DELETE`, `UPDATE`, `INSERT` ou `CREATE`;
- múltiplas instruções na mesma resposta;
- uso de funções perigosas;
- consultas sem filtro que varrem grandes tabelas;
- `SELECT *`;
- subconsultas excessivamente caras;
- acesso a tabelas não autorizadas;
- leitura de arquivos locais por funções do DuckDB;
- tentativa de exploração via comentários SQL;
- consultas que retornam dados sensíveis, caso a base evolua;
- negação de serviço por produtos cartesianos e agregações enormes.

A proteção deve ocorrer em camadas:

1. banco ou conexão realmente read-only;
2. allowlist de tabelas e colunas;
3. parser ou analisador de AST, não apenas regex;
4. bloqueio de múltiplas instruções;
5. limites de tempo, linhas, memória e tamanho;
6. views agregadas pré-criadas;
7. execução em processo ou ambiente isolado;
8. logs do SQL e do resultado;
9. validação dos nomes de tabelas e colunas;
10. fallback para perguntas que não podem ser traduzidas com segurança.

## Viés da base

Há vários vieses que precisam aparecer na interface e no README:

- viés de seleção: apenas vídeos em alta;
- viés geográfico: países com maior cobertura ou maior volume;
- viés temporal: outubro de 2024 a maio de 2026;
- viés de categoria;
- viés de idioma;
- viés de sobrevivência;
- possível duplicação de vídeos por país e dia;
- possível diferença de qualidade na coleta entre países;
- correlação entre tamanho do canal e chance de aparecer em trending.

“Canais maiores têm mais views” não é uma recomendação acionável por si só. E “vídeos longos performam melhor” pode apenas refletir categorias ou canais específicos.

# 3. Três visões alternativas ou variações

## Alternativa 1 — Agente de auditoria e análise de hipóteses sobre YouTube

Em vez de focar em “sugerir o próximo título”, o sistema responderia perguntas analíticas e produziria uma resposta com:

- SQL executado;
- visualização;
- intervalo ou distribuição relevante;
- limitações;
- indicação de associação, não causalidade;
- hipótese que poderia ser testada.

Exemplo:

> “Vídeos de 8 a 12 minutos têm mais views em canais pequenos no Brasil?”

O agente teria que definir corretamente o grupo, explicitar filtros e apontar se a amostra é suficiente.

**Prós**

- Mais alinhado a AI Engineering e analytics;
- permite avaliar SQL, raciocínio e explicabilidade;
- reduz a promessa exagerada de prever sucesso;
- torna o viés da base parte central do produto;
- demo fácil de entender.

**Contras**

- Menos “vistoso” do que geração de títulos;
- exige maior cuidado com definição de métricas;
- ainda é, em parte, um agente de perguntas sobre dados.

**MVP estimado:** 3 a 5 dias.

## Alternativa 2 — Sistema de avaliação de títulos com múltiplos critérios

O LLM gera títulos, mas o produto não os ranqueia apenas por views previstas. Ele avalia cada alternativa em dimensões separadas:

- aderência ao tema;
- clareza;
- especificidade;
- potencial de curiosidade;
- risco de clickbait;
- adequação ao público;
- similaridade excessiva com títulos existentes;
- score estatístico do modelo de views.

A saída incluiria uma tabela de trade-offs, não apenas uma classificação única.

**Prós**

- Muito demonstrável para Marketing AI Builder;
- mostra composição de LLM com modelo clássico;
- introduz avaliação de qualidade e segurança de conteúdo;
- permite discussão madura sobre otimização multiobjetivo.

**Contras**

- Os scores do LLM podem ser subjetivos;
- o modelo de views pode não generalizar para títulos novos;
- exige uma política clara para “qualidade”;
- risco de parecer ferramenta de copywriting com IA.

**MVP estimado:** 4 a 7 dias.

## Alternativa 3 — Copiloto de monitoramento e diagnóstico de tendências

Essa é a alternativa mais diferente de “agente de perguntas sobre dados”.

O sistema analisaria snapshots agregados e detectaria:

- categorias ou países com mudança de volume;
- crescimento anormal;
- títulos ou termos emergentes;
- mudanças na duração, tags ou canais;
- possíveis mudanças de distribuição ao longo do tempo.

Em seguida, produziria um briefing periódico:

> “O que mudou nesta semana, qual evidência sustenta isso e quais hipóteses de conteúdo merecem investigação?”

**Prós**

- Demonstra pipeline, análise temporal, detecção de mudança e geração de relatório;
- é mais próximo de uma aplicação real de Marketing AI;
- reduz dependência de conversas abertas com o agente;
- permite uma demo muito visual e objetiva;
- evidencia que o sistema não depende apenas de prompting.

**Contras**

- exige pipeline de dados ou snapshots bem definidos;
- pode ser difícil provar que uma tendência é relevante;
- detecções falsas podem gerar relatórios enganosos;
- envolve mais trabalho de produto e visualização.

**MVP estimado:** 5 a 8 dias.

Minha recomendação seria a Alternativa 1 como núcleo técnico, com uma versão reduzida da Alternativa 2 como demonstração de marketing. A Alternativa 3 tem o maior potencial de diferenciação, mas é menos segura para um prazo extremamente curto.

# 4. Se mantiver a ideia original

## Escopo mínimo recomendado

O MVP deveria ter somente três capacidades principais:

1. responder perguntas analíticas sobre um conjunto pequeno de views agregadas;
2. gerar e comparar até três títulos;
3. explicar uma previsão com fatores relevantes e limitações.

Eu cortaria inicialmente:

- suporte a múltiplos agentes;
- memória longa de conversa;
- upload de arquivos;
- ingestão automática de dados;
- dezenas de ferramentas;
- personalização para vários perfis de usuário;
- deploy como prioridade;
- otimização sofisticada de prompts;
- geração automática de relatórios extensos;
- uso do dataset completo na nuvem.

## O que é obrigatório

- schema explícito e pequeno;
- catálogo das tabelas e colunas disponíveis;
- SQL com allowlist;
- conexão verdadeiramente read-only;
- limite de tempo, memória e linhas;
- bloqueio de múltiplas instruções;
- logs de pergunta, SQL, execução e custo;
- respostas com evidência;
- recusa de perguntas fora do escopo;
- aviso visível sobre associação, não causalidade;
- distinção entre dados observados e previsão;
- avaliação reproduzível;
- pelo menos um caso de erro demonstrado;
- fallback quando a API não estiver disponível;
- README com arquitetura, limitações e instruções de execução.

## Arquitetura sugerida

Fluxo recomendado:

1. **Interface Streamlit**
   - recebe a pergunta;
   - mostra resposta, gráfico, SQL e limitações;
   - permite abrir detalhes técnicos.

2. **Camada de orquestração**
   - classifica a intenção:
     - análise descritiva;
     - predição;
     - geração de títulos;
     - explicação;
     - fora de escopo;
   - decide quais ferramentas podem ser usadas;
   - impõe limite de chamadas.

3. **Catálogo semântico**
   - descreve tabelas, colunas, unidades, filtros válidos e limitações;
   - informa explicitamente que a base contém vídeos em alta.

4. **Camada segura de consulta**
   - recebe uma intenção estruturada ou SQL;
   - valida AST;
   - aplica allowlist;
   - executa apenas sobre views agregadas;
   - limita custo e tamanho do resultado.

5. **Camada analítica**
   - calcula agregações determinísticas;
   - chama o LightGBM;
   - calcula SHAP;
   - gera tabelas prontas para visualização.

6. **Camada de apresentação**
   - transforma resultados em narrativa;
   - inclui evidência numérica;
   - mostra incertezas e ressalvas;
   - impede que o LLM invente valores ausentes.

7. **Observabilidade e avaliação**
   - registra latência, tokens, custo, SQL, erros, ferramentas chamadas e resultado;
   - executa o eval set fora da interface.

Uma decisão importante: o LLM não deveria receber liberdade irrestrita para escrever qualquer SQL. Para as perguntas mais comuns, é preferível que ele produza uma intenção estruturada validada e que o sistema monte a consulta a partir de templates seguros. O SQL gerado pode existir como modo avançado, mas não precisa ser o único caminho.

# 5. Desenho da avaliação

## Como montar o eval set

Trinta perguntas são poucas para tirar conclusões fortes, mas suficientes para um MVP se forem cuidadosamente estratificadas.

Eu montaria entre 60 e 100 casos, divididos em categorias:

- 20 perguntas simples de agregação;
- 15 perguntas com filtros e agrupamentos;
- 10 comparações entre períodos ou países;
- 10 perguntas ambíguas;
- 10 perguntas fora do escopo;
- 10 perguntas que exigem recusa por causalidade;
- 10 perguntas adversariais para SQL;
- 5 casos de geração de títulos;
- 5 casos de explicação de previsão.

Cada caso deveria conter:

- pergunta do usuário;
- intenção esperada;
- tabelas e colunas permitidas;
- resposta numérica ou faixa esperada;
- SQL de referência, quando aplicável;
- ferramentas esperadas;
- comportamento esperado em caso de recusa;
- critérios de aceitação;
- limitações que deveriam ser mencionadas.

O conjunto deve ter perguntas geradas por humanos, não apenas variações automáticas de um mesmo template. Caso contrário, o agente pode memorizar padrões de linguagem.

## Métricas

### Correção da resposta

- erro absoluto ou erro percentual para números;
- correção de filtros, agrupamentos e período;
- acerto de unidades;
- concordância com a resposta de referência;
- correção do ranking, quando houver;
- presença de evidência que realmente sustenta a conclusão.

Não usaria “acurácia” como métrica genérica para tudo.

### Correção do SQL

- taxa de SQL válido;
- taxa de execução bem-sucedida;
- equivalência semântica com a consulta de referência;
- tabelas e colunas corretas;
- ausência de operações proibidas;
- custo e tempo dentro do limite;
- quantidade de linhas retornadas.

SQL executável não significa SQL correto.

### Uso de ferramentas

- ferramenta correta chamada;
- ordem correta quando houver dependência;
- ausência de chamadas desnecessárias;
- uso do modelo preditivo somente quando apropriado;
- não geração de gráfico a partir de dados incorretos.

### Recusa e segurança

- taxa de recusa correta para perguntas fora do escopo;
- taxa de não recusa em perguntas válidas;
- resistência a instruções maliciosas;
- bloqueio de SQL perigoso;
- não exposição de dados além do permitido.

Uma recusa excessiva também é falha de produto.

### Qualidade da resposta

Avaliação humana em escala simples para:

- clareza;
- fidelidade aos dados;
- utilidade;
- transparência;
- tom adequado;
- ausência de causalidade indevida;
- qualidade visual.

### Operação

- latência mediana e p95;
- custo médio por pergunta;
- número médio de chamadas;
- taxa de timeout;
- taxa de fallback;
- tamanho médio do resultado.

## Como evitar um eval enganoso

- manter um conjunto de teste separado e nunca usar suas respostas no prompt;
- incluir paráfrases e perguntas inéditas;
- variar nomes de países, períodos e categorias;
- testar casos com zero resultados;
- testar perguntas ambíguas;
- incluir números muito próximos para detectar confusão;
- verificar o significado da consulta, não apenas execução;
- incluir perguntas que exigem dizer “não é possível responder”;
- executar o eval várias vezes, pois respostas de LLM podem variar;
- fixar versão do modelo e temperatura quando possível;
- guardar exemplos de falha, não apenas a média;
- relatar intervalos ou distribuição das métricas;
- comparar contra baselines:
  - SQL determinístico sem LLM;
  - resposta manual;
  - agente sem catálogo;
  - agente sem validação.

O caso mais perigoso seria obter 95% de execução válida em 30 perguntas simples e concluir que o agente está pronto. Isso mede apenas que ele reconhece padrões básicos, não que raciocina corretamente sobre dados.

# 6. O que faria um avaliador dizer “uau” em 2 minutos

A demo deveria ser curta e roteirizada em três momentos.

## Momento 1: uma pergunta útil com evidência

Pergunta:

> “Compare a mediana de views de vídeos em alta de 8 a 12 minutos contra vídeos de 4 a 8 minutos no Brasil, no último período disponível.”

O sistema deveria mostrar:

- resposta direta;
- número da amostra;
- gráfico;
- SQL executado;
- tempo de execução;
- uma observação sobre a base conter apenas vídeos em alta.

Isso demonstra utilidade, rastreabilidade e controle.

## Momento 2: uma pergunta que o sistema recusa

Pergunta:

> “Qual duração causa mais views?”

A resposta ideal:

> “Não posso afirmar causalidade com esta base. Posso mostrar a associação observada, controlar por categoria ou comparar distribuições.”

Essa recusa provavelmente impressionaria mais do que uma resposta confiante. Ela mostra maturidade estatística e de produto.

## Momento 3: geração de títulos com trade-offs

Entrada:

> “Quero criar um vídeo sobre como estudar Python para entrevistas de dados.”

O sistema deveria retornar três opções com:

- título;
- score preditivo;
- fatores que influenciaram a previsão;
- risco de clickbait;
- observação de que o score é associativo;
- recomendação final explicando o trade-off.

O avaliador deveria conseguir ver que a recomendação não é simplesmente “o título com maior número”.

## O detalhe que mais diferencia

Mostrar, em uma aba ou painel, o rastro completo:

- intenção detectada;
- ferramenta usada;
- SQL validado;
- resultado;
- modelo chamado;
- custo;
- limitações;
- confiança ou motivo de recusa.

Em dois minutos, isso comunica que o candidato sabe construir um sistema controlável e avaliável, não apenas conversar com um modelo.

A mensagem final da demo deveria ser:

> “O LLM interpreta a intenção e orquestra ferramentas; os números vêm de consultas e modelos determinísticos; o sistema expõe suas limitações e é avaliado por casos reproduzíveis.”

Essa frase posiciona o projeto muito melhor do que chamá-lo simplesmente de “agente de insights do YouTube”.