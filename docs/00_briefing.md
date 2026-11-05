# Briefing: YouTube Insights Agent

## Quem e para quê
Eduardo Lima da Silva, estudante do 5º semestre de Ciência de Dados e IA (PUC-Campinas), vai se candidatar a um
programa de estágio com duas áreas escolhidas: **Tech - Engenharia de AI/ML (AI Engineer)** e **Marketing - AI Builder**.
No formulário, ele marcou que usa **prompt engineering avançado e orquestração de agentes (Claude, GPT, Gemini)**.
Ele já tem um projeto de ML clássico e precisa de um **segundo projeto de portfólio** que prove que ele constrói
**sistemas com LLMs**: agentes, uso de ferramentas, integração de APIs, avaliação e deploy.

## Ativos já existentes (projeto 1: `youtube-views-prediction`, público no GitHub)
- Base: 9,36 milhões de registros (19,5 GB CSV) de vídeos em alta do YouTube em 112 países (out/2024 a mai/2026),
  com uma linha por vídeo × país × dia em alta. Colunas: ids, datas de publicação e de trending, país, título,
  descrição, categoria, tags, duração, definição, views, likes, comentários e estatísticas do canal (inscritos, views, nº de vídeos, país).
- Parquet enxuto (`trending_slim.parquet`, 9,4 M linhas, ~420 MB) e Parquet por vídeo
  (`videos_features.parquet`, 956 mil vídeos, 37 features).
- Modelo LightGBM (`lgbm_views.txt`, 29 MB) que prevê log(views) após N dias, com R² de 0,85 em teste temporal e
  84% das previsões dentro de 2× do valor real. Também há um TF-IDF + Ridge sobre título/tags (R² de 0,38 isolado) e um `predict.py`.
- Achados: o modelo mede associação, não causalidade; a base só contém vídeos que entraram em alta (viés de seleção).

## Ideia proposta
Um **agente com Claude (API Anthropic, Python SDK, tool use)** que responde perguntas de estratégia de conteúdo em
português, usando ferramentas sobre os dados e o modelo:
1. `query_data(sql)`: SQL **somente leitura** no DuckDB sobre tabelas agregadas, com allowlist de tabelas, limite de linhas e timeout.
2. `make_chart(spec)`: gera um gráfico a partir do resultado de uma consulta.
3. `suggest_titles(idea, n)` + `predict_views(title, tags, category, duration, channel...)`: gera títulos com o LLM e
   ordena cada um com o LightGBM (deixando explícito que é associação, não causa).
4. `explain_prediction(...)`: contribuições SHAP da previsão.

Interface em **Streamlit**, com deploy gratuito no Streamlit Community Cloud. Como os dados completos não sobem lá,
o deploy usaria uma amostra ou agregados (<100 MB). Haveria uma **avaliação offline**: ~30 perguntas com resposta
calculada por SQL (gabarito determinístico), medindo a % de acerto numérico, a validade do SQL, o uso correto de
ferramentas e a recusa de perguntas fora do escopo, além de custo e latência por pergunta.

## Restrições
- Máquina Windows com 8 GB de RAM; Python 3.13; o usuário ainda não tem chave de API configurada.
- Precisa ser algo que um estudante consiga **explicar em entrevista** e que não pareça "wrapper de chat".
- Prazo curto: o primeiro MVP precisa funcionar em poucos dias.
- Honestidade: nada de inventar métricas de negócio.
