# Análises profundas: o que dá para tirar dos dados (e por que importa)

Oito perguntas de estratégia de conteúdo feitas **ao vivo** ao agente (Groq, `gpt-oss-120b`, plano gratuito) em 05/11/2026.
Cada análise foi respondida com **uma única consulta SQL** em 5 a 57 segundos e gravada em `demo_traces/05_*.json` até `12_*.json`, que podem ser reproduzidas no modo demo do app.

**Como os números foram validados:** cada número abaixo foi **recalculado por mim com SQL independente** no warehouse. Quando o agente errou ou simplificou algo, isso aparece na coluna "Revisão".

> ⚠️ **Leia antes:** a base só tem vídeos que **entraram em alta** (out/2024–mai/2026). Os achados são **associações dentro desse grupo**, não causas, e não valem para "todo o YouTube". As implicações são **hipóteses para testar**, por exemplo com testes A/B nos próprios vídeos.

| # | Pergunta | Achado (verificado) | Por que importa: o que dá para fazer | Revisão |
|---|---|---|---|---|
| 1 | Quais **palavras** aparecem mais em títulos de **carros** em alta no Brasil? | "novo" e "2025" (12× cada), "carro", marcas e modelos (Honda, Creta, Toyota, Polo, Fusca) | Lançamentos e modelos específicos dominam: dá para **planejar pautas e tags** em torno de lançamentos do ano e de modelos populares | ✅ números batem. ⚠️ Amostra pequena: só 69 vídeos de carros em alta no Brasil |
| 2 | Em **Gaming**, quais **palavras** estão associadas a **mais views** (≥ 50 vídeos)? | "reveal" tem mediana de **2,0 mi** contra **56,8 mil** da mediana geral do Gaming no BR (~35×); depois vêm "animation" (1,26 mi), "official" (1,0 mi) e nomes de criadores ("loko", "paulinho", ~0,9 mi) | Anúncios e revelações oficiais e criadores específicos concentram audiência: sugere **cobrir reveals rapidamente** e considerar **collabs** | ✅ top da lista bate. ⚠️ Em palavras repetidas no mesmo título o agente contou ocorrências, não vídeos ("granny": 473 contra 245 vídeos distintos) |
| 3 | Quais **categorias** entram em alta **mais rápido** depois de publicadas (BR)? | Mediana de horas até a alta: **Gaming ~12 h**, Sports e Entertainment ~14 h … **Music ~26 h** | Define **quando agir**: em Gaming a janela de impulsionamento (comunidade, ads, shorts de corte) são as primeiras horas; em Música o ciclo é mais longo | ✅ bate (calculado também com a data de entrada no Brasil: diferença < 1,5 h) |
| 4 | Em que **horário de Brasília** os vídeos de **Música** em alta foram publicados com mais views? | **2h: mediana de 2,1 mi** (n=96) e **1h: 1,8 mi** (n=170), contra ~100 mil no meio do dia (13h: 103 mil, n=355) | Hipótese: lançamentos de grandes artistas em horário global fixo. Para quem lança música, ajuda a **planejar o calendário** (evitar disputar atenção com grandes lançamentos ou surfar neles) | ✅ bate. ⚠️ Horários com n=28–43 têm pouca amostra |
| 5 | Quais categorias **viajam para mais países**? | Média de países por vídeo: **Howto & Style 3,1**, Film & Animation 2,9, Sports 2,8 … Music 2,0, News 1,7. A mediana é 1 em todas | Conteúdo "universal" (tutoriais, animação, esporte) tem mais potencial **internacional**: justifica **legendas, títulos traduzidos e distribuição multi-país** | ✅ bate exatamente |
| 6 | Títulos **com "?"** têm mais views (BR, por categoria)? | Depende da categoria: **People & Blogs +29%** e Science & Tech +8%, mas **Sports −70%**, **Film & Animation −88%**, Gaming −30%, Entertainment −29% | Não existe "regra universal de título": a **fórmula de título deve ser testada por nicho** (A/B), e não copiada de outra categoria | ✅ tabela bate. ❌ **o texto do agente errou a faixa** ("−30% a +8%"); a correta é de −88% a +29% |
| 7 | Qual categoria **engaja mais** (likes/view, BR)? | **Autos 7,5%**, Comedy 7,4%, Entertainment 7,1% … **Music 1,9%** | A meta de engajamento **deve ser relativa à categoria**: 3% de likes/view é ótimo para Música e fraco para Comédia. Serve como **benchmark** de desempenho | ✅ bate |
| 8 | Em quais categorias os **Shorts** superam os longos (BR)? | **Gaming: 573 mil × 55 mil (~10×)**, Entertainment ~4×, People & Blogs ~2×, Music ~1,8×. Em Sports há empate (378 × 365 mil) e em Film & Animation os longos ganham | A **estratégia de formato** muda por categoria: Shorts são alavanca clara em Gaming e Entertainment, mas não em Sports nem em Film | ✅ bate |

## O que essas análises mostram sobre o projeto

- **Vão além do "quantos" e do "qual a média":** palavras-chave (NLP simples em SQL com `unnest`/`string_split`), velocidade de viralização, sazonalidade por horário, alcance internacional e engajamento relativo.
- **Cada achado vira uma decisão de negócio testável:** pauta, SEO e tags, calendário de publicação, formato, internacionalização e benchmark de métricas.
- **Revisão honesta:** 8/8 análises responderam com dados corretos na tabela. Em 1/8 o texto interpretou errado a magnitude, e em 1/8 a contagem misturou ocorrências com vídeos. Isso reforça por que o sistema mostra o SQL, os dados brutos ("Dados da consulta") e checa os números automaticamente.

## Como reproduzir

No app, ative **"Ver execuções gravadas"** e escolha a análise, ou faça a mesma pergunta no modo ao vivo. Script usado: as perguntas estão nos arquivos `demo_traces/05_*.json` a `12_*.json` (campo `question`).
