# 01 · Recomendador de filmes por descrição

O usuário descreve o que quer assistir ("um suspense psicológico tipo Ilha do Medo, mas mais leve") e recebe filmes de sinopse parecida, com a explicação de por que combinam. A conversa permite refinar ("mais antigo", "me mostre outros") sem repetir filmes.

**Status:** funcional localmente, com 66 testes, tracing opcional no Langfuse, dois catálogos (amostra em português e **5000 filmes em inglês do CMU**), busca por descrição e por **filme de exemplo**, e avaliação da busca com 3 embedders. O agente foi testado à mão com o Gemini Flash-Lite nos dois modos. Falta avaliar o texto do agente, medir a busca por descrição no catálogo grande e publicar a demo.

## Como funciona

```
pergunta ─► agente (LLM + tool use) ─┬─► search_movies(descrição, filtros)      "um suspense tipo..."
                                       └─► similar_movies(título, filtros)        "algo como Matrix"
                                              │
                    ┌─────────────────────────┴───────────────────────┐
                    ▼                                                 ▼
          embedding da descrição ─► Qdrant (vetorial, com filtros)   BM25 (lexical, mesmos filtros)
                    └──────────────────► Reciprocal Rank Fusion ◄─────┘
                                              │
              cartões de filmes (dados do catálogo) ─► interface    +    texto do LLM explicando
```

- **Catálogo** (`app/catalog.py`): CSV com título, ano, gêneros, sinopse (nota e pôster opcionais). Três fontes: uma **amostra de 44 filmes em português** (sinopses escritas por mim) para rodar offline; o **CMU Movie Summary Corpus em inglês** (`scripts/prepare_cmu.py`, abaixo); e o TMDB (`scripts/fetch_tmdb.py`). `PORTFOLIO_FILMES_CATALOG` (`sample`, `cmu`, `tmdb`) força uma; sem isso vale a primeira que existir, nessa ordem: TMDB, CMU, amostra.
- **Idioma:** cada catálogo tem o seu (`en` para o CMU). O prompt manda o agente **escrever a consulta no idioma do catálogo, traduzindo o pedido do usuário** ("um suspense com final surpreendente" vira "a thriller with a surprising twist ending") e usar o título original em inglês (`Inception`, não "A Origem"); a resposta ao usuário continua no idioma dele. A tradução é feita pelo próprio modelo, sem uma chamada a mais.
- **Embeddings** (`app/embeddings.py`), mesma regra do Chatbot: Ollama local (`LOCAL_EMBEDDING_ENABLED`, `nomic-embed-text`) → Google (`gemini-embedding-001`, com `GOOGLE_API_KEY`) → `HashingEmbedder` offline, que **não tem semântica real** e só serve de linha de base e para testes.
- **Índice** (`app/index.py`): Qdrant em modo local (mesma API do servidor), com filtros por gênero, época e nota. Uma coleção por (embedder, catálogo); trocar de modelo recria o índice.
- **Busca híbrida:** vetor + BM25 (`portfolio_core.bm25`) combinados por RRF ponderado, com o peso do BM25 definido por embedder a partir da avaliação (0 = só vetor com o Google; 1 com os demais). Os filtros valem nos dois lados.
- **Agente** (`app/agent.py`): duas ferramentas. `search_movies` recebe uma **descrição** (o modelo reescreve o pedido numa descrição rica); `similar_movies` recebe o **título de um filme de exemplo** e compara a sinopse dele com as do catálogo: a parte vetorial reaproveita o embedding que o índice já guarda do filme (sem chamar o embedder de novo), a parte lexical usa a sinopse como consulta do BM25, e o próprio filme nunca volta. Se o título não existe, a ferramenta devolve títulos parecidos e manda descrever a trama com `search_movies`; se há refilmagens com o mesmo título, pede o ano. Filtros só se o usuário pedir; filmes já mostrados não se repetem (`include_shown` libera). O loop é o genérico do `portfolio_core`.
- **É um RAG:** a busca recupera sinopses do catálogo, elas entram no contexto do modelo como resultado da ferramenta, e o modelo escolhe e explica as recomendações a partir delas (a resposta é gerada sobre o que foi recuperado, não de memória).
- **Anti-alucinação:** os cartões vêm da ferramenta, não do texto do modelo; o prompt manda recomendar só filmes devolvidos e não inventar elenco, nota ou cenas.

## Rodar localmente

Na raiz do portfólio, com o venv criado:

```bash
.venv/Scripts/python -m pip install -e "_core[langchain]" -r 01-recomendador-filmes/requirements.txt pytest pytest-asyncio httpx
cd 01-recomendador-filmes
../.venv/Scripts/python -m uvicorn app.main:create_app --factory --port 8003
```

Abra `http://localhost:8003`. Sem chaves, o LLM é o `echo` (a interface abre, mas não conversa). Com `GOOGLE_API_KEY`, usa o Gemini Flash-Lite e os embeddings do Google; com `LOCAL_EMBEDDING_ENABLED=true`, embeddings no Ollama. O índice fica em `data/index/` (ignorado pelo git). **Se mudar de embedder, o índice é recriado sozinho.**

### Catálogo grande em inglês (CMU Movie Summary Corpus)

O corpus (Bamman, O'Connor e Smith, ACL 2013; licença **CC BY-SA**) tem as sinopses de 42 mil filmes extraídas da Wikipedia em inglês, com ano, gêneros e bilheteria. **Não tem notas nem pôsteres.** O IMDb não serve para esta busca: os datasets oficiais têm título, ano, gêneros e nota, mas não têm sinopse.

```powershell
# baixe https://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz (cerca de 46 MB) e extraia
..\.venv\Scripts\python scripts\prepare_cmu.py --src C:\caminho\MovieSummaries --max 5000
$env:PORTFOLIO_FILMES_CATALOG = "cmu"      # (opcional se não houver data/tmdb_movies.csv)
```
O script limpa a marcação da Wikipedia que sobrou nas sinopses (`{{plot}}`, `<ref>`, links), descarta sinopses com menos de 300 caracteres, corta as maiores (1500) em fim de frase e escreve `data/cmu_movies.csv` (ignorado pelo git). **`--max` importa:** o índice precisa de um embedding por filme; com os 42 mil filmes isso leva horas no Ollama ou estoura a cota gratuita do Google. Quando há mais filmes que vagas, ficam primeiro os que têm bilheteria registrada (um indício de popularidade, não de qualidade). O app mostra a atribuição exigida pela licença.

### Catálogo grande em português (TMDB)

```powershell
$env:TMDB_API_KEY = "sua-chave-v3"
..\.venv\Scripts\python scripts\fetch_tmdb.py --pages 50    # ~1000 filmes mais bem avaliados, em pt-BR
```

Reinicie o app: ele passa a usar `data/tmdb_movies.csv`, mostra notas e pôsteres e exibe a atribuição exigida pelo TMDB. Confira os [termos de uso](https://www.themoviedb.org/api-terms-of-use) (uso comercial é restrito) antes de publicar. O script não foi testado contra a API real.

## Avaliação da busca

`evaluation/` mede a busca **sem LLM**: 30 consultas (16 diretas, 6 por paráfrase, com pouca sobreposição de palavras, e 8 por nome ou palavra exata) com filmes relevantes julgados a partir das sinopses; compara vetorial × BM25 × híbrida em nDCG@10, recall@10 e MRR.

```powershell
cd 01-recomendador-filmes
..\.venv\Scripts\python -m evaluation.run_eval --embedder hash          # offline: linha de base
..\.venv\Scripts\python -m evaluation.run_eval --embedder ollama        # com nomic-embed-text
..\.venv\Scripts\python -m evaluation.run_eval --embedder google_genai  # com GOOGLE_API_KEY (~74 chamadas de embedding)
```

nDCG@10 por embedder, medido em 2026-11-01 (catálogo de 44 filmes). "Híbrida w" = RRF com peso `w` para o BM25 (o do vetor é 1):

| Embedder | Estratégia | todas | diretas | paráfrase | por nome |
|---|---|---|---|---|---|
| **Google `gemini-embedding-001`** | **só vetorial** | **0,996** | 1,000 | **0,981** | 1,000 |
| | híbrida w=0,2 | 0,965 | 1,000 | 0,824 | 1,000 |
| | híbrida w=1 | 0,954 | 1,000 | 0,769 | 1,000 |
| | só BM25 | 0,919 | 0,991 | 0,620 | 1,000 |
| **Ollama `nomic-embed-text`** | só vetorial | 0,883 | 0,914 | 0,645 | 1,000 |
| | **híbrida w=1** | **0,943** | 0,991 | **0,739** | 1,000 |
| | híbrida w=0,2 | 0,906 | 0,925 | 0,728 | 1,000 |
| | só BM25 | 0,919 | 0,991 | 0,620 | 1,000 |
| Hash (sem semântica) | híbrida w=1 | 0,906 | 0,982 | 0,577 | 1,000 |

O que isso mostra (e o que **não** mostra):
- **A melhor estratégia depende do embedder.** Com o embedder multilíngue do Google, misturar BM25 *piora* (a lista lexical, ruim nas paráfrases, arrasta a fusão); com o `nomic-embed-text`, que é focado em inglês, a híbrida é a melhor. Por isso o peso do BM25 é **configurável** (`PORTFOLIO_FILMES_TEXT_WEIGHT`) e o padrão é por embedder: **0** (só vetor) para o Google e **1** para os demais (`app/index.py`).
- **Correção:** uma versão anterior deste README dizia que a híbrida superava cada busca isolada. Isso vale para o `nomic-embed-text`, não para o Google; a medição com o Google desmentiu a generalização.
- **Buscas por nome ou palavra exata ("Totoro", "Keyser Söze") empatam em 1,000 em todas as estratégias**, inclusive só vetorial: com 44 filmes, o BM25 não traz ganho mensurável nem onde deveria ajudar. Com um catálogo grande (TMDB, ~1000 filmes), isso pode mudar; não foi medido.
- **Ressalvas:** catálogo pequeno; consultas escritas por mim conhecendo as sinopses (viés a favor da busca lexical); só 6 consultas de paráfrase e 8 por nome. Diferenças de poucos centésimos não são conclusivas, mas a diferença entre 0,98 e 0,77 nas paráfrases é grande.
- **Agente de ponta a ponta com o Gemini Flash-Lite** (`gemini-flash-lite-latest` + embeddings do Google), testado à mão com 2 mensagens: o modelo reescreveu o pedido numa descrição rica, devolveu "A Chegada" como primeira opção para "ficção científica sobre linguagem e comunicação com alienígenas", e no acompanhamento "me mostre outros, mais antigos que 2010" aplicou `year_max=2010` sozinho e não repetiu filmes. Não há avaliação automática do texto do agente.


## Avaliação da busca "parecidos com este filme" (catálogo CMU)

Com milhares de filmes não dá para julgar relevância à mão, então `evaluation/run_similar_eval.py` usa como gabarito **a mesma franquia**: tendo um filme de Toy Story, Matrix, Star Wars etc. como referência, os outros filmes da franquia devem aparecer entre os parecidos (28 franquias escolhidas por mim, 128 consultas, cada filme como referência uma vez). Catálogo de 5000 filmes, embeddings `nomic-embed-text` (Ollama), medido em 2026-11-01:

| Estratégia | precisão@5 | recall@10 |
|---|---|---|
| **só vetorial** | **0,716** | **0,999** |
| híbrida (BM25 da sinopse, peso 0,2) | 0,712 | 0,997 |
| híbrida (peso 1) | 0,687 | 0,989 |
| só BM25 (sinopse inteira como consulta) | 0,636 | 0,921 |

(a precisão@5 máxima possível é menor que 1 em franquias com menos de 6 filmes, então compare as linhas entre si, não com 1)

- **O vetor vence sozinho**, então `similar_movies` usa só o vetor por padrão, que ainda é mais barato (reaproveita o embedding que o índice guarda; ~0,09 s por consulta contra ~0,3 s com o BM25).
- **Ressalvas:** "mesma franquia" é um piso, não a medida do que importa: não mede se os filmes de outras franquias são bons parecidos. Ela favorece a busca lexical (sequências repetem nomes de personagens), e mesmo assim o vetor ganhou. Só medi o `nomic-embed-text`; o embedder do Google não foi rodado neste catálogo (custaria 5000 chamadas de embedding). **A busca por descrição (`search_movies`) no catálogo grande não foi medida**: não há gabarito, e o peso do BM25 padrão (1 para o nomic) vem da avaliação no catálogo pequeno em português.
- **Descoberta de desempenho:** a primeira versão levou ~3,5 s por consulta lexical, porque o BM25 do `portfolio-core` reanalisava a consulta para cada documento. Com sinopses inteiras como consulta isso é caro; agora ele usa um índice invertido (mesmo resultado, ~0,3 s, e há um teste que compara com a pontuação documento a documento).

### Leitura do agente real (Gemini Flash-Lite, catálogo CMU, `evaluation/try_agent.py`)

Sete pedidos em português, lidos à mão (não é uma avaliação com nota):
- A tradução funciona: "Um suspense psicológico com um final surpreendente" virou a consulta `a psychological thriller with a surprising twist ending`; "filmes de animação sobre brinquedos que ganham vida" trouxe os três Toy Story.
- "Algo como A Origem" foi para `similar_movies` com `Inception` (o modelo usou o título original, como mandei) e voltaram Dreamscape, The Matrix e Paycheck. "Algo parecido com Matrix, mas mais recente" usou o filtro de ano sozinho.
- **Fraquezas vistas:** no refinamento "me mostre outros, mais antigos que 2000" os resultados foram fracos (Blues Brothers 2000, Gumby) e o agente avisou que o catálogo é limitado; é o limite de 5000 filmes escolhidos por bilheteria, não por tema. As respostas listam só 3 filmes e os cartões mostram 6, então o texto nem sempre explica todos os cartões. Uma execução por pedido, sem medir variação.

## Testes

```bash
cd 01-recomendador-filmes && ../.venv/Scripts/python -m pytest
```

66 testes: catálogo (fontes, idioma, busca de título, gêneros mais comuns), `similar_movies` (nunca devolve o próprio filme, reaproveita o vetor, título ambíguo, título inexistente), preparação do CMU (limpeza da marcação da Wikipedia, filtros, limite por bilheteria), embedder, índice (filtros nos dois caminhos, exclusão, reaproveitamento e recriação do índice persistente), ferramenta (não repetir filmes, gênero inexistente, nota sem catálogo com notas, erros de argumento), loop do agente, API (SSE, taxa, validação), métricas, e o tracing (estrutura do trace, notas, falha do provedor, tracer desligado). **Nenhum teste chama um LLM ou um embedder real.**

## Observabilidade (Langfuse, opcional)

Sem `LANGFUSE_PUBLIC_KEY` e `LANGFUSE_SECRET_KEY` nada é enviado e nada muda. Com as chaves (use as do **projeto Langfuse deste app**; veja o README do `portfolio-langfuse`), cada pergunta vira um trace `movie-turn`, agrupado pela sessão, com a estrutura do agente:

| Observação | Tipo | O que registra |
|---|---|---|
| `generation` (uma por passo do agente) | generation | prompt completo daquele passo, resposta, tokens, tempo até o primeiro token, `stop_reason` e as ferramentas pedidas |
| `search_movies` | tool | a descrição e os filtros que o modelo escolheu, os filmes devolvidos e os avisos |

O trace guarda também o embedder, o catálogo e o número do turno. **Notas automáticas** (`app/guardrails.py`; heurísticas determinísticas, sinais e não bloqueios; o `comment` diz o que disparou):

| Nota | O que mede |
|---|---|
| `no_search` | o agente respondeu sem chamar `search_movies` |
| `empty_search` | alguma busca não devolveu nenhum filme (e qual era a consulta) |
| `unretrieved_titles` | quantos filmes do catálogo a resposta cita que a busca **não** devolveu nem estavam já na sessão ou na pergunta (candidato a filme inventado ou fora do contexto) |

Limites: o `unretrieved_titles` só vê títulos do catálogo escritos por inteiro (≥ 5 caracteres, palavras inteiras); um título citado por apelido ou muito curto passa, e um título inventado que não existe no catálogo também não é detectado. A interface ainda não envia o feedback do visitante (o servidor já devolve o `trace_id` no evento `meta`). Os testes usam observações falsas e o SDK real com exportador em memória (no `portfolio-core`); **o envio foi confirmado em 2026-11-01 contra um Langfuse 4.72.0 local** com `python -m evaluation.langfuse_smoke` (provedor roteirizado, sem LLM: chegaram `movie-turn`, as 2 `generation`, `search_movies` com a hierarquia correta e as 3 notas). Não foi testado com tráfego real nem na VPS.

## Decisões e limites conhecidos

- **Qdrant em modo local** em vez de servidor: zero infraestrutura extra na VPS e a mesma API. O modo local é para volumes pequenos (milhares de filmes) e só um processo pode abrir a pasta do índice. Para escalar, aponte o cliente para um Qdrant em container.
- **RRF** em vez de somar notas: as escalas de cosseno e BM25 não são comparáveis; só os postos importam. O peso do BM25 foi calibrado por embedder (ver a avaliação) em vez de ser fixo.
- **Os cartões são sempre os 6 primeiros resultados da busca**, não a escolha do LLM: pode haver cartões que o texto não recomenda (ou diz que nada serve). Melhoria possível: o modelo escolher quais cartões mostrar.
- **Histórico guarda só o texto** das respostas; os filmes já mostrados ficam num conjunto por sessão (com TTL).
- O catálogo de amostra é pequeno: com poucos filmes, "nada combina" é uma resposta comum e correta.

## Próximos passos

- [ ] Avaliação automática do texto do agente (hoje só a busca é medida) e o modelo escolher quais cartões mostrar.
- [ ] Medir a busca por descrição no catálogo grande (precisa de gabarito: consultas julgadas por outra pessoa) e re-medir o peso do BM25; rodar a avaliação de "parecidos" também com o embedder do Google.
- [ ] Catálogo TMDB real e consultas de avaliação sobre ele (idealmente julgadas por outra pessoa).
- [ ] Reranking por LLM dos top-20 e comparação do ganho contra o custo.
- [ ] Validar os traces contra um servidor Langfuse no ar e montar o painel (taxa de `empty_search` e de `unretrieved_titles`).

## Docker e deploy

Este repositório é autossuficiente (repositório GitHub: `portfolio-filmes`). Arquivos:

| Arquivo | Para quê |
|---|---|
| `Dockerfile` | Imagem em 2 estágios, usuário sem privilégios, `HEALTHCHECK` em `/health`. O `portfolio-core` vem da tag de `requirements-core.txt` |
| `compose.yml` | Produção: puxa `ghcr.io/hmarcio2/portfolio-filmes`, entra na rede `edge`, `read_only`, sem capabilities, limite de memória |
| `compose.local.yml` | Desenvolvimento: constrói a imagem com o `../_core` local e publica a porta 8003 |
| `deploy/filmes.caddy` | Trecho de Caddy instalado em `/opt/edge/sites/filmes.caddy` pelo deploy: o subdomínio `filmes.SEUDOMINIO` |
| `.github/workflows/ci.yml` | Em todo push/PR: testes (Python 3.12), build da imagem e `/health` no container |
| `.github/workflows/deploy.yml` | Build → GHCR → SSH → `docker compose up --wait` → recarrega o Caddy. **Manual por enquanto** (`workflow_dispatch`); para automatizar, troque o gatilho por push na `main` |
| `.env.example` | Variáveis do app. Na VPS, copie para `/opt/apps/filmes/.env` |

Rodar com Docker na sua máquina:

```bash
cp .env.example .env                      # preencha GOOGLE_API_KEY
docker compose -f compose.yml -f compose.local.yml up --build
```
Abra `http://localhost:8003`.

**Atenção ao índice do catálogo grande:** no primeiro start o app embute todos os filmes, ou seja, 5000 chamadas ao embedder. Com o Ollama local isso levou 5 minutos; com a API do Google no plano gratuito pode estourar a cota (não testei), e a VPS não tem GPU para o Ollama. O índice só vale para o embedder que o criou (a consulta precisa do mesmo modelo), então não dá para gerar com o Ollama e consultar com o Google. O índice fica no volume `filmes_index` e não é recalculado a cada deploy. Medido em Docker com o embedder de hash (sem chaves): os 5000 filmes cabem no teto de 768 MiB (cerca de 320 MiB em repouso).

**Catálogo grande em produção:** gere `cmu_movies.csv` (`scripts/prepare_cmu.py`) ou `tmdb_movies.csv` (`scripts/fetch_tmdb.py`) localmente e copie para `/opt/apps/filmes/catalog/` na VPS (para forçar um, `PORTFOLIO_FILMES_CATALOG=cmu` no `.env`). O compose monta essa pasta em `/srv/catalog` e o app passa a usar o arquivo (e mostrar a atribuição da fonte, que a licença exige). O índice vetorial fica no volume `filmes_index`, então os embeddings não são recalculados a cada deploy; trocar de embedder recria o índice sozinho.

**Deploy na VPS:** configure uma vez a VPS e os segredos do ambiente `production` conforme o README do `portfolio-edge`
(`VPS_HOST`, `VPS_USER`, `VPS_SSH_KEY`, `VPS_KNOWN_HOSTS`), crie `/opt/apps/filmes/.env` e rode `Actions > Deploy`.
Para voltar a uma versão: na VPS, `cd /opt/apps/filmes && IMAGE_TAG=sha-<commit-antigo> docker compose up -d --wait`.
