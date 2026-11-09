import pytest

from app.catalog import DATA_DIR, load_csv, norm
from app.embeddings import HashingEmbedder, get_embedder
from app.index import Filters, MovieIndex
from evaluation.cases import QUERIES
from evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank
from evaluation.run_eval import evaluate, summarize


@pytest.fixture(scope="module")
def catalog():
    return load_csv(DATA_DIR / "sample_movies.csv", source="amostra")


@pytest.fixture(scope="module")
def index(catalog):
    return MovieIndex(catalog, HashingEmbedder())


# ---------- catálogo ----------

def test_sample_catalog_loads_with_quoted_titles(catalog):
    titles = {m.title for m in catalog.movies}
    assert len(catalog.movies) == 44 and "Se Beber, Não Case" in titles
    assert all(m.overview and m.year for m in catalog.movies)
    assert catalog.attribution is None  # só o TMDB exige atribuição


def test_match_genres_ignores_accents_and_reports_unknown(catalog):
    ok, unknown = catalog.match_genres(["Ficcao cientifica", "SUSPENSE", "faroeste espacial"])
    assert ok == ["ficção científica", "suspense"] and unknown == ["faroeste espacial"]


def test_csv_without_overview_is_skipped(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("id,title,year,genres,rating,votes,overview,poster_path\n1,A,2000,drama,7.5,10,,\n2,B,2001,drama,8,5,Sinopse,/x.jpg\n", encoding="utf-8")
    cat = load_csv(p, source="tmdb")
    assert [m.title for m in cat.movies] == ["B"]
    assert cat.movies[0].poster_url.endswith("/x.jpg") and cat.attribution


CSV_HEADER = "id,title,year,genres,rating,votes,overview,poster_path\n"


def _write(folder, name, title):
    (folder / name).write_text(f"{CSV_HEADER}7,{title},2001,drama,8,5,Sinopse,\n", encoding="utf-8")


def test_default_catalog_picks_tmdb_then_cmu_then_sample(tmp_path, monkeypatch):
    from app.catalog import load_default_catalog

    monkeypatch.delenv("PORTFOLIO_FILMES_CATALOG", raising=False)
    monkeypatch.setenv("PORTFOLIO_FILMES_CATALOG_DIR", str(tmp_path))  # pasta isolada: não depende do que há em data/
    assert load_default_catalog().source == "amostra"
    _write(tmp_path, "cmu_movies.csv", "Y")
    cat = load_default_catalog()
    assert cat.source == "cmu" and cat.language == "en" and "CC BY-SA" in cat.attribution
    _write(tmp_path, "tmdb_movies.csv", "Z")
    cat = load_default_catalog()
    assert cat.source == "tmdb" and [m.title for m in cat.movies] == ["Z"] and cat.language == "pt"
    monkeypatch.setenv("PORTFOLIO_FILMES_CATALOG_DIR", str(tmp_path / "vazia"))  # pasta sem os arquivos
    assert load_default_catalog().source == "amostra"


def test_catalog_can_be_forced_and_language_overridden(tmp_path, monkeypatch):
    from app.catalog import load_default_catalog

    monkeypatch.setenv("PORTFOLIO_FILMES_CATALOG_DIR", str(tmp_path))
    _write(tmp_path, "cmu_movies.csv", "Y")
    _write(tmp_path, "tmdb_movies.csv", "Z")
    monkeypatch.setenv("PORTFOLIO_FILMES_CATALOG", "cmu")
    assert load_default_catalog().source == "cmu"
    monkeypatch.setenv("PORTFOLIO_FILMES_CATALOG", "sample")
    assert load_default_catalog().source == "amostra"
    monkeypatch.setenv("PORTFOLIO_FILMES_CATALOG_LANG", "pt")
    assert load_default_catalog().language == "pt"
    monkeypatch.setenv("PORTFOLIO_FILMES_CATALOG", "nada")
    with pytest.raises(ValueError):
        load_default_catalog()
    monkeypatch.setenv("PORTFOLIO_FILMES_CATALOG", "tmdb")
    monkeypatch.setenv("PORTFOLIO_FILMES_CATALOG_DIR", str(tmp_path / "vazia"))
    with pytest.raises(FileNotFoundError):
        load_default_catalog()


def _cat(*specs):
    from app.catalog import Catalog, Movie

    return Catalog([Movie(i, t, y, tuple(g), "sinopse") for i, (t, y, g) in enumerate(specs, 1)], source="cmu")


def test_top_genres_keeps_the_most_frequent_in_alphabetical_order():
    cat = _cat(("A", 2000, ["Drama", "Raro"]), ("B", 2001, ["Drama", "Comédia"]), ("C", 2002, ["Comédia", "Ação"]), ("D", 2003, ["Drama"]))
    assert cat.top_genres(2) == ["Comédia", "Drama"] and cat.top_genres(10) == ["Ação", "Comédia", "Drama", "Raro"]


def test_find_matches_titles_ignoring_case_and_accents_and_disambiguates_remakes():
    cat = _cat(("Dune", 1984, []), ("Dune", 2021, []), ("Amélie", 2001, []), ("Interstellar", 2014, []))
    movies, _ = cat.find("AMELIE")
    assert [m.title for m in movies] == ["Amélie"]
    assert [m.year for m in cat.find("dune")[0]] == [1984, 2021]            # ambíguo: devolve os dois, por ano
    assert [m.year for m in cat.find("Dune", year=2021)[0]] == [2021]       # o ano desfaz a ambiguidade
    assert [m.year for m in cat.find("Dune", year=1999)[0]] == [1984, 2021]  # ano sem filme: continua ambíguo


def test_find_suggests_similar_titles_when_nothing_matches():
    cat = _cat(("Interstellar", 2014, []), ("The Matrix", 1999, []), ("Inception", 2010, []))
    movies, suggestions = cat.find("Interstelar")
    assert movies == [] and suggestions[0] == "Interstellar"
    assert cat.find("zzzzzz")[1] == []


# ---------- embedder ----------

def test_hashing_embedder_is_deterministic_and_normalized():
    e = HashingEmbedder()
    a, b = e.embed_query("filme de terror"), e.embed_query("filme de terror")
    assert a == b and abs(sum(x * x for x in a) - 1) < 1e-9
    sim = lambda u, v: sum(x * y for x, y in zip(u, v))
    assert sim(a, e.embed_query("terror e suspense")) > sim(a, e.embed_query("comédia romântica"))


def test_get_embedder_default_is_hash_without_keys(monkeypatch):
    for k in ("EMBEDDING_PROVIDER", "GOOGLE_API_KEY", "LOCAL_EMBEDDING_ENABLED"):
        monkeypatch.delenv(k, raising=False)
    assert get_embedder().id.startswith("hash")


def test_get_embedder_builds_ollama_and_google(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "chave-falsa")
    assert get_embedder("ollama").id == "ollama-nomic-embed-text"
    assert get_embedder("google_genai").id == "google-gemini-embedding-001"
    with pytest.raises(ValueError):
        get_embedder("skynet")


# ---------- índice ----------

def titles(hits):
    return [h.movie.title for h in hits]


def test_search_finds_obvious_match_in_every_mode(index):
    for mode in ("vector", "text", "hybrid"):
        assert titles(index.search("a realidade é uma simulação criada por máquinas", k=3, mode=mode))[0] == "Matrix"


def test_hybrid_fuses_both_rankings(index):
    hits = index.search("assassino em série perseguido por detetives", k=5)
    assert any(h.vector_rank and h.text_rank for h in hits)
    assert {"Se7en: Os Sete Crimes Capitais", "O Silêncio dos Inocentes"} <= set(titles(hits))


def test_filters_apply_to_vector_and_text_paths(index):
    f = Filters(genres=["animação"], year_min=2005, year_max=2016)
    for mode in ("vector", "text", "hybrid"):
        hits = index.search("uma aventura", f, k=10, mode=mode)
        assert hits and all("animação" in h.movie.genres and 2005 <= h.movie.year <= 2016 for h in hits)


def test_exclude_ids_and_empty_result(index):
    first = index.search("viagem espacial", k=1)[0].movie
    again = index.search("viagem espacial", Filters(exclude_ids={first.id}), k=5)
    assert first.title not in titles(again)
    assert index.search("qualquer coisa", Filters(genres=["faroeste"]), k=5) == []


def test_rating_filter_excludes_movies_without_rating(index):
    assert index.search("filme", Filters(min_rating=7.0), k=5) == []  # a amostra não tem notas


def test_collection_is_rebuilt_when_embedder_changes(catalog, tmp_path):
    a = MovieIndex(catalog, HashingEmbedder(64), tmp_path / "idx")
    names_a = a.collection
    a._client.close()
    b = MovieIndex(catalog, HashingEmbedder(128), tmp_path / "idx")
    assert b.collection != names_a
    assert {c.name for c in b._client.get_collections().collections} == {b.collection}  # a antiga foi removida
    b._client.close()


def test_persistent_index_is_reused(catalog, tmp_path):
    a = MovieIndex(catalog, HashingEmbedder(), tmp_path / "idx")
    a._client.close()

    class Boom(HashingEmbedder):
        def embed_documents(self, texts):
            raise AssertionError("não deveria recalcular os embeddings")

    b = MovieIndex(catalog, Boom(), tmp_path / "idx")
    assert titles(b.search("matrix simulação", k=1)) == ["Matrix"]
    b._client.close()


# ---------- avaliação ----------

def test_metrics():
    gains = {"A": 2, "B": 1}
    assert ndcg_at_k(["A", "B"], gains) == pytest.approx(1.0)
    assert ndcg_at_k(["X", "A"], gains) < 1.0
    assert recall_at_k(["A", "X", "B"], gains, 2) == 0.5
    assert reciprocal_rank(["X", "Y", "B"], gains) == pytest.approx(1 / 3)
    assert ndcg_at_k([], gains) == 0.0


def test_every_judged_title_exists_in_sample(catalog):
    titles_ = {m.title for m in catalog.movies}
    for q in QUERIES:
        assert set(q.relevant) <= titles_, q.text


def test_baseline_evaluation_runs_and_direct_queries_work(index):
    results = evaluate(index)
    assert summarize(results["híbr. w=1"], "direta")[0] > 0.9  # linha de base: busca lexical + hash em consultas diretas
    assert summarize(results["híbr. w=1"])[1] > 0.9
    assert summarize(results["vetorial"], "lexical")[0] == 1.0  # buscas por nome acham o filme mesmo sem BM25


def test_default_text_weight_depends_on_embedder_and_env(monkeypatch):
    from app.index import default_text_weight

    monkeypatch.delenv("PORTFOLIO_FILMES_TEXT_WEIGHT", raising=False)
    assert default_text_weight("google-gemini-embedding-001") == 0.0  # medido: só vetor foi o melhor
    assert default_text_weight("ollama-nomic-embed-text") == 1.0 and default_text_weight("hash-512") == 1.0
    monkeypatch.setenv("PORTFOLIO_FILMES_TEXT_WEIGHT", "0.3")
    assert default_text_weight("google-gemini-embedding-001") == 0.3


def test_zero_text_weight_is_vector_only_and_weight_changes_fusion(catalog):
    idx = MovieIndex(catalog, HashingEmbedder(), text_weight=0.0)
    q = "assassino em série perseguido por detetives"
    assert [h.movie.id for h in idx.search(q, k=5)] == [h.movie.id for h in idx.search(q, k=5, mode="vector")]
    assert all(h.text_rank is None for h in idx.search(q, k=5))
    assert any(h.text_rank for h in idx.search(q, k=5, text_weight=1.0))


def test_text_mode_ignores_the_fusion_weight(catalog):
    """Regressão: com peso 0 (padrão do Google) o modo só-BM25 zerava as pontuações e a ordem virava acaso."""
    q = "assassino em série perseguido por detetives"
    ref = [h.movie.id for h in MovieIndex(catalog, HashingEmbedder(), text_weight=1.0).search(q, k=5, mode="text")]
    got = [h.movie.id for h in MovieIndex(catalog, HashingEmbedder(), text_weight=0.0).search(q, k=5, mode="text")]
    assert got == ref and ref  # (o BM25 só devolve filmes que casam com algum termo)
