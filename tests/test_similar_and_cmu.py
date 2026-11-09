"""Busca por filme de exemplo (similar_movies), idioma do catálogo no prompt e a preparação do catálogo CMU."""

import importlib.util
import json
from pathlib import Path

import pytest

from app.agent import TOOLS, build_system, make_executor, run_agent
from app.catalog import DATA_DIR, Catalog, Movie, load_csv
from app.embeddings import HashingEmbedder
from app.index import Filters, MovieIndex
from portfolio_core import LLMChunk, ToolUse, Usage

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def index():
    return MovieIndex(load_csv(DATA_DIR / "sample_movies.csv", source="amostra"), HashingEmbedder())


def movie(index, title):
    (m,), _ = index.catalog.find(title)
    return m


# ---------- índice ----------

def test_similar_never_returns_the_reference_and_respects_filters(index):
    ref = movie(index, "Matrix")
    hits = index.similar(ref, k=6)
    assert len(hits) == 6 and ref.id not in {h.movie.id for h in hits}
    old = index.similar(ref, Filters(year_max=2005), k=6)
    assert old and all(h.movie.year <= 2005 for h in old) and ref.id not in {h.movie.id for h in old}
    excluded = hits[0].movie.id
    assert excluded not in {h.movie.id for h in index.similar(ref, Filters(exclude_ids={excluded}), k=6)}


def test_similar_reuses_the_stored_vector_and_does_not_call_the_embedder(index, monkeypatch):
    calls = []
    monkeypatch.setattr(index.embedder, "embed_query", lambda text: calls.append(text) or [0.0] * 512)
    index.similar(movie(index, "Interestelar"), k=3)
    assert calls == []
    assert len(index.vector_of(movie(index, "Interestelar"))) == index.embedder.dim


def test_vector_only_and_text_only_modes(index):
    ref = movie(index, "Matrix")
    for mode in ("vector", "text"):
        hits = index.similar(ref, k=4, mode=mode)
        assert hits and ref.id not in {h.movie.id for h in hits}


# ---------- ferramenta ----------

def run(index, **args):
    shown: set[int] = set()
    return make_executor(index, shown)("similar_movies", args), shown


def test_tool_returns_example_movie_results_and_a_reference_event(index):
    res, shown = run(index, title="matrix")
    payload = json.loads(res.content)
    assert not res.is_error and payload["filme_de_exemplo"]["titulo"] == "Matrix"
    assert payload["resultados"] and "Matrix" not in {r["titulo"] for r in payload["resultados"]}
    assert res.events[0].data["reference"] == {"title": "Matrix", "year": 1999}
    assert res.meta["tool"] == "similar_movies" and "Matrix" in res.meta["query"] and res.meta["titles"]
    assert shown == {m["id"] for m in res.events[0].data["movies"]}  # o filme de exemplo não conta como mostrado


def test_tool_applies_filters_and_does_not_repeat_shown(index):
    first, shown = run(index, title="Matrix", year_max=2010)
    again = make_executor(index, shown)("similar_movies", {"title": "Matrix", "year_max": 2010})
    t1 = {r["titulo"] for r in json.loads(first.content)["resultados"]}
    t2 = {r["titulo"] for r in json.loads(again.content)["resultados"]}
    assert t1 and t2 and not t1 & t2


def test_unknown_title_is_an_error_with_suggestions_and_a_way_out(index):
    res, shown = run(index, title="Interestelarr")
    assert res.is_error and "Interestelar" in res.content and "search_movies" in res.content and not shown


def test_ambiguous_title_asks_for_the_year():
    movies = [Movie(1, "Dune", 1984, ("Sci-Fi",), "Sinopse um."), Movie(2, "Dune", 2021, ("Sci-Fi",), "Sinopse dois."),
              Movie(3, "Outro", 2000, ("Drama",), "Sinopse três.")]
    idx = MovieIndex(Catalog(movies, source="cmu"), HashingEmbedder())
    execute = make_executor(idx, set())
    ambiguous = execute("similar_movies", {"title": "Dune"})
    assert ambiguous.is_error and "[1984, 2021]" in ambiguous.content and "year" in ambiguous.content
    ok = execute("similar_movies", {"title": "Dune", "year": 2021})
    assert not ok.is_error and json.loads(ok.content)["filme_de_exemplo"]["ano"] == 2021


def test_bad_arguments_are_errors_not_exceptions(index):
    execute = make_executor(index, set())
    assert execute("similar_movies", {"title": ""}).is_error
    assert execute("similar_movies", {"title": 5}).is_error
    assert execute("similar_movies", "texto").is_error


def test_both_tools_are_offered_and_have_required_fields():
    names = {t["name"]: t["input_schema"]["required"] for t in TOOLS}
    assert names == {"search_movies": ["description"], "similar_movies": ["title"]}


# ---------- idioma e prompt ----------

def test_prompt_tells_the_model_to_write_the_query_in_the_catalog_language(index):
    assert "português" in build_system(index) and "INGLÊS" not in build_system(index)
    english = MovieIndex(Catalog([Movie(1, "Alpha", 2000, ("Drama",), "An english synopsis.")], source="cmu"), HashingEmbedder())
    prompt = build_system(english)
    assert "INGLÊS" in prompt and "a thriller with a surprising twist ending" in prompt and "similar_movies" in prompt


def test_prompt_lists_only_the_most_common_genres_for_big_catalogs():
    movies = [Movie(i, f"F{i}", 2000, (f"G{i:03d}",), "Sinopse.") for i in range(100)]
    prompt = build_system(MovieIndex(Catalog(movies, source="cmu"), HashingEmbedder()))
    assert "os 40 mais comuns" in prompt and prompt.count("G0") <= 40


@pytest.mark.asyncio
async def test_agent_calls_similar_movies_then_answers(index):
    class Scripted:
        def __init__(self):
            self.turns = [
                [LLMChunk(done=True, stop_reason="tool_use", content=["x"], usage=Usage(5, 2),
                          tool_uses=[ToolUse("t1", "similar_movies", {"title": "Matrix"})])],
                [LLMChunk(text="Veja Blade Runner 2049."), LLMChunk(done=True, stop_reason="end_turn", usage=Usage(7, 3))],
            ]

        async def stream(self, *, system, messages, tools=None):
            for c in self.turns.pop(0):
                yield c

    events = [e async for e in run_agent(Scripted(), index, set(), [{"role": "user", "content": "algo como Matrix"}])]
    assert [e.kind for e in events] == ["tool", "movies", "delta", "done"]
    assert events[-1].data["searches"][0]["tool"] == "similar_movies"


# ---------- preparação do catálogo CMU ----------

def load_prepare():
    spec = importlib.util.spec_from_file_location("prepare_cmu", ROOT / "scripts" / "prepare_cmu.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PLOT = "A detective hunts a killer across the city. " * 12  # > 300 caracteres


def meta_row(movie_id, name, date, revenue, genres):
    return "\t".join([str(movie_id), "/m/x", name, date, revenue, "100.0", "{}", "{}", json.dumps(genres)])


@pytest.fixture
def cmu_dir(tmp_path):
    rows = [
        meta_row(1, "Rich Film", "2001-05-04", "900000", {"/m/a": "Drama", "/m/b": "World cinema", "/m/c": "Thriller"}),
        meta_row(2, "Poor Film", "1999", "", {"/m/a": "Comedy", "/m/b": "Indie"}),
        meta_row(3, "Mid Film", "2005-01-01", "5000", {"/m/a": "Action"}),
        meta_row(4, "No Plot", "2005", "", {}),
        meta_row(5, "Short Plot", "2005", "7000", {}),
        meta_row(6, "Bad Year", "0000-00-00", "100", {"/m/a": "Drama"}),
    ]
    (tmp_path / "movie.metadata.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    plots = [f"1\t{{{{Plot}}}} {PLOT}[[Link]] text.<ref name=x>cite</ref> More&nbsp;text.", f"2\t{PLOT}", f"3\t{PLOT}",
             "5\tToo short.", f"6\t{PLOT}"]
    (tmp_path / "plot_summaries.txt").write_text("\n".join(plots) + "\n", encoding="utf-8")
    return tmp_path


def test_prepare_cleans_markup_filters_and_converts_fields(cmu_dir):
    mod = load_prepare()
    rows = {r["id"]: r for r in mod.build(cmu_dir, limit=100, min_chars=300, max_chars=1500)}
    assert set(rows) == {1, 2, 3, 6}  # sem sinopse (4) e sinopse curta (5) ficam de fora
    one = rows[1]
    assert "{{" not in one["overview"] and "[[" not in one["overview"] and "<ref" not in one["overview"]
    assert "&nbsp;" not in one["overview"] and "Link text." in one["overview"] and "More text." in one["overview"]
    assert one["genres"] == "Drama|Thriller" and one["year"] == 2001 and rows[2]["genres"] == "Comedy"  # ruído removido
    assert rows[6]["year"] == ""  # ano inválido


def test_prepare_limit_keeps_films_with_revenue_first_and_output_is_sorted_by_id(cmu_dir):
    mod = load_prepare()
    rows = mod.build(cmu_dir, limit=2, min_chars=300, max_chars=1500)
    assert [r["id"] for r in rows] == [1, 3]  # bilheterias 900000 e 5000; o 6 (100) e o 2 (sem) ficam de fora


def test_prepare_cuts_long_plots_at_a_sentence_end():
    mod = load_prepare()
    text = " ".join(f"Sentence number {i} ends here." for i in range(100))
    cut = mod.clean_overview(text, 200)
    assert len(cut) <= 200 and cut.endswith("here.") and not cut.endswith(" ")


def test_prepared_csv_loads_as_a_cmu_catalog(cmu_dir, tmp_path):
    mod = load_prepare()
    out = tmp_path / "out" / "cmu_movies.csv"
    import sys

    argv = sys.argv
    sys.argv = ["prepare_cmu.py", "--src", str(cmu_dir), "--out", str(out), "--max", "10"]
    try:
        assert mod.main() == 0
    finally:
        sys.argv = argv
    cat = load_csv(out, source="cmu")
    assert {m.title for m in cat.movies} == {"Rich Film", "Poor Film", "Mid Film", "Bad Year"}
    assert cat.language == "en" and all(m.rating is None and m.poster_url is None for m in cat.movies)


def test_prepare_complains_about_a_wrong_folder(tmp_path, capsys, monkeypatch):
    mod = load_prepare()
    monkeypatch.setattr("sys.argv", ["prepare_cmu.py", "--src", str(tmp_path)])
    assert mod.main() == 2 and "MovieSummaries" in capsys.readouterr().err
