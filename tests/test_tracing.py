"""Tracing do recomendador: estrutura do trace e notas automáticas."""

import json

import pytest
from fastapi.testclient import TestClient

from app.catalog import DATA_DIR, load_csv
from app.embeddings import HashingEmbedder
from app.guardrails import evaluate_turn, mentioned_titles
from app.index import MovieIndex
from app.main import create_app
from portfolio_core import LLMChunk, Settings, Span, ToolUse, Tracer, Usage

TITLES = ["Interestelar", "Matrix", "Up: Altas Aventuras", "Ilha do Medo", "Clube da Luta"]


# ---------- notas ----------

def scores(**kw):
    base = dict(question="quero algo", answer="Veja Matrix.", searches=[{"query": "q", "titles": ["Matrix"]}],
                catalog_titles=TITLES, shown_titles=["Matrix"])
    return {s.name: s for s in evaluate_turn(**{**base, **kw})}


def test_clean_turn_has_no_alerts():
    s = scores()
    assert {n: x.value for n, x in s.items()} == {"no_search": 0, "empty_search": 0, "unretrieved_titles": 0}


def test_answer_without_search_is_flagged():
    assert scores(searches=[])["no_search"].value == 1


def test_empty_search_is_flagged_with_the_query():
    s = scores(searches=[{"query": "algo impossível", "titles": []}])
    assert s["empty_search"].value == 1 and "algo impossível" in s["empty_search"].comment


def test_title_in_answer_that_search_did_not_return_is_counted_and_named():
    s = scores(answer="Veja Matrix e também Interestelar.")
    assert s["unretrieved_titles"].value == 1 and "Interestelar" in s["unretrieved_titles"].comment
    assert s["unretrieved_titles"].data_type == "NUMERIC"


def test_titles_already_shown_or_named_by_the_user_are_not_counted():
    assert scores(answer="Como já mostrei Interestelar...", shown_titles=["Interestelar"])["unretrieved_titles"].value == 0
    assert scores(question="Gostei de Interestelar", answer="Parecido com Interestelar: Matrix.")["unretrieved_titles"].value == 0


def test_title_matching_ignores_accents_and_punctuation_but_needs_whole_words_and_the_same_capitals():
    assert mentioned_titles("recomendo Up: Altas Aventuras!", TITLES) == ["Up: Altas Aventuras"]
    assert mentioned_titles("recomendo Ilha do Medo, um clássico", TITLES) == ["Ilha do Medo"]
    assert mentioned_titles("a Matrixx é outra coisa", TITLES) == []
    assert mentioned_titles("ela up", ["Up", "Ela"]) == []  # títulos curtos demais não entram


def test_common_word_titles_only_count_when_written_as_in_the_catalog():
    titles = ["Alien", "Speed", "Crash"]
    assert mentioned_titles("Um alien chega, a speed é alta e o crash sobe", titles) == []
    assert mentioned_titles("Veja Alien e depois Crash.", titles) == ["Alien", "Crash"]


def test_failed_or_empty_turn_has_no_scores():
    assert evaluate_turn(question="q", answer="", searches=[], catalog_titles=TITLES, shown_titles=[]) == []
    assert evaluate_turn(question="q", answer="x", searches=[], catalog_titles=TITLES, shown_titles=[], failed=True) == []


# ---------- trace no fluxo da API ----------

class Rec(Span):
    def __init__(self, name="root", kind="span", **kw):
        self.name, self.kind, self.kw, self.updates, self.children, self.scored, self.ended = name, kind, kw, [], [], {}, False

    def child(self, name, *, as_type="span", **kw):
        c = Rec(name, as_type, **kw)
        self.children.append(c)
        return c

    def update(self, **kw):
        self.updates.append({k: v for k, v in kw.items() if v is not None})

    def end(self):
        self.ended = True

    def score(self, name, value, *, data_type="NUMERIC", comment=None):
        self.scored[name] = (value, comment)

    @property
    def trace_id(self):
        return "a" * 32

    @property
    def merged(self):
        out = {}
        for u in self.updates:
            out.update(u)
        return out


class RecTracer(Tracer):
    enabled = True

    def __init__(self):
        self.roots, self.shut = [], False

    def start(self, name, *, session_id=None, input=None, metadata=None, tags=None):
        root = Rec(name, session_id=session_id, input=input, metadata=metadata)
        self.roots.append(root)
        return root

    def shutdown(self):
        self.shut = True


class Scripted:
    def __init__(self, *turns):
        self.turns = list(turns)

    async def stream(self, *, system, messages, tools=None):
        for c in self.turns.pop(0):
            yield c


def tool_turn(**args):
    return [LLMChunk(done=True, stop_reason="tool_use", content=["x"], tool_uses=[ToolUse("t1", "search_movies", args)],
                     usage=Usage(5, 2))]


def text_turn(text):
    return [LLMChunk(text=text), LLMChunk(done=True, stop_reason="end_turn", usage=Usage(7, 3))]


@pytest.fixture(scope="module")
def index():
    return MovieIndex(load_csv(DATA_DIR / "sample_movies.csv", source="amostra"), HashingEmbedder())


def chat(index, provider, message="robôs e simulação"):
    tracer = RecTracer()
    with TestClient(create_app(Settings(provider="echo", model="m-test"), provider, index, tracer)) as client:
        text = client.post("/chat", json={"message": message}).text
    return tracer, text


def test_turn_trace_has_generations_a_search_tool_and_scores(index):
    tracer, text = chat(index, Scripted(tool_turn(description="robôs e simulação"), text_turn("Veja Matrix.")))
    (root,) = tracer.roots
    assert root.name == "movie-turn" and root.kw["input"] == "robôs e simulação" and root.kw["session_id"]
    assert root.kw["metadata"]["embedder"].startswith("hash") and root.kw["metadata"]["catalog"] == "amostra"
    assert [(c.name, c.kind) for c in root.children] == [("generation", "generation"), ("search_movies", "tool"),
                                                         ("generation", "generation")]
    assert root.children[0].kw["model"] == "m-test"
    assert "Matrix" in root.children[1].merged["output"]
    assert root.merged["output"] == "Veja Matrix." and root.merged["metadata"]["searches"] == 1
    assert root.scored["no_search"][0] == 0 and root.scored["unretrieved_titles"][0] == 0
    assert root.ended and all(c.ended for c in root.children)
    meta = next(json.loads(b.split("\ndata: ")[1]) for b in text.split("\n\n") if b.startswith("event: meta"))
    assert meta["trace_id"] == "a" * 32  # a interface pode usar para enviar feedback


def test_hallucinated_title_shows_up_as_a_score(index):
    tracer, _ = chat(index, Scripted(tool_turn(description="robôs e simulação"), text_turn("Veja Matrix e Interestelar.")))
    value, comment = tracer.roots[0].scored["unretrieved_titles"]
    assert value == 1 and "Interestelar" in comment


def test_answer_without_search_is_scored(index):
    tracer, _ = chat(index, Scripted(text_turn("Pode me contar mais?")), message="oi")
    assert tracer.roots[0].scored["no_search"][0] == 1


def test_provider_failure_marks_the_trace_as_error_without_scores(index):
    from portfolio_core import LLMError

    class Broken:
        async def stream(self, **kw):
            raise LLMError("fora do ar")
            yield  # pragma: no cover

    tracer, text = chat(index, Broken())
    root = tracer.roots[0]
    assert "event: error" in text and root.merged["level"] == "ERROR" and "fora do ar" in root.merged["status_message"]
    assert root.scored == {} and root.ended


def test_tracer_is_shut_down_with_the_app(index):
    tracer, _ = chat(index, Scripted(text_turn("ok")), message="oi")
    assert tracer.shut


def test_default_tracer_is_a_noop_without_keys(index, monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    with TestClient(create_app(Settings(provider="echo"), Scripted(text_turn("ok")), index)) as client:
        text = client.post("/chat", json={"message": "oi"}).text
    assert "trace_id" not in text and "event: done" in text
