import json

import pytest
from fastapi.testclient import TestClient

from app.agent import TOOLS, build_system, make_executor, run_agent
from app.catalog import DATA_DIR, load_csv
from app.embeddings import HashingEmbedder
from app.index import MovieIndex
from app.main import create_app
from portfolio_core import LLMChunk, Settings, ToolUse, Usage


@pytest.fixture(scope="module")
def index():
    return MovieIndex(load_csv(DATA_DIR / "sample_movies.csv", source="amostra"), HashingEmbedder())


class Scripted:
    def __init__(self, *turns):
        self.turns = list(turns)
        self.calls = []

    async def stream(self, *, system, messages, tools=None):
        self.calls.append({"system": system, "messages": list(messages), "tools": tools})
        for c in self.turns.pop(0):
            yield c


def tool_turn(**args):
    return [LLMChunk(done=True, stop_reason="tool_use", content=["x"], tool_uses=[ToolUse("t1", "search_movies", args)],
                     usage=Usage(5, 2))]


def text_turn(text):
    return [LLMChunk(text=text), LLMChunk(done=True, stop_reason="end_turn", usage=Usage(7, 3))]


# ---------- ferramenta ----------

def search(index, shown=None, **args):
    shown = set() if shown is None else shown
    return make_executor(index, shown)("search_movies", args), shown


def test_tool_returns_results_card_event_and_marks_shown(index):
    res, shown = search(index, description="suspense num hospital psiquiátrico numa ilha")
    payload = json.loads(res.content)
    assert not res.is_error and payload["resultados"][0]["titulo"] == "Ilha do Medo"
    assert res.events[0].kind == "movies" and res.events[0].data["movies"][0]["title"] == "Ilha do Medo"
    assert "Ilha do Medo" in res.meta["titles"] and len(shown) == len(payload["resultados"])


def test_second_search_does_not_repeat_shown_movies(index):
    shown: set[int] = set()
    first, _ = search(index, shown, description="animação para a família")
    second, _ = search(index, shown, description="animação para a família")
    t1 = {r["titulo"] for r in json.loads(first.content)["resultados"]}
    t2 = {r["titulo"] for r in json.loads(second.content)["resultados"]}
    assert t1 and t2 and not t1 & t2
    again, _ = search(index, shown, description="animação para a família", include_shown=True)
    assert {r["titulo"] for r in json.loads(again.content)["resultados"]} & t1


def test_exclude_titles_filters_and_unknown_genre_is_reported(index):
    res, _ = search(index, description="viagem espacial", exclude_titles=["interestelar"], genres=["faroeste", "ficção científica"])
    payload = json.loads(res.content)
    assert "Interestelar" not in {r["titulo"] for r in payload["resultados"]}
    assert any("faroeste" in a for a in payload["avisos"])
    assert all("ficção científica" in r["generos"] for r in payload["resultados"])


def test_min_rating_is_ignored_with_note_when_catalog_has_no_ratings(index):
    res, _ = search(index, description="drama", min_rating=8)
    payload = json.loads(res.content)
    assert payload["resultados"] and any("notas" in a for a in payload["avisos"])


def test_no_results_gives_actionable_note_and_no_event(index):
    res, _ = search(index, description="algo", genres=["ficção científica"], year_min=2030)
    assert json.loads(res.content)["resultados"] == [] and not res.events and "Relaxe" in json.loads(res.content)["avisos"][-1]


def test_bad_arguments_are_errors_not_exceptions(index):
    run = make_executor(index, set())
    assert run("search_movies", {"description": ""}).is_error
    assert run("search_movies", "texto").is_error
    assert run("outra", {}).is_error


def test_system_prompt_lists_catalog_facts(index):
    s = build_system(index)
    assert "44 filmes" in s and "ficção científica" in s and TOOLS[0]["name"] == "search_movies"


# ---------- agente ----------

async def collect(agen):
    return [e async for e in agen]


@pytest.mark.asyncio
async def test_agent_searches_then_answers(index):
    provider = Scripted(tool_turn(description="suspense numa ilha"), text_turn("Recomendo Ilha do Medo."))
    events = await collect(run_agent(provider, index, set(), [{"role": "user", "content": "suspense numa ilha"}]))
    assert [e.kind for e in events] == ["tool", "movies", "delta", "done"]
    done = events[-1].data
    assert done["answer"] == "Recomendo Ilha do Medo." and done["searches"][0]["query"] == "suspense numa ilha"
    result = provider.calls[1]["messages"][-1]["content"][0]
    assert result["type"] == "tool_result" and "Ilha do Medo" in result["content"]


# ---------- API ----------

def parse_sse(text):
    out = []
    for block in text.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        out.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def test_chat_flow_streams_cards_and_answer(index):
    provider = Scripted(tool_turn(description="robôs e simulação"), text_turn("Veja Matrix."))
    with TestClient(create_app(Settings(provider="echo"), provider, index)) as client:
        events = parse_sse(client.post("/chat", json={"message": "robôs e simulação"}).text)
        kinds = [n for n, _ in events]
        assert kinds[0] == "meta" and kinds[-1] == "done" and "movies" in kinds
        assert events[kinds.index("movies")][1]["movies"][0]["title"] == "Matrix"


def test_info_health_and_ui(index):
    with TestClient(create_app(Settings(provider="echo"), Scripted(), index)) as client:
        assert client.get("/health").json()["movies"] == 44
        info = client.get("/info").json()
        assert info["source"] == "amostra" and info["attribution"] is None and info["embedder"].startswith("hash")
        assert "Recomendador" in client.get("/").text


def test_rate_limit_and_validation(index):
    provider = Scripted(text_turn("a"), text_turn("b"))
    with TestClient(create_app(Settings(provider="echo", rate_limit_per_minute=1, max_message_chars=20), provider, index)) as client:
        assert client.post("/chat", json={"message": "x" * 21}).status_code == 422
        assert client.post("/chat", json={"message": "oi"}).status_code == 200
        assert client.post("/chat", json={"message": "oi"}).status_code == 429
