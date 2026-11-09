"""Teste de fumaça do Langfuse para o recomendador: manda UM trace de teste pelo pipeline real (agente com uma busca,
guardrails) e confere se o trace, a ferramenta e as notas chegaram ao servidor.

Uso (na pasta 01-recomendador-filmes), com as chaves do projeto Langfuse DESTE app e o servidor no ar:

    $env:LANGFUSE_PUBLIC_KEY = "pk-lf-..."
    $env:LANGFUSE_SECRET_KEY = "sk-lf-..."
    $env:LANGFUSE_BASE_URL   = "http://localhost:3300"
    ../.venv/Scripts/python -m evaluation.langfuse_smoke

Não chama nenhum LLM nem embedder real (provedor roteirizado e embedder de hash). O trace vai para o ambiente
``portfolio-smoke``, que dá para filtrar ou apagar sem misturar com dados reais.
"""

from __future__ import annotations

import os
import sys
import time

from fastapi.testclient import TestClient

from app.catalog import DATA_DIR, load_csv
from app.embeddings import HashingEmbedder
from app.index import MovieIndex
from app.main import create_app
from portfolio_core import LLMChunk, Settings, ToolUse, Usage, get_tracer
from portfolio_core.config import load_dotenv_if_present

EXPECTED = {"movie-turn", "generation", "search_movies"}
EXPECTED_SCORES = {"no_search", "empty_search", "unretrieved_titles"}


class ScriptedProvider:
    def __init__(self):
        self.turns = [
            [LLMChunk(done=True, stop_reason="tool_use", content=[{"type": "tool_use"}], usage=Usage(150, 20),
                      tool_uses=[ToolUse("t1", "search_movies", {"description": "robôs e simulação da realidade"})])],
            [LLMChunk(text="Teste de fumaça: veja "), LLMChunk(text="Matrix."),
             LLMChunk(done=True, stop_reason="end_turn", usage=Usage(900, 12))],
        ]

    async def stream(self, *, system, messages, tools=None):
        for chunk in self.turns.pop(0):
            yield chunk


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    load_dotenv_if_present()
    os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "portfolio-smoke")
    tracer = get_tracer()
    if not tracer.enabled:
        print("Langfuse desligado: defina LANGFUSE_PUBLIC_KEY e LANGFUSE_SECRET_KEY (e LANGFUSE_BASE_URL).", file=sys.stderr)
        return 2
    client = tracer._client  # noqa: SLF001
    host = os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST") or "https://cloud.langfuse.com"
    try:
        ok = client.auth_check()
    except Exception as exc:
        print(f"Não consegui falar com o Langfuse em {host}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    if not ok:
        print(f"O Langfuse em {host} recusou as chaves (auth_check falhou).", file=sys.stderr)
        return 3
    print(f"Conectado ao Langfuse em {host} (ambiente: {os.environ['LANGFUSE_TRACING_ENVIRONMENT']}).")

    index = MovieIndex(load_csv(DATA_DIR / "sample_movies.csv", source="amostra"), HashingEmbedder())
    settings = Settings(provider="echo", model="smoke-test-model", rate_limit_per_minute=100)
    with TestClient(create_app(settings, ScriptedProvider(), index, tracer)) as app_client:
        reply = app_client.post("/chat", json={"message": "Teste de fumaça: filme sobre robôs"})
        session_id = reply.text.split('"session_id": "')[1].split('"')[0]
        trace_id = reply.text.split('"trace_id": "')[1].split('"')[0]
    tracer.flush()
    print(f"Trace enviado (sessão {session_id}). Esperando o servidor processar...")

    deadline, seen = time.monotonic() + 60, {}
    while time.monotonic() < deadline and not EXPECTED <= set(seen):
        try:
            page = client.api.observations.get_many(session_id=session_id, limit=20, fields="core,basic,usage,model")
            seen = {o.name: o for o in page.data}
        except Exception as exc:
            print(f"  (leitura ainda falhou: {type(exc).__name__}; tentando de novo)")
        if not EXPECTED <= set(seen):
            time.sleep(3)
    if not EXPECTED <= set(seen):
        print(f"FALHA: faltam as observações {sorted(EXPECTED - set(seen))}. Chegaram: {', '.join(sorted(seen)) or 'nenhuma'}",
              file=sys.stderr)
        return 1
    print("OK: chegaram as observações (movie-turn, 2 generations, search_movies).")

    deadline, found = time.monotonic() + 60, set()
    while time.monotonic() < deadline and not EXPECTED_SCORES <= found:
        try:
            found = {s.name for s in client.api.scores.get_many(trace_id=trace_id, limit=50).data}
        except Exception:
            pass
        if not EXPECTED_SCORES <= found:
            time.sleep(3)
    if EXPECTED_SCORES <= found:
        print(f"OK: chegaram as {len(EXPECTED_SCORES)} notas.")
    else:
        print(f"AVISO: ainda não consigo ler as notas {sorted(EXPECTED_SCORES - found)} (foram enviadas; confira na interface).")
    try:
        print("  veja em:", client.get_trace_url(trace_id=trace_id))
    except Exception:
        pass
    tracer.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
