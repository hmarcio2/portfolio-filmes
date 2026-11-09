"""Roda algumas conversas pelo agente REAL (LLM + busca) e mostra o que ele fez: as chamadas de ferramenta (com a
consulta que o modelo escreveu), os filmes devolvidos e a resposta. Serve para ler, não para pontuar.

Uso (na pasta 01-recomendador-filmes), com um LLM (GOOGLE_API_KEY) e o catálogo escolhido:

    $env:PORTFOLIO_FILMES_CATALOG = "cmu"; $env:EMBEDDING_PROVIDER = "ollama"; $env:GOOGLE_API_KEY = "..."
    ../.venv/Scripts/python -m evaluation.try_agent
    ../.venv/Scripts/python -m evaluation.try_agent "Quero um filme de terror com uma casa mal-assombrada"
"""

from __future__ import annotations

import json
import sys

from fastapi.testclient import TestClient

from app.catalog import load_default_catalog
from app.embeddings import get_embedder
from app.index import MovieIndex
from app.main import _index_path, create_app
from portfolio_core import Settings, get_tracer  # noqa: F401  (get_tracer: sem chaves do Langfuse é um no-op)

DEFAULT_CONVERSATIONS = [
    ["Quero algo parecido com Matrix, mas mais recente"],
    ["Um suspense psicológico com um final surpreendente"],
    ["filmes de animação sobre brinquedos que ganham vida", "me mostre outros, mais antigos que 2000"],
    ["Algo como A Origem"],  # título em português: o modelo precisa usar o original ("Inception")
    ["Algo como Dune"],       # título ambíguo: refilmagens
]


def parse_sse(text: str):
    for block in text.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        yield name.removeprefix("event: "), json.loads(data.removeprefix("data: "))


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    conversations = [[q] for q in sys.argv[1:]] or DEFAULT_CONVERSATIONS
    settings = Settings.from_env().resolved()
    if settings.provider == "echo":
        print("Nenhum LLM configurado: defina GOOGLE_API_KEY (ou outro provedor).", file=sys.stderr)
        return 2
    catalog = load_default_catalog()
    index = MovieIndex(catalog, get_embedder(), _index_path())
    print(f"catálogo: {catalog.source} ({len(catalog.movies)} filmes, {catalog.language}); LLM: {settings.provider}/{settings.model}\n")
    with TestClient(create_app(settings, index=index)) as client:
        for messages in conversations:
            sid = None
            for text in messages:
                print(f"USUÁRIO: {text}")
                reply = client.post("/chat", json={"message": text, "session_id": sid})
                answer = ""
                for name, data in parse_sse(reply.text):
                    if name == "meta":
                        sid = data["session_id"]
                    elif name == "tool":
                        print(f"  ferramenta: {data['name']} {json.dumps(data['input'], ensure_ascii=False)}")
                    elif name == "movies":
                        ref = f" (parecidos com {data['reference']['title']})" if data.get("reference") else ""
                        print(f"  filmes{ref}: " + "; ".join(f"{m['title']} ({m['year']})" for m in data["movies"]))
                    elif name == "delta":
                        answer += data["text"]
                    elif name == "error":
                        print(f"  ERRO: {data['error']}")
                print(f"ASSISTENTE: {answer.strip()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
