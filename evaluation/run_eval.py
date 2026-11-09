"""Mede a qualidade da busca (sem LLM): vetorial vs BM25 vs híbrida, por embedder.

Uso (na pasta 01-recomendador-filmes):

    ../.venv/Scripts/python -m evaluation.run_eval                       # embedder automático (ver app/embeddings.py)
    ../.venv/Scripts/python -m evaluation.run_eval --embedder hash       # offline, sem semântica real (linha de base)
    ../.venv/Scripts/python -m evaluation.run_eval --embedder google_genai
    ../.venv/Scripts/python -m evaluation.run_eval --embedder ollama

Usa o catálogo de data/ (a amostra de 44 filmes, ou data/tmdb_movies.csv se existir). As consultas e os
julgamentos de relevância valem só para a amostra embutida.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from app.catalog import DATA_DIR, load_csv
from app.embeddings import get_embedder
from app.index import MovieIndex

from .cases import QUERIES
from .metrics import ndcg_at_k, recall_at_k, reciprocal_rank

RESULTS_DIR = Path(__file__).resolve().parent / "results"
# nome -> argumentos de MovieIndex.search. "híbrida" usa o peso padrão do embedder (app/index.py); as outras são explícitas.
MODES = {
    "vetorial": dict(mode="vector"),
    "bm25": dict(mode="text"),
    "híbr. w=1": dict(mode="hybrid", text_weight=1.0),
    "híbr. w=0.2": dict(mode="hybrid", text_weight=0.2),
}


class CachedQueries:
    """Memoriza o embedding de cada consulta: as variantes de busca repetem as mesmas consultas, e APIs com
    limite por minuto (plano gratuito) não aguentam a rajada."""

    def __init__(self, inner):
        self._inner, self.id, self._cache = inner, inner.id, {}

    def embed_documents(self, texts):
        return self._inner.embed_documents(texts)

    def embed_query(self, text):
        if text not in self._cache:
            self._cache[text] = self._inner.embed_query(text)
        return self._cache[text]


def evaluate(index: MovieIndex, k: int = 10) -> dict:
    out: dict = {}
    for mode, kwargs in MODES.items():
        rows = []
        for q in QUERIES:
            ranked = [h.movie.title for h in index.search(q.text, k=k, **kwargs)]
            rows.append({
                "query": q.text, "tag": q.tag, "top": ranked[:3],
                "ndcg": ndcg_at_k(ranked, q.relevant, k), "recall": recall_at_k(ranked, q.relevant, k),
                "mrr": reciprocal_rank(ranked, q.relevant),
            })
        out[mode] = rows
    return out


def summarize(rows: list[dict], tag: str | None = None) -> tuple[float, float, float]:
    rows = [r for r in rows if tag is None or r["tag"] == tag]
    n = len(rows) or 1
    return sum(r["ndcg"] for r in rows) / n, sum(r["recall"] for r in rows) / n, sum(r["mrr"] for r in rows) / n


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embedder", help="hash | ollama | google_genai (padrão: automático)")
    p.add_argument("-k", type=int, default=10)
    args = p.parse_args()

    catalog = load_csv(DATA_DIR / "sample_movies.csv", source="amostra")
    embedder = CachedQueries(get_embedder(args.embedder))
    print(f"Embedder: {embedder.id} · catálogo: {len(catalog.movies)} filmes · {len(QUERIES)} consultas\n")
    index = MovieIndex(catalog, embedder)  # em memória: não disputa a pasta do índice com o servidor

    results = evaluate(index, args.k)
    print(f"Peso padrão do BM25 para este embedder (o que o app usa): {index.text_weight}\n")
    print(f"{'modo':<12} {'grupo':<10} {'nDCG@'+str(args.k):>8} {'recall@'+str(args.k):>10} {'MRR':>6}")
    for mode in MODES:
        for tag in (None, "direta", "parafrase", "lexical"):
            ndcg, rec, mrr = summarize(results[mode], tag)
            print(f"{mode:<12} {tag or 'todas':<10} {ndcg:>8.3f} {rec:>10.3f} {mrr:>6.3f}")
    print("\nPiores consultas (híbrida, w=1):")
    for r in sorted(results["híbr. w=1"], key=lambda r: r["ndcg"])[:3]:
        print(f"  nDCG {r['ndcg']:.2f}  {r['query']!r} -> {r['top']}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{embedder.id}.json"
    out.write_text(json.dumps({"embedder": embedder.id, "k": args.k, "results": results}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\nDetalhes: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
