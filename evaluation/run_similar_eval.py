"""Mede a busca "filmes parecidos com este" (``MovieIndex.similar``) no catálogo CMU, sem LLM.

Não há como julgar à mão milhares de filmes, então o gabarito é **a mesma franquia**: para um filme de uma franquia
(Toy Story, Matrix, Star Wars...), os outros filmes dela devem aparecer entre os parecidos. É uma medida honesta mas
limitada:

- é um piso, não o teto: a busca também deve achar filmes parecidos de outras franquias, e isso não é medido;
- favorece a busca lexical: sequências repetem nomes de personagens e lugares, o que o BM25 aproveita muito bem;
- as franquias foram escolhidas por mim, e os grupos pelo início do título.

Uso (na pasta 01-recomendador-filmes; o índice do catálogo precisa existir ou será criado, o que chama o embedder):

    $env:PORTFOLIO_FILMES_CATALOG = "cmu"; $env:EMBEDDING_PROVIDER = "ollama"     # ou google_genai, hash
    ../.venv/Scripts/python -m evaluation.run_similar_eval
"""

from __future__ import annotations

import argparse
import os
import sys

from app.catalog import load_default_catalog
from app.embeddings import get_embedder
from app.index import MovieIndex
from app.main import _index_path

# Início do título de cada franquia (o título inteiro igual ao prefixo, ou o prefixo seguido de espaço ou ':')
FRANCHISES = [
    "Toy Story", "The Matrix", "Star Wars", "The Godfather", "Die Hard", "The Lord of the Rings", "Harry Potter and the",
    "Pirates of the Caribbean", "Shrek", "Rocky", "Jaws", "Terminator", "Indiana Jones and the", "Mission: Impossible",
    "Batman", "Spider-Man", "Back to the Future", "Ice Age", "Scream", "Saw", "Lethal Weapon", "Home Alone",
    "Austin Powers", "The Bourne", "Halloween", "A Nightmare on Elm Street", "Friday the 13th", "X-Men",
]
K = 10
STRATEGIES = {  # nome: (mode, text_weight)
    "só vetorial": ("vector", None),
    "só BM25 (sinopse)": ("text", None),
    "híbrida peso 1": ("hybrid", 1.0),
    "híbrida peso 0,2": ("hybrid", 0.2),
}


def in_franchise(title: str, prefix: str) -> bool:
    return title == prefix or title.startswith(prefix + " ") or title.startswith(prefix + ":")


def groups(catalog) -> dict[str, list]:
    out = {}
    for prefix in FRANCHISES:
        members = [m for m in catalog.movies if in_franchise(m.title, prefix)]
        if len(members) >= 3:
            out[prefix] = members
    return out


def evaluate(index: MovieIndex) -> dict[str, dict[str, float]]:
    """Por estratégia: precisão@5, recall@10 (os outros filmes da franquia achados entre os 10 primeiros, com
    denominador ``min(10, tamanho - 1)``) e o número de consultas. Cada filme da franquia é a referência uma vez."""
    result = {}
    franchises = groups(index.catalog)
    for name, (mode, weight) in STRATEGIES.items():
        p5, r10, n = 0.0, 0.0, 0
        for members in franchises.values():
            ids = {m.id for m in members}
            for ref in members:
                relevant = ids - {ref.id}
                hits = [h.movie.id for h in index.similar(ref, k=K, mode=mode, text_weight=weight)]
                p5 += sum(1 for h in hits[:5] if h in relevant) / 5
                r10 += sum(1 for h in hits if h in relevant) / min(K, len(relevant))
                n += 1
        result[name] = {"precision@5": p5 / n, "recall@10": r10 / n, "consultas": n}
    return result


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    os.environ.setdefault("PORTFOLIO_FILMES_CATALOG", "cmu")
    catalog = load_default_catalog()
    if catalog.source != "cmu":
        print(f"Esta avaliação é para o catálogo CMU (catálogo atual: {catalog.source}). Rode scripts/prepare_cmu.py.", file=sys.stderr)
        return 2
    embedder = get_embedder()
    index = MovieIndex(catalog, embedder, _index_path())
    franchises = groups(catalog)
    print(f"{len(catalog.movies)} filmes, embedder {embedder.id}; {len(franchises)} franquias, "
          f"{sum(len(g) for g in franchises.values())} consultas (cada filme como referência uma vez).\n")
    print(f"{'estratégia':<20} {'precisão@5':>11} {'recall@10':>10}")
    for name, r in evaluate(index).items():
        print(f"{name:<20} {r['precision@5']:>11.3f} {r['recall@10']:>10.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
