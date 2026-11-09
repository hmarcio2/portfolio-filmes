"""Baixa os filmes mais bem avaliados do TMDB (em português) para data/tmdb_movies.csv.

Uso (na pasta 01-recomendador-filmes), com uma chave gratuita da API v3 do TMDB:

    $env:TMDB_API_KEY = "sua-chave"
    ../.venv/Scripts/python scripts/fetch_tmdb.py --pages 50        # ~1000 filmes

Depois, reinicie o app: ele prefere data/tmdb_movies.csv à amostra embutida e recria o índice.
Os termos do TMDB exigem atribuição (o app a mostra quando usa este arquivo) e restringem o uso comercial;
confira-os em https://www.themoviedb.org/api-terms-of-use antes de publicar.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.themoviedb.org/3"
OUT = Path(__file__).resolve().parent.parent / "data" / "tmdb_movies.csv"


def get(path: str, key: str, **params) -> dict:
    url = f"{API}{path}?{urllib.parse.urlencode({'api_key': key, 'language': 'pt-BR', **params})}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 3:  # limite de taxa: espera e tenta de novo
                time.sleep(2 * (attempt + 1))
                continue
            raise SystemExit(f"TMDB respondeu {exc.code} em {path}. Confira a chave TMDB_API_KEY.") from None
    raise SystemExit("TMDB: muitas tentativas.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pages", type=int, default=50, help="páginas de 20 filmes (padrão: 50)")
    args = p.parse_args()
    key = os.environ.get("TMDB_API_KEY")
    if not key:
        print("Defina TMDB_API_KEY (chave v3 gratuita do TMDB).", file=sys.stderr)
        return 2

    genres = {g["id"]: g["name"].lower() for g in get("/genre/movie/list", key)["genres"]}
    rows, seen = [], set()
    for page in range(1, args.pages + 1):
        data = get("/movie/top_rated", key, page=page)
        for m in data["results"]:
            overview = (m.get("overview") or "").strip()
            if m["id"] in seen or not overview:
                continue
            seen.add(m["id"])
            rows.append({
                "id": m["id"], "title": m["title"], "year": (m.get("release_date") or "")[:4],
                "genres": "|".join(genres[g] for g in m.get("genre_ids", []) if g in genres),
                "rating": m.get("vote_average", ""), "votes": m.get("vote_count", ""),
                "overview": overview, "poster_path": m.get("poster_path") or "",
            })
        if page >= data.get("total_pages", page):
            break
        time.sleep(0.1)

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "title", "year", "genres", "rating", "votes", "overview", "poster_path"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} filmes gravados em {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
