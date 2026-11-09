"""Converte o CMU Movie Summary Corpus para o catálogo do app (data/cmu_movies.csv).

O corpus (Bamman, O'Connor e Smith, ACL 2013; licença CC BY-SA) tem as sinopses em inglês de 42 mil filmes, extraídas
da Wikipedia, e metadados do Freebase (título, data, bilheteria, gêneros). Não tem nota nem pôster.

Baixe e extraia você mesmo (cerca de 46 MB):

    https://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz

Uso (na pasta 01-recomendador-filmes):

    ../.venv/Scripts/python scripts/prepare_cmu.py --src <pasta MovieSummaries> --max 5000

Por que ``--max``: o índice precisa de um embedding por filme. Com 42 mil filmes, isso leva horas no Ollama local ou
estoura a cota gratuita da API do Google. Quando há menos vagas que filmes, ficam primeiro os que têm bilheteria
registrada (um indício de popularidade), do maior para o menor, e depois os de sinopse mais longa. Não é uma lista
dos "melhores" filmes: o corpus não tem notas.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "cmu_movies.csv"
# Gêneros do Freebase que não dizem nada sobre o filme em si
NOISE_GENRES = {"World cinema", "Indie", "Black-and-white"}
MAX_GENRES = 6


_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_REF = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>|<[^>]+>", re.DOTALL)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_MARKUP_LEFTOVERS = ("<", "|", "http", "{{", "}}", "ref name", "&amp")


def strip_wiki_markup(text: str) -> str:
    """O corpus vem de um dump da Wikipedia: tira modelos ({{plot}}), notas (<ref>) e links ([[...]]) que sobraram,
    e descarta as frases em que ainda restou marcação quebrada (ex.: uma citação cortada no meio)."""
    text = html.unescape(text).replace(" ", " ")
    for _ in range(3):  # modelos aninhados
        text = _TEMPLATE.sub(" ", text)
    text = _REF.sub(" ", text).replace("[[", "").replace("]]", "")
    sentences = [s for s in _SENTENCE_END.split(re.sub(r"\s+", " ", text).strip())
                 if not any(mark in s for mark in _MARKUP_LEFTOVERS)]
    return " ".join(sentences)


def clean_overview(text: str, max_chars: int) -> str:
    """Sem marcação da Wikipedia, espaços normalizados e corte na última frase inteira que cabe em ``max_chars``."""
    text = strip_wiki_markup(text)
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    return (cut[:end + 1] if end > max_chars // 2 else cut.rstrip()).strip()


def parse_year(date: str) -> int | None:
    m = re.match(r"(\d{4})", date.strip())
    year = int(m.group(1)) if m else None
    return year if year and 1870 <= year <= 2030 else None


def parse_genres(raw: str) -> list[str]:
    try:
        names = list(json.loads(raw).values()) if raw.strip() else []
    except json.JSONDecodeError:
        return []
    return [g for g in dict.fromkeys(names) if g not in NOISE_GENRES][:MAX_GENRES]


def read_metadata(path: Path) -> dict[int, dict]:
    movies: dict[int, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or not cols[0].isdigit() or not cols[2].strip():
                continue
            try:
                revenue = float(cols[4]) if cols[4].strip() else 0.0
            except ValueError:
                revenue = 0.0
            movies[int(cols[0])] = {"title": cols[2].strip(), "year": parse_year(cols[3]), "revenue": revenue,
                                    "genres": parse_genres(cols[8])}
    return movies


def read_plots(path: Path) -> dict[int, str]:
    plots: dict[int, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            movie_id, _, text = line.partition("\t")
            if movie_id.isdigit() and text.strip():
                plots[int(movie_id)] = text
    return plots


def build(src: Path, *, limit: int, min_chars: int, max_chars: int) -> list[dict]:
    meta = read_metadata(src / "movie.metadata.tsv")
    plots = read_plots(src / "plot_summaries.txt")
    rows = []
    for movie_id, plot in plots.items():
        m = meta.get(movie_id)
        overview = clean_overview(plot, max_chars)
        if m is None or len(overview) < min_chars:
            continue
        rows.append({"id": movie_id, "title": m["title"], "year": m["year"] or "", "genres": "|".join(m["genres"]),
                     "rating": "", "votes": "", "overview": overview, "poster_path": "",
                     "_revenue": m["revenue"], "_len": len(overview)})
    rows.sort(key=lambda r: (r["_revenue"] > 0, r["_revenue"], r["_len"]), reverse=True)
    return sorted(rows[:limit], key=lambda r: r["id"])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, type=Path, help="pasta extraída (com movie.metadata.tsv e plot_summaries.txt)")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--max", type=int, default=5000, dest="limit", help="quantos filmes manter (padrão: 5000)")
    p.add_argument("--min-chars", type=int, default=300, help="descarta sinopses menores que isto (padrão: 300)")
    p.add_argument("--max-chars", type=int, default=1500, help="corta sinopses maiores, em fim de frase (padrão: 1500)")
    args = p.parse_args()
    for name in ("movie.metadata.tsv", "plot_summaries.txt"):
        if not (args.src / name).is_file():
            print(f"Não achei {args.src / name}. Aponte --src para a pasta MovieSummaries extraída.", file=sys.stderr)
            return 2
    rows = build(args.src, limit=args.limit, min_chars=args.min_chars, max_chars=args.max_chars)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "title", "year", "genres", "rating", "votes", "overview", "poster_path"]
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with_revenue = sum(1 for r in rows if r["_revenue"] > 0)
    print(f"{len(rows)} filmes gravados em {args.out} ({with_revenue} com bilheteria registrada).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
