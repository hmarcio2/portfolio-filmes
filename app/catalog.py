"""Catálogo de filmes: leitura de CSV (amostra embutida, CMU Movie Summary Corpus ou exportação do TMDB), gêneros e
busca de um filme pelo título."""

from __future__ import annotations

import csv
import difflib
import os
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
POSTER_BASE = "https://image.tmdb.org/t/p/w342"

TMDB_ATTRIBUTION = "Este produto usa a API do TMDB, mas não é endossado nem certificado pelo TMDB."
CMU_ATTRIBUTION = ("Sinopses e metadados: CMU Movie Summary Corpus (Bamman, O'Connor e Smith, ACL 2013), "
                   "licença CC BY-SA; as sinopses vêm da Wikipedia em inglês.")
# Idioma das sinopses de cada fonte: o agente escreve a consulta nesse idioma. A amostra embutida é em português;
# o script do TMDB baixa em pt-BR; o CMU é em inglês. ``PORTFOLIO_FILMES_CATALOG_LANG`` (pt, en) sobrescreve.
SOURCE_LANGUAGE = {"amostra": "pt", "tmdb": "pt", "cmu": "en"}


def norm(text: str) -> str:
    """Minúsculas e sem acentos, para comparar gêneros e títulos."""
    t = unicodedata.normalize("NFKD", text.strip().lower())
    return "".join(c for c in t if not unicodedata.combining(c))


@dataclass(frozen=True)
class Movie:
    id: int
    title: str
    year: int | None
    genres: tuple[str, ...]
    overview: str
    rating: float | None = None
    votes: int | None = None
    poster_url: str | None = None

    @property
    def document(self) -> str:
        """Texto que vira embedding: título, gêneros e sinopse."""
        return f"{self.title}. {', '.join(self.genres)}. {self.overview}"

    def as_dict(self, overview_chars: int = 300) -> dict:
        overview = self.overview if len(self.overview) <= overview_chars else self.overview[:overview_chars].rstrip() + "…"
        return {
            "id": self.id, "title": self.title, "year": self.year, "genres": list(self.genres),
            "overview": overview, "rating": self.rating, "votes": self.votes, "poster_url": self.poster_url,
        }


@dataclass
class Catalog:
    movies: list[Movie]
    source: str = "amostra"  # amostra | tmdb | cmu | csv
    genres: list[str] = field(init=False)

    def __post_init__(self):
        counts: dict[str, int] = {}
        for m in self.movies:
            for g in m.genres:
                counts[g] = counts.get(g, 0) + 1
        self.genres = sorted(counts)
        self._genre_counts = counts
        self._titles: dict[str, list[Movie]] | None = None

    @property
    def attribution(self) -> str | None:
        return {"tmdb": TMDB_ATTRIBUTION, "cmu": CMU_ATTRIBUTION}.get(self.source)

    @property
    def language(self) -> str:
        """Idioma das sinopses (``pt`` ou ``en``)."""
        return os.environ.get("PORTFOLIO_FILMES_CATALOG_LANG") or SOURCE_LANGUAGE.get(self.source, "en")

    def top_genres(self, n: int = 40) -> list[str]:
        """Os ``n`` gêneros mais frequentes, em ordem alfabética (o CMU tem centenas, muitos raros demais para o prompt)."""
        return sorted(sorted(self._genre_counts, key=lambda g: -self._genre_counts[g])[:n])

    def find(self, title: str, year: int | None = None) -> tuple[list[Movie], list[str]]:
        """Filmes com o título dado (sem maiúsculas nem acentos; ``year`` filtra remakes) e, se não houver nenhum,
        até 5 títulos parecidos para sugerir."""
        if self._titles is None:
            self._titles = {}
            for m in self.movies:
                self._titles.setdefault(norm(m.title), []).append(m)
        key = norm(title)
        found = self._titles.get(key, [])
        if year is not None and len(found) > 1:
            found = [m for m in found if m.year == year] or found
        if found:
            return sorted(found, key=lambda m: m.year or 0), []
        close = difflib.get_close_matches(key, list(self._titles), n=5, cutoff=0.6)
        return [], [self._titles[c][0].title for c in close]

    def by_id(self) -> dict[int, Movie]:
        return {m.id: m for m in self.movies}

    def match_genres(self, wanted: list[str]) -> tuple[list[str], list[str]]:
        """Mapeia gêneros pedidos para os existentes (sem acento, por igualdade ou prefixo). Retorna (ok, desconhecidos)."""
        known = {norm(g): g for g in self.genres}
        found, unknown = [], []
        for w in wanted:
            n = norm(w)
            hit = known.get(n) or next((g for k, g in known.items() if k.startswith(n) or n.startswith(k)), None)
            (found if hit else unknown).append(hit or w)
        return list(dict.fromkeys(found)), unknown


def _opt_float(v: str) -> float | None:
    try:
        return float(v) if v.strip() else None
    except ValueError:
        return None


def _opt_int(v: str) -> int | None:
    try:
        return int(float(v)) if v.strip() else None
    except ValueError:
        return None


def load_csv(path: Path, source: str = "csv") -> Catalog:
    movies: list[Movie] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            overview = (row.get("overview") or "").strip()
            title = (row.get("title") or "").strip()
            if not title or not overview:  # sem sinopse não há o que comparar
                continue
            poster = (row.get("poster_path") or "").strip()
            movies.append(
                Movie(
                    id=int(row["id"]),
                    title=title,
                    year=_opt_int(row.get("year") or ""),
                    genres=tuple(g.strip() for g in (row.get("genres") or "").split("|") if g.strip()),
                    overview=overview,
                    rating=_opt_float(row.get("rating") or ""),
                    votes=_opt_int(row.get("votes") or ""),
                    poster_url=f"{POSTER_BASE}{poster}" if poster.startswith("/") else None,
                )
            )
    return Catalog(movies, source=source)


def load_default_catalog() -> Catalog:
    """Escolhe o catálogo. ``PORTFOLIO_FILMES_CATALOG`` = ``sample``, ``cmu`` ou ``tmdb`` força um; sem isso vale o
    primeiro que existir entre ``tmdb_movies.csv`` (scripts/fetch_tmdb.py), ``cmu_movies.csv`` (scripts/prepare_cmu.py)
    e a amostra embutida.

    Os arquivos são procurados em ``PORTFOLIO_FILMES_CATALOG_DIR`` (no Docker, uma pasta montada do host) e, por
    padrão, em ``data/``.
    """
    folder = Path(os.environ.get("PORTFOLIO_FILMES_CATALOG_DIR") or DATA_DIR)
    files = {"tmdb": folder / "tmdb_movies.csv", "cmu": folder / "cmu_movies.csv", "sample": DATA_DIR / "sample_movies.csv"}
    wanted = (os.environ.get("PORTFOLIO_FILMES_CATALOG") or "auto").strip().lower()
    if wanted not in {"auto", *files}:
        raise ValueError(f"PORTFOLIO_FILMES_CATALOG inválido: {wanted!r} (use sample, cmu, tmdb ou auto)")
    if wanted != "auto":
        if not files[wanted].is_file():
            raise FileNotFoundError(f"PORTFOLIO_FILMES_CATALOG={wanted}, mas {files[wanted]} não existe.")
        return load_csv(files[wanted], source="amostra" if wanted == "sample" else wanted)
    for source in ("tmdb", "cmu"):
        if files[source].is_file():
            return load_csv(files[source], source=source)
    return load_csv(files["sample"], source="amostra")
