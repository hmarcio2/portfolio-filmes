"""Índice de filmes: banco vetorial (Qdrant) + BM25, combinados por Reciprocal Rank Fusion.

O Qdrant roda em modo local (em memória ou numa pasta), com a mesma API do servidor; para
escalar, basta apontar o cliente para um servidor Qdrant. Cada combinação (embedder, catálogo)
tem a sua coleção; mudar de modelo de embedding recria o índice.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from qdrant_client import QdrantClient, models

from portfolio_core.bm25 import BM25Index

from .catalog import Catalog, Movie
from .embeddings import Embedder

RRF_K = 60


def default_text_weight(embedder_id: str) -> float:
    """Peso do BM25 na busca híbrida. Medido em evaluation/ (README): com o embedder multilíngue do Google a
    busca só vetorial foi a melhor (nDCG 0,996) e misturar BM25 piorou; com embedders mais fracos em português
    (nomic-embed-text, hash) a híbrida com peso 1 foi a melhor. ``PORTFOLIO_FILMES_TEXT_WEIGHT`` sobrescreve."""
    override = os.environ.get("PORTFOLIO_FILMES_TEXT_WEIGHT")
    if override:
        return float(override)
    return 0.0 if embedder_id.startswith("google") else 1.0


@dataclass
class Filters:
    genres: list[str] = field(default_factory=list)  # qualquer um dos gêneros (OR), nomes canônicos do catálogo
    year_min: int | None = None
    year_max: int | None = None
    min_rating: float | None = None
    exclude_ids: set[int] = field(default_factory=set)

    def accepts(self, m: Movie) -> bool:
        if m.id in self.exclude_ids:
            return False
        if self.genres and not set(self.genres) & set(m.genres):
            return False
        if self.year_min is not None and (m.year is None or m.year < self.year_min):
            return False
        if self.year_max is not None and (m.year is None or m.year > self.year_max):
            return False
        if self.min_rating is not None and (m.rating is None or m.rating < self.min_rating):
            return False
        return True

    def to_qdrant(self) -> models.Filter | None:
        must: list = []
        if self.genres:
            must.append(models.FieldCondition(key="genres", match=models.MatchAny(any=self.genres)))
        if self.year_min is not None or self.year_max is not None:
            must.append(models.FieldCondition(key="year", range=models.Range(gte=self.year_min, lte=self.year_max)))
        if self.min_rating is not None:
            must.append(models.FieldCondition(key="rating", range=models.Range(gte=self.min_rating)))
        must_not = [models.HasIdCondition(has_id=sorted(self.exclude_ids))] if self.exclude_ids else []
        return models.Filter(must=must or None, must_not=must_not or None) if (must or must_not) else None


@dataclass
class Hit:
    movie: Movie
    score: float
    vector_rank: int | None = None
    text_rank: int | None = None


class MovieIndex:
    def __init__(self, catalog: Catalog, embedder: Embedder, path: Path | None = None, text_weight: float | None = None):
        self.catalog = catalog
        self.embedder = embedder
        # peso do BM25 na busca híbrida; 0 = só vetorial (ver evaluation/ e o README)
        self.text_weight = default_text_weight(embedder.id) if text_weight is None else text_weight
        self._by_id = catalog.by_id()
        self._bm25 = BM25Index([m.document for m in catalog.movies])
        self._client = QdrantClient(path=str(path)) if path else QdrantClient(":memory:")
        self.collection = self._collection_name()
        self._ensure_collection()

    # ---------- construção ----------

    def _collection_name(self) -> str:
        fingerprint = hashlib.blake2b(
            "|".join(f"{m.id}:{m.title}:{len(m.overview)}" for m in self.catalog.movies).encode(), digest_size=4
        ).hexdigest()
        return re.sub(r"[^A-Za-z0-9_.-]", "_", f"movies_{self.embedder.id}_{fingerprint}")

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection in existing:
            if self._client.count(self.collection).count == len(self.catalog.movies):
                return
            self._client.delete_collection(self.collection)
        movies = self.catalog.movies
        vectors = self.embedder.embed_documents([m.document for m in movies])
        self._client.create_collection(
            self.collection, vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE)
        )
        self._client.upsert(
            self.collection,
            points=[
                models.PointStruct(
                    id=m.id, vector=v,
                    payload={"genres": list(m.genres), "year": m.year, "rating": m.rating, "title": m.title},
                )
                for m, v in zip(movies, vectors)
            ],
        )
        for name in existing:  # coleções de outros embedders/catálogos ficam obsoletas
            if name.startswith("movies_") and name != self.collection:
                self._client.delete_collection(name)

    def __len__(self) -> int:
        return len(self.catalog.movies)

    # ---------- busca ----------

    def _vector_search(self, query: str, filters: Filters, limit: int, vector: list[float] | None = None) -> list[int]:
        result = self._client.query_points(
            self.collection, query=vector if vector is not None else self.embedder.embed_query(query),
            query_filter=filters.to_qdrant(), limit=limit, with_payload=False,
        )
        return [int(p.id) for p in result.points]

    def vector_of(self, movie: Movie) -> list[float]:
        """O embedding que o índice já guarda do filme (não chama o embedder de novo)."""
        (point,) = self._client.retrieve(self.collection, ids=[movie.id], with_vectors=True)
        return list(point.vector)

    def _text_search(self, query: str, filters: Filters, limit: int) -> list[int]:
        allowed = {i for i, m in enumerate(self.catalog.movies) if filters.accepts(m)}
        return [self.catalog.movies[i].id for i, _ in self._bm25.search(query, limit, allowed)]

    def similar(self, movie: Movie, filters: Filters | None = None, k: int = 8, mode: str = "vector",
                text_weight: float | None = None) -> list[Hit]:
        """Filmes de sinopse parecida com a de ``movie`` (que nunca aparece no resultado).

        Por padrão só vetorial: usa o embedding que o índice já guarda do filme (título, gêneros e sinopse), sem
        chamar o embedder. Medido no catálogo CMU (``evaluation/run_similar_eval.py``), misturar o BM25 da sinopse
        não ajudou. ``mode="hybrid"`` ou ``"text"`` usam a sinopse como consulta do BM25.
        """
        base = filters or Filters()
        filters = Filters(base.genres, base.year_min, base.year_max, base.min_rating, set(base.exclude_ids) | {movie.id})
        return self.search(movie.overview, filters, k, mode, text_weight, query_vector=self.vector_of(movie))

    def search(self, query: str, filters: Filters | None = None, k: int = 8, mode: str = "hybrid",
               text_weight: float | None = None, query_vector: list[float] | None = None) -> list[Hit]:
        """``mode``: ``hybrid`` (vetor + BM25 por RRF ponderado), ``vector`` ou ``text``.

        ``text_weight`` é o peso do BM25 na fusão (o do vetor é 1); padrão: ``self.text_weight``.
        ``query_vector`` é o embedding já pronto da consulta (usado por ``similar``); sem ele, ``query`` é embutida.
        """
        filters = filters or Filters()
        weight = self.text_weight if text_weight is None else text_weight
        if mode == "hybrid" and weight == 0:
            mode = "vector"
        elif mode == "text":
            weight = 1.0  # busca puramente lexical: o peso só vale na fusão com o vetor
        pool = max(k * 4, 30)
        vec = self._vector_search(query, filters, pool, query_vector) if mode in {"hybrid", "vector"} else []
        txt = self._text_search(query, filters, pool) if mode in {"hybrid", "text"} else []
        v_rank = {mid: r for r, mid in enumerate(vec, 1)}
        t_rank = {mid: r for r, mid in enumerate(txt, 1)}
        fused = {
            mid: (1 / (RRF_K + v_rank[mid]) if mid in v_rank else 0.0)
            + (weight / (RRF_K + t_rank[mid]) if mid in t_rank else 0.0)
            for mid in set(v_rank) | set(t_rank)
        }
        ranked = sorted(fused, key=lambda mid: fused[mid], reverse=True)[:k]
        return [Hit(self._by_id[mid], fused[mid], v_rank.get(mid), t_rank.get(mid)) for mid in ranked]
