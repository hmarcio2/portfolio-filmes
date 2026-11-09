"""Embeddings das sinopses. Mesma regra do Chatbot: local (Ollama, ``LOCAL_EMBEDDING_*``) ou API do Google.

Ordem quando ``EMBEDDING_PROVIDER`` não é informado: Ollama se ``LOCAL_EMBEDDING_ENABLED``, Google se
houver ``GOOGLE_API_KEY``, senão o ``HashingEmbedder`` offline (sem semântica real: só serve para
rodar sem chave e nos testes).
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Protocol, Sequence

from portfolio_core.bm25 import tokenize


class Embedder(Protocol):
    id: str  # identifica o modelo; um índice só é válido para o embedder que o criou

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Bag-of-words + trigramas de caracteres projetados por hash em ``dim`` dimensões (determinístico, offline)."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.id = f"hash-{dim}"

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in tokenize(text):
            for feature, weight in [(f"w:{token}", 1.0), *[(f"c:{token[i:i + 3]}", 0.3) for i in range(max(1, len(token) - 2))]]:
                h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
                vec[h % self.dim] += weight if (h >> 40) & 1 else -weight
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class LangChainEmbedder:
    def __init__(self, model, embedder_id: str):
        self._model = model
        self.id = embedder_id

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):  # lotes pequenos: respeita limites de requisição
            out.extend(self._model.embed_documents(list(texts[i:i + 64])))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._model.embed_query(text)


def _bool(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def get_embedder(provider: str | None = None) -> Embedder:
    env = os.environ
    provider = provider or env.get("EMBEDDING_PROVIDER") or (
        "ollama" if _bool(env.get("LOCAL_EMBEDDING_ENABLED")) else "google_genai" if env.get("GOOGLE_API_KEY") else "hash"
    )
    if provider == "hash":
        return HashingEmbedder()
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        model = env.get("LOCAL_EMBEDDING_MODEL", "nomic-embed-text")
        base_url = env.get("LOCAL_LLM_BASE_URL", "http://localhost:11434")
        return LangChainEmbedder(OllamaEmbeddings(model=model, base_url=base_url), f"ollama-{model}")
    if provider == "google_genai":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        model = env.get("EMBEDDING_MODEL", "gemini-embedding-001")
        return LangChainEmbedder(GoogleGenerativeAIEmbeddings(model=model), f"google-{model}")
    raise ValueError(f"EMBEDDING_PROVIDER desconhecido: {provider!r}")
