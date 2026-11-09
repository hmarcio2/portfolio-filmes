"""Notas automáticas (scores) de cada turno do recomendador, gravadas no trace do Langfuse.

São **sinais de observabilidade**, não bloqueios: servem para ver quantas vezes o agente respondeu sem buscar, buscou
e não achou nada, ou citou um filme do catálogo que a busca não devolveu. As verificações são determinísticas e
baratas (sem outro LLM) e são heurísticas: o ``comment`` de cada nota diz o que disparou, para auditar.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .catalog import norm

MIN_TITLE_CHARS = 5  # títulos curtos ("Up", "Ela") dariam falsos positivos em qualquer frase


@dataclass(frozen=True)
class Score:
    name: str
    value: float
    data_type: str = "BOOLEAN"  # BOOLEAN (0/1) ou NUMERIC
    comment: str | None = None


def _words(text: str) -> str:
    """Tira acentos e troca a pontuação por espaço, com espaços nas pontas, para casar só palavras inteiras.
    Mantém as maiúsculas: muitos títulos são palavras comuns (``Alien``, ``Speed``, ``Heat``) e só contam como
    citação quando escritos como no catálogo."""
    folded = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return " " + re.sub(r"[^A-Za-z0-9]+", " ", folded) + " "


def mentioned_titles(answer: str, titles: list[str]) -> list[str]:
    """Títulos do catálogo citados na resposta (palavras inteiras, com a mesma grafia das maiúsculas; acentos e
    pontuação não contam)."""
    text = _words(answer)
    return [t for t in titles if len(norm(t)) >= MIN_TITLE_CHARS and _words(t) in text]


def evaluate_turn(*, question: str, answer: str, searches: list[dict], catalog_titles: list[str],
                  shown_titles: list[str], failed: bool = False) -> list[Score]:
    """Notas de um turno. ``searches`` é o ``meta`` de cada chamada de busca; ``shown_titles`` são os filmes já
    mostrados na sessão (inclui os deste turno): citá-los não é um problema."""
    if failed or not answer:
        return []
    empty = [s.get("query", "") for s in searches if not s.get("titles")]
    known = {norm(t) for t in shown_titles} | {norm(t) for t in mentioned_titles(question, catalog_titles)}
    unretrieved = [t for t in mentioned_titles(answer, catalog_titles) if norm(t) not in known]
    return [
        Score("no_search", not searches,
              comment="o agente respondeu sem chamar search_movies" if not searches else None),
        Score("empty_search", bool(empty),
              comment=("buscas sem resultado: " + "; ".join(q[:80] for q in empty)) if empty else None),
        Score("unretrieved_titles", len(unretrieved), "NUMERIC",
              ("filmes do catálogo citados que a busca não devolveu: " + ", ".join(unretrieved)) if unretrieved else None),
    ]
