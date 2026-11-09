"""Consultas de teste com filmes relevantes (julgados a partir das sinopses da amostra embutida).

Ganho 2 = é o filme procurado; 1 = serviria como resposta aceitável. ``tag="parafrase"`` marca consultas
que quase não repetem palavras da sinopse: é onde embeddings semânticos devem superar o BM25.
``tag="lexical"`` marca buscas por nome ou palavra exata (personagem, sigla, título curto), onde o BM25 deveria ajudar.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Query:
    text: str
    relevant: dict[str, int]  # título -> ganho
    tag: str = "direta"


QUERIES: list[Query] = [
    Query("um filme sobre viagem espacial para salvar a humanidade", {"Interestelar": 2, "Gravidade": 1}),
    Query("invadir a mente de alguém dentro de sonhos", {"A Origem": 2}),
    Query("a realidade é uma simulação criada por máquinas", {"Matrix": 2}),
    Query("suspense psicológico num hospital psiquiátrico numa ilha", {"Ilha do Medo": 2}),
    Query("assassino em série perseguido por detetives", {"Se7en: Os Sete Crimes Capitais": 2, "O Silêncio dos Inocentes": 2}),
    Query("uma família pobre se infiltra na casa de uma família rica", {"Parasita": 2}),
    Query("crime e tráfico numa favela do Rio de Janeiro", {"Cidade de Deus": 2, "Tropa de Elite": 2}),
    Query("animação sobre emoções dentro da cabeça de uma criança", {"Divertida Mente": 2}),
    Query("um idoso viaja numa casa com balões", {"Up: Altas Aventuras": 2}),
    Query("irmãs descobrem espíritos mágicos da floresta", {"Meu Amigo Totoro": 2, "A Viagem de Chihiro": 1}),
    Query("deserto pós-apocalíptico com perseguição de carros", {"Mad Max: Estrada da Fúria": 2}),
    Query("um casal passa a noite andando e conversando pela cidade", {"Antes do Amanhecer": 2}),
    Query("linguista tenta se comunicar com alienígenas", {"A Chegada": 2}),
    Query("família precisa viver em silêncio por causa de criaturas", {"Um Lugar Silencioso": 2}),
    Query("amigos acordam sem lembrar da despedida de solteiro", {"Se Beber, Não Case": 2}),
    Query("família mafiosa e a luta pelo poder", {"O Poderoso Chefão": 2}),
    # paráfrases: pouca sobreposição de palavras com as sinopses
    Query("história de superação de um músico exigente com um mestre abusivo", {"Whiplash: Em Busca da Perfeição": 2}, "parafrase"),
    Query("o mundo inteiro é um palco montado para um só homem", {"O Show de Truman": 2}, "parafrase"),
    Query("o planeta precisa ser abandonado e um pai deixa os filhos para trás", {"Interestelar": 2}, "parafrase"),
    Query("garoto brinca de amizade sem entender a barbárie nazista", {"O Menino do Pijama Listrado": 2, "A Vida é Bela": 1}, "parafrase"),
    Query("um pai transforma uma tragédia numa brincadeira para proteger o filho", {"A Vida é Bela": 2, "O Menino do Pijama Listrado": 1}, "parafrase"),
    Query("criança mexicana descobre o passado da família no além", {"Viva: A Vida é uma Festa": 2}, "parafrase"),
    # buscas por nome ou palavra exata: onde a busca lexical deveria ajudar
    Query("Keyser Söze", {"Os Suspeitos": 2}, "lexical"),
    Query("Totoro", {"Meu Amigo Totoro": 2}, "lexical"),
    Query("Amélie", {"O Fabuloso Destino de Amélie Poulain": 2}, "lexical"),
    Query("hobbit anel", {"O Senhor dos Anéis: A Sociedade do Anel": 2}, "lexical"),
    Query("BOPE", {"Tropa de Elite": 2}, "lexical"),
    Query("Se7en", {"Se7en: Os Sete Crimes Capitais": 2}, "lexical"),
    Query("Joãozinho Grilo e Chicó", {"O Auto da Compadecida": 2}, "lexical"),
    Query("Truman", {"O Show de Truman": 2}, "lexical"),
]
