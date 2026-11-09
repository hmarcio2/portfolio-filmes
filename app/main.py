"""API do recomendador de filmes: chat com streaming (SSE) e cartões de filmes."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from portfolio_core import (
    DailyBudget,
    LLMError,
    LLMProvider,
    RateLimiter,
    SessionStore,
    Settings,
    Tracer,
    Usage,
    get_provider,
    get_tracer,
    sse_event,
)
from portfolio_core.tracing import log_llm_call

from .agent import run_agent
from .catalog import load_default_catalog
from .embeddings import get_embedder
from .guardrails import evaluate_turn
from .index import MovieIndex

logger = logging.getLogger("portfolio.filmes")

APP_NAME = "01-recomendador-filmes"
BASE_DIR = Path(__file__).resolve().parent.parent


class ChatRequest(BaseModel):
    session_id: str | None = Field(default=None, max_length=64)
    message: str = Field(min_length=1, max_length=4000)


class ShownStore:
    """Filmes já mostrados por sessão (para "me mostre outros" não repetir), com TTL."""

    def __init__(self, ttl: float):
        self._ttl = ttl
        self._data: dict[str, tuple[float, set[int]]] = {}

    def get(self, session_id: str) -> set[int]:
        now = time.monotonic()
        for sid in [s for s, (seen, _) in self._data.items() if now - seen > self._ttl]:
            del self._data[sid]
        _, shown = self._data.get(session_id, (now, set()))
        self._data[session_id] = (now, shown)
        return shown


def _index_path() -> Path | None:
    raw = os.environ.get("PORTFOLIO_FILMES_INDEX_DIR", "")
    if raw.lower() == "memory":
        return None
    return Path(raw) if raw else BASE_DIR / "data" / "index"


def create_app(
    settings: Settings | None = None,
    provider: LLMProvider | None = None,
    index: MovieIndex | None = None,
    tracer: Tracer | None = None,
) -> FastAPI:
    settings = (settings or Settings.from_env()).resolved()
    provider = provider or get_provider(settings)
    tracer = tracer or get_tracer()  # Langfuse se houver LANGFUSE_PUBLIC_KEY/SECRET_KEY; senão, no-op
    limiter = RateLimiter(settings.rate_limit_per_minute)
    budget = DailyBudget(settings.daily_message_cap)
    sessions = SessionStore(ttl=settings.session_ttl_seconds, max_messages=settings.max_history_messages)
    shown_store = ShownStore(settings.session_ttl_seconds)
    state: dict = {"index": index}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if state["index"] is None:  # construir o índice pode chamar a API de embeddings: fora do loop de eventos
            def build() -> MovieIndex:
                return MovieIndex(load_default_catalog(), get_embedder(), _index_path())

            state["index"] = await asyncio.to_thread(build)
            logger.info("Índice pronto: %d filmes, embedder %s", len(state["index"]), state["index"].embedder.id)
        yield
        await asyncio.to_thread(tracer.shutdown)  # envia os traces pendentes antes de encerrar

    app = FastAPI(title="Recomendador de filmes", docs_url=None, redoc_url=None, lifespan=lifespan)

    def client_key(request: Request) -> str:
        if settings.trust_proxy:
            forwarded = request.headers.get("x-forwarded-for", "")
            if forwarded:
                return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    @app.get("/health")
    def health():
        idx = state["index"]
        return {"status": "ok", "movies": len(idx) if idx else 0, "provider": type(provider).__name__}

    @app.get("/info")
    def info():
        idx: MovieIndex = state["index"]
        return {
            "movies": len(idx), "source": idx.catalog.source, "attribution": idx.catalog.attribution,
            "embedder": idx.embedder.id, "genres": idx.catalog.genres,
        }

    @app.post("/chat")
    async def chat(body: ChatRequest, request: Request):
        message = body.message.strip()
        if not message or len(message) > settings.max_message_chars:
            return JSONResponse({"error": f"A mensagem deve ter entre 1 e {settings.max_message_chars} caracteres."},
                                status_code=422)
        allowed, retry_after = limiter.check(client_key(request))
        if not allowed:
            return JSONResponse({"error": "Muitas mensagens em pouco tempo. Aguarde um instante."}, status_code=429,
                                headers={"Retry-After": str(int(retry_after) + 1)})
        if not budget.try_consume():
            return JSONResponse({"error": "O limite diário de buscas desta demo foi atingido. Volte amanhã!"},
                                status_code=429)

        sid = body.session_id or sessions.new_id()
        sessions.append(sid, "user", message)
        history = sessions.history(sid)
        shown = shown_store.get(sid)
        idx: MovieIndex = state["index"]

        # Um trace por pergunta, agrupado por sessão (sem identificar o visitante: nem IP, nem user_id).
        trace = tracer.start(
            "movie-turn", session_id=sid, input=message,
            metadata={"app": APP_NAME, "provider": settings.provider, "model": settings.model,
                      "embedder": idx.embedder.id, "catalog": idx.catalog.source, "turn": sum(m["role"] == "user" for m in history)},
        )

        async def events():
            meta = {"session_id": sid}
            if trace.trace_id:
                meta["trace_id"] = trace.trace_id
            yield sse_event("meta", meta)
            started = time.monotonic()
            usage, stop_reason = None, None
            answer, searches = "", []
            outcome = {"level": None, "status": None}
            try:
                async for ev in run_agent(provider, idx, shown, history, trace=trace, model=settings.model or None,
                                          model_parameters={"temperature": settings.temperature, "max_tokens": settings.max_tokens}):
                    if ev.kind == "done":
                        usage, stop_reason = ev.data["usage"], ev.data["stop_reason"]
                        answer, searches = ev.data["answer"], ev.data["searches"]
                        sessions.append(sid, "assistant", ev.data["answer"])
                        yield sse_event("done", {"searches": len(ev.data["searches"])})
                    elif ev.kind == "error":
                        sessions.drop_last(sid)
                        outcome.update(level="ERROR", status=str(ev.data.get("error"))[:300])
                        yield sse_event("error", ev.data)
                    else:
                        yield sse_event(ev.kind, ev.data)
            except LLMError as exc:
                logger.exception("Falha na chamada ao LLM")
                sessions.drop_last(sid)
                log_llm_call(app=APP_NAME, model=settings.model, latency_s=time.monotonic() - started,
                             usage=usage or Usage(), stop_reason=None, error=type(exc).__name__)
                outcome.update(level="ERROR", status=f"{type(exc).__name__}: {exc}"[:300])
                yield sse_event("error", {"error": "Não consegui buscar agora. Tente novamente em instantes."})
                return
            finally:
                # Também roda se o visitante fechar a página no meio (o gerador é cancelado).
                latency = round(time.monotonic() - started, 3)
                trace.update(output=answer or None, level=outcome["level"], status_message=outcome["status"],
                             metadata={"searches": len(searches), "stop_reason": stop_reason, "latency_s": latency,
                                       "queries": [s.get("query") for s in searches]})
                shown_titles = [m.title for m in idx.catalog.movies if m.id in shown]
                for score in evaluate_turn(question=message, answer=answer, searches=searches,
                                           catalog_titles=[m.title for m in idx.catalog.movies],
                                           shown_titles=shown_titles, failed=outcome["level"] == "ERROR"):
                    trace.score(score.name, score.value, data_type=score.data_type, comment=score.comment)
                trace.end()
            if usage is not None:
                log_llm_call(app=APP_NAME, model=settings.model, latency_s=time.monotonic() - started,
                             usage=usage, stop_reason=stop_reason)

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    app.mount("/", StaticFiles(directory=BASE_DIR / "web", html=True), name="web")
    return app
