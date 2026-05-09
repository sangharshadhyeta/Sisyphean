from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from engine.activity import recent_events
from engine.task_tracker import active_tasks

from engine.config import Config
from engine.llm.client import LlamaClient
from engine.llm.context import ContextManager
from engine.memory.graph import KnowledgeGraph, seed_graph
from engine.memory.store import ArtifactStore
from engine.memory.injector import MemoryInjector
from engine.memory.extractor import MemoryExtractor
from engine.translation.loop import TranslationLoop
from engine.translation.budget import BudgetTracker
from engine.permissions import PermissionGuard
from engine.compat.anthropic import AnthropicRequest, handle_messages
from engine.compat.openai_compat import OAIRequest, handle_chat_completions

logger = logging.getLogger(__name__)
_START = time.time()
_REQUEST_COUNT = 0


def _to_display_path(path: str) -> str:
    """Convert Git Bash path /c/Users/... back to Windows C:/Users/... for display."""
    import re as _re
    return _re.sub(r"^/([a-zA-Z])/", lambda m: f"{m.group(1).upper()}:/", path)


def create_app(config: Config) -> FastAPI:
    ext = config.llm.external_api
    if ext.enabled and ext.base_url and ext.api_key:
        # Use an external OpenAI-compatible provider instead of llama-server
        logger.info(
            "Using external LLM API: %s  model=%s",
            ext.base_url, ext.model,
        )
        client = LlamaClient(
            base_url=ext.base_url,
            api_key=ext.api_key,
            model=ext.model,
            mock=False,
        )
    else:
        llm_url = f"http://{config.llm.server.host}:{config.llm.server.port}"
        if config.llm.local_model:
            logger.info("Using Ollama at %s  model=%s", llm_url, config.llm.local_model)
        client = LlamaClient(llm_url, mock=config.mock, model=config.llm.local_model)
    ctx = ContextManager(client, config.llm.server.context_size)

    # ── Memory system ────────────────────────────────────────────────────────
    mem_path = Path(config.memory.path)
    mem_path.mkdir(parents=True, exist_ok=True)

    graph = KnowledgeGraph(mem_path / "graph.json")
    store = ArtifactStore(
        mem_path / "artifacts.jsonl",
        embedding_model=config.memory.embedding_model,
    )
    injector = MemoryInjector(
        graph, store,
        token_budget=config.memory.injection_budget,
        top_n_nodes=config.memory.top_n_nodes,
        top_n_artifacts=config.memory.top_n_artifacts,
    )
    extractor = MemoryExtractor(graph, store, client)

    budget_tracker = BudgetTracker(mem_path)
    permission_guard = PermissionGuard.from_config(config.permissions)
    translation_loop = TranslationLoop(
        client=client,
        ctx_manager=ctx,
        budget_tracker=budget_tracker,
        workspace=config.workspace,
        permission_guard=permission_guard,
        injector=injector,   # enables per-step memory refresh
    )

    # Seed graph with engine policy and empty stubs on first run
    policy_path = Path(config.memory.engine_policy_file)
    policy_text = policy_path.read_text(encoding="utf-8") if policy_path.exists() else ""
    if policy_text:
        seed_graph(graph, policy_text)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        _app.state.client = client
        _app.state.ctx = ctx
        _app.state.injector = injector
        _app.state.extractor = extractor
        _app.state.translation_loop = translation_loop
        _app.state.soul_text = soul_text
        _app.state.graph = graph          # exposed for soul remember action
        logger.info("Sisyphean engine started")
        yield
        await client.close()
        logger.info("Sisyphean engine stopped")

    app = FastAPI(
        title="Sisyphean",
        version="0.1.0",
        description="Local AI agent engine — Gemma 4B with persistent memory",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve assets folder (logo, favicon) as static files
    _assets_dir = Path(__file__).parent.parent.parent / "assets"
    if _assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")

    # ── Global error handler ─────────────────────────────────────────────────

    @app.exception_handler(Exception)
    async def _global_error(_req: Request, exc: Exception):
        logger.error("Unhandled error: %s", exc, exc_info=True)
        return JSONResponse(
            {"error": {"type": "api_error", "message": str(exc)}},
            status_code=500,
        )

    # ── Info + health ────────────────────────────────────────────────────────

    @app.get("/", tags=["info"])
    async def root():
        return {
            "name": "Sisyphean",
            "version": "0.1.0",
            "uptime_seconds": round(time.time() - _START),
            "mock": config.mock and not (ext.enabled and ext.api_key),
            "memory": {
                "graph_nodes": len(graph.all_nodes()),
                "artifacts": len(store._entries),
            },
        }

    @app.get("/health", tags=["info"])
    async def health():
        return {"status": "ok"}

    # ── Models ───────────────────────────────────────────────────────────────

    @app.get("/v1/models", tags=["models"])
    async def list_models():
        models = [{"id": config.llm.model_name, "object": "model", "created": 0, "owned_by": "sisyphean"}]
        if config.embedding.enabled:
            models.append({"id": config.embedding.model_name, "object": "model", "created": 0, "owned_by": "sisyphean"})
        return {"object": "list", "data": models}

    # ── Status dashboard API ─────────────────────────────────────────────────

    _llm_ready_cache: dict = {"ts": 0.0, "ready": False}

    async def _check_llm_ready() -> bool:
        now = time.time()
        if now - _llm_ready_cache["ts"] < 8:
            return _llm_ready_cache["ready"]
        ext_cfg = config.llm.external_api
        if config.mock:
            result = True
        elif ext_cfg.enabled and ext_cfg.base_url and ext_cfg.api_key:
            result = True
        else:
            llm_url = f"http://{config.llm.server.host}:{config.llm.server.port}"
            health_path = "/api/tags" if config.llm.local_model else "/health"
            try:
                async with httpx.AsyncClient(timeout=1.5) as c:
                    r = await c.get(llm_url + health_path)
                    result = r.status_code == 200
            except Exception:
                result = False
        _llm_ready_cache["ts"] = now
        _llm_ready_cache["ready"] = result
        return result

    @app.get("/api/status", tags=["info"])
    async def api_status():
        global _REQUEST_COUNT
        ext_cfg = config.llm.external_api
        if ext_cfg.enabled and ext_cfg.model:
            model_name = ext_cfg.model
        elif config.llm.local_model:
            model_name = config.llm.local_model
        else:
            model_name = config.llm.model_name or "unknown"

        llm_ready = await _check_llm_ready()

        return {
            "status": "running",
            "uptime_seconds": round(time.time() - _START),
            "model": model_name,
            "mock": config.mock,
            "llm_ready": llm_ready,
            "requests_total": _REQUEST_COUNT,
            "graph_nodes": len(graph.all_nodes()),
            "artifacts": len(store._entries),
            "workspace": _to_display_path(translation_loop.workspace or str(config.workspace)),
            "recent": recent_events(30),
        }

    # ── Task flowchart API ───────────────────────────────────────────────────

    @app.get("/api/tasks", tags=["info"])
    async def api_tasks():
        """Live task state for the dashboard flowchart panel."""
        return {"tasks": active_tasks(10)}

    # ── Dashboard HTML ───────────────────────────────────────────────────────

    _DASHBOARD_HTML = (Path(__file__).parent.parent.parent / "dashboard.html")

    @app.get("/dashboard", tags=["info"], response_class=HTMLResponse)
    async def dashboard():
        if _DASHBOARD_HTML.exists():
            return HTMLResponse(_DASHBOARD_HTML.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>Dashboard not found. Place dashboard.html in project root.</h1>")

    # Redirect / to dashboard
    @app.get("/", tags=["info"])
    async def root_redirect():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/dashboard")

    # ── Anthropic Messages API ───────────────────────────────────────────────

    @app.post("/v1/messages", tags=["anthropic"])
    async def anthropic_messages(req: AnthropicRequest, request: Request):
        global _REQUEST_COUNT
        _REQUEST_COUNT += 1
        return await handle_messages(
            req,
            request.app.state.client,
            request.app.state.ctx,
            request.app.state.injector,
            request.app.state.extractor,
            request.app.state.translation_loop,
            http_request=request,
        )

    # ── OpenAI Chat Completions API ──────────────────────────────────────────

    @app.post("/v1/chat/completions", tags=["openai"])
    async def oai_chat(req: OAIRequest, request: Request):
        global _REQUEST_COUNT
        _REQUEST_COUNT += 1
        return await handle_chat_completions(
            req,
            request.app.state.client,
            request.app.state.ctx,
        )

    return app
