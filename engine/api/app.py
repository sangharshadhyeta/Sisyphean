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
from engine.llm.embeddings import EmbeddingClient, EmbeddingCache
from engine.memory.graph import knowledge_graph, seed_knowledge_graph, sync_personality_to_graph, seed_skill_graph
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
        if config.llm.local_model:
            ollama_port = getattr(config.llm.server, "ollama_port", 11434)
            llm_url = f"http://{config.llm.server.host}:{ollama_port}"
            logger.info("Using Ollama at %s  model=%s", llm_url, config.llm.local_model)
        else:
            llm_url = f"http://{config.llm.server.host}:{config.llm.server.port}"
        client = LlamaClient(llm_url, mock=config.mock, model=config.llm.local_model)
    ctx = ContextManager(client, config.llm.server.context_size)

    # ── Memory system ────────────────────────────────────────────────────────
    mem_path = Path(config.memory.path)
    mem_path.mkdir(parents=True, exist_ok=True)

    # ── Embedding client (Ollama) ─────────────────────────────────────────────
    # Shared cache so extractor and graph search reuse the same vectors.
    # Falls back to Jaccard transparently when Ollama embedding is unavailable.
    _embed_cache  = EmbeddingCache()
    _ollama_port  = getattr(config.llm.server, "ollama_port", 11434)
    _embed_model  = getattr(config.embedding, "ollama_model", "nomic-embed-text")
    embed_client: EmbeddingClient | None = None
    if config.embedding.enabled:
        embed_client = EmbeddingClient(
            ollama_url=f"http://127.0.0.1:{_ollama_port}",
            model=_embed_model,
            cache=_embed_cache,
        )
        logger.info("EmbeddingClient configured: model=%s", _embed_model)

    # Use the module-level GraphStore singleton — single graph shared by
    # injector (reads) and extractor (writes) so extracted facts are immediately
    # visible in the next request's injected context.
    graph = knowledge_graph
    graph._embed_cache = _embed_cache   # invalidated by upsert_node on summary change
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
    extractor = MemoryExtractor(graph, store, client, embed_client=embed_client)

    budget_tracker = BudgetTracker(mem_path)
    permission_guard = PermissionGuard.from_config(config.permissions)
    translation_loop = TranslationLoop(
        client=client,
        ctx_manager=ctx,
        budget_tracker=budget_tracker,
        workspace=config.workspace,
        permission_guard=permission_guard,
        injector=injector,
        knowledge_graph=graph,
    )

    # Sync personality (engine_policy.md) and user prefs (user_prefs.md) into
    # the graph on every startup — unconditionally, so edits to either file are
    # picked up after a restart without needing to wipe the graph.
    policy_path = Path(config.memory.engine_policy_file)
    policy_text = policy_path.read_text(encoding="utf-8") if policy_path.exists() else ""
    prefs_path  = mem_path / "user_prefs.md"
    sync_personality_to_graph(graph, policy_path, prefs_path)

    # Seed skill nodes from the skills/ directory into the knowledge graph.
    skills_path = Path(config.skills_path) if hasattr(config, "skills_path") else Path("skills")
    seed_skill_graph(graph, skills_path)

    # Mtime state for live pickup — middleware checks these on every request
    # and re-syncs if either file has changed since last sync.
    def _get_mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    _file_mtimes: dict = {
        "soul":  _get_mtime(policy_path),
        "prefs": _get_mtime(prefs_path),
    }

    # ── OS detection — runs once per process start ────────────────────────────
    # Saves OS / Python info to the knowledge graph (injected into every request
    # as a "### System" memory section) AND to memory/system_info.md for human
    # inspection and BirdClaw to read.
    import platform as _platform, sys as _sys
    _os_name    = _platform.system()          # "Windows", "Linux", "Darwin"
    _os_ver     = _platform.release()         # "11", "22.04", …
    _os_machine = _platform.machine()         # "AMD64", "x86_64", …
    _py_ver     = (f"{_sys.version_info.major}.{_sys.version_info.minor}"
                   f".{_sys.version_info.micro}")
    if _os_name == "Windows":
        _shell_hint = (
            "Shell: PowerShell/cmd. "
            "Use systeminfo, Get-Process, Get-PSDrive, tasklist for system state. "
            "Do NOT use Linux/bash commands: printf, cat, grep, find, chmod, touch, "
            "uptime, sed, awk, ls (→ Get-ChildItem or dir). "
            "To write a file: use Python (python -c \"open('f','w').write('text')\") "
            "or Write-Output / Out-File in PowerShell. "
            "Never use 'printf text > file' or 'cat > file' — they do not exist on Windows."
        )
    elif _os_name == "Darwin":
        _shell_hint = (
            "Shell: bash/zsh (macOS). "
            "Use top -l1, vm_stat, df -h, uptime for system state."
        )
    else:
        _shell_hint = (
            "Shell: bash (Linux). "
            "Use top -bn1, free -h, df -h, uptime, systemctl for system state."
        )
    _os_summary = (
        f"{_os_name} {_os_ver} ({_os_machine}), Python {_py_ver}. {_shell_hint}"
    )
    graph.upsert_node("System", "system", summary=_os_summary)
    _sysinfo_path = mem_path / "system_info.md"
    _sysinfo_path.write_text(
        f"# System Info\n\n"
        f"OS: {_os_name} {_os_ver} ({_os_machine})\n"
        f"Python: {_py_ver}\n"
        f"Shell hint: {_shell_hint}\n",
        encoding="utf-8",
    )
    logger.info(
        "OS detected: %s %s (%s), Python %s",
        _os_name, _os_ver, _os_machine, _py_ver,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        _app.state.client = client
        _app.state.ctx = ctx
        _app.state.injector = injector
        _app.state.extractor = extractor
        _app.state.translation_loop = translation_loop
        _app.state.soul_text = policy_text
        _app.state.graph = graph          # exposed for soul remember action
        logger.info("Sisyphean engine started")
        yield
        await client.close()
        if embed_client is not None:
            await embed_client.close()
        logger.info("Sisyphean engine stopped")

    _model_display = (
        ext.model if (ext.enabled and ext.model)
        else config.llm.local_model or config.llm.model_name or "unknown"
    )
    app = FastAPI(
        title="Sisyphean",
        version="0.1.0",
        description=f"Local AI agent engine — {_model_display} with persistent memory",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _sync_personality_on_change(request: Request, call_next):
        """Re-sync personality files into the graph whenever they change on disk.

        Fires on every /v1/messages and /v1/chat/completions request — just two
        stat() calls, so the overhead is negligible.  When either file's mtime
        has changed since the last sync, sync_personality_to_graph() is called
        before the request is processed, giving true live pickup without restart.
        """
        if request.url.path in ("/v1/messages", "/v1/chat/completions"):
            soul_mtime  = _get_mtime(policy_path)
            prefs_mtime = _get_mtime(prefs_path)
            if soul_mtime != _file_mtimes["soul"] or prefs_mtime != _file_mtimes["prefs"]:
                logger.info(
                    "Personality files changed on disk — resyncing graph "
                    "(soul_changed=%s prefs_changed=%s)",
                    soul_mtime  != _file_mtimes["soul"],
                    prefs_mtime != _file_mtimes["prefs"],
                )
                sync_personality_to_graph(graph, policy_path, prefs_path)
                _file_mtimes["soul"]  = soul_mtime
                _file_mtimes["prefs"] = prefs_mtime
        return await call_next(request)

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
        elif ext_cfg.enabled and ext_cfg.base_url:
            # Actually probe the external API — don't assume it's reachable
            probe_url = ext_cfg.base_url.rstrip("/") + "/models"
            try:
                headers = {}
                if ext_cfg.api_key:
                    headers["Authorization"] = f"Bearer {ext_cfg.api_key}"
                async with httpx.AsyncClient(timeout=3.0) as c:
                    r = await c.get(probe_url, headers=headers)
                    result = r.status_code in (200, 401)  # 401 = server up, wrong key
            except Exception:
                result = False
        else:
            if config.llm.local_model:
                # Ollama: use ollama_port, not llama-server port
                port = config.llm.server.ollama_port
                health_path = "/api/tags"
            else:
                port = config.llm.server.port
                health_path = "/health"
            llm_url = f"http://{config.llm.server.host}:{port}"
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

    # ── Memory graph API ─────────────────────────────────────────────────────

    @app.get("/api/graph", tags=["info"])
    async def api_graph():
        """Return a lightweight snapshot of the knowledge graph for visualisation.

        Returns up to 120 most-recently-updated nodes and their edges.
        Each node: {id, label, type, summary}
        Each edge: {source, target}
        """
        all_n = graph.all_nodes()
        # Sort by recency (last_seen is a float unix timestamp from upsert_node),
        # fall back to created_at string, cap at 120
        def _sort_ts(n: dict) -> float:
            v = n.get("last_seen", 0)
            if isinstance(v, str):
                try:
                    from datetime import datetime, timezone as _tz
                    return datetime.fromisoformat(v).timestamp()
                except Exception:
                    return 0.0
            return float(v) if v else 0.0

        sorted_n = sorted(all_n, key=_sort_ts, reverse=True)[:120]
        # "key" field is the graph node key (= name.lower().strip())
        key_set = {n.get("key", "") for n in sorted_n}

        nodes = [
            {
                "id":         n.get("key", n.get("name", "?")),
                "label":      (n.get("name") or n.get("label", n.get("key", "?")))[:32],
                "type":       n.get("type", "fact"),
                "summary":    (n.get("summary") or n.get("content", ""))[:120],
                "confidence": round(float(n.get("confidence", 0.5)), 2),
                "observations": int(n.get("observations", 1)),
                "ts":         _sort_ts(n),
            }
            for n in sorted_n
        ]

        # Edges from the graph's adjacency (GraphStore uses _graph, not _g).
        # Only include edges where both endpoints are in our capped node set.
        edges = []
        try:
            for src, dst in graph._graph.edges():
                if src in key_set and dst in key_set:
                    edges.append({"source": src, "target": dst})
        except Exception:
            pass

        return {"nodes": nodes, "edges": edges[:300]}

    @app.post("/api/graph/purge", tags=["admin"])
    async def api_graph_purge():
        """Purge every node and edge from the knowledge graph then re-seed.

        Re-seeds:
          1. Identity anchor nodes  (seed_knowledge_graph)
          2. Skill nodes from skills/  (seed_skill_graph)

        Returns { nodes_before, nodes_after }.
        """
        nodes_before = graph.node_count()

        # Clear all nodes and edges atomically
        with graph._lock:
            graph._graph.clear()

        # Re-seed identity anchors (guard removed because graph is empty)
        seed_knowledge_graph(graph, "")

        # Re-seed skill nodes
        _sp = Path(config.skills_path) if hasattr(config, "skills_path") else Path("skills")
        seed_skill_graph(graph, _sp)

        # Persist to disk
        try:
            graph.save()
        except Exception as exc:
            logger.warning("graph purge: save failed: %s", exc)

        nodes_after = graph.node_count()
        logger.info("api: graph purged %d -> %d nodes", nodes_before, nodes_after)
        return {"status": "ok", "nodes_before": nodes_before, "nodes_after": nodes_after}

    # ── Task flowchart API ───────────────────────────────────────────────────

    @app.get("/api/tasks", tags=["info"])
    async def api_tasks():
        """Live task state for the dashboard flowchart panel.

        ephemeral=true — state resets on server restart; callers must not
        persist or cache this data across sessions.
        """
        return {"tasks": active_tasks(10), "ephemeral": True}

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
        if not await _check_llm_ready():
            from fastapi import HTTPException
            raise HTTPException(
                status_code=503,
                detail="AI model unavailable — LLM backend is not responding. "
                       "Start the model server and retry.",
            )
        _REQUEST_COUNT += 1
        return await handle_messages(
            req,
            request.app.state.client,
            request.app.state.ctx,
            request.app.state.injector,
            request.app.state.extractor,
            request.app.state.translation_loop,
            http_request=request,
            dump_requests=config.debug.dump_requests,
        )

    # ── OpenAI Chat Completions API ──────────────────────────────────────────

    @app.post("/v1/chat/completions", tags=["openai"])
    async def oai_chat(req: OAIRequest, request: Request):
        global _REQUEST_COUNT
        if not await _check_llm_ready():
            from fastapi import HTTPException
            raise HTTPException(
                status_code=503,
                detail="AI model unavailable — LLM backend is not responding. "
                       "Start the model server and retry.",
            )
        _REQUEST_COUNT += 1
        return await handle_chat_completions(
            req,
            request.app.state.client,
            request.app.state.ctx,
        )

    return app
