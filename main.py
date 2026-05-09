"""Sisyphean — entry point.

Starts llama-server as a managed subprocess (unless mock mode is on),
waits for it to be healthy, then serves the FastAPI engine via uvicorn.

Usage:
    python main.py                         # start engine (uses config.yaml)
    python main.py tray                    # Windows system tray watchdog
    python main.py setup                   # interactive first-time setup wizard
    python main.py config                  # re-run setup wizard to change settings
    python main.py launch birdclaw         # open BirdClaw web UI in browser
    python main.py launch claude           # start Claude Code CLI

Dream (offline memory consolidation):
    python main.py dream                   # memorise + cleanup
    python main.py dream --dry-run         # report only, no writes
    python main.py dream --memorise-only   # skip cleanup
    python main.py dream --cleanup-only    # skip memorise
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import uvicorn

from engine.config import load_config
from engine.api.app import create_app
from engine.llm.client import LlamaClient

_LOG_DIR  = Path.home() / ".sisyphean" / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "engine.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_FILE, encoding="utf-8"),
    ],
)
# Suppress httpx INFO noise (every /tokenize 404, every routine HTTP request)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("sisyphean")

_procs: list[subprocess.Popen] = []


# ── llama-server discovery ────────────────────────────────────────────────────

def _find_llama_server() -> str:
    """Locate the llama-server binary. Checks PATH then common build locations."""
    for name in ("llama-server", "llama-server.exe"):
        found = shutil.which(name)
        if found:
            return found

    candidates = [
        "./llama.cpp/build/bin/llama-server",
        "./llama.cpp/build/bin/Release/llama-server.exe",
        "./llama.cpp/build/Release/llama-server.exe",
        "./llama-server",
        "./llama-server.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return str(Path(c).resolve())

    raise FileNotFoundError(
        "llama-server binary not found.\n"
        "Build llama.cpp (https://github.com/ggerganov/llama.cpp#build) "
        "and ensure the binary is on PATH or in the project root."
    )


# ── Subprocess management ─────────────────────────────────────────────────────

def _start_server(config, *, embedding: bool = False) -> subprocess.Popen:
    binary = _find_llama_server()
    scfg = config.embedding.server if embedding else config.llm.server
    model = config.embedding.model_path if embedding else config.llm.model_path

    cmd: list[str] = [
        binary,
        "--model", model,
        "--host", scfg.host,
        "--port", str(scfg.port),
        "--ctx-size", str(scfg.context_size),
        "--threads", str(scfg.threads),
        "--n-gpu-layers", str(scfg.gpu_layers),
    ]
    if embedding:
        cmd.append("--embedding")
    cmd.extend(scfg.extra_args)

    label = "embedding server" if embedding else "llama-server"
    logger.info("Starting %s: %s", label, " ".join(cmd))

    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _stop_all(_sig=None, _frame=None) -> None:
    for proc in _procs:
        try:
            proc.terminate()
        except Exception:
            pass
    if _sig is not None:
        sys.exit(0)


# ── Main ──────────────────────────────────────────────────────────────────────

def _free_port(port: int) -> None:
    """Kill any process already listening on *port* so we can bind cleanly.

    Prevents 'Only one usage of each socket address' errors when restarting.
    Works on Windows (netstat + taskkill) and Linux/macOS (lsof/fuser).
    """
    import socket
    # Quick check — if nothing is listening, skip the OS-specific logic
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        if s.connect_ex(("127.0.0.1", port)) != 0:
            return  # port is free

    try:
        if sys.platform == "win32":
            import subprocess as _sp
            out = _sp.check_output(
                ["netstat", "-ano"], text=True, stderr=_sp.DEVNULL
            )
            for line in out.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    pid = int(line.split()[-1])
                    if pid and pid != os.getpid():
                        logger.info("Freeing port %d — killing PID %d", port, pid)
                        _sp.call(["taskkill", "/F", "/PID", str(pid)],
                                 stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        else:
            import subprocess as _sp
            out = _sp.check_output(
                ["lsof", "-ti", f"tcp:{port}"], text=True, stderr=_sp.DEVNULL
            )
            for pid_str in out.split():
                pid = int(pid_str)
                if pid != os.getpid():
                    logger.info("Freeing port %d — killing PID %d", port, pid)
                    os.kill(pid, signal.SIGKILL)
    except Exception as exc:
        logger.debug("_free_port: %s", exc)


async def main() -> None:
    config_path = os.environ.get("SISYPHEAN_CONFIG", "config.yaml")
    config = load_config(config_path)

    Path(config.workspace).mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGINT, _stop_all)
    signal.signal(signal.SIGTERM, _stop_all)

    ext = config.llm.external_api
    using_ollama = bool(config.llm.local_model) and not config.mock
    using_external = ext.enabled and ext.base_url and ext.api_key

    if not config.mock and not using_ollama and not using_external:
        # llama-server mode: validate model file exists before launching
        model = Path(config.llm.model_path)
        if not model.exists():
            logger.error(
                "Model file not found: %s\n"
                "  * Download a 4B GGUF model and set llm.model_path in config.yaml\n"
                "  * Set llm.local_model: <name> to use a running Ollama instance\n"
                "  * Or set mock: true to run without a model",
                config.llm.model_path,
            )
            sys.exit(1)

        _procs.append(_start_server(config))

        if config.embedding.enabled:
            embed_model = Path(config.embedding.model_path)
            if not embed_model.exists():
                logger.warning(
                    "Embedding model not found (%s) -- embedding server will not start",
                    config.embedding.model_path,
                )
            else:
                _procs.append(_start_server(config, embedding=True))

    # ── Start the API server immediately ─────────────────────────────────────
    # Dashboard and health endpoint are accessible right away.
    # Ollama/llama-server health is checked in the background — inference
    # requests made before the model is ready will fail gracefully.

    app = create_app(config)

    logger.info("━" * 52)
    logger.info("  Sisyphean engine  →  http://%s:%d", config.api.host, config.api.port)
    logger.info("  Anthropic compat  →  POST /v1/messages")
    logger.info("  OpenAI compat     →  POST /v1/chat/completions")
    logger.info("  Dashboard         →  http://%s:%d/dashboard", config.api.host, config.api.port)
    if config.mock:
        logger.info("  ⚠  MOCK MODE — no real model loaded")
    logger.info("━" * 52)

    _free_port(config.api.port)

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=config.api.host,
            port=config.api.port,
            log_level="warning",
        )
    )

    async def _wait_for_model():
        """Background task: poll until the LLM backend is ready, then log status."""
        if config.mock or using_external:
            return
        llm_url = f"http://{config.llm.server.host}:{config.llm.server.port}"
        probe = LlamaClient(llm_url, model=config.llm.local_model)
        if using_ollama:
            logger.info(
                "Waiting for Ollama at %s (model: %s) — start it with: ollama serve",
                llm_url, config.llm.local_model,
            )
        # Retry indefinitely in short bursts so the loop stays alive
        attempt = 0
        while True:
            ready = await probe.health_check(retries=15, interval=2.0, ollama=using_ollama)
            if ready:
                logger.info("✓ Model backend ready at %s", llm_url)
                await probe.close()
                return
            attempt += 1
            if using_ollama:
                logger.warning(
                    "Ollama not yet reachable (attempt %d) — "
                    "start it with: ollama serve  |  ollama pull %s",
                    attempt, config.llm.local_model,
                )
            else:
                logger.warning("llama-server not yet ready (attempt %d)", attempt)
            await asyncio.sleep(5)

    try:
        # Run server and model-wait concurrently — server starts immediately,
        # model poller runs in background until Ollama/llama-server comes up.
        await asyncio.gather(server.serve(), _wait_for_model())
    finally:
        _stop_all()


# ── Dream subcommand ─────────────────────────────────────────────────────────

async def dream_main(args: list[str]) -> None:
    from engine.memory.dream import dream_cli

    config_path = os.environ.get("SISYPHEAN_CONFIG", "config.yaml")
    dry_run = "--dry-run" in args
    memorise = "--cleanup-only" not in args
    cleanup = "--memorise-only" not in args

    code = await dream_cli(
        config_path=config_path,
        memorise=memorise,
        cleanup=cleanup,
        dry_run=dry_run,
    )
    sys.exit(code)


# ── Setup wizard ─────────────────────────────────────────────────────────────

def _setup_wizard(config_path: str = "config.yaml") -> None:
    """Interactive first-time setup / config editor.

    Walks the user through the essential settings and writes config.yaml.
    Can be re-run at any time with:  python main.py config
    """
    import yaml  # type: ignore[import]

    print("\n━━━  Sisyphean Setup  ━━━")

    # Load existing config so we can show current values as defaults
    cfg_file = Path(config_path)
    existing: dict = {}
    if cfg_file.exists():
        try:
            import yaml as _y
            with cfg_file.open(encoding="utf-8") as f:
                existing = _y.safe_load(f) or {}
        except Exception:
            pass

    def _ask(prompt: str, default: str) -> str:
        shown = f" [{default}]" if default else ""
        val = input(f"  {prompt}{shown}: ").strip()
        return val if val else default

    # ── LLM backend ───────────────────────────────────────────────────────────
    print("\n[1/4]  LLM backend")
    print("  Options:  a) Local Ollama   b) External API (llama.cpp, LM Studio, OpenRouter…)")
    mode = input("  Choose [b]: ").strip().lower() or "b"

    if mode == "a":
        ollama_model = _ask("Ollama model name", existing.get("llm", {}).get("local_model", "qwen3:0.6b"))
        ollama_port  = _ask("Ollama port", str(existing.get("llm", {}).get("server", {}).get("port", 11434)))
        api_url  = ""
        api_key  = ""
        api_model = ""
    else:
        current_url = (existing.get("llm", {}).get("external_api", {}).get("base_url", "") or
                       "http://192.168.29.37:8081/v1")
        raw_url = _ask("API base URL (host:port or full URL)", current_url)
        # Normalise: ensure it ends with /v1
        if raw_url and not raw_url.startswith("http"):
            raw_url = "http://" + raw_url
        if raw_url and not raw_url.rstrip("/").endswith("/v1"):
            raw_url = raw_url.rstrip("/") + "/v1"
        api_url   = raw_url
        api_key   = _ask("API key (leave blank for local servers)", existing.get("llm", {}).get("external_api", {}).get("api_key", "local"))
        api_model = _ask("Model name", existing.get("llm", {}).get("external_api", {}).get("model", "gemma-4-E4B-it-Q8_0.gguf"))
        ollama_model = ""
        ollama_port  = "11434"

    # ── Engine port ───────────────────────────────────────────────────────────
    print("\n[2/4]  Engine API port")
    engine_port = _ask("Sisyphean engine port", str(existing.get("api", {}).get("port", 47291)))

    # ── Workspace ─────────────────────────────────────────────────────────────
    print("\n[3/4]  Workspace directory")
    workspace = _ask("Workspace path", existing.get("workspace", "./workspace"))

    # ── BirdClaw engine URL ───────────────────────────────────────────────────
    print("\n[4/4]  BirdClaw integration (optional)")
    bc_default = existing.get("birdclaw", {}).get("engine_url", f"http://127.0.0.1:{engine_port}")
    bc_url = _ask("BirdClaw will call Sisyphean at", bc_default)

    # ── Write config ──────────────────────────────────────────────────────────
    new_cfg: dict = {
        "llm": {
            "model_path": existing.get("llm", {}).get("model_path", ""),
            "model_name": existing.get("llm", {}).get("model_name", "sisyphean-gemma4"),
            "local_model": ollama_model,
            "server": {
                "host": "127.0.0.1",
                "port": int(ollama_port),
                "context_size": existing.get("llm", {}).get("server", {}).get("context_size", 8192),
                "threads": existing.get("llm", {}).get("server", {}).get("threads", 8),
                "gpu_layers": existing.get("llm", {}).get("server", {}).get("gpu_layers", 0),
                "extra_args": [],
            },
            "external_api": {
                "enabled": bool(api_url),
                "base_url": api_url,
                "api_key": api_key,
                "model": api_model,
            },
        },
        "embedding": existing.get("embedding", {"enabled": False}),
        "api": {
            "host": "127.0.0.1",
            "port": int(engine_port),
            "cors_origins": [
                "http://localhost:5173",
                "http://localhost:3000",
                "http://localhost:47293",
            ],
        },
        "memory": existing.get("memory", {
            "path": "./memory",
            "engine_policy_file": "./engine_policy.md",
            "injection_budget": 1500,
            "top_n_nodes": 5,
            "top_n_artifacts": 3,
            "embedding_model": None,
        }),
        "mock": existing.get("mock", False),
        "workspace": workspace,
        "birdclaw": {"engine_url": bc_url},
    }

    import yaml as _y
    with cfg_file.open("w", encoding="utf-8") as f:
        _y.dump(new_cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\n✓ Config saved to {cfg_file.resolve()}")
    print(f"  Engine will be at  http://127.0.0.1:{engine_port}")
    print(f"  Start with:  python main.py")
    print(f"  Or tray:     python main.py tray\n")


# ── Launch subcommand ─────────────────────────────────────────────────────────

def _launch(target: str) -> None:
    """Open another tool from Sisyphean's launcher."""
    import webbrowser

    if target == "birdclaw":
        url = "http://127.0.0.1:47293"
        print(f"Opening BirdClaw at {url}")
        webbrowser.open(url)

    elif target == "claude":
        import shutil
        cli = shutil.which("claude") or shutil.which("claude.cmd")
        if not cli:
            print("Claude Code not found on PATH.")
            print("Install it: npm install -g @anthropic-ai/claude-code")
            sys.exit(1)
        print("Launching Claude Code CLI…")
        os.execv(cli, [cli])   # replace process — no subprocess overhead

    else:
        print(f"Unknown launch target: {target!r}")
        print("Usage:  python main.py launch birdclaw | claude")
        sys.exit(1)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _args = sys.argv[1:]
    if not _args:
        asyncio.run(main())
    elif _args[0] == "dream":
        asyncio.run(dream_main(_args[1:]))
    elif _args[0] == "tray":
        import tray as _tray
        _tray.main()
    elif _args[0] in ("setup", "config"):
        cfg = os.environ.get("SISYPHEAN_CONFIG", "config.yaml")
        _setup_wizard(cfg)
    elif _args[0] == "launch":
        target = _args[1] if len(_args) > 1 else ""
        if not target:
            print("Usage:  python main.py launch birdclaw | claude")
            sys.exit(1)
        _launch(target)
    elif _args[0] == "serve":
        asyncio.run(main())
    else:
        asyncio.run(main())
