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

Note: Dream cycle (memory consolidation, inner_self merging, session log processing)
is handled by BirdClaw — run it from the BirdClaw directory.
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
        "C:/llama.cpp/llama-server.exe",
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

def _check_port(port: int) -> None:
    """Exit with an error if another process is already listening on *port*.

    Prevents silently killing a running Sisyphean instance when a second one
    is accidentally started. The caller (tray watchdog, installer) should
    check the return code and surface the error to the user.
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        if s.connect_ex(("127.0.0.1", port)) == 0:
            logger.error(
                "Port %d is already in use — Sisyphean may already be running.\n"
                "  Stop the existing instance before starting a new one,\n"
                "  or use  python main.py tray  which manages restarts automatically.",
                port,
            )
            sys.exit(1)


_HERE = Path(__file__).parent
_DEFAULT_CONFIG = str(_HERE / "config.yaml")
_BC_WEB_PORT = 47293  # BirdClaw web UI port


async def main() -> None:
    config_path = os.environ.get("SISYPHEAN_CONFIG", _DEFAULT_CONFIG)
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

    _check_port(config.api.port)

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


# Dream cycle is BirdClaw's responsibility — it orchestrates memory consolidation,
# inner_self merging, session log processing, and graph enrichment.


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
                       "http://localhost:8080/v1")
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
            "model_name": existing.get("llm", {}).get("model_name", "sisyphean-4b"),
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
                f"http://localhost:{_BC_WEB_PORT}",
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

def _ensure_sisyphean_running(config) -> None:
    """Start Sisyphean in the background if it is not already listening."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        already_up = s.connect_ex(("127.0.0.1", config.api.port)) == 0
    if already_up:
        print(f"  Sisyphean already running on port {config.api.port}.")
        return
    print(f"  Sisyphean not detected on port {config.api.port} — starting it…")
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_CONSOLE
    subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "serve"],
        cwd=str(Path(__file__).parent),
        creationflags=flags,
    )
    # Brief wait so it can bind before callers try to connect
    import time as _t
    _t.sleep(2)


def _find_birdclaw_dir() -> Path | None:
    """Auto-detect the BirdClaw installation directory.

    Search order:
      1. Sibling of Sisyphean: ../BirdClaw  (most common layout)
      2. Lowercase variant:    ../birdclaw
      3. Home directory:       ~/BirdClaw
    """
    candidates = [
        _HERE.parent / "BirdClaw",
        _HERE.parent / "birdclaw",
        Path.home() / "BirdClaw",
        Path.home() / "birdclaw",
    ]
    for d in candidates:
        if (d / "main.py").exists():
            return d
    return None


def _is_port_open(port: int, host: str = "127.0.0.1") -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _start_birdclaw_web(bc_dir: Path) -> bool:
    """Launch BirdClaw web server in a new visible console window (Windows)
    or background process (Linux/Mac). Returns True when the port is open."""
    import time as _t

    python = sys.executable

    if sys.platform == "win32":
        # Start in a new console so BirdClaw output is visible and the process
        # stays alive after this script exits.
        subprocess.Popen(
            ["cmd", "/c", "start", "BirdClaw", "cmd", "/k",
             python, str(bc_dir / "main.py"), "web"],
            cwd=str(bc_dir),
            shell=False,
        )
    else:
        subprocess.Popen(
            [python, str(bc_dir / "main.py"), "web"],
            cwd=str(bc_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # Poll up to 15 s for BirdClaw to bind its port
    for _ in range(30):
        _t.sleep(0.5)
        if _is_port_open(_BC_WEB_PORT):
            return True
    return False


def _open_in_new_terminal(cmd: list[str], cwd: str) -> None:
    """Open a command in a new terminal window (Windows) or foreground (other)."""
    if sys.platform == "win32":
        subprocess.Popen(
            ["cmd", "/c", "start", "cmd", "/k"] + cmd,
            cwd=cwd,
            shell=False,
        )
    else:
        subprocess.Popen(cmd, cwd=cwd)


def _launch(target: str) -> None:
    """Ensure Sisyphean is running, then open the requested tool."""
    config_path = os.environ.get("SISYPHEAN_CONFIG", _DEFAULT_CONFIG)
    config = load_config(config_path)
    _ensure_sisyphean_running(config)

    here = _HERE

    if target == "birdclaw":
        import webbrowser, urllib.parse

        _BC_PORT = _BC_WEB_PORT

        if _is_port_open(_BC_PORT):
            print(f"  BirdClaw already running on port {_BC_PORT}.")
        else:
            bc_dir = _find_birdclaw_dir()
            if bc_dir is None:
                print("  BirdClaw not found. Install it alongside Sisyphean:")
                print("    git clone https://github.com/sangharshadhyeta/BirdClaw ../BirdClaw")
                print("    cd ../BirdClaw && pip install -e .")
                print("  Then retry:  python main.py launch birdclaw")
                sys.exit(1)
            print(f"  Starting BirdClaw from {bc_dir} …")
            ready = _start_birdclaw_web(bc_dir)
            if not ready:
                print(f"  Warning: BirdClaw did not start within 15 s. Opening browser anyway.")
            else:
                print(f"  BirdClaw web UI ready.")

        caller_cwd = os.getcwd()
        cwd_param  = urllib.parse.quote(caller_cwd, safe="")
        url = f"http://127.0.0.1:{_BC_PORT}/?cwd={cwd_param}"
        print(f"  Opening  http://127.0.0.1:{_BC_PORT}/")
        webbrowser.open(url)

    elif target == "claude":
        import shutil
        cli = shutil.which("claude") or shutil.which("claude.cmd")
        if not cli:
            print("  Claude Code not found on PATH.")
            print("  Install it: npm install -g @anthropic-ai/claude-code")
            sys.exit(1)
        # Run Claude Code in the caller's working directory (not Sisyphean's dir)
        caller_cwd = os.getcwd()
        print(f"  Launching Claude Code in {caller_cwd}")
        env = os.environ.copy()
        env.setdefault("ANTHROPIC_BASE_URL", f"http://127.0.0.1:{config.api.port}")
        env.setdefault("ANTHROPIC_API_KEY", "sisyphean-local")
        result = subprocess.run([cli], cwd=caller_cwd, env=env)
        sys.exit(result.returncode)

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
        print("Dream cycle is handled by BirdClaw — run it from the BirdClaw directory.")
        sys.exit(0)
    elif _args[0] == "tray":
        import tray as _tray
        _tray.main()
    elif _args[0] in ("setup", "config"):
        cfg = os.environ.get("SISYPHEAN_CONFIG", _DEFAULT_CONFIG)
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
