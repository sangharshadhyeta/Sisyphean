"""Sisyphean — unified system tray manager.

Manages both the Sisyphean engine and the BirdClaw agent from a single tray
icon.  Run via install.bat or directly:

    pythonw tray.py          (no console window)
    python  tray.py          (with console, useful for debugging)
    python  main.py tray     (via main.py entry point)

Icon dot colour:
  Green  — engine running AND AI model ready
  Amber  — engine running but AI model unavailable (backend down)
  Red    — engine stopped

Requires: pip install pystray Pillow
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

import pystray
from PIL import Image, ImageDraw

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).parent
ICON_PATH = HERE / "assets" / "sisyphean.png"
MAIN_PY   = HERE / "main.py"
PYTHON    = Path(sys.executable)
PYTHONW   = PYTHON.parent / "pythonw.exe"

_DEFAULT_ENGINE_PORT = 47291
_DEFAULT_BC_PORT     = 47293

CREATE_NO_WINDOW = 0x08000000  # Windows: suppress console popup


def _read_engine_port() -> int:
    try:
        import yaml  # type: ignore[import]
        with (HERE / "config.yaml").open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return int(data.get("api", {}).get("port", _DEFAULT_ENGINE_PORT))
    except Exception:
        return _DEFAULT_ENGINE_PORT


_ENGINE_PORT = _read_engine_port()
ENGINE_DASH  = f"http://127.0.0.1:{_ENGINE_PORT}/dashboard"
_LOG_DIR     = Path.home() / ".sisyphean" / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE     = _LOG_DIR / "engine_out.txt"
BC_LOG_FILE  = _LOG_DIR / "birdclaw_out.txt"

# ── Singleton guard ────────────────────────────────────────────────────────────
_singleton_mutex = None
if sys.platform == "win32":
    import ctypes
    _singleton_mutex = ctypes.windll.kernel32.CreateMutexW(
        None, False, "Global\\SisypheanTray"
    )
    if ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
        sys.exit(0)

# ── Process handles ────────────────────────────────────────────────────────────
_engine_proc: subprocess.Popen | None = None
_bc_proc:     subprocess.Popen | None = None
_lock = threading.Lock()

# When True, watchdog will not auto-restart that component
_engine_paused = False
_bc_paused     = False


# ── Port helpers ───────────────────────────────────────────────────────────────

def _port_open(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _engine_running() -> bool:
    with _lock:
        alive = _engine_proc is not None and _engine_proc.poll() is None
    return alive or _port_open(_ENGINE_PORT)


def _find_birdclaw_dir() -> Path | None:
    candidates = [
        HERE.parent / "BirdClaw",
        HERE.parent / "birdclaw",
        Path.home() / "BirdClaw",
        Path.home() / "birdclaw",
    ]
    return next((d for d in candidates if (d / "main.py").exists()), None)


_BC_DIR = _find_birdclaw_dir()
_BC_PORT = _DEFAULT_BC_PORT
if _BC_DIR:
    try:
        sys.path.insert(0, str(_BC_DIR))
        from birdclaw.config import settings as _bc_cfg  # type: ignore[import]
        _BC_PORT = _bc_cfg.web_port
    except Exception:
        pass
    finally:
        if str(_BC_DIR) in sys.path:
            sys.path.remove(str(_BC_DIR))

BC_UI_URL = f"http://127.0.0.1:{_BC_PORT}"


def _bc_running() -> bool:
    with _lock:
        alive = _bc_proc is not None and _bc_proc.poll() is None
    return alive or _port_open(_BC_PORT)


# ── Status probe ───────────────────────────────────────────────────────────────
# "ready"      — engine up, model responding
# "model_down" — engine up, model unavailable
# "offline"    — engine not reachable

def _engine_status() -> str:
    if not _port_open(_ENGINE_PORT):
        return "offline"
    try:
        url = f"http://127.0.0.1:{_ENGINE_PORT}/api/status"
        with urllib.request.urlopen(url, timeout=2) as r:
            data = json.loads(r.read())
        return "ready" if data.get("llm_ready") else "model_down"
    except Exception:
        return "offline"


# ── Engine lifecycle ───────────────────────────────────────────────────────────

def start_engine():
    global _engine_proc, _engine_paused
    _engine_paused = False
    with _lock:
        if _engine_proc and _engine_proc.poll() is None:
            return
    if _port_open(_ENGINE_PORT):
        return
    with _lock:
        flags = CREATE_NO_WINDOW if sys.platform == "win32" else 0
        log = open(LOG_FILE, "a", encoding="utf-8")
        _engine_proc = subprocess.Popen(
            [str(PYTHON), str(MAIN_PY)],
            cwd=str(HERE),
            stdout=log,
            stderr=log,
            creationflags=flags,
        )


def stop_engine():
    global _engine_proc, _engine_paused
    _engine_paused = True          # tell watchdog not to restart
    with _lock:
        if _engine_proc:
            _engine_proc.terminate()
            try:
                _engine_proc.wait(timeout=6)
            except subprocess.TimeoutExpired:
                _engine_proc.kill()
            _engine_proc = None


def restart_engine():
    stop_engine()
    time.sleep(0.8)
    start_engine()


# ── BirdClaw lifecycle ─────────────────────────────────────────────────────────

def start_birdclaw():
    global _bc_proc, _bc_paused
    if _BC_DIR is None:
        return
    _bc_paused = False
    with _lock:
        if _bc_proc and _bc_proc.poll() is None:
            return
    if _port_open(_BC_PORT):
        return
    with _lock:
        flags = CREATE_NO_WINDOW if sys.platform == "win32" else 0
        exe = str(PYTHONW) if PYTHONW.exists() else str(PYTHON)
        log = open(BC_LOG_FILE, "a", encoding="utf-8")
        _bc_proc = subprocess.Popen(
            [exe, str(_BC_DIR / "main.py"), "web"],
            cwd=str(_BC_DIR),
            stdout=log,
            stderr=log,
            creationflags=flags,
        )


def stop_birdclaw():
    global _bc_proc, _bc_paused
    _bc_paused = True
    with _lock:
        if _bc_proc:
            _bc_proc.terminate()
            try:
                _bc_proc.wait(timeout=6)
            except subprocess.TimeoutExpired:
                _bc_proc.kill()
            _bc_proc = None


def restart_birdclaw():
    stop_birdclaw()
    time.sleep(0.8)
    start_birdclaw()


# ── Icon rendering ─────────────────────────────────────────────────────────────

_DOT_COLORS = {
    "ready":      (80,  200,  80, 255),   # green
    "model_down": (240, 160,  30, 255),   # amber
    "offline":    (200,  60,  60, 255),   # red
}


def _load_icon(status: str) -> Image.Image:
    """Render the Sisyphean icon with a coloured status dot (bottom-right)."""
    try:
        img = Image.open(ICON_PATH).convert("RGBA").resize((64, 64), Image.LANCZOS)
    except Exception:
        img = Image.new("RGBA", (64, 64), (30, 30, 30, 255))

    draw = ImageDraw.Draw(img)
    dot = _DOT_COLORS.get(status, _DOT_COLORS["offline"])
    draw.ellipse([48, 48, 62, 62], fill=dot, outline=(0, 0, 0, 180), width=1)
    return img


def _status_label(status: str) -> str:
    return {
        "ready":      "● Running — AI ready",
        "model_down": "⚠ Running — AI model unavailable",
        "offline":    "○ Engine stopped",
    }.get(status, "○ Engine stopped")


# ── Tray icon ──────────────────────────────────────────────────────────────────

def build_tray() -> pystray.Icon:
    _status_cache: dict = {"status": _engine_status()}

    icon = pystray.Icon(
        name="Sisyphean",
        icon=_load_icon(_status_cache["status"]),
        title="Sisyphean",
        on_activate=lambda i, _: webbrowser.open(ENGINE_DASH),
    )

    # ── Refresh helpers ────────────────────────────────────────────────────────

    def _refresh(icon):
        s = _engine_status()
        _status_cache["status"] = s
        icon.icon  = _load_icon(s)
        icon.title = f"Sisyphean — {_status_label(s)}"

    def _refresh_async(icon):
        threading.Thread(target=lambda: _refresh(icon), daemon=True).start()

    # ── Sisyphean callbacks ────────────────────────────────────────────────────

    def on_open_dash(icon, item):
        webbrowser.open(ENGINE_DASH)

    def on_start_engine(icon, item):
        def _do():
            start_engine()
            time.sleep(1.5)
            _refresh(icon)
            icon.notify("Sisyphean", "Engine started.")
        threading.Thread(target=_do, daemon=True).start()

    def on_restart_engine(icon, item):
        def _do():
            icon.notify("Sisyphean", "Restarting engine…")
            restart_engine()
            time.sleep(1.5)
            _refresh(icon)
        threading.Thread(target=_do, daemon=True).start()

    def on_stop_engine(icon, item):
        def _do():
            stop_engine()
            _refresh(icon)
            icon.notify("Sisyphean", "Engine stopped.")
        threading.Thread(target=_do, daemon=True).start()

    def on_view_engine_logs(icon, item):
        if sys.platform == "win32":
            os.startfile(str(LOG_FILE))
        else:
            subprocess.Popen(["xdg-open", str(LOG_FILE)])

    def on_update_engine(icon, item):
        def _do():
            icon.notify("Sisyphean", "Updating — pulling latest code…")
            stop_engine()
            flags = CREATE_NO_WINDOW if sys.platform == "win32" else 0
            try:
                r = subprocess.run(
                    ["git", "pull"], cwd=str(HERE),
                    capture_output=True, text=True, creationflags=flags,
                )
                if r.returncode != 0:
                    icon.notify("Sisyphean", f"git pull failed: {r.stderr.strip()[:80]}")
                else:
                    subprocess.run(
                        [str(PYTHON), "-m", "pip", "install", "-r",
                         "requirements.txt", "--upgrade", "--quiet"],
                        cwd=str(HERE), creationflags=flags,
                    )
                    icon.notify("Sisyphean", "Update complete. Restarting…")
            except Exception as exc:
                icon.notify("Sisyphean", f"Update error: {exc}")
            time.sleep(1)
            start_engine()
            time.sleep(1.5)
            _refresh(icon)
        threading.Thread(target=_do, daemon=True).start()

    # ── BirdClaw callbacks ─────────────────────────────────────────────────────

    def on_open_bc(icon, item):
        webbrowser.open(BC_UI_URL)

    def on_start_bc(icon, item):
        def _do():
            if _BC_DIR is None:
                icon.notify("Sisyphean", "BirdClaw not found.")
                return
            start_birdclaw()
            time.sleep(2)
            icon.notify("Sisyphean", f"BirdClaw started — localhost:{_BC_PORT}")
        threading.Thread(target=_do, daemon=True).start()

    def on_restart_bc(icon, item):
        def _do():
            if _BC_DIR is None:
                return
            icon.notify("Sisyphean", "Restarting BirdClaw…")
            restart_birdclaw()
            time.sleep(2)
        threading.Thread(target=_do, daemon=True).start()

    def on_stop_bc(icon, item):
        def _do():
            stop_birdclaw()
            icon.notify("Sisyphean", "BirdClaw stopped.")
        threading.Thread(target=_do, daemon=True).start()

    def on_view_bc_logs(icon, item):
        if sys.platform == "win32":
            os.startfile(str(BC_LOG_FILE))
        else:
            subprocess.Popen(["xdg-open", str(BC_LOG_FILE)])

    # ── Quit — stops tray only; engine/BirdClaw keep running ──────────────────

    def on_quit_tray(icon, item):
        """Close the tray icon. Engine and BirdClaw are left running."""
        icon.stop()

    def on_quit_all(icon, item):
        """Stop everything and close the tray."""
        stop_engine()
        stop_birdclaw()
        icon.stop()

    # ── Menu ───────────────────────────────────────────────────────────────────

    def _sisyphean_status(_):
        return _status_label(_status_cache.get("status", "offline"))

    def _bc_status(_):
        if _BC_DIR is None:
            return "BirdClaw — not installed"
        return f"{'● BirdClaw running' if _bc_running() else '○ BirdClaw stopped'}"

    bc_items = []
    if _BC_DIR is not None:
        bc_items = [
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(_bc_status, None, enabled=False),
            pystray.MenuItem(f"Open BirdClaw (localhost:{_BC_PORT})", on_open_bc),
            pystray.MenuItem("Start BirdClaw",   on_start_bc),
            pystray.MenuItem("Restart BirdClaw", on_restart_bc),
            pystray.MenuItem("Stop BirdClaw",    on_stop_bc),
            pystray.MenuItem("BirdClaw Logs",    on_view_bc_logs),
        ]

    icon.menu = pystray.Menu(
        pystray.MenuItem(_sisyphean_status, None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem(
            lambda _: f"Open Dashboard  (localhost:{_ENGINE_PORT}/dashboard)",
            on_open_dash,
            default=True,
        ),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Start Engine",          on_start_engine),
        pystray.MenuItem("Restart Engine",        on_restart_engine),
        pystray.MenuItem("Stop Engine",           on_stop_engine),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Engine Logs",           on_view_engine_logs),
        pystray.MenuItem("Update Engine (git pull)", on_update_engine),
        *bc_items,
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit Tray",             on_quit_tray),
        pystray.MenuItem("Quit All",              on_quit_all),
    )

    return icon


# ── Watchdog ───────────────────────────────────────────────────────────────────

def _watchdog(icon: pystray.Icon):
    """Polls engine + BirdClaw health every 15 s.  Restarts crashed processes,
    but respects intentional stops (_engine_paused / _bc_paused flags)."""
    while True:
        time.sleep(15)

        # Sisyphean engine
        if not _engine_paused and not _engine_running():
            start_engine()
            time.sleep(2)

        # BirdClaw (only if installed and was running before)
        if _BC_DIR and not _bc_paused and not _bc_running():
            # Don't auto-start BirdClaw — user must start it explicitly
            pass

        # Refresh icon from live status
        s = _engine_status()
        icon.icon  = _load_icon(s)
        icon.title = f"Sisyphean — {_status_label(s)}"


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    import traceback
    _tray_log = HERE / "tray.log"
    try:
        start_engine()          # start engine on tray launch
        icon = build_tray()
        threading.Thread(target=_watchdog, args=(icon,), daemon=True).start()
        icon.run()              # blocks until icon.stop() is called
    except Exception:
        with open(_tray_log, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*40}\n")
            traceback.print_exc(file=f)
        raise


if __name__ == "__main__":
    main()
