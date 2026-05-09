"""Sisyphean Engine — system tray manager.

Run via install.bat. Shows a tray icon in the notification area.
Right-click for options. Double-click to open the API dashboard.

Requires: pip install pystray Pillow
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

import pystray
from PIL import Image, ImageDraw

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
ICON_PATH  = HERE / "assets" / "sisyphean.png"
MAIN_PY    = HERE / "main.py"
PYTHON     = Path(sys.executable)
PYTHONW    = PYTHON.parent / "pythonw.exe"  # windowless Python
ENGINE_URL = "http://127.0.0.1:47291/dashboard"
_LOG_DIR   = Path.home() / ".sisyphean" / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE   = _LOG_DIR / "engine_out.txt"

CREATE_NO_WINDOW = 0x08000000  # Windows flag — no console popup

_proc: subprocess.Popen | None = None
_lock = threading.Lock()


# ── Engine lifecycle ───────────────────────────────────────────────────────────

def _running() -> bool:
    with _lock:
        return _proc is not None and _proc.poll() is None


def start_engine():
    global _proc
    with _lock:
        if _proc and _proc.poll() is None:
            return
        log = open(LOG_FILE, "a", encoding="utf-8")
        _proc = subprocess.Popen(
            [str(PYTHON), str(MAIN_PY)],
            cwd=str(HERE),
            stdout=log,
            stderr=log,
            creationflags=CREATE_NO_WINDOW,
        )


def stop_engine():
    global _proc
    with _lock:
        if _proc:
            _proc.terminate()
            try:
                _proc.wait(timeout=6)
            except subprocess.TimeoutExpired:
                _proc.kill()
            _proc = None


def restart_engine():
    stop_engine()
    time.sleep(0.8)
    start_engine()


# ── Icon helpers ───────────────────────────────────────────────────────────────

def _load_icon(running: bool) -> Image.Image:
    """Load the Sisyphean icon, adding a small status dot overlay."""
    try:
        img = Image.open(ICON_PATH).convert("RGBA").resize((64, 64), Image.LANCZOS)
    except Exception:
        img = Image.new("RGBA", (64, 64), (30, 30, 30, 255))

    # Draw a small status dot in the bottom-right corner
    draw = ImageDraw.Draw(img)
    dot_color = (80, 200, 80, 255) if running else (200, 60, 60, 255)
    draw.ellipse([48, 48, 62, 62], fill=dot_color, outline=(0, 0, 0, 180), width=1)
    return img


# ── Tray icon ──────────────────────────────────────────────────────────────────

def build_tray() -> pystray.Icon:
    icon_img = _load_icon(_running())

    # ── Callbacks ──────────────────────────────────────────────────────────────

    def open_dashboard(icon=None, item=None):
        webbrowser.open(ENGINE_URL)

    icon = pystray.Icon(
        name="Sisyphean",
        icon=icon_img,
        title="Sisyphean Engine",
        on_activate=open_dashboard,   # left-click / double-click opens dashboard
    )

    def on_start(icon, item):
        threading.Thread(target=_do_start, args=(icon,), daemon=True).start()

    def _do_start(icon):
        start_engine()
        time.sleep(1)
        _refresh_icon(icon)
        icon.notify("Sisyphean", "Engine started.")

    def on_restart(icon, item):
        threading.Thread(target=_do_restart, args=(icon,), daemon=True).start()

    def _do_restart(icon):
        icon.notify("Sisyphean", "Restarting engine…")
        restart_engine()
        time.sleep(1)
        _refresh_icon(icon)

    def on_stop(icon, item):
        stop_engine()
        _refresh_icon(icon)
        icon.notify("Sisyphean", "Engine stopped.")

    def on_view_logs(icon, item):
        os.startfile(str(LOG_FILE))

    def on_update(icon, item):
        threading.Thread(target=_do_update, args=(icon,), daemon=True).start()

    def _do_update(icon):
        icon.notify("Sisyphean", "Updating — pulling latest code…")
        stop_engine()
        try:
            result = subprocess.run(
                ["git", "pull"],
                cwd=str(HERE),
                capture_output=True, text=True,
                creationflags=CREATE_NO_WINDOW,
            )
            if result.returncode != 0:
                icon.notify("Sisyphean", f"git pull failed: {result.stderr.strip()[:80]}")
            else:
                subprocess.run(
                    [str(PYTHON), "-m", "pip", "install", "-r", "requirements.txt",
                     "--upgrade", "--quiet"],
                    cwd=str(HERE),
                    creationflags=CREATE_NO_WINDOW,
                )
                icon.notify("Sisyphean", "Update complete. Restarting engine…")
        except Exception as e:
            icon.notify("Sisyphean", f"Update error: {e}")
        time.sleep(1)
        start_engine()
        time.sleep(1)
        _refresh_icon(icon)

    def on_quit(icon, item):
        stop_engine()
        icon.stop()

    def _refresh_icon(icon):
        icon.icon = _load_icon(_running())
        icon.title = f"Sisyphean Engine — {'Running' if _running() else 'Stopped'}"

    # ── Menu ───────────────────────────────────────────────────────────────────

    icon.menu = pystray.Menu(
        pystray.MenuItem(
            lambda _: f"{'● Running' if _running() else '○ Stopped'}",
            None,
            enabled=False,
        ),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Open Dashboard (localhost:47291/dashboard)", open_dashboard, default=True),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Start Engine",   on_start),
        pystray.MenuItem("Restart Engine", on_restart),
        pystray.MenuItem("Stop Engine",    on_stop),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("View Logs", on_view_logs),
        pystray.MenuItem("Update (git pull)", on_update),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", on_quit),
    )

    return icon


# ── Watchdog ───────────────────────────────────────────────────────────────────

def _watchdog(icon: pystray.Icon):
    """Restart the engine automatically if it crashes."""
    while True:
        time.sleep(15)
        if not _running():
            start_engine()
            time.sleep(2)
            icon.icon = _load_icon(_running())


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    import traceback
    _tray_log = HERE / "tray.log"
    try:
        start_engine()
        icon = build_tray()
        threading.Thread(target=_watchdog, args=(icon,), daemon=True).start()
        icon.run()
    except Exception:
        # pythonw swallows all output — write errors to tray.log
        with open(_tray_log, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*40}\n")
            traceback.print_exc(file=f)
        raise


if __name__ == "__main__":
    main()
