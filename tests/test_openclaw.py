"""OpenClaw integration test for the Sisyphean engine.

Tests Sisyphean as a backend provider for OpenClaw, covering:
  1. Raw protocol compatibility (what OpenClaw sends / expects)
  2. Tool_use round-trip via the Anthropic adapter
  3. OpenAI-completions adapter (simpler fallback path)
  4. Optionally: live OpenClaw CLI one-shot via subprocess

Usage:
    # Start Sisyphean first (mock mode is fine for protocol tests):
    #   python main.py
    # Then:
    python test_openclaw.py
    python test_openclaw.py --live   # also runs openclaw agent --local
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

INTER_TEST_PAUSE = 0  # local Ollama — no rate limiting needed

SISYPHEAN_URL = "http://127.0.0.1:47291"
OPENCLAW_DIR = Path(__file__).parent.parent / "openclaw"
OPENCLAW_CONFIG = Path.home() / ".openclaw" / "openclaw.sisyphean.json"

_failures = 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def post(url: str, body: dict, extra_headers: dict | None = None) -> dict:
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as r:   # 3 min for retry backoff
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return {"_http_error": e.code, **json.loads(raw)}
        except Exception:
            return {"_http_error": e.code, "raw": raw}


def ok(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    safe_detail = detail.encode("ascii", errors="replace").decode("ascii") if detail else ""
    print(f"  [{status}] {label}" + (f"  - {safe_detail}" if safe_detail else ""))
    if not condition:
        global _failures
        _failures += 1


# ── Protocol constants that OpenClaw sends ────────────────────────────────────
# OpenClaw uses the Anthropic Messages API protocol.
# These are the exact headers and request structure it sends.

OPENCLAW_HEADERS = {
    "x-api-key": "sisyphean-local",
    "anthropic-version": "2023-06-01",
}

# OpenClaw's built-in tool definitions (simplified subset)
OPENCLAW_TOOLS = [
    {
        "name": "Bash",
        "description": "Execute a bash command in the workspace and return the output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to run",
                },
                "timeout": {
                    "type": "number",
                    "description": "Optional timeout in milliseconds",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "Read",
        "description": "Read the contents of a file at a given path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the file to read",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "Write",
        "description": "Write content to a file at a given path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to write to",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write",
                },
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "WebSearch",
        "description": "Search the web for current information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                },
            },
            "required": ["query"],
        },
    },
]


# ── Test 1: OpenAI compat adapter ─────────────────────────────────────────────

def test_oai_compat():
    """OpenClaw's openai-completions adapter hits /v1/chat/completions."""
    print("\n=== OpenAI-completions adapter (openclaw-oai provider) ===")
    r = post(
        f"{SISYPHEAN_URL}/v1/chat/completions",
        {
            "model": "sisyphean",
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 32,
        },
        extra_headers={"Authorization": "Bearer sisyphean-local"},
    )
    ok("No HTTP error", "_http_error" not in r, str(r.get("_http_error", "")))
    choices = r.get("choices", [])
    ok("choices present", len(choices) > 0)
    if choices:
        msg = choices[0].get("message", {})
        ok("role=assistant", msg.get("role") == "assistant")
        ok("content non-empty", bool(str(msg.get("content", "")).strip()), repr(msg.get("content")))


# ── Test 2: Anthropic adapter — simple message ────────────────────────────────

def test_anthropic_simple():
    """OpenClaw's anthropic-messages adapter for simple queries (no tools)."""
    print("\n=== Anthropic-messages adapter - simple query ===")
    r = post(
        f"{SISYPHEAN_URL}/v1/messages",
        {
            "model": "sisyphean",
            "messages": [{"role": "user", "content": "What is 2 + 2?"}],
            "max_tokens": 64,
        },
        extra_headers=OPENCLAW_HEADERS,
    )
    ok("No HTTP error", "_http_error" not in r, str(r.get("_http_error", "")))
    ok("type=message", r.get("type") == "message", str(r.get("type")))
    content = r.get("content", [])
    texts = [b.get("text", "") for b in content if b.get("type") == "text"]
    ok("text block returned", bool(texts and texts[0].strip()), str(content))
    ok("stop_reason present", bool(r.get("stop_reason")))


# ── Test 3: Full tool_use round-trip (openclaw agent protocol) ────────────────

def test_tool_use_roundtrip():
    """Simulate exactly what OpenClaw's agent loop does:
      Turn 1 → send user message + tools → expect tool_use or end_turn
      Turn 2 → send tool_result → expect next action or end_turn
    """
    print("\n=== Tool_use round-trip (openclaw agent protocol) ===")

    # Turn 1: user message with tools
    r1 = post(
        f"{SISYPHEAN_URL}/v1/messages",
        {
            "model": "sisyphean",
            "messages": [
                {"role": "user", "content": "List the Python files in the current directory."}
            ],
            "max_tokens": 256,
            "tools": OPENCLAW_TOOLS,
        },
        extra_headers=OPENCLAW_HEADERS,
    )
    ok("Turn 1: no HTTP error", "_http_error" not in r1, str(r1.get("_http_error", "")))
    stop1 = r1.get("stop_reason", "")
    content1 = r1.get("content", [])
    ok("Turn 1: content present", len(content1) > 0)

    if stop1 == "tool_use":
        tool_blocks = [b for b in content1 if b.get("type") == "tool_use"]
        ok("Turn 1: tool_use block present", len(tool_blocks) > 0)

        if tool_blocks:
            tb = tool_blocks[0]
            tool_name = tb.get("name", "")
            tool_input = tb.get("input", {})
            tool_id = tb.get("id", "")
            ok("Turn 1: tool has id", bool(tool_id))
            ok("Turn 1: tool has name", bool(tool_name))
            ok(f"Turn 1: tool name is known ({tool_name})", tool_name in {t["name"] for t in OPENCLAW_TOOLS})
            ok("Turn 1: tool has input dict", isinstance(tool_input, dict))
            print(f"     -> {tool_name}({json.dumps(tool_input)[:80]})")

            # Check thinking block has SISYPHEAN_STATE (state persistence)
            think_blocks = [b for b in content1 if b.get("type") == "thinking"]
            state_blocks = [b for b in think_blocks if "SISYPHEAN_STATE:" in (b.get("thinking") or "")]
            ok(
                "Turn 1: SISYPHEAN_STATE in thinking block",
                len(state_blocks) > 0,
                f"{len(think_blocks)} thinking blocks, {len(state_blocks)} with state",
            )

            # Turn 2: send tool_result back (simulates OpenClaw executing the tool)
            print("\n=== Tool_use round-trip - Turn 2 (tool_result) ===")
            r2 = post(
                f"{SISYPHEAN_URL}/v1/messages",
                {
                    "model": "sisyphean",
                    "messages": [
                        {"role": "user", "content": "List the Python files in the current directory."},
                        {"role": "assistant", "content": content1},  # include thinking + tool_use
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_id,
                                    "content": "main.py\ntest_api.py\ntest_openclaw.py\nengine/__init__.py",
                                }
                            ],
                        },
                    ],
                    "max_tokens": 256,
                    "tools": OPENCLAW_TOOLS,
                },
                extra_headers=OPENCLAW_HEADERS,
            )
            ok("Turn 2: no HTTP error", "_http_error" not in r2, str(r2.get("_http_error", "")))
            stop2 = r2.get("stop_reason", "")
            ok(
                f"Turn 2: stop_reason is tool_use or end_turn (got {stop2!r})",
                stop2 in ("tool_use", "end_turn"),
            )
            content2 = r2.get("content", [])
            ok("Turn 2: content present", len(content2) > 0)

            if stop2 == "end_turn":
                texts2 = [b.get("text", "") for b in content2 if b.get("type") == "text"]
                ok("Turn 2: final text present", bool(texts2 and texts2[0].strip()))
                print(f"     Final answer: {texts2[0][:120] if texts2 else '(none)'}")
            elif stop2 == "tool_use":
                tb2 = next((b for b in content2 if b.get("type") == "tool_use"), None)
                if tb2:
                    print(f"     Next tool: {tb2.get('name')}({json.dumps(tb2.get('input', {}))[:80]})")

    elif stop1 == "end_turn":
        ok("Turn 1: answered directly (no tool needed)", True)
        texts1 = [b.get("text", "") for b in content1 if b.get("type") == "text"]
        ok("Turn 1: text present", bool(texts1 and texts1[0].strip()))


# ── Test 4: Remember / soul routing ──────────────────────────────────────────

def test_remember_routing():
    """Soul router should intercept 'remember' messages without entering the loop."""
    print("\n=== Soul routing - remember interceptor ===")
    r = post(
        f"{SISYPHEAN_URL}/v1/messages",
        {
            "model": "sisyphean",
            "messages": [
                {"role": "user", "content": "Remember that I prefer tabs over spaces."}
            ],
            "max_tokens": 64,
            "tools": OPENCLAW_TOOLS,  # tools present — soul router must still intercept
        },
        extra_headers=OPENCLAW_HEADERS,
    )
    ok("No HTTP error", "_http_error" not in r, str(r.get("_http_error", "")))
    ok("stop_reason=end_turn (no tool call for remember)", r.get("stop_reason") == "end_turn")
    content = r.get("content", [])
    tool_blocks = [b for b in content if b.get("type") == "tool_use"]
    ok("No tool_use blocks (remember is handled inline)", len(tool_blocks) == 0, str(content))
    texts = [b.get("text", "") for b in content if b.get("type") == "text"]
    ok("Acknowledgement text present", bool(texts and texts[0].strip()))


# ── Test 5: OpenClaw CLI live run (optional) ─────────────────────────────────

def test_openclaw_cli_live():
    """Run OpenClaw agent --local pointing at Sisyphean via ANTHROPIC_BASE_URL."""
    print("\n=== OpenClaw CLI live integration ===")

    if not OPENCLAW_DIR.exists():
        ok("openclaw dir exists at " + str(OPENCLAW_DIR), False, "repo not found - skipping")
        return

    entry = OPENCLAW_DIR / "openclaw.mjs"
    if not entry.exists():
        ok("openclaw.mjs found", False, "build not done - skipping")
        return

    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = SISYPHEAN_URL
    env["ANTHROPIC_API_KEY"] = "sisyphean-local"

    try:
        result = subprocess.run(
            ["node", str(entry),
             "agent", "--local",
             "--message", "What is the capital of France?",
             "--model", "sisyphean/sisyphean",
             "--json"],
            capture_output=True, text=True, timeout=60, env=env,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        ok("CLI exited 0", result.returncode == 0, f"exit={result.returncode} stderr={stderr[:200]}")

        if stdout:
            try:
                data = json.loads(stdout)
                reply = data.get("reply") or data.get("content") or data.get("text") or str(data)
                ok("CLI JSON response has content", bool(reply), repr(reply[:120]))
                print(f"     Reply: {str(reply)[:120]}")
            except json.JSONDecodeError:
                ok("CLI output is plain text (non-JSON mode)", bool(stdout))
                print(f"     Output: {stdout[:200]}")

    except subprocess.TimeoutExpired:
        ok("CLI completed within 60s", False, "timed out")
    except FileNotFoundError:
        ok("node executable found", False, "node not in PATH")


# ── Runner ────────────────────────────────────────────────────────────────────

def main():
    live = "--live" in sys.argv
    print(f"OpenClaw <-> Sisyphean integration test")
    print(f"  Sisyphean: {SISYPHEAN_URL}")
    print(f"  OpenClaw:  {OPENCLAW_DIR}")
    print(f"  Mode:      {'live CLI' if live else 'protocol only'}")
    print("=" * 56)

    # Check Sisyphean is up
    try:
        with urllib.request.urlopen(f"{SISYPHEAN_URL}/health", timeout=5) as r:
            h = json.loads(r.read())
            if h.get("status") != "ok":
                raise RuntimeError("health != ok")
    except Exception as exc:
        print(f"\nERROR: Sisyphean engine not reachable at {SISYPHEAN_URL}")
        print(f"  Details: {exc}")
        print("  Start it first: python main.py")
        sys.exit(1)

    def pace():
        if INTER_TEST_PAUSE:
            print(f"\n  [pace] waiting {INTER_TEST_PAUSE}s (free-tier rate limit)...")
            time.sleep(INTER_TEST_PAUSE)

    test_oai_compat()
    pace(); test_anthropic_simple()
    pace(); test_tool_use_roundtrip()
    pace(); test_remember_routing()

    if live:
        pace(); test_openclaw_cli_live()

    print("\n" + "=" * 56)
    if _failures == 0:
        print("All tests passed.")
    else:
        print(f"{_failures} test(s) FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
