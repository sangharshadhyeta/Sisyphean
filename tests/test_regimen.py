"""Sisyphean 11-test regimen.

Tests (in order):
  T1  hi                            soul greeting (text reply, no stale memory junk)
  T2  2+2                               symbolic math (digit-operator regex)
  T3  what is 12 times 7?               spoken math (word-operator regex)
  T4  100/4                             symbolic math (digit-operator regex)
  T5  what is the square root of 144?   spoken math (keyword regex)
  T6  capital of France      research (Paris, via internal web_search)
  T7  latest Python version  web search required (version number in answer)
  T8  create folder test123  bash mkdir
  T9  remember I prefer vim  save_memory (confirmation in reply)
  T10 ok standalone          soul response (no hollow filler, no tool calls)
  T11 thanks                 acknowledgement -- NOT saved to memory, no tool call

Failure in any test resets to T1. Run: python test_regimen.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:47291"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def post(path, body, timeout=180):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return {"_http_error": e.code, **json.loads(raw)}
        except Exception:
            return {"_http_error": e.code, "raw": raw}
    except Exception as e:
        return {"_http_error": "timeout", "raw": str(e)}


def get(path):
    try:
        with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.URLError as e:
        return {"error": str(e)}


def _s(s, n=200):
    """Truncate + ASCII-safe for Windows cp1252 stdout."""
    return str(s)[:n].encode("ascii", errors="replace").decode("ascii")


# ---------------------------------------------------------------------------
# Tool definitions (outer Claude Code tools)
# ---------------------------------------------------------------------------

BASH = {
    "name": "Bash",
    "description": "Execute a shell command and return stdout/stderr.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}

WEBSEARCH = {
    "name": "WebSearch",
    "description": "Search the web for current information.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# Multi-turn driver — simulates the Claude Code tool execution round-trip
# ---------------------------------------------------------------------------

def _drive(messages, tools, sim, max_rounds=8):
    """Drive multi-turn tool_use until end_turn or max_rounds.

    sim(block) -> str  maps a tool_use block to a simulated/real result.

    Runs sim() on pending tool_use blocks BEFORE each engine call so the
    engine always receives a proper tool_result in the last user message.

    Returns (last_response, full_history).
    """
    last_r = {}
    for _ in range(max_rounds):
        # Execute any pending tool_use in the last assistant message first
        if messages and messages[-1].get("role") == "assistant":
            pending = [b for b in messages[-1].get("content", [])
                       if b.get("type") == "tool_use"]
            if pending:
                results = [
                    {"type": "tool_result", "tool_use_id": b["id"], "content": sim(b)}
                    for b in pending
                ]
                messages = messages + [{"role": "user", "content": results}]

        r = post("/v1/messages", {
            "model": "sisyphean",
            "messages": messages,
            "max_tokens": 1024,
            "tools": tools,
        })
        last_r = r
        if "_http_error" in r:
            return r, messages
        content = r.get("content", [])
        stop = r.get("stop_reason", "")
        messages = messages + [{"role": "assistant", "content": content}]
        if stop == "end_turn":
            return r, messages
        if not any(b.get("type") == "tool_use" for b in content):
            return r, messages
    return last_r, messages


def _texts(r):
    """Extract the answer text from a response — skips the *Sisyphean stages:* trace block."""
    parts = []
    for b in r.get("content", []):
        if b.get("type") != "text":
            continue
        t = b.get("text", "")
        if t.startswith("*Sisyphean stages:"):
            continue  # skip the internal trace block; only check the actual answer
        parts.append(t)
    return " ".join(parts)


def _tool_names(r):
    """Names of tool_use blocks in a single response."""
    return [b.get("name", "") for b in r.get("content", []) if b.get("type") == "tool_use"]


# ---------------------------------------------------------------------------
# Simulators
# ---------------------------------------------------------------------------

def _sim_bash_real(block):
    """Execute Bash tool_use blocks for real (python -c only; mkdir simulated)."""
    cmd = block.get("input", {}).get("command", "")
    # Only execute python -c expression evaluation — safe, no side effects
    if cmd.strip().startswith("python") and "-c" in cmd:
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=10
            )
            out = (result.stdout + result.stderr).strip()
            print(f"      [bash] {_s(cmd, 80)} -> {_s(out, 40)}")
            return out
        except Exception as e:
            return f"Error: {e}"
    # mkdir / folder creation — simulate success
    if any(k in cmd.lower() for k in ("mkdir", "md ")):
        print(f"      [bash] {_s(cmd, 80)} -> (simulated ok)")
        return ""  # success (empty stdout)
    print(f"      [bash] {_s(cmd, 80)} -> (not executed)")
    return ""


def _sim_mkdir(block):
    """Simulate mkdir — always succeeds."""
    return ""


# ---------------------------------------------------------------------------
# Individual tests
# ---------------------------------------------------------------------------

def _run(label, msg, tools, sim, checks):
    """
    Send msg, drive multi-turn, apply checks(final_r, history) -> list[tuple[str,bool,str]].
    Returns (passed, final_r) so callers can print the reply without a second request.
    """
    r1 = post("/v1/messages", {
        "model": "sisyphean",
        "messages": [{"role": "user", "content": msg}],
        "max_tokens": 512,
        "tools": tools,
    }, timeout=600)

    if "_http_error" in r1:
        print(f"    FAIL  [{label}] HTTP error: {r1.get('_http_error')}")
        return False, r1

    stop1 = r1.get("stop_reason", "")
    c1 = r1.get("content", [])

    if stop1 == "tool_use":
        msgs = [{"role": "user", "content": msg}, {"role": "assistant", "content": c1}]
        final_r, history = _drive(msgs, tools, sim)
    else:
        final_r = r1
        history = [{"role": "user", "content": msg}, {"role": "assistant", "content": c1}]

    passed = True
    for check_label, cond, detail in checks(final_r, history):
        status = "PASS" if cond else "FAIL"
        line = f"    {status}  {check_label}"
        if detail:
            line += f"  -- {_s(detail, 100)}"
        print(line)
        if not cond:
            passed = False
    return passed, final_r


# ---------------------------------------------------------------------------
# T1 — hi (soul greeting)
# ---------------------------------------------------------------------------

def t1_hi():
    print("\n[T1] hi")
    msg = "hi"

    def checks(r, _hist):
        stop = r.get("stop_reason", "")
        text = _texts(r).strip()
        tools = _tool_names(r)
        is_greeting = any(k in text.lower() for k in (
            "hello", "hey", "welcome", "good to", "what can", "how can",
            "what's up", "what would", "ready", "back",
        )) or text.lower().startswith("hi")
        # Reject web-search-result replies masquerading as greetings
        is_web_result = any(k in text.lower() for k in (
            "web search", "search results", "here are the results",
        ))
        is_greeting = is_greeting and not is_web_result
        hollow = any(h in text.lower() for h in ("certainly!", "great!", "of course!", "absolutely!"))
        return [
            ("stop=end_turn",           stop == "end_turn",              stop),
            ("no outer tool calls",     len(tools) == 0,                 str(tools)),
            ("has text reply",          bool(text),                      ""),
            ("not stale/stage junk",    "atomic-clock" not in text.lower()
                                        and "checker.py" not in text.lower()
                                        and "sisyphean stages" not in text.lower(), _s(text, 120)),
            ("is actual greeting",      is_greeting,                     _s(text, 120)),
            ("no hollow filler",        not hollow,                      _s(text, 80)),
        ]

    passed, final_r = _run("hi", msg, [BASH, WEBSEARCH], lambda b: "", checks)
    if passed:
        print(f"    reply: {_s(_texts(final_r), 120)}")
        return True
    return False


# ---------------------------------------------------------------------------
# T2-T5 — math via bash
# ---------------------------------------------------------------------------

def _math_test(label, expr, expected_str):
    print(f"\n[{label}] {expr}")
    msg = expr

    def checks(r, hist):
        stop = r.get("stop_reason", "")
        text = _texts(r).strip()
        # Check final answer contains expected value
        has_answer = expected_str in text.replace(".0", "").replace(" ", "")
        # Also accept if the answer says the number anywhere (e.g. "The result is 4")
        has_answer = has_answer or expected_str in text
        return [
            ("stop=end_turn",       stop == "end_turn",     stop),
            ("has text reply",      bool(text),              ""),
            (f"answer contains {expected_str!r}",
                                    has_answer,             _s(text, 100)),
        ]

    passed, final_r = _run(label, msg, [BASH, WEBSEARCH], _sim_bash_real, checks)
    if passed:
        print(f"    reply: {_s(_texts(final_r), 120)}")
    return passed


def t2_2plus2():
    # symbolic form — tests the digit-operator-digit regex path
    return _math_test("T2", "2+2", "4")


def t3_12times7():
    # spoken form — tests the word-operator regex path
    return _math_test("T3", "what is 12 times 7?", "84")


def t4_100div4():
    # symbolic form — tests the digit-operator-digit regex path
    return _math_test("T4", "100/4", "25")


def t5_sqrt144():
    print("\n[T5] what is the square root of 144?")
    msg = "what is the square root of 144?"

    def checks(r, hist):
        stop = r.get("stop_reason", "")
        text = _texts(r).strip()
        has_12 = "12" in text
        return [
            ("stop=end_turn",       stop == "end_turn",  stop),
            ("has text reply",      bool(text),           ""),
            ("answer contains 12",  has_12,              _s(text, 100)),
        ]

    passed, final_r = _run("T5", msg, [BASH, WEBSEARCH], _sim_bash_real, checks)
    if passed:
        print(f"    reply: {_s(_texts(final_r), 120)}")
    return passed


# ---------------------------------------------------------------------------
# T6 — capital of France (research; internal web_search)
# ---------------------------------------------------------------------------

def t6_capital_of_france():
    print("\n[T6] capital of France")
    msg = "what is the capital of France?"

    def checks(r, hist):
        stop = r.get("stop_reason", "")
        text = _texts(r).strip().lower()
        return [
            ("stop=end_turn",       stop == "end_turn",     stop),
            ("has text reply",      bool(text),              ""),
            ("answer contains paris", "paris" in text,      _s(text, 120)),
        ]

    passed, final_r = _run("T6", msg, [BASH, WEBSEARCH], lambda b: "Paris is the capital of France.", checks)
    if passed:
        print(f"    reply: {_s(_texts(final_r), 120)}")
    return passed


# ---------------------------------------------------------------------------
# T7 — latest Python version (must search web; answer has a version number)
# ---------------------------------------------------------------------------

def t7_latest_python():
    import re as _re
    print("\n[T7] latest Python version")
    msg = "what is the latest Python version?"

    def sim_ws(block):
        name = block.get("name", "")
        if name == "WebSearch":
            return "Python 3.13.3 is the latest stable release as of May 2025."
        return ""

    def checks(r, hist):
        stop = r.get("stop_reason", "")
        text = _texts(r).strip()
        # A version like 3.x.y or just "3." is sufficient
        has_version = bool(_re.search(r'3\.\d+', text))
        return [
            ("stop=end_turn",            stop == "end_turn",  stop),
            ("has text reply",           bool(text),           ""),
            ("answer has Python version", has_version,        _s(text, 120)),
        ]

    passed, final_r = _run("T7", msg, [BASH, WEBSEARCH], sim_ws, checks)
    if passed:
        print(f"    reply: {_s(_texts(final_r), 120)}")
    return passed


# ---------------------------------------------------------------------------
# T8 — create folder test123 (bash mkdir)
# ---------------------------------------------------------------------------

def t8_create_folder():
    print("\n[T8] create folder test123")
    msg = "create a folder called test123"

    def checks(r, hist):
        stop = r.get("stop_reason", "")
        text = _texts(r).strip().lower()
        has_confirm = any(k in text for k in ("test123", "folder", "created", "directory", "mkdir"))
        # Check if bash was called at any point in history
        bash_called = any(
            b.get("type") == "tool_use" and b.get("name") == "Bash"
            for msg in hist
            for b in (msg.get("content") if isinstance(msg.get("content"), list) else [])
        )
        return [
            ("stop=end_turn",           stop == "end_turn",    stop),
            ("bash was called",         bash_called,           str([
                b.get("input", {}).get("command", "")[:60]
                for msg in hist
                for b in (msg.get("content") if isinstance(msg.get("content"), list) else [])
                if isinstance(b, dict) and b.get("type") == "tool_use"
            ])),
            ("answer confirms creation", has_confirm,          _s(text, 120)),
        ]

    passed, final_r = _run("T8", msg, [BASH, WEBSEARCH], _sim_mkdir, checks)
    if passed:
        print(f"    reply: {_s(_texts(final_r), 120)}")
    return passed


# ---------------------------------------------------------------------------
# T9 — remember I prefer vim (save_memory; no outer tool call)
# ---------------------------------------------------------------------------

def t9_remember_vim():
    print("\n[T9] remember I prefer vim")
    msg = "remember I prefer vim"

    def checks(r, hist):
        stop = r.get("stop_reason", "")
        text = _texts(r).strip().lower()
        tools = _tool_names(r)
        # Engine handles save_memory internally; outer response should be end_turn text
        # The reply should acknowledge "vim" or "noted" or "preference"
        has_ack = any(k in text for k in ("vim", "noted", "remember", "preference", "got it",
                                           "saved", "stored", "will remember"))
        return [
            ("stop=end_turn",       stop == "end_turn",   stop),
            ("no outer tool calls", len(tools) == 0,      str(tools)),
            ("has text reply",      bool(text),            ""),
            ("acknowledges vim",    has_ack,              _s(text, 120)),
        ]

    passed, final_r = _run("T9", msg, [BASH, WEBSEARCH], lambda b: "", checks)
    if passed:
        print(f"    reply: {_s(_texts(final_r), 120)}")
    return passed


# ---------------------------------------------------------------------------
# T10 — ok standalone (soul response; brief, no hollow filler, no tools)
# ---------------------------------------------------------------------------

def t10_ok():
    print("\n[T10] ok")
    msg = "ok"

    def checks(r, hist):
        stop = r.get("stop_reason", "")
        text = _texts(r).strip()
        tools = _tool_names(r)
        hollow = any(h in text.lower() for h in ("certainly!", "great!", "of course!", "absolutely!"))
        return [
            ("stop=end_turn",       stop == "end_turn",   stop),
            ("no outer tool calls", len(tools) == 0,      str(tools)),
            ("has text reply",      bool(text),            ""),
            ("no hollow filler",    not hollow,           _s(text, 80)),
        ]

    passed, final_r = _run("T10", msg, [BASH, WEBSEARCH], lambda b: "", checks)
    if passed:
        print(f"    reply: {_s(_texts(final_r), 80)}")
    return passed


# ---------------------------------------------------------------------------
# T11 — thanks (NOT saved to memory; brief acknowledgement)
# ---------------------------------------------------------------------------

def t11_thanks():
    print("\n[T11] thanks")
    msg = "thanks"

    def checks(r, hist):
        stop = r.get("stop_reason", "")
        text = _texts(r).strip()
        tools = _tool_names(r)
        hollow = any(h in text.lower() for h in ("certainly!", "great!", "of course!", "absolutely!"))
        # "thanks" must NOT trigger a memory save — check that the reply
        # does not say "saved", "noted", "remembered", "I'll remember that"
        saved_phrases = ("saved", "noted that", "i'll remember", "storing", "i've noted")
        looks_saved = any(p in text.lower() for p in saved_phrases)
        return [
            ("stop=end_turn",       stop == "end_turn",    stop),
            ("no outer tool calls", len(tools) == 0,       str(tools)),
            ("has text reply",      bool(text),             ""),
            ("no hollow filler",    not hollow,            _s(text, 80)),
            ("NOT saved to memory", not looks_saved,       _s(text, 120)),
        ]

    passed, final_r = _run("T11", msg, [BASH, WEBSEARCH], lambda b: "", checks)
    if passed:
        print(f"    reply: {_s(_texts(final_r), 80)}")
    return passed


# ---------------------------------------------------------------------------
# Runner — failure resets to T1
# ---------------------------------------------------------------------------

TESTS = [
    ("T1",  t1_hi),
    ("T2",  t2_2plus2),
    ("T3",  t3_12times7),
    ("T4",  t4_100div4),
    ("T5",  t5_sqrt144),
    ("T6",  t6_capital_of_france),
    ("T7",  t7_latest_python),
    ("T8",  t8_create_folder),
    ("T9",  t9_remember_vim),
    ("T10", t10_ok),
    ("T11", t11_thanks),
]


def main():
    print(f"Sisyphean 11-test regimen  ->  {BASE_URL}")
    print("=" * 56)

    r = get("/health")
    if "error" in r or r.get("status") != "ok":
        print(f"ERROR: engine not reachable at {BASE_URL}")
        print("  Start with: python main.py")
        sys.exit(1)

    attempt = 1
    i = 0
    while i < len(TESTS):
        label, fn = TESTS[i]
        print(f"\n{'-' * 56}")
        print(f"  Attempt {attempt}  Running {label} ...")
        time.sleep(0.5)  # brief pause between tests

        passed = fn()

        if not passed:
            print(f"\n  !! {label} FAILED -- resetting to T1 (waiting 15s for queue to clear)")
            i = 0
            attempt += 1
            if attempt > 33:  # safety limit: 3 resets per test max
                print("\n  Too many resets. Aborting.")
                sys.exit(1)
            time.sleep(15)  # let any queued Ollama request finish before retrying
        else:
            print(f"  OK {label} passed")
            i += 1

    print(f"\n{'=' * 56}")
    print(f"All 11 tests passed (attempt {attempt}).")


if __name__ == "__main__":
    main()
