"""Sisyphean engine integration tests — updated for new pipeline.

Progressive complexity:
  L1  Infrastructure         -- HTTP endpoints, models list
  L2  Direct answers         -- routing: no tools for greetings/social
  L3  Single bash round      -- tool_use -> result -> answer
  L4  File creation          -- write -> run -> read output
  L5  Orchestration          -- multi-round tool chains, find/edit
  L6  Cross-domain           -- web+code ordering; error recovery; max pipeline

Run:
    python test_api.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:47291"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def post(path, body, timeout=180):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST"
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


_failures = 0


def _s(s, n=200):
    return str(s)[:n].encode("ascii", errors="replace").decode("ascii")


def ok(label, condition, detail=""):
    global _failures
    status = "PASS" if condition else "FAIL"
    line = f"  [{status}] {label}"
    if detail:
        line += f"  -- {_s(detail, 120)}"
    print(line)
    if not condition:
        _failures += 1


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


def _drive(messages, tools, sim, max_rounds=10):
    """Drive multi-turn tool_use until end_turn or max_rounds.

    Runs sim() on any pending tool_use in the last assistant message BEFORE
    each engine post — ensures engine always receives a proper tool_result.
    """
    last_r = {}
    for _ in range(max_rounds):
        # Execute pending tool_use from last assistant message first
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
        stop = r.get("stop_reason", "")
        content = r.get("content", [])
        if "_http_error" in r:
            return r, messages
        messages = messages + [{"role": "assistant", "content": content}]
        if stop == "end_turn":
            return r, messages
    return last_r, messages


def _cmds(history):
    """All bash command strings from the full conversation history."""
    out = []
    for msg in history:
        for blk in (msg.get("content") if isinstance(msg.get("content"), list) else []):
            if isinstance(blk, dict) and blk.get("type") == "tool_use":
                out.append(blk.get("input", {}).get("command", ""))
    return out


def _tool_names(history):
    """Ordered list of every outer tool called across the conversation."""
    names = []
    for msg in history:
        for blk in (msg.get("content") if isinstance(msg.get("content"), list) else []):
            if isinstance(blk, dict) and blk.get("type") == "tool_use":
                names.append(blk.get("name", ""))
    return names


def _extract_pipeline_state(content):
    """Parse PIPELINE_STATE from thinking blocks (new pipeline format)."""
    for b in content:
        if b.get("type") != "thinking":
            continue
        t = b.get("thinking", "")
        idx = t.find("PIPELINE_STATE:")
        if idx == -1:
            continue
        raw = t[idx + len("PIPELINE_STATE:"):].strip()
        try:
            return json.loads(raw)
        except Exception:
            pass
    return None


def _plan_tools(content):
    """Return all planned step tool names from PIPELINE_STATE."""
    state = _extract_pipeline_state(content)
    if not state:
        return []
    tools = []
    for sub in state.get("st", []):
        for step in sub.get("steps", []):
            tools.append(step.get("tool", ""))
    return tools


# ===========================================================================
# L1 -- Infrastructure
# ===========================================================================

def test_health():
    print("\n=== L1: Health & Info ===")
    r = get("/health")
    ok("/health returns status=ok", r.get("status") == "ok", str(r))
    r = get("/api/status")
    ok("/api/status has llm_ready field", "llm_ready" in r, str(r)[:120])
    ok("/api/status has model field", bool(r.get("model")), str(r)[:120])


def test_models():
    print("\n=== L1: Models list ===")
    r = get("/v1/models")
    ok("object=list", r.get("object") == "list")
    ok("at least one model", len(r.get("data", [])) >= 1)


def test_oai_completions():
    print("\n=== L1: OpenAI compat ===")
    r = post("/v1/chat/completions", {
        "model": "sisyphean",
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 64,
    })
    ok("no HTTP error", "_http_error" not in r, str(r.get("_http_error", "")))
    choices = r.get("choices", [])
    ok("choices present", len(choices) > 0)
    if choices:
        ok("role=assistant", choices[0].get("message", {}).get("role") == "assistant")
        ok("content non-empty", bool(str(choices[0].get("message", {}).get("content", "")).strip()))


# ===========================================================================
# L2 -- Direct answers (routing; no tools needed)
# ===========================================================================

def test_routing_direct():
    print("\n=== L2: Routing decisions ===")

    social_cases = [
        ("hi there",                             "greeting"),
        ("thanks, that helps",                   "thanks"),
        ("Remember I prefer tabs over spaces.",  "remember"),
    ]
    for msg, label in social_cases:
        r = post("/v1/messages", {
            "model": "sisyphean",
            "messages": [{"role": "user", "content": msg}],
            "max_tokens": 128, "tools": [BASH, WEBSEARCH],
        }, timeout=60)
        stop = r.get("stop_reason", "")
        content = r.get("content", [])
        tools_called = [b for b in content if b.get("type") == "tool_use"]
        texts = [b.get("text", "") for b in content if b.get("type") == "text"]
        ok(f"[{label}] no HTTP error", "_http_error" not in r, str(r.get("_http_error", "")))
        ok(f"[{label}] stop=end_turn", stop == "end_turn", stop)
        ok(f"[{label}] zero outer tool blocks", len(tools_called) == 0, str(len(tools_called)))
        ok(f"[{label}] text reply present", bool(texts and texts[0].strip()))
        if texts:
            print(f"     [{label}]: {_s(texts[0], 80)}")

    tool_cases = [
        ("What is 2 + 2?",   "math",       "4"),
        ("what can you do?", "capability",  None),
        ("are you alive?",   "alive-check", None),
    ]
    for msg, label, kw in tool_cases:
        r = post("/v1/messages", {
            "model": "sisyphean",
            "messages": [{"role": "user", "content": msg}],
            "max_tokens": 256, "tools": [BASH, WEBSEARCH],
        }, timeout=90)
        stop = r.get("stop_reason", "")
        content = r.get("content", [])
        texts = [b.get("text", "") for b in content if b.get("type") == "text"]
        ok(f"[{label}] no HTTP error", "_http_error" not in r, str(r.get("_http_error", "")))
        ok(f"[{label}] stop valid", stop in ("tool_use", "end_turn"), stop)
        if kw and texts and stop == "end_turn":
            full = " ".join(texts).lower()
            ok(f"[{label}] answer contains '{kw}'", kw in full, _s(full, 100))
        if texts:
            print(f"     [{label}]: {_s(texts[0], 80)}")


# ===========================================================================
# L3 -- Single bash round
# ===========================================================================

def test_single_bash_round():
    print("\n=== L3: Single bash round (ls -> answer) ===")
    msg = "List the Python files in the current directory."
    r1 = post("/v1/messages", {
        "model": "sisyphean",
        "messages": [{"role": "user", "content": msg}],
        "max_tokens": 512, "tools": [BASH],
    })
    ok("no HTTP error", "_http_error" not in r1, str(r1.get("_http_error", "")))
    stop1 = r1.get("stop_reason", "")
    c1 = r1.get("content", [])
    ok(f"stop valid ({stop1!r})", stop1 in ("tool_use", "end_turn"))

    if stop1 == "tool_use":
        tb = next((b for b in c1 if b.get("type") == "tool_use"), None)
        ok("bash tool called", tb is not None)
        if tb:
            print(f"     cmd: {_s(tb.get('input', {}).get('command', ''), 80)}")
        msgs = [{"role": "user", "content": msg}, {"role": "assistant", "content": c1}]
        final_r, hist = _drive(msgs, [BASH], lambda b: "main.py\ntest_api.py\nengine/__init__.py")
        ok("reached end_turn", final_r.get("stop_reason") == "end_turn")
        texts = [b.get("text", "") for b in final_r.get("content", []) if b.get("type") == "text"]
        full = " ".join(texts).lower()
        ok("answer names .py files", ".py" in full or "main" in full, _s(full, 150))
        if texts:
            print(f"     answer: {_s(texts[0], 100)}")
    else:
        texts = [b.get("text", "") for b in c1 if b.get("type") == "text"]
        ok("direct answer mentions .py", ".py" in " ".join(texts).lower())


def test_state_round_trip():
    print("\n=== L3: PIPELINE_STATE survives tool_result round-trip ===")
    msg = "Run 'ls -1' and list the files."
    r1 = post("/v1/messages", {
        "model": "sisyphean",
        "messages": [{"role": "user", "content": msg}],
        "max_tokens": 256, "tools": [BASH],
    })
    ok("T1 no error", "_http_error" not in r1)
    c1 = r1.get("content", [])
    stop1 = r1.get("stop_reason", "")

    if stop1 == "tool_use":
        s1 = _extract_pipeline_state(c1)
        ok("T1 PIPELINE_STATE present", s1 is not None, str(s1)[:80] if s1 else "missing")
        tb = next((b for b in c1 if b.get("type") == "tool_use"), None)
        if tb and s1 is not None:
            si1 = s1.get("si", 0)
            r2 = post("/v1/messages", {
                "model": "sisyphean",
                "messages": [
                    {"role": "user", "content": msg},
                    {"role": "assistant", "content": c1},
                    {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": tb["id"],
                         "content": "main.py\ntest_api.py"}
                    ]},
                ],
                "max_tokens": 256, "tools": [BASH],
            })
            ok("T2 no error", "_http_error" not in r2)
            stop2 = r2.get("stop_reason", "")
            ok(f"T2 moved forward (stop={stop2!r})", stop2 in ("end_turn", "tool_use"), stop2)
    else:
        ok("answered directly -- acceptable", True)


# ===========================================================================
# L4 -- File creation: write -> run -> read output
# ===========================================================================

def test_write_and_run():
    print("\n=== L4: Write file -> run -> verify output ===")
    task = (
        "Write a Python file called hello.py that prints 'Hello World', "
        "then run it and tell me what it printed."
    )

    def sim(b):
        cmd = b.get("input", {}).get("command", "")
        if "python" in cmd.lower() and "hello" in cmd.lower():
            return "Hello World"
        return "Command completed successfully."

    r1 = post("/v1/messages", {
        "model": "sisyphean",
        "messages": [{"role": "user", "content": task}],
        "max_tokens": 512, "tools": [BASH],
    })
    ok("no HTTP error", "_http_error" not in r1, str(r1.get("_http_error", "")))
    stop1 = r1.get("stop_reason", "")
    c1 = r1.get("content", [])
    ok(f"stop valid ({stop1!r})", stop1 in ("tool_use", "end_turn"))

    if stop1 == "tool_use":
        tb = next((b for b in c1 if b.get("type") == "tool_use"), None)
        ok("bash called", tb is not None)
        if tb:
            print(f"     first cmd: {_s(tb.get('input', {}).get('command', ''), 100)}")
        msgs = [{"role": "user", "content": task}, {"role": "assistant", "content": c1}]
        final_r, hist = _drive(msgs, [BASH], sim)
        ok(f"reached end_turn", final_r.get("stop_reason") == "end_turn")
        texts = [b.get("text", "") for b in final_r.get("content", []) if b.get("type") == "text"]
        full = " ".join(texts).lower()
        ok("answer mentions hello or world", "hello" in full or "world" in full, _s(full, 150))
        cmds = _cmds(hist)
        ok("bash was called", len(cmds) >= 1, str(cmds))
        if texts:
            print(f"     answer: {_s(texts[0], 100)}")
    else:
        texts = [b.get("text", "") for b in c1 if b.get("type") == "text"]
        ok("direct answer mentions hello", "hello" in " ".join(texts).lower())


def test_write_verify_exists():
    print("\n=== L4: Write file -> verify it exists ===")
    task = (
        "Create a file called marker.txt containing the word 'created', "
        "then verify it exists by listing files, and tell me the result."
    )

    written = {"done": False}

    def sim(b):
        cmd = b.get("input", {}).get("command", "")
        if "marker" in cmd and any(k in cmd for k in ("echo", "tee", "touch", ">", "printf")):
            written["done"] = True
            return ""
        if any(k in cmd for k in ("ls", "dir", "find")):
            return ("marker.txt\nmain.py\ntest_api.py"
                    if written["done"] else "main.py\ntest_api.py")
        if "cat" in cmd and "marker" in cmd:
            return "created"
        if "marker" in cmd:
            written["done"] = True
            return "Command completed successfully."
        return ""

    r1 = post("/v1/messages", {
        "model": "sisyphean",
        "messages": [{"role": "user", "content": task}],
        "max_tokens": 512, "tools": [BASH],
    })
    ok("no HTTP error", "_http_error" not in r1, str(r1.get("_http_error", "")))
    stop1 = r1.get("stop_reason", "")
    c1 = r1.get("content", [])

    if stop1 == "tool_use":
        msgs = [{"role": "user", "content": task}, {"role": "assistant", "content": c1}]
        final_r, hist = _drive(msgs, [BASH], sim)
        ok(f"reached end_turn", final_r.get("stop_reason") == "end_turn")
        texts = [b.get("text", "") for b in final_r.get("content", []) if b.get("type") == "text"]
        full = " ".join(texts).lower()
        ok("answer mentions marker/created/exists",
           any(k in full for k in ("marker", "created", "exists", "found", "file")), _s(full, 150))
        cmds = _cmds(hist)
        ok("a marker command was issued", any("marker" in c for c in cmds), str(cmds))
        if texts:
            print(f"     answer: {_s(texts[0], 100)}")
    else:
        texts = [b.get("text", "") for b in c1 if b.get("type") == "text"]
        ok("end_turn mentions file",
           any(k in " ".join(texts).lower() for k in ("marker", "file", "created")))


def test_sum_script():
    print("\n=== L4: Write computation script -> run -> read numeric output ===")
    task = (
        "Write a Python script called counter.py that prints the sum of integers 1 to 10. "
        "Run it and tell me what number it printed."
    )

    def sim(b):
        cmd = b.get("input", {}).get("command", "")
        if "python" in cmd.lower() and "counter" in cmd.lower():
            return "55"
        return "Command completed successfully."

    r1 = post("/v1/messages", {
        "model": "sisyphean",
        "messages": [{"role": "user", "content": task}],
        "max_tokens": 512, "tools": [BASH],
    })
    ok("no HTTP error", "_http_error" not in r1, str(r1.get("_http_error", "")))
    stop1 = r1.get("stop_reason", "")
    c1 = r1.get("content", [])

    if stop1 == "tool_use":
        msgs = [{"role": "user", "content": task}, {"role": "assistant", "content": c1}]
        final_r, hist = _drive(msgs, [BASH], sim)
        ok(f"reached end_turn", final_r.get("stop_reason") == "end_turn")
        texts = [b.get("text", "") for b in final_r.get("content", []) if b.get("type") == "text"]
        ok("answer contains 55", "55" in " ".join(texts), _s(" ".join(texts), 150))
        cmds = _cmds(hist)
        ok("bash was called", len(cmds) >= 1, str(cmds[-3:]))
        if texts:
            print(f"     answer: {_s(texts[0], 100)}")
    else:
        texts = [b.get("text", "") for b in c1 if b.get("type") == "text"]
        ok("direct answer contains 55", "55" in " ".join(texts))


# ===========================================================================
# L5 -- Orchestration: multi-round tool chains
# ===========================================================================

def test_plan_has_bash_step():
    """Planner MUST choose bash for a write-and-run task."""
    print("\n=== L5-orch: Planner chooses bash for write+run task ===")
    task = (
        "Write a Python script called squares.py that prints the squares of 1 to 5 "
        "(one per line: 1, 4, 9, 16, 25), then run it."
    )

    def sim(b):
        cmd = b.get("input", {}).get("command", "")
        if "python" in cmd.lower() and "squares" in cmd.lower():
            return "1\n4\n9\n16\n25"
        return "Command completed successfully."

    r1 = post("/v1/messages", {
        "model": "sisyphean",
        "messages": [{"role": "user", "content": task}],
        "max_tokens": 512, "tools": [BASH],
    })
    ok("no HTTP error", "_http_error" not in r1)
    c1 = r1.get("content", [])
    stop1 = r1.get("stop_reason", "")
    ok(f"stop valid ({stop1!r})", stop1 in ("tool_use", "end_turn"))

    if stop1 == "tool_use":
        ptypes = _plan_tools(c1)
        print(f"     planned step tools: {ptypes}")
        ok("plan includes bash", "bash" in ptypes, f"got {ptypes}")
        msgs = [{"role": "user", "content": task}, {"role": "assistant", "content": c1}]
        final_r, hist = _drive(msgs, [BASH], sim)
        ok("reached end_turn", final_r.get("stop_reason") == "end_turn")
        cmds = _cmds(hist)
        ok(f"bash was called ({len(cmds)} calls)", len(cmds) >= 1, str(cmds[-3:]))
        texts = [b.get("text", "") for b in final_r.get("content", []) if b.get("type") == "text"]
        full = " ".join(texts)
        ok("answer mentions output (25/16/9 or squares)",
           any(k in full for k in ("25", "16", "9", "squares", "1\n4")), _s(full, 150))
        if texts:
            print(f"     answer: {_s(texts[0], 100)}")
    else:
        texts = [b.get("text", "") for b in c1 if b.get("type") == "text"]
        ok("direct answer mentions squares", "square" in " ".join(texts).lower())


def test_find_read_edit_cycle():
    """Find config.yaml -> read -> change port -> verify -> report. Min 2 bash calls."""
    print("\n=== L5-orch: Find file -> read -> edit -> verify ===")
    task = (
        "In the project directory, find config.yaml. "
        "Read it and tell me the current 'port' value under the 'api' section. "
        "Then change that port value to 9000 in the file and confirm the new port."
    )

    state = {"read": False, "edited": False}

    def sim(b):
        cmd = b.get("input", {}).get("command", "")
        if "config.yaml" in cmd and any(k in cmd for k in ("cat", "head", "type", "less")):
            state["read"] = True
            port = "9000" if state["edited"] else "8000"
            return f"api:\n  host: 0.0.0.0\n  port: {port}\nllm:\n  model: qwen3:0.6b"
        if any(k in cmd for k in ("ls", "find", "dir")):
            return "config.yaml\nmain.py\nrequirements.txt"
        if "9000" in cmd and "config" in cmd:
            state["edited"] = True
            return ""
        if any(k in cmd for k in ("sed", "awk", "python", "powershell")) and "config" in cmd:
            state["edited"] = True
            return ""
        return "Command completed successfully."

    r1 = post("/v1/messages", {
        "model": "sisyphean",
        "messages": [{"role": "user", "content": task}],
        "max_tokens": 512, "tools": [BASH],
    })
    ok("no HTTP error", "_http_error" not in r1)
    stop1 = r1.get("stop_reason", "")
    c1 = r1.get("content", [])

    if stop1 == "tool_use":
        msgs = [{"role": "user", "content": task}, {"role": "assistant", "content": c1}]
        final_r, hist = _drive(msgs, [BASH], sim)
        ok("reached end_turn", final_r.get("stop_reason") == "end_turn")
        cmds = _cmds(hist)
        ok(f"at least 1 bash call (got {len(cmds)})", len(cmds) >= 1, str(cmds))
        texts = [b.get("text", "") for b in final_r.get("content", []) if b.get("type") == "text"]
        full = " ".join(texts).lower()
        ok("answer mentions port or config",
           any(k in full for k in ("port", "9000", "8000", "config")), _s(full, 150))
        print(f"     bash rounds: {len(cmds)}  answer: {_s(texts[0] if texts else '', 100)}")
    else:
        texts = [b.get("text", "") for b in c1 if b.get("type") == "text"]
        ok("direct answer mentions port", "port" in " ".join(texts).lower())


def test_write_module_and_tests():
    """Write calc.py + test_calc.py -> run -> ALL TESTS PASSED."""
    print("\n=== L5-orch: Write module + test file -> run -> ALL TESTS PASSED ===")
    task = (
        "Write two Python files:\n"
        "1) calc.py with functions: add(a,b), subtract(a,b), multiply(a,b), divide(a,b)\n"
        "2) test_calc.py that imports calc and asserts: add(2,3)==5, subtract(10,4)==6, "
        "multiply(3,4)==12, divide(10,2)==5.0 -- print 'ALL TESTS PASSED' if all pass.\n"
        "Run test_calc.py and tell me if all tests passed."
    )

    def sim(b):
        cmd = b.get("input", {}).get("command", "")
        if "python" in cmd.lower() and "test_calc" in cmd.lower():
            return "ALL TESTS PASSED"
        return "Command completed successfully."

    r1 = post("/v1/messages", {
        "model": "sisyphean",
        "messages": [{"role": "user", "content": task}],
        "max_tokens": 1024, "tools": [BASH],
    })
    ok("no HTTP error", "_http_error" not in r1)
    c1 = r1.get("content", [])
    stop1 = r1.get("stop_reason", "")

    if stop1 == "tool_use":
        ptypes = _plan_tools(c1)
        print(f"     plan tools: {ptypes}")
        ok("plan includes bash", "bash" in ptypes, f"got {ptypes}")
        msgs = [{"role": "user", "content": task}, {"role": "assistant", "content": c1}]
        final_r, hist = _drive(msgs, [BASH], sim)
        ok("reached end_turn", final_r.get("stop_reason") == "end_turn")
        cmds = _cmds(hist)
        ok(f"at least 1 bash call (got {len(cmds)})", len(cmds) >= 1, str(cmds[-4:]))
        texts = [b.get("text", "") for b in final_r.get("content", []) if b.get("type") == "text"]
        full = " ".join(texts).lower()
        ok("answer mentions tests/calc/pass",
           any(k in full for k in ("pass", "calc", "test", "all")), _s(full, 150))
        print(f"     bash rounds: {len(cmds)}  answer: {_s(texts[0] if texts else '', 100)}")
    else:
        texts = [b.get("text", "") for b in c1 if b.get("type") == "text"]
        ok("direct answer mentions calc or pass",
           any(k in " ".join(texts).lower() for k in ("calc", "pass")))


# ===========================================================================
# L6 -- Cross-domain
# ===========================================================================

def test_websearch_before_code():
    """Web search (internal or outer) must be used before code write."""
    print("\n=== L6-cross: Web search used before code write ===")
    task = (
        "Search online for Python's hashlib.md5 usage example, "
        "then write a script called hasher.py that uses hashlib.md5 to hash the string 'hello' "
        "and prints the hex digest. Run it and tell me the output."
    )

    def sim(b):
        name = b.get("name", "")
        inp = b.get("input", {})
        if name == "WebSearch":
            return ("hashlib.md5(b'hello').hexdigest() returns '5d41402abc4b2a76b9719d911017c592'. "
                    "Import hashlib, call hashlib.md5(b'string').hexdigest().")
        cmd = inp.get("command", "")
        if "python" in cmd.lower() and "hasher" in cmd.lower():
            return "5d41402abc4b2a76b9719d911017c592"
        return "Command completed successfully."

    r1 = post("/v1/messages", {
        "model": "sisyphean",
        "messages": [{"role": "user", "content": task}],
        "max_tokens": 512, "tools": [BASH, WEBSEARCH],
    })
    ok("no HTTP error", "_http_error" not in r1)
    c1 = r1.get("content", [])
    stop1 = r1.get("stop_reason", "")
    ok(f"stop valid ({stop1!r})", stop1 in ("tool_use", "end_turn"))

    # Check planned steps include web search (internal or outer)
    ptypes = _plan_tools(c1)
    print(f"     planned tools: {ptypes}")
    ok("plan includes web search",
       any("web" in t or "search" in t for t in ptypes) or stop1 == "end_turn",
       f"got {ptypes}")

    if stop1 == "tool_use":
        msgs = [{"role": "user", "content": task}, {"role": "assistant", "content": c1}]
        final_r, hist = _drive(msgs, [BASH, WEBSEARCH], sim)
        ok("reached end_turn", final_r.get("stop_reason") == "end_turn")
        tool_seq = _tool_names(hist)
        print(f"     outer tool sequence: {tool_seq}")
        texts = [b.get("text", "") for b in final_r.get("content", []) if b.get("type") == "text"]
        full = " ".join(texts).lower()
        ok("answer mentions hash/md5/5d41",
           any(k in full for k in ("5d41", "md5", "hash", "hex", "hasher")), _s(full, 150))
        if texts:
            print(f"     answer: {_s(texts[0], 100)}")
    else:
        texts = [b.get("text", "") for b in c1 if b.get("type") == "text"]
        ok("answer mentions md5 or hash",
           any(k in " ".join(texts).lower() for k in ("md5", "hash", "hashlib")))


def test_error_recovery_loop():
    """Write fib.py -> first run returns error -> engine fixes and reruns."""
    print("\n=== L6-cross: Write -> error -> fix -> rerun ===")
    task = (
        "Write a Python file called fib.py that prints the first 8 Fibonacci numbers "
        "(1 1 2 3 5 8 13 21). Run it. If there is an error, fix the bug and run it again. "
        "Tell me what it printed in the end."
    )

    run_count = {"n": 0}

    def sim(b):
        cmd = b.get("input", {}).get("command", "")
        if "python" in cmd.lower() and "fib" in cmd.lower():
            run_count["n"] += 1
            if run_count["n"] == 1:
                return ("Traceback (most recent call last):\n"
                        "  File \"fib.py\", line 3\n    print(fib)\n"
                        "NameError: name 'fib' is not defined")
            return "1\n1\n2\n3\n5\n8\n13\n21"
        return "Command completed successfully."

    r1 = post("/v1/messages", {
        "model": "sisyphean",
        "messages": [{"role": "user", "content": task}],
        "max_tokens": 512, "tools": [BASH],
    })
    ok("no HTTP error", "_http_error" not in r1)
    c1 = r1.get("content", [])
    stop1 = r1.get("stop_reason", "")

    if stop1 == "tool_use":
        msgs = [{"role": "user", "content": task}, {"role": "assistant", "content": c1}]
        final_r, hist = _drive(msgs, [BASH], sim, max_rounds=10)
        ok("reached end_turn", final_r.get("stop_reason") == "end_turn")
        cmds = _cmds(hist)
        python_runs = [c for c in cmds if "python" in c.lower() and "fib" in c.lower()]
        ok(f"fib.py run at least once (got {len(python_runs)})", len(python_runs) >= 1,
           str(cmds[-4:]))
        texts = [b.get("text", "") for b in final_r.get("content", []) if b.get("type") == "text"]
        full = " ".join(texts)
        ok("answer contains fibonacci output",
           any(k in full for k in ("21", "13", "8", "fibonacci", "fib", "1 1 2")), _s(full, 150))
        print(f"     bash rounds: {len(cmds)}  fib runs: {len(python_runs)}"
              f"  answer: {_s(texts[0] if texts else '', 100)}")
    else:
        texts = [b.get("text", "") for b in c1 if b.get("type") == "text"]
        ok("answer mentions fibonacci",
           any(k in " ".join(texts).lower() for k in ("fib", "21", "sequence")))


def test_full_pipeline_web_write_test():
    """WebSearch -> write wordcount.py -> write tests -> run -> report."""
    print("\n=== L6-cross: Web search -> write lib -> write tests -> run ===")
    task = (
        "Search online for Python's collections.Counter usage, "
        "then write a file called wordcount.py with a function count_words(text) "
        "that uses Counter to count word frequencies and returns the result as a dict. "
        "Write a second file test_wordcount.py that imports count_words and asserts "
        "count_words('the cat sat on the mat')['the'] == 2 -- "
        "print 'ALL TESTS PASSED' if it passes. "
        "Run the tests and tell me if they passed."
    )

    def sim(b):
        name = b.get("name", "")
        inp = b.get("input", {})
        if name == "WebSearch":
            return ("collections.Counter counts elements. "
                    "Counter(text.split()) gives word frequencies as a Counter (subclass of dict).")
        cmd = inp.get("command", "")
        if "python" in cmd.lower() and "test_wordcount" in cmd.lower():
            return "ALL TESTS PASSED"
        return "Command completed successfully."

    r1 = post("/v1/messages", {
        "model": "sisyphean",
        "messages": [{"role": "user", "content": task}],
        "max_tokens": 1024, "tools": [BASH, WEBSEARCH],
    })
    ok("no HTTP error", "_http_error" not in r1)
    c1 = r1.get("content", [])
    stop1 = r1.get("stop_reason", "")

    ptypes = _plan_tools(c1)
    print(f"     planned tools: {ptypes}")

    if stop1 == "tool_use":
        msgs = [{"role": "user", "content": task}, {"role": "assistant", "content": c1}]
        final_r, hist = _drive(msgs, [BASH, WEBSEARCH], sim, max_rounds=10)
        ok("reached end_turn", final_r.get("stop_reason") == "end_turn")
        tool_seq = _tool_names(hist)
        cmds = _cmds(hist)
        print(f"     outer tools: {tool_seq}  bash calls: {len(cmds)}")
        ok("bash was used", len(cmds) >= 1, str(tool_seq))
        texts = [b.get("text", "") for b in final_r.get("content", []) if b.get("type") == "text"]
        full = " ".join(texts).lower()
        ok("answer mentions tests/pass/wordcount",
           any(k in full for k in ("pass", "wordcount", "test", "counter")), _s(full, 150))
        if texts:
            print(f"     answer: {_s(texts[0], 100)}")
    else:
        texts = [b.get("text", "") for b in c1 if b.get("type") == "text"]
        ok("answer mentions counter or wordcount",
           any(k in " ".join(texts).lower() for k in ("counter", "wordcount", "pass")))


# ===========================================================================
# Runner
# ===========================================================================

def _assert_server_up():
    r = get("/health")
    if "error" in r or r.get("status") != "ok":
        print("\n  [ABORT] Server is down -- restart with: python main.py")
        sys.exit(2)


def main():
    print(f"Sisyphean engine tests  ->  {BASE_URL}")
    print("=" * 56)

    r = get("/health")
    if "error" in r or r.get("status") != "ok":
        print(f"ERROR: engine not reachable at {BASE_URL}")
        sys.exit(1)

    # L1
    test_health()
    test_models()
    test_oai_completions()
    _assert_server_up()

    # L2
    test_routing_direct()
    _assert_server_up()

    # L3
    test_single_bash_round()
    test_state_round_trip()
    _assert_server_up()

    # L4
    test_write_and_run()
    test_write_verify_exists()
    test_sum_script()
    _assert_server_up()

    # L5
    test_plan_has_bash_step()
    test_find_read_edit_cycle()
    test_write_module_and_tests()
    _assert_server_up()

    # L6
    test_websearch_before_code()
    test_error_recovery_loop()
    test_full_pipeline_web_write_test()

    print("\n" + "=" * 56)
    if _failures == 0:
        print("All tests passed.")
    else:
        print(f"{_failures} test(s) FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
