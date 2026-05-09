# Sisyphean

A local AI agent that runs as a background service and integrates with Claude Code as a custom model provider. Built on small local models (currently qwen3:0.6b via Ollama, targeting gemma4 e4b via llama.cpp).

Exposes both Anthropic Messages API (`/v1/messages`) and OpenAI Chat Completions (`/v1/chat/completions`) so it works as a drop-in model for Claude Code and any OpenAI-compatible client.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the engine (Ollama must be running with your model pulled)
python main.py

# Or launch with the system tray manager
python main.py tray

# Run offline memory consolidation
python main.py dream
```

Engine runs at `http://127.0.0.1:8000` by default.

---

## Configuration

Edit `config.yaml`:

```yaml
llm:
  local_model: "qwen3:0.6b"    # Ollama model name
  server:
    port: 11434                  # Ollama port

api:
  host: 127.0.0.1
  port: 8000

mock: false                      # true = no model needed (for testing)
workspace: ./workspace           # where the agent writes files
```

To use an external API instead of a local model:

```yaml
llm:
  external_api:
    enabled: true
    base_url: "https://openrouter.ai/api/v1"
    api_key: "sk-..."
    model: "google/gemma-3-4b-it:free"
```

---

## Claude Code Integration

Add Sisyphean as a custom model in Claude Code's settings:

```json
{
  "customApiUrl": "http://127.0.0.1:8000",
  "customApiKey": "local"
}
```

Claude Code sends requests to `/v1/messages` (Anthropic format). Sisyphean translates them through its translation loop and returns tool_use / end_turn responses.

---

## Architecture

```
main.py            Entry point — starts uvicorn + manages llama-server subprocess
tray.py            System tray manager — watchdog, start/stop/restart from taskbar

engine/
  api/app.py       FastAPI app — mounts all routes
  compat/
    anthropic.py   POST /v1/messages  — Anthropic Messages API handler
    openai_compat.py  POST /v1/chat/completions — OpenAI compat passthrough
  llm/
    client.py      LlamaClient — async HTTP client for Ollama / llama.cpp / external API
    context.py     ContextManager — token-aware sliding window
  translation/
    loop.py        TranslationLoop — the main agent loop (MicroState, _micro_loop)
    executor.py    decide() — single-step LLM decision (plan/execute/answer)
    decomposer.py  decompose() — breaks task into typed stage steps
    manifest.py    TaskManifest / Instruction — stage queue dataclasses
    prompts.py     SYSTEM prompt, soul sections, stage action prompts
    web_search.py  DuckDuckGo search + HTML fetch + condenser
    subtask/
      planner.py   plan() — breaks write goal into named items (functions/sections)
      manifest.py  SubtaskManifest / SubtaskItem — per-item status tracking
      verifier.py  run() — parses written file, scores each item (pure regex)
      writer.py    run_write_step() — item-by-item write with verify + retry
  memory/
    graph.py       GraphStore — NetworkX knowledge graph (facts, concepts, entities)
    store.py       Artifact store — JSONL artifact log
    session_log.py Session log — per-session conversation history
    injector.py    MemoryInjector — injects graph context into requests
    extractor.py   MemoryExtractor — extracts facts from responses
    dream.py       Dream cycle — graph merge + reflection + cleanup
    cleanup.py     Retention policy — prunes old sessions and artifacts
  permissions.py   Permission guard — protects workspace boundaries
  config.py        Config dataclasses + YAML loader
```

---

## Agent Loop

Every request goes through `TranslationLoop.process()`:

1. **Continuation check** — if the last user message has `tool_result` blocks, extract `MicroState` from the preceding thinking block and resume
2. **`_micro_loop()`** — runs up to `MAX_STEPS=12` inner steps:
   - **Write stage dispatch** — if a `write_code`/`write_doc` step is queued from the plan, runs the subtask pipeline instead of `decide()`
   - **`decide()`** — one LLM call that returns an action (plan, bash, search, answer, etc.)
   - Internal tools (plan_task, search_knowledge, search_history, save_memory, web_search, list_workspace, read_file) are handled inside the loop
   - Outer tools (Bash) are returned to Claude Code as `tool_use` blocks; state is encoded in a `SISYPHEAN_STATE` thinking block
3. **Forced answer** — if MAX_STEPS exceeded, a final synthesis call produces an answer

**Stage detection** — `_detect_stage()` reads `internal_messages` to determine:
- `plan` — first step, no plan yet → offers plan_task + answer
- `execute` — plan exists → offers all tools
- `answer` — budget almost exhausted → forces answer only

**Semantic history** — rather than sending raw conversation history, one LLM call per outer turn summarises relevant past exchanges (Jaccard similarity ≥ 0.15 threshold). Injected as `[Relevant prior work]` block.

**Stall guard** — duplicate internal tool calls (same tool + same input) are blocked after first hit; model gets a nudge toward answering.

---

## Subtask Pipeline (write_code / write_doc)

When the decomposer produces a `write_code` or `write_doc` step:

1. **Planner** (`subtask/planner.py`) — one LLM call → list of named items (function names for code, heading names for docs). Vague output triggers an automatic retry with a stricter prompt.
2. **Writer** (`subtask/writer.py`) — iterates items one at a time:
   - Builds a focused write prompt: goal + completed items + file tail + what to write next
   - LLM outputs raw text; written/appended to file
   - Verifier scores the result
   - Up to 2 retries per item with gap analysis as context
3. **Verifier** (`subtask/verifier.py`) — pure regex; parses `def`/`class` blocks (code) or `##` headings (docs), scores each manifest item as complete/partial/missing/regressed

---

## Internal Tools

| Tool | When to use |
|------|-------------|
| `plan_task` | Complex multi-step tasks — produces typed stage list |
| `list_workspace` | Before mkdir or file writes — shows what exists |
| `read_file` | Before modifying a file — reads relevant portion (query-scoped) |
| `search_memory` | Domain knowledge or past research from the graph |
| `search_history` | What was done in previous sessions |
| `save_memory` | Persist a fact or preference to the knowledge graph |
| `web_search` | Current information not in model knowledge |

---

## Memory

- **Knowledge graph** (`memory/graph.py`) — NetworkX graph of facts, concepts, and entities. Persisted as `memory/graph.json`.
- **Artifact store** (`memory/store.py`) — JSONL log of significant outputs. Persisted as `memory/artifacts.jsonl`.
- **Session log** (`memory/session_log.py`) — per-session JSONL conversation log. Used by semantic history.
- **Dream cycle** (`python main.py dream`) — consolidates session logs into the graph, runs reflection, and prunes stale entries.

---

## System Tray

```bash
python main.py tray
```

Launches the engine in the background and shows a tray icon in the Windows notification area. Right-click for Start / Stop / Restart / View Logs / Update.

> **If the icon doesn't appear:** Windows hides new tray icons in the overflow area. Click the `^` arrow in the taskbar notification area and drag the Sisyphean icon to the visible bar, or go to **Settings → Personalisation → Taskbar → Other system tray icons** and enable it.

The tray includes a watchdog that automatically restarts the engine if it crashes.

---

## Development

```bash
# Run tests (requires engine running on :8000)
python test_api.py

# Mock mode — no model needed
# Set mock: true in config.yaml, then:
python main.py
```

Disable the semantic history LLM call during testing to speed things up:

```python
# In loop.py, _semantic_history_summary returns "" immediately
```

---

## Roadmap

- [ ] Switch to llama.cpp + gemma4 e4b (4B model for better reasoning)
- [ ] Telegram / Discord channel adapters
- [ ] React web frontend (replace direct API calls)
- [ ] PC control (screenshot, mouse/keyboard)
- [ ] Self-modification (agent edits own source, runs tests, commits)
- [ ] Dream cycle improvements — auto-ingest research stage findings
