# Sisyphean

A local AI agent engine that runs as a background service. It exposes both the Anthropic Messages API (`/v1/messages`) and OpenAI Chat Completions (`/v1/chat/completions`), making it a drop-in model provider for Claude Code, BirdClaw, and any OpenAI-compatible client.

Sisyphean is the **engine** — stateless, reasoning-focused, no personality. Pair it with [BirdClaw](https://github.com/sangharshadhyeta/BirdClaw) for a full autonomous agent with soul, dreaming, and a persistent web UI.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# First-time setup (LLM backend, port, workspace)
python main.py setup

# Start the engine
python main.py

# Or start with the system tray icon (recommended on Windows)
python main.py tray
```

Engine runs at `http://127.0.0.1:47291` by default.

---

## Starting BirdClaw

If you have [BirdClaw](https://github.com/sangharshadhyeta/BirdClaw) installed alongside Sisyphean,
you can start it directly from here:

```bash
# Start Sisyphean first (if not already running)
python main.py tray

# Then in another terminal, launch BirdClaw web UI:
python main.py launch birdclaw
```

`launch birdclaw` will:
1. Check if BirdClaw is already running on port 47293
2. If not, auto-detect the BirdClaw directory (`../BirdClaw` next to Sisyphean) and start it
3. Open your browser to `http://127.0.0.1:47293/`

> **Install BirdClaw** (if not already):
> ```bash
> git clone https://github.com/sangharshadhyeta/BirdClaw ../BirdClaw
> cd ../BirdClaw && pip install -e .
> ```

---

## Claude Code Integration

Add Sisyphean as a custom model in Claude Code's settings:

```json
{
  "customApiUrl": "http://127.0.0.1:47291",
  "customApiKey": "local"
}
```

Claude Code sends requests to `/v1/messages` (Anthropic format). Sisyphean routes them through its agent loop and returns `tool_use` / `end_turn` responses.

Or launch Claude Code via Sisyphean (sets `ANTHROPIC_BASE_URL` automatically):

```bash
python main.py launch claude
```

---

## Configuration

Edit `config.yaml` (created by `python main.py setup`):

```yaml
llm:
  local_model: "qwen3:0.6b"    # Ollama model name (if using Ollama)
  server:
    port: 11434                  # Ollama port

  external_api:                  # Use OpenRouter, LM Studio, llama.cpp, etc.
    enabled: true
    base_url: "http://192.168.1.x:8081/v1"
    api_key: "local"
    model: "gemma-4-E4B-it-Q8_0.gguf"

api:
  host: 127.0.0.1
  port: 47291

mock: false                      # true = no model needed (for UI/API testing)
workspace: ./workspace           # sandboxed directory; agent writes only here
```

---

## Architecture

```
main.py                  Entry point — uvicorn + optional llama-server subprocess
tray.py                  Windows system tray watchdog (start/stop/restart/update)

engine/
  api/app.py             FastAPI app factory — all routes
  config.py              Config dataclasses + YAML loader
  permissions.py         Permission guard — enforces workspace boundary

  compat/
    anthropic.py         POST /v1/messages (Anthropic Messages API)
    openai_compat.py     POST /v1/chat/completions (OpenAI compat passthrough)

  llm/
    client.py            LlamaClient — async HTTP client (Ollama / llama.cpp / external)
    context.py           ContextManager — token-aware sliding window

  translation/           Core agent loop
    loop.py              TranslationLoop — process() → _micro_loop() (up to 12 steps)
    executor.py          decide() — single LLM call, unified action menu
    decomposer.py        decompose() — breaks task into typed stage steps
    manifest.py          TaskManifest / Instruction dataclasses
    prompts.py           SYSTEM prompt, engine policy sections, per-stage action menus
    web_search.py        DuckDuckGo search + HTML fetch + condenser
    subtask/             Activated for write_code / write_doc plan steps
      planner.py         LLM call → named items (function names / section headings)
      manifest.py        SubtaskManifest / SubtaskItem status tracking
      writer.py          Item-by-item write with MAX_ITEM_RETRIES=2
      verifier.py        Pure-regex scoring (complete/partial/missing/regressed)

  memory/
    graph.py             NetworkX knowledge graph (shared with BirdClaw)
    store.py             JSONL artifact store → memory/artifacts.jsonl
    session_log.py       Per-session conversation log (used by semantic history)
    injector.py          Injects graph context into LLM requests
    extractor.py         Extracts facts from LLM responses
    cleanup.py           Retention policy — prunes old sessions/artifacts

  core/
    pipeline.py          Task execution pipeline
    recall.py            Memory recall / retrieval

  activity.py            Recent events tracking
  task_tracker.py        Active task management

engine_policy.md         Reasoning discipline injected into every LLM call
workspace/               Default sandboxed directory; agent may only write here
```

---

## Agent Loop

Every `/v1/messages` request enters `TranslationLoop.process()`:

1. **Continuation check** — if the previous message has `tool_result` blocks, decode `SISYPHEAN_STATE:<base64-json>` from a thinking block and resume `MicroState` (step count, internal messages, summary, pending write steps).
2. **`_micro_loop()`** — up to 12 inner steps:
   - If a `write_code`/`write_doc` step is queued in `pending_write_steps`, dispatch to the subtask pipeline (planner → writer → verifier).
   - Otherwise call `decide()` — one LLM call returns an action.
   - **Internal tools** handled inside the loop: `plan_task`, `search_knowledge`, `search_history`, `save_memory`, `web_search`, `list_workspace`, `read_file`.
   - **Outer tools** (e.g., `bash`) returned as `tool_use` blocks to the caller; current state encoded in a thinking block for the next turn.
3. **Forced answer** — if MAX_STEPS exceeded, a final synthesis call produces the answer.

No stage gating — the model decides freely at every step. All actions always available: think, search_memory, search_history, web_search, list_workspace, read_file, save_memory, bash, answer.

Stall guard blocks duplicate (tool, input) calls. Semantic history uses Jaccard similarity ≥ 0.15 over session logs with one LLM summary call per outer turn.

---

## Internal Tools

| Tool | When to use |
|------|-------------|
| `plan_task` | Complex multi-step tasks — produces typed stage list |
| `list_workspace` | Before mkdir or file writes — shows what exists |
| `read_file` | Before modifying a file — reads relevant portion |
| `search_memory` | Domain knowledge or past research from the graph |
| `search_history` | What was done in previous sessions |
| `save_memory` | Persist a fact or preference to the knowledge graph |
| `web_search` | Current information not in model knowledge |

---

## Memory

Sisyphean owns the single shared knowledge graph used by both Sisyphean and BirdClaw:

- **Knowledge graph** (`~/.sisyphean/memory/knowledge_graph.json`) — NetworkX graph of facts, concepts, and entities extracted during task execution. BirdClaw reads from and writes to this same file.
- **Artifact store** (`memory/artifacts.jsonl`) — JSONL log of significant outputs.
- **Session log** — per-session JSONL conversation log used by semantic history.

> Memory consolidation (dreaming) lives in **BirdClaw**, not in Sisyphean. Sisyphean's `MemoryExtractor` populates the graph during task execution; BirdClaw's dream cycle enriches it overnight.

---

## System Tray

```bash
python main.py tray
```

Launches the engine in the background and shows a tray icon in the Windows notification area. Right-click for Start / Stop / Restart / View Logs / Update (git pull + restart).

> **If the icon doesn't appear:** Windows hides new tray icons in the overflow area. Click the `^` arrow in the taskbar notification area and drag the Sisyphean icon to the visible bar, or go to **Settings → Personalisation → Taskbar → Other system tray icons** and enable it.

The tray includes a watchdog that automatically restarts the engine if it crashes.

---

## Dashboard

Browse to `http://127.0.0.1:47291/dashboard` for a live status page showing uptime, graph nodes, artifact count, active tasks, and recent events.

---

## Commands

```bash
python main.py              # start engine
python main.py tray         # Windows tray watchdog (recommended)
python main.py setup        # first-time setup wizard
python main.py config       # re-run setup to change settings
python main.py launch birdclaw   # start BirdClaw web UI + open browser
python main.py launch claude     # start Claude Code pointed at Sisyphean
```

---

## Development

```bash
# Mock mode — no model needed, useful for testing API shape
# Set mock: true in config.yaml, then:
python main.py

# Run tests (engine must be running on port 47291)
python tests/test_regimen.py
python tests/test_api.py
python tests/test_openclaw.py

# Note: tests hardcode port 8000 — update BASE_URL or temporarily change
# config.yaml api.port to 8000 before running.
```

---

## Roadmap

- [ ] Switch to llama.cpp + gemma4 e4b (better reasoning on consumer GPU)
- [ ] Telegram / Discord channel adapters
- [ ] PC control (screenshot, mouse/keyboard) via BirdClaw tool bridge
- [ ] Self-modification (agent edits own source, runs tests, commits)
