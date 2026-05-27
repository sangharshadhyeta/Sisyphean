<p align="center">
  <img src="https://raw.githubusercontent.com/sangharshadhyeta/Sisyphean/master/assets/sisyphean.png" alt="Sisyphean Logo" width="90" height="90" onerror="this.style.display='none'" />
</p>

<h1 align="center">Sisyphean</h1>

<p align="center">
  <strong>Universal local AI engine — Anthropic + OpenAI compatible API.</strong><br/>
  Reasoning · Planning · Tool execution · Web search — works with Claude Code and BirdClaw.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/compatible-Anthropic%20%7C%20OpenAI-blue?style=flat-square" />
  <img src="https://img.shields.io/badge/runs-locally-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/works%20with-BirdClaw-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" />
</p>

---

A local AI agent engine that runs as a persistent background service. It exposes the Anthropic Messages API (`/v1/messages`) and OpenAI Chat Completions (`/v1/chat/completions`), making it a drop-in model provider for Claude Code, BirdClaw, and any OpenAI-compatible client.

Sisyphean is a **full agent** — it has a personality (`soul.md`), a persistent knowledge graph, a bash execution environment, a web search pipeline, a skills library, and a dream cycle for offline memory consolidation. It is not a dumb proxy. Every request passes through a multi-step reasoning loop that can search its own memory, invoke skills, run code, fetch the web, and write files before answering.

---

## Validated on gemma4:latest (Ollama)

All agent capabilities — routing, tool chaining, graph memory injection, web search, bash execution, skill dispatch, and subtask writing — have been validated against `gemma4:latest` served locally via **Ollama**. This is a 9.6 GB model running entirely offline. Sisyphean is specifically designed to get reliable agentic behaviour out of local models by routing decisions through structured prompts, progressive context disclosure, and fallback retries rather than trusting the model to free-form reason.

External API mode (llama.cpp, OpenRouter, Groq, Google AI Studio) is also supported — see [Configuration](#configuration).

---

## Quick Start

> **BirdClaw users:** run `BirdClaw/install.bat` instead — it installs Sisyphean, SearXNG, and BirdClaw together as a single stack. Come back here only if you want Sisyphean standalone.

```bash
# Install dependencies
pip install -r requirements.txt

# Start the engine (Ollama must be running with your model pulled)
python main.py

# Or start with the Windows system tray watchdog (recommended)
python main.py tray
```

Engine runs at `http://127.0.0.1:47291` by default.

To pull the validated model via Ollama:
```bash
ollama pull gemma4:latest
```

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

Or launch Claude Code with the base URL pre-set:

```bash
python main.py launch claude
```

---

## Configuration

Edit `config.yaml`:

```yaml
llm:
  local_model: "gemma4:latest"   # Ollama model name (recommended)

  # ── Alternative: external OpenAI-compatible API ────────────────────
  # Use llama.cpp, LM Studio, OpenRouter, Groq, or Google AI Studio.
  external_api:
    enabled: false
    base_url: "http://192.168.1.x:8081/v1"
    api_key: "local"
    model: "gemma-4-E4B-it-Q8_0.gguf"   # or any model name

api:
  host: 127.0.0.1
  port: 47291

search:
  searxng_url: "http://localhost:8888"   # set to "" to disable (falls back to Jina)

mock: false                    # true = no model needed (for API/UI testing)
workspace: ./workspace         # sandboxed directory; agent writes only here
skills_path: ./skills          # permanent skill scripts library
```

> **Web search tiers:** SearXNG (local instance, port 8888) → Jina AI reader (public, no key required). BirdClaw's `install.bat` sets up SearXNG automatically.

> **Free cloud tiers that work out of the box:** [OpenRouter](https://openrouter.ai) (`google/gemma-3-4b-it:free`), [Groq](https://console.groq.com) (`gemma2-9b-it`), [Google AI Studio](https://aistudio.google.com) (`gemini-2.0-flash-lite`).

---

## Design Philosophy — Progressive Disclosure

Every LLM call in Sisyphean receives the **minimum context required to make one correct decision**. Call count scales with task complexity, not with a fixed budget.

| Stage | What the model sees |
|-------|-------------------|
| **Route** | Query + skill name list only. No history, no graph. Classifies intent in one call. |
| **Decompose** | Query + 400 chars task context + 400 chars graph excerpt. Produces typed stage list. |
| **Plan** | Stage goal + skill index + 600 chars graph. Decides steps for one stage. |
| **Execute** | Full stage context including tool results from prior steps. |
| **Synthesise** | Accumulated results. Produces the final answer. |

This means a simple greeting resolves in 1–2 calls; a multi-file coding task might use 8–12. The token budget per call is small and predictable, which is what keeps small local models (4–9B parameters) reliable.

### Need-to-Know Context Guards

Two filters enforce the principle at the skill index level:

**Calc skill filter** — The `calc` skill is only shown to the model when the query contains digits or math operators (`0–9 + - * / ^ % ( )`). Without this, small models sometimes select `calc` for queries like "latest Python version" or "create a folder" because the pattern match on "Python" fools them.

**Placeholder guard** — When the model echoes back a template token (e.g. `EXPRESSION`, `QUERY`, `ARGS`) as the argument to `run_skill`, the match is discarded. This prevents template noise from being dispatched as a real skill invocation.

---

## Architecture

```
main.py                  Entry point — uvicorn + optional Ollama/llama-server subprocess
tray.py                  Windows system tray watchdog (start/stop/restart/update)
soul.md                  Agent personality — edit freely, live-synced on every request

engine/
  api/app.py             FastAPI app factory — routes + mtime-based personality sync
  config.py              Config dataclasses + YAML loader
  permissions.py         Permission guard — enforces workspace boundary

  compat/
    anthropic.py         POST /v1/messages (Anthropic Messages API handler)
    openai_compat.py     POST /v1/chat/completions (OpenAI compat passthrough)

  llm/
    client.py            LlamaClient — async HTTP client (Ollama / llama.cpp / external)
    context.py           ContextManager — token-aware sliding window

  translation/           Core agent loop
    loop.py              TranslationLoop — thin adapter → delegates to core Pipeline
    executor.py          decide() — single LLM call, unified action menu
    decomposer.py        decompose() — maps task to typed stage steps
    manifest.py          TaskManifest / Instruction dataclasses
    prompts.py           SYSTEM prompt, per-stage action menus
    web_search.py        SearXNG → Jina AI fetch + condenser
    subtask/             Activated for write_code / write_doc plan steps
      planner.py         LLM call → named items (function names / headings)
      manifest.py        SubtaskManifest / SubtaskItem status tracking
      writer.py          Item-by-item write with MAX_ITEM_RETRIES=2
      verifier.py        Pure-regex scoring (complete/partial/missing/regressed)

  memory/
    graph.py             NetworkX knowledge graph — personality, user, facts, concepts
    store.py             JSONL artifact store → memory/artifacts.jsonl
    session_log.py       Per-session conversation log (used by semantic history)
    injector.py          Builds relevance-ranked memory context for every request
    extractor.py         Extracts facts from LLM responses and writes to graph
    dream.py             Dream cycle: merge logs → graph + reflection + cleanup
    cleanup.py           Retention policy — prunes old sessions and artifacts

  core/
    pipeline.py          Task execution pipeline — full agent loop with graph access
    recall.py            Memory recall / retrieval
    consolidator.py      Knowledge consolidation
    synthesizer.py       Result synthesis

  activity.py            Recent events tracking
  task_tracker.py        Active task management

skills/                  Permanent skill scripts (stdlib-first, always available)
  calc.py                Safe expression evaluator (math.* functions)
  arxiv.py               arXiv paper search
  read_pdf.py            PDF text extraction (pymupdf)
  github_ops.py          GitHub issues/PRs/clone/search (gh CLI)
  youtube.py             YouTube transcript fetcher (yt-dlp)
  obsidian.py            Obsidian vault search/read/create
  maps.py                Geocoding, routing, nearby POIs (Nominatim/OSRM/Overpass)
  hf_hub.py              HuggingFace model/dataset search
  ocr.py                 Image OCR (pytesseract or easyocr)
  web.py                 URL fetch → condensed plain text

memory/                  Persisted knowledge (graph.json + artifacts.jsonl)
workspace/               Default sandboxed directory; agent may only write here
```

---

## Agent Loop

Every `/v1/messages` request enters `TranslationLoop.process()`:

1. **Continuation check** — if the previous message carries `tool_result` blocks, decode `PIPELINE_STATE:<base64-json>` from a thinking block and resume `MicroState` (step count, internal messages, summary, pending write steps).
2. **Graph injection** — before the first LLM call, the knowledge graph is queried with the user message and the top-N relevant nodes are prepended to the system prompt.
3. **Route** — a lightweight classifier call maps the query to a route hint (`bash`, `search`, `memory`, `answer`, or a skill name). Pure arithmetic is fast-patched to `bash` before the LLM call to avoid misrouting.
4. **Decompose** — the query is broken into typed stages (`think`, `search`, `write_code`, `write_doc`, `bash`, `answer`). Gets 400 chars task context + 400 chars graph excerpt.
5. **`_micro_loop()`** — up to 12 inner steps:
   - If a `write_code`/`write_doc` step is queued in `pending_write_steps`, dispatch to the subtask pipeline (planner → writer → verifier).
   - Otherwise call `decide()` — one LLM call returns an action.
   - **Internal tools** handled inside the loop: `plan_task`, `search_knowledge`, `search_history`, `save_memory`, `web_search`, `list_workspace`, `read_file`, `run_skill`.
   - **Outer tools** (e.g., `bash`) returned as `tool_use` blocks to the caller; current state encoded in a thinking block for the next turn.
6. **Forced answer** — if MAX_STEPS exceeded, a final synthesis call produces the answer.

No stage gating — the model decides freely at every step. Stall guard blocks duplicate `(tool, input)` calls. Semantic history uses Jaccard similarity ≥ 0.15 over session logs with one LLM summary call per outer turn.

---

## Skills

Skills are self-contained Python scripts in `skills/`. The agent invokes them via bash: `python skills/SCRIPT.py ARGS`.

The skill index is injected into route and plan calls as a compact name + summary list. The calc skill is filtered out when the query contains no digits or math operators (need-to-know guard).

| Script | Purpose | Deps |
|--------|---------|------|
| `calc.py` | Safe math expressions (`sqrt`, `sin`, `log`, `pi`, …) | stdlib |
| `arxiv.py` | Search arXiv papers | stdlib |
| `read_pdf.py` | Extract text from PDF, optional page range | `pymupdf` |
| `github_ops.py` | Issues, PRs, clone, search via GitHub | `gh` CLI |
| `youtube.py` | Fetch YouTube transcript | `yt-dlp` |
| `obsidian.py` | Search/read/create Obsidian vault notes | stdlib |
| `maps.py` | Geocoding, routes, nearby POIs | stdlib |
| `hf_hub.py` | HuggingFace model/dataset search | stdlib |
| `ocr.py` | Image OCR | `pytesseract` or `easyocr` |
| `web.py` | URL fetch → condensed plain text | stdlib |

See `skills/README.md` for setup notes and usage examples.

---

## Knowledge Graph

The knowledge graph (`memory/graph.json`) is the architectural heart of Sisyphean. Everything the agent learns — preferences, research, project state, web facts — gets written into a persistent NetworkX graph. On every incoming request, the graph is queried with the user message and the top-N most relevant nodes are prepended to the system prompt.

### Node types

| Type | What lives here |
|------|----------------|
| `soul` | Agent personality — loaded from `soul.md` on every startup |
| `user` | User preferences, working style, remembered facts |
| `project` | Active projects, goals, current status |
| `fact` | Discrete facts extracted from conversations and web research |
| `concept` | Technical and domain concepts with relationships |
| `system` | OS, Python version, shell type |
| `skill` | Skill runbooks synced from BirdClaw (via dream cycle) |

### Live sync

Both `soul.md` (agent personality) and `memory/user_prefs.md` (user preferences) are synced to the graph bidirectionally:

- **File → graph:** on every startup and on every request where either file's mtime has changed. Edit the file, send the next message — the graph reflects it immediately, no restart.
- **Graph → file:** when `save_memory` fires during a conversation, the fact is appended to `user_prefs.md` in addition to being written to the graph.

---

## Internal Tools

| Tool | When used |
|------|-----------|
| `plan_task` | Complex multi-step tasks — produces typed stage list |
| `list_workspace` | Before any file write — shows what already exists |
| `read_file` | Before modifying a file — reads relevant portion |
| `search_memory` | Domain knowledge or past research from the graph |
| `search_history` | What was done in previous sessions |
| `save_memory` | Persist a fact or preference — writes to graph AND `user_prefs.md` |
| `web_search` | Current information; results extracted to graph as `fact` nodes |
| `run_skill` | Invoke a skill script from the `skills/` library |
| `bash` | Shell execution — code runs, file writes, system queries |

---

## Memory Injection

The injector builds the context prepended to every LLM call. Priority order (highest → lowest, truncated to token budget of 1500 tokens):

1. **Soul / personality** — relevant section of `soul.md` matched to query
2. **System info** — OS, Python version, shell type
3. **User knowledge** — all `user` nodes (preferences, remembered facts)
4. **Active project** — most recently touched project nodes (last 14 days)
5. **Relevant facts/concepts** — keyword + recency scored graph search against the incoming message
6. **Research knowledge** — extracted facts from past conversations
7. **Past artifacts** — significant outputs from the artifact store, query-matched

The soul section is never dropped. Lower-priority sections are truncated first when the budget is exceeded.

---

## Dream Cycle — Offline Memory Consolidation

```bash
python main.py dream
python main.py dream --dry-run       # preview without writing
python main.py dream --cleanup-only  # prune stale sessions only
python main.py dream --memorise-only # ingest logs to graph only
```

The dream cycle runs offline (no live requests):
1. Reads all session logs since the last dream
2. Merges new facts into the knowledge graph with deduplication
3. Strengthens high-confidence nodes, prunes low-confidence ones
4. Updates `soul`, `user`, and `project` nodes with accumulated insights
5. Writes a reflection summary

---

## Routing

| Input type | Route | Step chain |
|------------|-------|-----------|
| Pure arithmetic | `bash` (fast-patched) | write `calc.py` → run → return result |
| Math expression | `bash` | `run_skill:calc EXPRESSION` |
| Factual question | `search` | `web_search KEYWORDS` |
| Code task | `bash` | `list_workspace` → `write_code FILENAME` |
| File operation | `bash` | `bash COMMAND` |
| Memory | `memory` | `save_memory FACT` |
| Greeting / conversational | `answer` | direct answer, no tools |

Math is always written to a Python script and executed — never answered inline — so the result is verifiable.

---

## System Tray

```bash
python main.py tray
```

Launches the engine in the background with a Windows system tray icon. Right-click for Start / Stop / Restart / View Logs / Update (git pull + restart). Includes a watchdog that automatically restarts the engine on crash.

> **If the icon doesn't appear:** Windows hides new tray icons in the overflow area. Click `^` in the taskbar, or go to **Settings → Personalisation → Taskbar → Other system tray icons** and enable Sisyphean.

---

## Dashboard

Browse to `http://127.0.0.1:47291/dashboard` for a live status page: uptime, graph node count, artifact count, active tasks, recent events.

---

## Commands

```bash
python main.py              # start engine
python main.py tray         # Windows tray watchdog (recommended)
python main.py dream        # offline memory consolidation
python main.py dream --dry-run
python main.py dream --cleanup-only
python main.py launch birdclaw   # start BirdClaw web UI + open browser
python main.py launch claude     # start Claude Code pointed at Sisyphean
```

---

## Development & Tests

```bash
# Mock mode — no model needed, tests API shape only
# Set mock: true in config.yaml, then:
python main.py

# Run tests (engine must be running on port 47291)
python tests/test_suite.py --regimen    # behavioural regimen (T1–T9)
```

### Regimen results (gemma4:latest via Ollama)

| Test | Description | Result |
|------|-------------|--------|
| T1 | Greeting | ✅ Pass |
| T2 | sqrt(144) via calc skill | ✅ Pass |
| T3 | Save user memory preference | ✅ Pass |
| T4 | Read file from workspace | ✅ Pass |
| T5 | Multi-step calc (100 + 200 * 3) | ✅ Pass |
| T6 | Capital of France (web search) | ⚠️ SearXNG offline — routes correctly |
| T7 | Latest Python version (web search) | ⚠️ SearXNG offline — routes correctly |
| T8 | Create a folder via bash | ✅ Pass |
| T9 | Remember preference (vim) | ✅ Pass |

T6 and T7 fail only because SearXNG is not running in CI. Routing is correct; results would be correct with a live search instance.

---

## Roadmap

- [x] gemma4:latest validated — full capability suite passing
- [x] Math routed through bash (write + run Python, never inline)
- [x] `save_memory` writes to graph AND `memory/user_prefs.md`
- [x] `soul.md` and `user_prefs.md` live-synced to graph (mtime-based, no restart needed)
- [x] Graph search ISO timestamp fix — recency scoring correct
- [x] `save_memory` merges into related nodes (no duplicate accumulation)
- [x] Skills library — 10 built-in scripts, stdlib-first
- [x] Calc skill guard — need-to-know context filter
- [x] Placeholder guard — discard template-echo `run_skill` invocations
- [x] Progressive disclosure — right-sized context per stage, call count scales with complexity
- [ ] History timeline node — serial chronological graph structure
- [ ] Incremental write + verify for calculation scripts
- [ ] Real tokeniser (replace `len(text) // 4` approximation)
- [ ] `asyncio.Lock` on knowledge graph mutations
- [ ] Streaming test coverage
- [ ] Telegram / Discord channel adapters
- [ ] PC control (screenshot, mouse/keyboard) via BirdClaw tool bridge
- [ ] Self-modification (agent edits own source, runs tests, commits)

---

## Known Limitations

**Token counting is approximate.** `ContextManager` falls back to `len(text) // 4`. For Gemma/Qwen the real ratio is 3–3.5 chars/token, so the sliding window may overflow or over-compress on very long sessions.

**Knowledge graph has no asyncio lock.** `graph.py` uses atomic file writes but no `asyncio.Lock` on in-memory state. Two concurrent tasks can produce a last-write-wins race. Low risk in single-user local use.

**Embedding store rebuilds from scratch on every insert.** `ArtifactStore._rebuild_embeddings()` re-encodes the entire corpus each time. Performance degrades linearly as the store grows.

**Streaming is untested.** The SSE streaming path in `anthropic.py` is implemented but has no test coverage.

**Graph corruption is silent data loss.** If `memory/graph.json` is corrupted (partial write, disk full), `KnowledgeGraph._load()` starts fresh and discards all stored knowledge. A `.bak` fallback exists in `GraphStore` but not in `KnowledgeGraph`.
