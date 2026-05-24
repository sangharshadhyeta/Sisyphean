# Sisyphean

A local AI agent engine that runs as a persistent background service. It exposes the Anthropic Messages API (`/v1/messages`) and OpenAI Chat Completions (`/v1/chat/completions`), making it a drop-in model provider for Claude Code, BirdClaw, and any OpenAI-compatible client.

Sisyphean is a **full agent** — it has a personality (`engine_policy.md`), a persistent knowledge graph, a bash execution environment, web search, and a dream cycle for offline memory consolidation. It is not a dumb proxy. Every request passes through a multi-step reasoning loop that can search its own memory, run code, search the web, and write files before answering.

---

## **Tested on Gemma 4 E4B (gemma-4-E4B-it-Q8_0.gguf)**

All agent capabilities — multi-step reasoning, tool chaining, graph memory injection, web search, bash execution, subtask writing — have been validated against `gemma-4-E4B-it-Q8_0.gguf` (Q8 quantisation, served via llama.cpp). This is a 4-billion-parameter model running entirely offline on local hardware. Sisyphean is specifically designed to get reliable agentic behaviour out of small local models by routing decisions through structured prompts and fallback retries rather than trusting the model to free-form reason.

---

## Quick Start

> **BirdClaw users:** run `BirdClaw/install.bat` instead — it installs Sisyphean, SearXNG, and BirdClaw together as a single stack. Come back here only if you want Sisyphean standalone.

```bash
# Install dependencies
pip install -r requirements.txt

# First-time setup (LLM backend, port, workspace)
python main.py setup

# Start the engine
python main.py

# Or start with the Windows system tray watchdog (recommended)
python main.py tray
```

Engine runs at `http://127.0.0.1:47291` by default.

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

Edit `config.yaml` (created by `python main.py setup`):

```yaml
llm:
  local_model: "qwen3:0.6b"   # Ollama model name

  external_api:                # Use llama.cpp, LM Studio, OpenRouter, etc.
    enabled: true
    base_url: "http://192.168.1.x:8081/v1"
    api_key: "local"
    model: "gemma-4-E4B-it-Q8_0.gguf"

api:
  host: 127.0.0.1
  port: 47291

search:
  searxng_url: "http://localhost:8888"   # set to "" to disable (falls back to Jina)

mock: false                    # true = no model needed (for API/UI testing)
workspace: ./workspace         # sandboxed directory; agent writes only here
```

> **Web search tiers:** SearXNG (local Docker container, port 8888) → Jina AI reader (public, no key required). DuckDuckGo has been removed. BirdClaw's `install.bat` sets up SearXNG automatically.

---

## How It Thinks — The Graph

The knowledge graph is the architectural heart of Sisyphean. Everything the agent learns, every preference it records, every piece of research it does gets written into a persistent NetworkX graph (`~/.sisyphean/memory/knowledge_graph.json`). On every incoming request, the graph is queried with the user's message and the most relevant nodes are injected into the model's context window before it begins reasoning.

This solves two problems at once: **context overflow** and **persistent continuity**.

### Node types and what they hold

| Type | What lives here |
|------|----------------|
| `soul` | The agent's personality — loaded from `engine_policy.md` on every startup |
| `user` | User preferences, working style, tools preferred — loaded from `memory/user_prefs.md` and accumulated via `save_memory` |
| `project` | Active projects, goals, current status |
| `fact` | Discrete facts extracted from conversations and web research |
| `concept` | Technical and domain concepts with relationships to other nodes |
| `system` | OS, Python version, shell type — injected so the model always knows what commands to use |

### Self and identity

The `soul` node is the agent's self-concept. When you ask "are you alive?", the retrieval pulls the identity section of `engine_policy.md`, which gives the model the grounding to answer from its own written character rather than hallucinating a generic AI response. The soul node links outward to facts about AI cognition, continuity, and the nature of local vs cloud inference — so "are you alive?" becomes a traversal through everything the agent knows about itself.

### User node and preferences

Every time you say "remember I prefer X" or "save this", the fact is written to both the graph (as a `user` node) and to `memory/user_prefs.md`. On every startup, both files are re-synced to the graph. On every request, if either file has changed on disk, the graph is updated before the request is processed — meaning you can edit `engine_policy.md` or `user_prefs.md` directly and the agent picks it up on the next message, no restart needed.

### Context management and overflow prevention

The graph acts as a selective context buffer. Without it, every long-running task would need to fit its entire history into one context window — impossible on a 4B model with an 8K context. With the graph, anything worth keeping gets written to a node. The injector retrieves only the top-N most relevant nodes (scored by keyword match + recency) and prepends them to the system prompt. Web search results, file reads, past session summaries — all go through the same graph, so the model's effective memory is unbounded while its live context stays small.

### Graph + web search synergy

When the model searches the web, extracted facts are written to the graph as `fact` nodes. The next time a related question arrives, those facts are already in the graph and get injected directly — no re-search needed. The model can also choose to do a fresh web search to update or extend what it already knows in the graph, building deeper coverage of a topic over multiple sessions.

### Timestamps and temporal awareness

Every node carries a `last_seen` timestamp. The injector applies a recency bonus: nodes updated in the last 7 days score higher. This means the model naturally surfaces fresh research over stale facts. The dream cycle (offline consolidation) reads session logs by timestamp to build a chronological picture of what was done and when — closing the loop on the agent's own timeline.

### Dreaming — offline memory consolidation

```bash
python main.py dream
```

The dream cycle runs offline (no live requests). It:
1. Reads all session logs since the last dream
2. Merges new facts into the knowledge graph with deduplication
3. Strengthens high-confidence nodes, prunes low-confidence ones
4. Updates the `soul`, `user`, and `project` nodes with accumulated insights
5. Writes a reflection summary

Running the dream cycle regularly keeps the graph clean and the agent's self-knowledge current. The more sessions it processes, the more coherent the agent's world model becomes.

---

## Architecture

```
main.py                  Entry point — uvicorn + optional llama-server subprocess
tray.py                  Windows system tray watchdog (start/stop/restart/update)

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
    web_search.py        SearXNG → Jina AI fetch + condenser (no DuckDuckGo)
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

  soul/                  (legacy) Soul router — section search on engine_policy.md
  policy/                Policy router — parses and matches engine_policy.md sections

  activity.py            Recent events tracking
  task_tracker.py        Active task management

engine_policy.md         Agent personality and reasoning discipline — edit freely,
                         picked up on next startup (or next request if mtime changes)
memory/user_prefs.md     User preferences — edit freely, same live-sync as above
workspace/               Default sandboxed directory; agent may only write here
```

---

## Agent Loop

Every `/v1/messages` request enters `TranslationLoop.process()`:

1. **Continuation check** — if the previous message carries `tool_result` blocks, decode `PIPELINE_STATE:<base64-json>` from a thinking block and resume `MicroState` (step count, internal messages, summary, pending write steps).
2. **Graph injection** — before the first LLM call, the knowledge graph is queried with the user message and the top-N relevant nodes are prepended to the system prompt.
3. **`_micro_loop()`** — up to 12 inner steps:
   - If a `write_code`/`write_doc` step is queued in `pending_write_steps`, dispatch to the subtask pipeline (planner → writer → verifier).
   - Otherwise call `decide()` — one LLM call returns an action.
   - **Internal tools** handled inside the loop: `plan_task`, `search_knowledge`, `search_history`, `save_memory`, `web_search`, `list_workspace`, `read_file`.
   - **Outer tools** (e.g., `bash`) returned as `tool_use` blocks to the caller; current state encoded in a thinking block for the next turn.
4. **Forced answer** — if MAX_STEPS exceeded, a final synthesis call produces the answer.

No stage gating — the model decides freely at every step. All actions always available: think, search_memory, search_history, web_search, list_workspace, read_file, save_memory, bash, answer.

Stall guard blocks duplicate (tool, input) calls. Semantic history uses Jaccard similarity ≥ 0.15 over session logs with one LLM summary call per outer turn.

---

## Memory Injection

The injector builds the memory context prepended to every LLM call. Priority order (highest → lowest, truncated to token budget):

1. **Soul / personality** — full `engine_policy.md` text from the `policy` node
2. **System info** — OS, Python version, shell type (so the model knows which commands to use)
3. **User knowledge** — all `user` nodes (preferences, working style, remembered facts)
4. **Active project** — most recently touched project nodes (last 14 days)
5. **Relevant facts/concepts** — keyword + recency scored graph search against the incoming message
6. **Research knowledge** — NER entities and extracted facts from past conversations
7. **Past artifacts** — significant outputs from the artifact store, query-matched

The budget defaults to 1500 tokens. Sections are dropped from the bottom if the budget is exceeded. The soul section is never dropped.

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
| `bash` | Shell execution — code runs, file writes, system queries |

---

## Routing

The task router maps each incoming message to a step chain before any LLM call:

| Input type | Step chain |
|------------|-----------|
| Math / calculation | `Run ls \| Write calc.py \| Run python calc.py` |
| Factual question | `Search KEYWORDS` |
| Code task | `Run ls \| Write FILENAME` |
| File operation | `Run COMMAND` |
| Memory | `Save: FACT` |
| Greeting / conversational | *(direct answer, no tools)* |

Math always writes and runs a Python script — never answered inline — so the result is verifiable.

---

## Personality and User Preferences — Live Sync

Both `engine_policy.md` (agent personality) and `memory/user_prefs.md` (user preferences) are part of the knowledge graph. They are synced bidirectionally:

- **File → graph**: on every startup, and on every incoming request if either file's mtime has changed. Edit the file, send the next message, the graph reflects it.
- **Graph → file**: when `save_memory` fires from a conversation, the fact is appended to `user_prefs.md` in addition to being written to the graph.

To change the agent's personality, edit `engine_policy.md` directly. No restart needed — the next message picks it up.

---

## System Tray

```bash
python main.py tray
```

Launches the engine in the background with a Windows system tray icon. Right-click for Start / Stop / Restart / View Logs / Update (git pull + restart). Includes a watchdog that automatically restarts the engine if it crashes.

> **If the icon doesn't appear:** Windows hides new icons in the overflow area. Click `^` in the taskbar notification area, or go to **Settings → Personalisation → Taskbar → Other system tray icons** and enable Sisyphean.

---

## Dashboard

Browse to `http://127.0.0.1:47291/dashboard` for a live status page: uptime, graph node count, artifact count, active tasks, recent events.

---

## Commands

```bash
python main.py              # start engine
python main.py tray         # Windows tray watchdog (recommended)
python main.py setup        # first-time setup wizard
python main.py config       # re-run setup to change settings
python main.py dream        # offline memory consolidation (dream cycle)
python main.py dream --dry-run
python main.py dream --cleanup-only
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
python tests/test_api.py        # L1-L6 capability suite (147 tests)
python tests/test_openclaw.py   # OpenClaw adapter + tool round-trip
python tests/test_regimen.py    # 11-test behavioural regimen

# Tests expect port 47291. If you changed api.port, update BASE_URL
# at the top of each test file to match.
```

All three test suites pass against `gemma-4-E4B-it-Q8_0.gguf`.

---

## Roadmap

- [x] Gemma 4 E4B validated — full capability suite passing
- [x] Math routed through Bash (write + run Python, never inline)
- [x] `save_memory` writes to graph AND `memory/user_prefs.md`
- [x] `engine_policy.md` and `user_prefs.md` live-synced to graph (mtime-based, no restart needed)
- [x] Graph search ISO timestamp fix — recency scoring works correctly
- [x] save_memory merges into related nodes (no duplicate accumulation)
- [x] Workspace listed before file writes (agent sees what exists)
- [ ] History timeline node — serial chronological graph structure where session events are linked to a central timeline node, then branch outward to knowledge produced in that session
- [ ] Incremental write + verify for calculation scripts (subtask pipeline applied to calc.py, not just complex modules)
- [ ] Fix token counting — use real tokenizer or `/tokenize` endpoint
- [ ] Add `asyncio.Lock` to knowledge graph mutations
- [ ] Incremental embedding updates (don't rebuild full corpus on each insert)
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

**Graph corruption is silent data loss.** If `knowledge_graph.json` is corrupted (partial write, disk full), `KnowledgeGraph._load()` starts fresh and discards all stored knowledge. The `GraphStore` layer checks a `.bak` file; `KnowledgeGraph` does not.
