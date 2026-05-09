# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Sisyphean is a local AI agent that runs as a FastAPI background service, acting as a custom model provider for Claude Code. It exposes both Anthropic Messages API (`/v1/messages`) and OpenAI Chat Completions (`/v1/chat/completions`) so it integrates as a drop-in model with Claude Code and any OpenAI-compatible client.

It was rebuilt from scratch after BirdClaw (predecessor) had reliability issues (stale cache, context truncation, no bash timeouts, graph corruption). Key principles: no tool result caching, context summarisation rather than dropping, workspace boundary enforcement.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Start the engine (Ollama must be running with the configured model pulled)
python main.py

# Start with Windows system tray watchdog
python main.py tray

# Offline memory consolidation (dream cycle)
python main.py dream
python main.py dream --dry-run
python main.py dream --cleanup-only
python main.py dream --memorise-only

# Run tests (engine must be running on :8000)
python test_api.py
python test_regimen.py
python test_openclaw.py
```

Engine runs at `http://127.0.0.1:8000` by default. Set `mock: true` in `config.yaml` to test without a running model.

## Configuration

`config.yaml` is the single config file. Key options:

```yaml
llm:
  local_model: "qwen3:0.6b"    # Ollama model name
  server:
    port: 11434                  # Ollama port
  external_api:                  # Alternative: use OpenRouter, Groq, etc.
    enabled: false
    base_url: "https://openrouter.ai/api/v1"
    api_key: "sk-..."
    model: "google/gemma-3-4b-it:free"

api:
  host: 127.0.0.1
  port: 8000

mock: false                      # true = no model needed (for testing)
workspace: ./workspace           # sandboxed dir; agent writes here only
```

## Architecture

```
main.py                  Entry point — uvicorn + llama-server subprocess management
tray.py                  Windows system tray watchdog

engine/
  api/app.py             FastAPI app factory
  config.py              Config dataclasses + YAML loader
  permissions.py         Permission guard — enforces workspace boundary

  compat/
    anthropic.py         POST /v1/messages handler (Anthropic format)
    openai_compat.py     POST /v1/chat/completions (passthrough)

  llm/
    client.py            LlamaClient — async HTTP client (Ollama/llama.cpp/external)
    context.py           ContextManager — token-aware sliding window

  translation/           Core agent loop
    loop.py              TranslationLoop — process() → _micro_loop() (up to 12 steps)
    executor.py          decide() — single LLM call, unified action menu
    decomposer.py        decompose() — breaks task into typed stage steps
    manifest.py          TaskManifest / Instruction dataclasses
    prompts.py           SYSTEM prompt, soul sections, per-stage action menus
    web_search.py        DuckDuckGo search + HTML fetch + condenser
    subtask/             Activated for write_code / write_doc plan steps
      planner.py         LLM call → named items (function names / section headings)
      manifest.py        SubtaskManifest / SubtaskItem status tracking
      writer.py          Item-by-item write with MAX_ITEM_RETRIES=2
      verifier.py        Pure-regex scoring (complete/partial/missing/regressed)

  memory/
    graph.py             NetworkX knowledge graph → memory/graph.json
    store.py             JSONL artifact store
    session_log.py       Per-session conversation log (used by semantic history)
    injector.py          Injects graph context into LLM requests
    extractor.py         Extracts facts from LLM responses
    dream.py             Dream cycle: merge logs → graph + reflection + cleanup
    cleanup.py           Retention policy — prunes old sessions/artifacts

  core/
    pipeline.py          Task execution pipeline
    recall.py            Memory recall / retrieval
    consolidator.py      Knowledge consolidation
    synthesizer.py       Result synthesis

  activity.py            Recent events tracking
  task_tracker.py        Active task management

soul.md                  Agent personality — edit to change behaviour
workspace/               Default sandboxed directory; agent may only write here
```

## Agent loop internals

Every `/v1/messages` request enters `TranslationLoop.process()`:

1. **Continuation check** — if the previous message has `tool_result` blocks, decode `SISYPHEAN_STATE:<base64-json>` from a thinking block and resume `MicroState` (step count, internal messages, summary, pending write steps).
2. **`_micro_loop()`** — up to 12 inner steps:
   - If a `write_code`/`write_doc` step is queued in `pending_write_steps`, dispatch to the subtask pipeline (planner → writer → verifier).
   - Otherwise call `decide()` which makes one LLM call and returns an action.
   - **Internal tools** handled inside the loop: `plan_task`, `search_knowledge`, `search_history`, `save_memory`, `web_search`, `list_workspace`, `read_file`.
   - **Outer tools** (e.g., `bash`) returned as `tool_use` blocks to the caller; current state encoded in a thinking block for the next turn.
3. **Forced answer** — if MAX_STEPS exceeded, a final synthesis call produces the answer.

No stage gating — the model decides freely at every step. All actions are always available: think, search_memory, search_history, web_search, list_workspace, read_file, save_memory, bash, answer.

Stall guard blocks duplicate (tool, input) calls. Semantic history uses Jaccard similarity ≥ 0.15 over session logs with one LLM summary call per outer turn.

## Claude Code integration

```json
{
  "customApiUrl": "http://127.0.0.1:8000",
  "customApiKey": "local"
}
```

Claude Code sends Anthropic-format requests to `/v1/messages`. Sisyphean translates through the agent loop and returns `tool_use` / `end_turn` responses.
