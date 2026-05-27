from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator


class MemoryConfig(BaseModel):
    path: str = "./memory"
    engine_policy_file: str = "./engine_policy.md"
    injection_budget: int = 1500    # max tokens injected per request
    top_n_nodes: int = 5            # graph nodes retrieved per query
    top_n_artifacts: int = 3        # artifact store results per query
    embedding_model: str | None = "all-MiniLM-L6-v2"  # None = keyword search only


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8080
    ollama_port: int = 11434
    context_size: int = 8192
    threads: int = 4
    gpu_layers: int = 0
    extra_args: list[str] = []


class ExternalAPIConfig(BaseModel):
    """Optional external OpenAI-compatible LLM provider (OpenRouter, Groq, etc.).

    When enabled=True, the engine calls base_url/v1/chat/completions with the
    given api_key and model, skipping llama-server entirely.  Useful for
    testing with a free cloud model (e.g. google/gemma-3-4b-it:free on
    OpenRouter) before a local GPU is available.

    Example providers:
      OpenRouter:  base_url=https://openrouter.ai/api/v1
                   model=google/gemma-3-4b-it:free      (free, no credits needed)
      Groq:        base_url=https://api.groq.com/openai/v1
                   model=gemma2-9b-it
      Google AI:   base_url=https://generativelanguage.googleapis.com/v1beta/openai
                   model=gemini-2.0-flash-lite
    """
    enabled: bool = False
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    model: str = "google/gemma-3-4b-it:free"


class LLMConfig(BaseModel):
    model_path: str = ""
    model_name: str = "sisyphean-4b"
    # If set, the engine connects to a pre-running OpenAI-compatible server
    # (e.g. Ollama) instead of launching llama-server.  The value is the
    # exact model name sent in every completion request.
    # Example: local_model: "qwen2.5:1.5b"  (Ollama)
    local_model: str = ""
    server: ServerConfig = ServerConfig()
    external_api: ExternalAPIConfig = ExternalAPIConfig()


class EmbeddingConfig(BaseModel):
    enabled: bool = False
    # Ollama embedding model — used when local_model is set in LLMConfig.
    # Pull with:  ollama pull nomic-embed-text
    ollama_model: str = "nomic-embed-text"
    # llama-server embedding config (only used when local_model is NOT set)
    model_path: str = ""
    model_name: str = "sisyphean-embed"
    server: ServerConfig = ServerConfig(port=8081, context_size=2048)


class APIConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 47291
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:47291"]


class PermissionsConfig(BaseModel):
    # File/directory patterns the agent must never write without user approval.
    # Glob-style: "engine/**" matches any file under engine/.
    # Checked by the permission guard before the loop returns write actions.
    protected_paths: list[str] = [
        "engine/**",        # all core engine files
        "main.py",
        "config.yaml",
        "requirements.txt",
    ]
    # Bash substrings that trigger a warning (shown to user, not silently blocked)
    dangerous_commands: list[str] = [
        "rm -rf",
        "sudo ",
        "chmod 777",
        "dd if=",
        "mkfs.",
        "> /dev/",
        "format c:",
        "del /f",
        "rmdir /s",
    ]


class SearchConfig(BaseModel):
    """Web search backend configuration.

    Priority order when executing a search:
      1. SearXNG  — fast, private, supports Google/Bing/DDG simultaneously.
                   Requires a local SearXNG instance. Set searxng_url to enable.
      2. DuckDuckGo package (ddgs / duckduckgo_search) — if installed.
      3. DuckDuckGo Instant Answers API — pure httpx, no extra package, limited results.
    """
    searxng_url: str = ""   # e.g. "http://localhost:8888" — empty = skip SearXNG
    max_results: int = 5
    timeout: float = 15.0


class DebugConfig(BaseModel):
    dump_requests: bool = False  # write last_request.json on every fresh request


class Config(BaseModel):
    llm: LLMConfig
    embedding: EmbeddingConfig = EmbeddingConfig()
    api: APIConfig = APIConfig()
    memory: MemoryConfig = MemoryConfig()
    permissions: PermissionsConfig = PermissionsConfig()
    search: SearchConfig = SearchConfig()
    debug: DebugConfig = DebugConfig()
    mock: bool = False
    workspace: str = "./workspace"
    skills_path: str = "./skills"

    @field_validator("workspace")
    @classmethod
    def expand_workspace(cls, v: str) -> str:
        return str(Path(v).expanduser().resolve())

    @field_validator("skills_path")
    @classmethod
    def expand_skills_path(cls, v: str) -> str:
        return str(Path(v).expanduser().resolve())


def load_config(path: str | Path = "config.yaml") -> Config:
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(**data)
