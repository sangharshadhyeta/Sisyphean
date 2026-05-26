"""AST-based codebase indexer — populates the knowledge graph with structural
code intelligence without any LLM calls.

For every Python file it finds, it extracts:
  - Module node  (type='module')  — path, docstring summary, import deps
  - Function nodes (type='function') — signature, docstring, parent module
  - Class nodes    (type='class')    — bases, methods list, docstring

Edges created:
  module   ──contains──►  function / class
  module   ──imports──►   other_module      (dependency edge)
  project  ──has_module──► module           (when project_name given)

Design principles
-----------------
- Zero LLM calls — pure ast + stdlib, runs fast enough for file-watch hooks
- Incremental — skips files whose mtime hasn't changed since last index
- Idempotent — safe to re-run; uses upsert_node / upsert_edge throughout
- Non-invasive — imports nothing from the pipeline; graph is the only coupling
- BirdClaw-hookable — call index_path(path, graph) from any watcher
- CLI — python -m engine.memory.code_indexer PATH [--project NAME]

Node naming convention
----------------------
  module   : "module:<stem>"          e.g. "module:pipeline"
  function : "fn:<stem>.<name>"       e.g. "fn:pipeline._start"
  class    : "cls:<stem>.<name>"      e.g. "cls:graph.GraphStore"
  method   : "fn:<stem>.<Cls>.<name>" e.g. "fn:graph.GraphStore.upsert_node"

Using prefixed names prevents collisions with conversation-fact nodes that
happen to share a name with a code entity.
"""
from __future__ import annotations

import ast
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.memory.graph import GraphStore

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

_MAX_SUMMARY   = 300   # chars — kept short so many nodes fit in injection budget
_MAX_FUNCTIONS = 60    # per file — guard against auto-generated monster files
_MAX_CLASSES   = 20    # per file
_SKIP_DIRS     = frozenset({"__pycache__", ".git", ".venv", "venv", "node_modules",
                             ".mypy_cache", ".pytest_cache", "dist", "build", "eggs"})
_SKIP_FILES    = frozenset({"setup.py", "conftest.py"})


# ── Internal data model ──────────────────────────────────────────────────────

@dataclass
class _FnInfo:
    name: str
    qualname: str          # e.g. "MyClass.my_method"
    lineno: int
    args: str              # "(self, x: int, y: str = '') -> bool"
    docstring: str


@dataclass
class _ClsInfo:
    name: str
    lineno: int
    bases: list[str]
    docstring: str
    methods: list[str]     # just names, not full signatures


@dataclass
class _ModuleInfo:
    path: Path
    stem: str              # file stem, used as short identifier
    docstring: str
    imports: list[str]     # module names this file imports
    functions: list[_FnInfo]  = field(default_factory=list)
    classes:   list[_ClsInfo] = field(default_factory=list)


# ── AST helpers ──────────────────────────────────────────────────────────────

def _docstring(node: ast.AST) -> str:
    """Extract docstring from a module/function/class node, or '' if absent."""
    if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.ClassDef)):
        return ""
    if (node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)):
        raw = node.body[0].value.value.strip()
        # First non-blank line only — keeps summaries tight
        first = next((ln.strip() for ln in raw.splitlines() if ln.strip()), raw)
        return first[:_MAX_SUMMARY]
    return ""


def _arg_annotation(ann: ast.expr | None) -> str:
    if ann is None:
        return ""
    try:
        return ast.unparse(ann)
    except Exception:
        return ""


def _signature(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Reconstruct a compact signature string: (arg: type, ...) -> ret"""
    args = func.args
    parts: list[str] = []

    # positional-only, regular args, *args, keyword-only, **kwargs
    all_args: list[tuple[ast.arg, ast.expr | None]] = []
    for arg in args.posonlyargs:
        all_args.append((arg, arg.annotation))
    for arg in args.args:
        all_args.append((arg, arg.annotation))
    if args.vararg:
        all_args.append((args.vararg, args.vararg.annotation))
    for arg in args.kwonlyargs:
        all_args.append((arg, arg.annotation))
    if args.kwarg:
        all_args.append((args.kwarg, args.kwarg.annotation))

    for arg, ann in all_args:
        name = arg.arg
        ann_str = _arg_annotation(ann)
        parts.append(f"{name}: {ann_str}" if ann_str else name)

    ret = ""
    if func.returns:
        ret = f" -> {_arg_annotation(func.returns)}"

    prefix = "async " if isinstance(func, ast.AsyncFunctionDef) else ""
    return f"{prefix}({', '.join(parts)}){ret}"


def _extract_imports(tree: ast.Module) -> list[str]:
    """Return a deduplicated list of top-level module names imported."""
    seen: set[str] = set()
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in seen:
                    seen.add(top)
                    names.append(top)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top not in seen:
                    seen.add(top)
                    names.append(top)
    return names


def _parse_file(path: Path) -> _ModuleInfo | None:
    """Parse a single Python file into a _ModuleInfo. Returns None on error."""
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.debug("code_indexer: cannot read %s: %s", path, exc)
        return None

    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as exc:
        logger.debug("code_indexer: syntax error in %s: %s", path, exc)
        return None

    stem = path.stem
    doc  = _docstring(tree)
    imports = _extract_imports(tree)

    functions: list[_FnInfo] = []
    classes: list[_ClsInfo]  = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and len(classes) < _MAX_CLASSES:
            bases    = [_arg_annotation(b) for b in node.bases if b]
            methods  = [
                n.name for n in ast.walk(node)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not n.name.startswith("__")
            ]
            classes.append(_ClsInfo(
                name      = node.name,
                lineno    = node.lineno,
                bases     = bases,
                docstring = _docstring(node),
                methods   = methods[:20],
            ))

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if len(functions) >= _MAX_FUNCTIONS:
                continue
            # Skip private dunder methods at module level — low value
            if node.name.startswith("__") and node.name.endswith("__"):
                continue
            functions.append(_FnInfo(
                name      = node.name,
                qualname  = node.name,   # ast.walk loses class context; qualname set below
                lineno    = node.lineno,
                args      = _signature(node),
                docstring = _docstring(node),
            ))

    # Fix qualnames for methods (walk loses class context)
    # Re-walk at top level only
    functions.clear()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("__") and node.name.endswith("__"):
                continue
            if len(functions) >= _MAX_FUNCTIONS:
                break
            functions.append(_FnInfo(
                name     = node.name,
                qualname = node.name,
                lineno   = node.lineno,
                args     = _signature(node),
                docstring= _docstring(node),
            ))
        elif isinstance(node, ast.ClassDef):
            for mnode in node.body:
                if not isinstance(mnode, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if mnode.name.startswith("__") and mnode.name.endswith("__"):
                    continue
                if len(functions) >= _MAX_FUNCTIONS:
                    break
                functions.append(_FnInfo(
                    name     = mnode.name,
                    qualname = f"{node.name}.{mnode.name}",
                    lineno   = mnode.lineno,
                    args     = _signature(mnode),
                    docstring= _docstring(mnode),
                ))

    return _ModuleInfo(
        path      = path,
        stem      = stem,
        docstring = doc,
        imports   = imports,
        functions = functions,
        classes   = classes,
    )


# ── Graph writer ─────────────────────────────────────────────────────────────

def _write_module(info: _ModuleInfo, graph: "GraphStore",
                   project_name: str | None, root: Path) -> None:
    """Write a _ModuleInfo into the graph using upsert_node / upsert_edge."""

    rel_path = str(info.path.relative_to(root)) if root else str(info.path)
    module_node = f"module:{info.stem}"

    # ── Module node ──────────────────────────────────────────────────────────
    summary = info.docstring or f"Python module at {rel_path}"
    graph.upsert_node(
        module_node, "module",
        summary  = summary[:_MAX_SUMMARY],
        sources  = [rel_path],
        path     = rel_path,
    )

    # ── Project → module edge ────────────────────────────────────────────────
    if project_name:
        proj_node = f"project:{project_name}"
        graph.upsert_node(proj_node, "project",
                          summary=f"Project {project_name}",
                          sources=[rel_path])
        graph.upsert_edge(proj_node, "has_module", module_node, weight=1.0)

    # ── Import edges (module → other_module) ─────────────────────────────────
    for imp in info.imports:
        # Only wire edges to modules that exist in the same project (avoid
        # cluttering the graph with stdlib / third-party package nodes).
        # We'll link them lazily — if the target module node exists, add edge.
        target = f"module:{imp}"
        if graph.get_node(target):
            graph.upsert_edge(module_node, "imports", target, weight=0.5)

    # ── Function nodes ───────────────────────────────────────────────────────
    for fn in info.functions:
        fn_node = f"fn:{info.stem}.{fn.qualname}"
        fn_summary = fn.docstring or f"{fn.qualname}{fn.args}"
        graph.upsert_node(
            fn_node, "function",
            summary  = fn_summary[:_MAX_SUMMARY],
            sources  = [f"{rel_path}:{fn.lineno}"],
            path     = rel_path,
            lineno   = fn.lineno,
            signature= fn.args,
        )
        graph.upsert_edge(module_node, "contains", fn_node, weight=1.0)

    # ── Class nodes ──────────────────────────────────────────────────────────
    for cls in info.classes:
        cls_node = f"cls:{info.stem}.{cls.name}"
        bases_str = f"({', '.join(cls.bases)})" if cls.bases else ""
        methods_str = ", ".join(cls.methods[:10])
        cls_summary = (cls.docstring
                       or f"class {cls.name}{bases_str} — {methods_str}"
                       or f"class {cls.name}")
        graph.upsert_node(
            cls_node, "class",
            summary  = cls_summary[:_MAX_SUMMARY],
            sources  = [f"{rel_path}:{cls.lineno}"],
            path     = rel_path,
            lineno   = cls.lineno,
            bases    = cls.bases,
            methods  = cls.methods,
        )
        graph.upsert_edge(module_node, "contains", cls_node, weight=1.0)

    logger.debug(
        "code_indexer: %s → %d fn, %d cls",
        info.stem, len(info.functions), len(info.classes),
    )


# ── Mtime cache — skips unchanged files ─────────────────────────────────────

class _MtimeCache:
    """Lightweight file → mtime map, stored as a side-car JSON file.

    Stored at <root>/.sisyphean_index_cache.json so it stays close to the
    code it describes and survives engine restarts.
    """

    def __init__(self, root: Path) -> None:
        self._path = root / ".sisyphean_index_cache.json"
        self._data: dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        try:
            import json as _json
            if self._path.exists():
                self._data = _json.loads(self._path.read_text())
        except Exception:
            self._data = {}

    def _save(self) -> None:
        try:
            import json as _json
            self._path.write_text(_json.dumps(self._data, indent=2))
        except Exception:
            pass

    def is_fresh(self, path: Path) -> bool:
        """Return True if file hasn't changed since last index."""
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return False
        return self._data.get(str(path), 0.0) >= mtime

    def mark(self, path: Path) -> None:
        try:
            self._data[str(path)] = path.stat().st_mtime
        except OSError:
            pass

    def flush(self) -> None:
        self._save()


# ── Public API ───────────────────────────────────────────────────────────────

def index_file(
    path: Path,
    graph: "GraphStore",
    project_name: str | None = None,
    root: Path | None = None,
) -> bool:
    """Index a single Python file into graph. Returns True if file was parsed."""
    path = Path(path)
    if not path.exists() or path.suffix != ".py":
        return False
    root = root or path.parent
    info = _parse_file(path)
    if info is None:
        return False
    _write_module(info, graph, project_name, root)
    return True


def index_path(
    root: Path | str,
    graph: "GraphStore",
    project_name: str | None = None,
    incremental: bool = True,
    extensions: tuple[str, ...] = (".py",),
) -> dict[str, int]:
    """Recursively index all Python files under root into graph.

    Parameters
    ----------
    root        : directory (or single file) to index
    graph       : GraphStore to write into
    project_name: if set, creates a project node and wires module edges to it
    incremental : skip files whose mtime matches the cache (default True)
    extensions  : file extensions to process (default .py only)

    Returns a stats dict: {files_scanned, files_indexed, functions, classes}
    """
    root = Path(root)
    if root.is_file():
        ok = index_file(root, graph, project_name, root.parent)
        return {"files_scanned": 1, "files_indexed": int(ok),
                "functions": 0, "classes": 0}

    if not root.is_dir():
        logger.warning("code_indexer: %s is not a file or directory", root)
        return {"files_scanned": 0, "files_indexed": 0,
                "functions": 0, "classes": 0}

    cache = _MtimeCache(root) if incremental else None
    stats = {"files_scanned": 0, "files_indexed": 0, "functions": 0, "classes": 0}
    t0    = time.monotonic()

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skipped dirs in-place so os.walk doesn't descend into them
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fname in filenames:
            if not any(fname.endswith(ext) for ext in extensions):
                continue
            if fname in _SKIP_FILES:
                continue
            fpath = Path(dirpath) / fname
            stats["files_scanned"] += 1

            if cache and cache.is_fresh(fpath):
                continue

            info = _parse_file(fpath)
            if info is None:
                continue

            _write_module(info, graph, project_name, root)
            stats["files_indexed"] += 1
            stats["functions"]     += len(info.functions)
            stats["classes"]       += len(info.classes)

            if cache:
                cache.mark(fpath)

    if cache:
        cache.flush()

    elapsed = time.monotonic() - t0
    logger.info(
        "code_indexer: indexed %d/%d files  fn=%d  cls=%d  in %.2fs",
        stats["files_indexed"], stats["files_scanned"],
        stats["functions"], stats["classes"], elapsed,
    )
    return stats


def reindex_file(
    path: Path,
    graph: "GraphStore",
    project_name: str | None = None,
    root: Path | None = None,
) -> bool:
    """Force re-index a single file, ignoring the mtime cache.

    Called by BirdClaw's file-watcher hook on every save event.
    Returns True on success.
    """
    path = Path(path)
    root = root or path.parent
    # Invalidate cache entry so next full index doesn't skip it
    cache = _MtimeCache(root)
    cache._data.pop(str(path), None)
    cache.flush()
    return index_file(path, graph, project_name, root)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli() -> None:
    import argparse, sys

    parser = argparse.ArgumentParser(
        description="Index Python source files into the Sisyphean knowledge graph."
    )
    parser.add_argument("path", help="Directory or file to index")
    parser.add_argument("--project", "-p", default=None,
                        help="Project name to attach module nodes to")
    parser.add_argument("--full", action="store_true",
                        help="Force full re-index (ignore mtime cache)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse files but do not write to graph")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    root = Path(args.path).resolve()
    if not root.exists():
        print(f"ERROR: {root} does not exist", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        # Dry run: parse only, print summary
        total_fn = total_cls = total_files = 0
        for dirpath, dirnames, filenames in os.walk(root) if root.is_dir() else [("", [], [root.name])]:
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for fname in filenames:
                if not fname.endswith(".py"):
                    continue
                fpath = Path(dirpath) / fname if root.is_dir() else root
                info = _parse_file(fpath)
                if info:
                    total_files += 1
                    total_fn    += len(info.functions)
                    total_cls   += len(info.classes)
                    print(f"  {info.stem:30s}  fn={len(info.functions):3d}  "
                          f"cls={len(info.classes):2d}  doc={bool(info.docstring)}")
        print(f"\n  Total: {total_files} files  {total_fn} functions  {total_cls} classes")
        return

    # Live run — import the shared graph
    try:
        from engine.memory.graph import knowledge_graph as graph
    except ImportError:
        print("ERROR: cannot import engine.memory.graph — run from project root",
              file=sys.stderr)
        sys.exit(1)

    stats = index_path(root, graph,
                       project_name=args.project or root.name,
                       incremental=not args.full)
    print(
        f"\nIndexed {stats['files_indexed']}/{stats['files_scanned']} files  "
        f"fn={stats['functions']}  cls={stats['classes']}"
    )


if __name__ == "__main__":
    _cli()
