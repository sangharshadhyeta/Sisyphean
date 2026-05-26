# Sisyphean skill — read, search, and create notes in an Obsidian vault
import sys
import os
import re
from pathlib import Path


def get_vault() -> Path:
    env = os.environ.get("OBSIDIAN_VAULT", "")
    if env:
        return Path(env).expanduser()
    return Path.home() / "Documents" / "Obsidian"


def find_notes(vault: Path, query: str) -> list[Path]:
    """Return all .md files whose name or content matches query (case-insensitive)."""
    q = query.lower()
    matches = []
    try:
        for p in vault.rglob("*.md"):
            if q in p.stem.lower():
                matches.append(p)
                continue
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                if q in content.lower():
                    matches.append(p)
            except Exception:
                pass
    except Exception as e:
        print(f"Error scanning vault: {e}")
    return matches


def cmd_search(vault: Path, query: str) -> None:
    if not vault.exists():
        print(f"Vault not found: {vault}")
        print("Set OBSIDIAN_VAULT environment variable to the correct path.")
        return
    results = find_notes(vault, query)
    if not results:
        print(f"No notes matching '{query}' in {vault}")
        return
    print(f"Found {len(results)} note(s) matching '{query}':\n")
    for p in results[:30]:
        try:
            first_line = p.read_text(encoding="utf-8", errors="ignore").splitlines()[0][:80]
        except Exception:
            first_line = "(unreadable)"
        rel = p.relative_to(vault)
        print(f"  {rel}")
        print(f"    {first_line}")
        print()


def cmd_read(vault: Path, name: str) -> None:
    if not vault.exists():
        print(f"Vault not found: {vault}")
        return
    # Try exact match first, then fuzzy
    candidates = list(vault.rglob(f"{name}.md")) + list(vault.rglob(f"*{name}*.md"))
    if not candidates:
        print(f"Note not found: {name}")
        return
    target = candidates[0]
    try:
        content = target.read_text(encoding="utf-8", errors="ignore")
        print(f"# {target.stem}\n")
        print(content[:5000])
        if len(content) > 5000:
            print(f"\n[... truncated at 5000 chars — {len(content)} total]")
    except Exception as e:
        print(f"Error reading note: {e}")


def cmd_create(vault: Path, name: str, content: str) -> None:
    if not vault.exists():
        try:
            vault.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Could not create vault directory: {e}")
            return
    safe_name = re.sub(r'[<>:"/\\|?*]', "_", name)
    target = vault / f"{safe_name}.md"
    try:
        target.write_text(content, encoding="utf-8")
        print(f"Created: {target}")
    except Exception as e:
        print(f"Error creating note: {e}")


def main():
    vault = get_vault()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python skills/obsidian.py search QUERY")
        print("  python skills/obsidian.py read NOTE_NAME")
        print('  python skills/obsidian.py create NOTE_NAME "content"')
        print(f"\nVault: {vault}")
        return

    cmd = sys.argv[1].lower()

    if cmd == "search":
        if len(sys.argv) < 3:
            print("Usage: python skills/obsidian.py search QUERY")
            return
        cmd_search(vault, " ".join(sys.argv[2:]))

    elif cmd == "read":
        if len(sys.argv) < 3:
            print("Usage: python skills/obsidian.py read NOTE_NAME")
            return
        cmd_read(vault, " ".join(sys.argv[2:]))

    elif cmd == "create":
        if len(sys.argv) < 4:
            print('Usage: python skills/obsidian.py create NOTE_NAME "content"')
            return
        cmd_create(vault, sys.argv[2], " ".join(sys.argv[3:]))

    else:
        print(f"Unknown command: {cmd}")
        print("Available: search, read, create")


if __name__ == "__main__":
    main()
