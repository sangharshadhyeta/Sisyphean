# Sisyphean skill — GitHub operations via the gh CLI (issues, PRs, clone, search)
import sys
import subprocess
import shutil
import json
import os


def run_gh(*args) -> tuple[bool, str]:
    """Run gh command, return (ok, output)."""
    if not shutil.which("gh"):
        return False, (
            "gh CLI not found. Install from https://cli.github.com/ then run:\n"
            "    gh auth login"
        )
    try:
        result = subprocess.run(
            ["gh"] + list(args),
            capture_output=True, text=True, timeout=30
        )
        output = result.stdout + (("\n" + result.stderr) if result.stderr.strip() else "")
        return result.returncode == 0, output.strip()
    except subprocess.TimeoutExpired:
        return False, "Error: gh command timed out after 30s"
    except Exception as e:
        return False, f"Error running gh: {e}"


def cmd_issues(repo: str) -> None:
    ok, out = run_gh("issue", "list", "--repo", repo, "--state", "open",
                     "--json", "number,title,author,createdAt,url", "--limit", "20")
    if not ok:
        print(out)
        return
    try:
        issues = json.loads(out)
    except json.JSONDecodeError:
        print(out)
        return
    if not issues:
        print(f"No open issues for {repo}.")
        return
    print(f"Open issues for {repo} ({len(issues)}):\n")
    for issue in issues:
        print(f"  #{issue['number']} {issue['title']}")
        print(f"    By {issue['author']['login']} on {issue['createdAt'][:10]}")
        print(f"    {issue['url']}")
        print()


def cmd_pr(repo: str) -> None:
    ok, out = run_gh("pr", "list", "--repo", repo, "--state", "open",
                     "--json", "number,title,author,createdAt,url", "--limit", "20")
    if not ok:
        print(out)
        return
    try:
        prs = json.loads(out)
    except json.JSONDecodeError:
        print(out)
        return
    if not prs:
        print(f"No open PRs for {repo}.")
        return
    print(f"Open PRs for {repo} ({len(prs)}):\n")
    for pr in prs:
        print(f"  #{pr['number']} {pr['title']}")
        print(f"    By {pr['author']['login']} on {pr['createdAt'][:10]}")
        print(f"    {pr['url']}")
        print()


def cmd_clone(repo: str) -> None:
    dest = os.path.join("workspace", repo.replace("/", "_"))
    ok, out = run_gh("repo", "clone", repo, dest)
    if ok:
        print(f"Cloned {repo} → {dest}")
    else:
        print(out)


def cmd_search(query: str) -> None:
    ok, out = run_gh("search", "repos", query, "--limit", "10",
                     "--json", "fullName,description,stargazersCount,url")
    if not ok:
        print(out)
        return
    try:
        repos = json.loads(out)
    except json.JSONDecodeError:
        print(out)
        return
    if not repos:
        print("No repositories found.")
        return
    print(f"Top repos for '{query}':\n")
    for r in repos:
        desc = r.get("description") or "(no description)"
        print(f"  {r['fullName']} ★{r['stargazersCount']}")
        print(f"    {desc}")
        print(f"    {r['url']}")
        print()


COMMANDS = {
    "issues": (cmd_issues, "OWNER/REPO"),
    "pr":     (cmd_pr,     "OWNER/REPO"),
    "clone":  (cmd_clone,  "OWNER/REPO"),
    "search": (cmd_search, "QUERY"),
}


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        for name, (_, usage) in COMMANDS.items():
            print(f"  python skills/github_ops.py {name} {usage}")
        return

    cmd = sys.argv[1].lower()
    arg = " ".join(sys.argv[2:])

    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(COMMANDS)}")
        return

    COMMANDS[cmd][0](arg)


if __name__ == "__main__":
    main()
