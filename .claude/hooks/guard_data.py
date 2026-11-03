"""PreToolUse hook: keep the gated match data out of Claude's context.

The competition terms forbid giving the provided videos and datasets, "or any extracts thereof", to
a third-party hosted LLM -- and Claude Code is one. Code that Claude writes may read data/ when it
runs locally; Claude itself must not read, list, search or print it. This hook blocks the tool
calls that would, on top of the Read deny rule in .claude/settings.json.

It is a guard rail, not a sandbox: a script can still print what it read. CLAUDE.md covers that
side -- scripts report aggregates only.

Exit code 2 blocks the call and hands stderr back to Claude as the reason.
"""

import json
import os
import re
import sys

# Params
# The one file under data/ that is committed and public, so reading it gives nothing away.
ALLOWED = ("data/README.md",)

# Shell commands: a path segment called data (relative or absolute, either slash), a bare `data`
# argument at the end of the command or in quotes, the bundle's file stems, and the variable that
# relocates the bundle. Deliberately broad -- a false positive costs a rephrase, a false negative
# costs a breach of the terms.
SHELL_PATTERNS = (
    re.compile(r"(^|[\s\"'=(,:/\\])data[/\\]", re.IGNORECASE),
    re.compile(r"(^|\s)data\s*$", re.IGNORECASE),
    re.compile(r"[\"']data[\"']", re.IGNORECASE),
    re.compile(r"BAR-ATH_(tactical|broadcast)", re.IGNORECASE),
    re.compile(r"MTAH_DATA_DIR"),
)

# File tools: the input fields that name a path or a glob.
PATH_FIELDS = ("file_path", "path", "notebook_path")
GLOB_FIELDS = ("pattern", "glob")

REASON = ("Blocked by .claude/hooks/guard_data.py: the match data under data/ must not enter "
          "Claude's context (competition T&Cs, see CLAUDE.md). Write a script that processes it "
          "locally and prints aggregates only, and run that script without naming data/ on the "
          "command line -- games.asset_paths() resolves the files.")


def data_roots(project):
    roots = [os.path.join(project, "data")]
    if os.environ.get("MTAH_DATA_DIR"):
        roots.append(os.environ["MTAH_DATA_DIR"])
    return [os.path.normcase(os.path.realpath(root)) for root in roots]


def is_allowed(path, project):
    return any(os.path.normcase(os.path.realpath(os.path.join(project, allowed))) == path
               for allowed in ALLOWED)


def path_is_data(value, cwd, project):
    path = os.path.normcase(os.path.realpath(os.path.join(cwd, value)))
    if is_allowed(path, project):
        return False
    for root in data_roots(project):
        if path == root or path.startswith(root + os.sep):
            return True
    return False


def blocked(payload):
    tool = payload.get("tool_name", "")
    tool_input = payload.get("tool_input") or {}
    project = os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or os.getcwd()
    cwd = payload.get("cwd") or project

    if tool in ("Bash", "PowerShell"):
        command = tool_input.get("command", "")
        return any(pattern.search(command) for pattern in SHELL_PATTERNS)

    for field in PATH_FIELDS:
        value = tool_input.get(field)
        if value and path_is_data(value, cwd, project):
            return True
    for field in GLOB_FIELDS:
        value = tool_input.get(field)
        if value and SHELL_PATTERNS[0].search(" " + value.replace("\\", "/")):
            return True
    return False


def main():
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return 0  # not a tool call we understand; let Claude Code's own rules decide
    if blocked(payload):
        sys.stderr.write(REASON + "\n")
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
