"""Where the gitignored inputs live, including from a linked git worktree.

data/ and devset/ are gitignored, so a worktree made for a parallel experiment starts without
them. Rather than copying 2 GB of match data or naming its path on a command line (which the
data guard hook blocks), the loaders fall back to the main checkout's copies. Caches derived from
code -- devset/cache/ -- stay per worktree, so two experiments never overwrite each other's.
"""

import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main_checkout():
    """Root of the main working tree when running in a linked worktree, else None."""
    try:
        common = subprocess.check_output(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    main = os.path.dirname(os.path.normpath(common))
    if os.path.normcase(os.path.realpath(main)) == os.path.normcase(os.path.realpath(ROOT)):
        return None
    return main


def shared(relative, probe):
    """ROOT/relative if `probe` exists under it, else the main checkout's, else ROOT/relative."""
    local = os.path.join(ROOT, relative)
    if os.path.exists(os.path.join(local, probe)):
        return local
    main = main_checkout()
    if main and os.path.exists(os.path.join(main, relative, probe)):
        return os.path.join(main, relative)
    return local
