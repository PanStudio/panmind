"""Durable writes: atomic file replacement and git commits of profile history."""

import fcntl
import logging
import os
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def atomic_write_text(path: Path, text: str) -> None:
    """Write text so readers see either the old file or the new one, never a mix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        suffix=".tmp", prefix=f"{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        Path(temp_name).replace(path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def atomic_write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Serialize ``data`` as YAML and write it atomically."""
    atomic_write_text(
        path,
        yaml.safe_dump(
            data, default_flow_style=False, sort_keys=False, allow_unicode=True
        ),
    )


LOCK_TIMEOUT = 30.0
GIT_TIMEOUT = 30.0


@contextmanager
def flock(path: Path, timeout: float = LOCK_TIMEOUT) -> Iterator[None]:
    """Hold an exclusive flock on ``path``, waiting at most ``timeout`` seconds.

    Raises:
        TimeoutError: If another process holds the lock past the deadline.
    """
    deadline = time.monotonic() + timeout
    with path.open("a") as lock:
        while True:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"{path.name} is held by another PanMind process; "
                        f"gave up after {timeout:.0f} s"
                    ) from None
                time.sleep(0.05)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


_held = threading.local()


@contextmanager
def panmind_lock(panmind_dir: Path) -> Iterator[None]:
    """Serialize profile writes and their commits under ``panmind_dir``.

    A user accept, an auto-accept from another session's hook and an import
    each read, edit and commit the same files. The lock is an flock, so it
    holds across processes and threads, and it is reentrant within a thread
    so a locked caller can call ``git_commit``, which takes it too.
    """
    key = str(Path(panmind_dir).resolve())
    depth: dict[str, int] = _held.__dict__.setdefault("depth", {})
    if depth.get(key):
        depth[key] += 1
        try:
            yield
        finally:
            depth[key] -= 1
        return
    Path(key).mkdir(parents=True, exist_ok=True)
    with flock(Path(key) / "panmind.lock"):
        depth[key] = 1
        try:
            yield
        finally:
            depth[key] = 0


_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "panmind",
    "GIT_AUTHOR_EMAIL": "panmind@localhost",
    "GIT_COMMITTER_NAME": "panmind",
    "GIT_COMMITTER_EMAIL": "panmind@localhost",
}

_GITIGNORE = """\
# Derived or high-churn state; profiles and proposals are the history.
observations/
rendered/
temporary/
*.lock
*.tmp
"""


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    has_identity = subprocess.run(
        ["git", "-C", str(repo), "config", "user.email"],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False,
        env=env, timeout=GIT_TIMEOUT,
    ).stdout.strip()
    if not has_identity:
        env.update(_GIT_IDENTITY)
    # Signing or hooks can prompt on stdin, and this runs from hooks and the
    # stdio MCP server where stdin carries JSON-RPC frames.
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "commit.gpgsign=false", *args],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False,
        env=env, timeout=GIT_TIMEOUT,
    )


def git_commit(repo: Path, paths: Iterable[Path], message: str) -> str | None:
    """Commit ``paths`` in ``repo``, initializing it if needed.

    Returns the new commit sha, or None when git is unavailable, nothing
    changed, or the commit failed. Profile writes must not fail because
    history could not be recorded, so failures are logged, not raised.
    """
    with panmind_lock(repo):
        return _git_commit(repo, paths, message)


def _git_commit(repo: Path, paths: Iterable[Path], message: str) -> str | None:
    try:
        if not (repo / ".git").exists():
            repo.mkdir(parents=True, exist_ok=True)
            if _git(repo, "init", "-q").returncode != 0:
                return None
            gitignore = repo / ".gitignore"
            if not gitignore.exists():
                atomic_write_text(gitignore, _GITIGNORE)
            paths = [*paths, gitignore]
        rel = [str(Path(p).resolve().relative_to(repo.resolve())) for p in paths]
        _git(repo, "add", "--", *rel)
        if _git(repo, "diff", "--cached", "--quiet", "--", *rel).returncode == 0:
            return None
        # Only the given paths, so a file someone else staged never lands
        # under this message.
        result = _git(repo, "commit", "-q", "-m", message, "--", *rel)
        if result.returncode != 0:
            logger.warning("git commit in %s failed: %s", repo, result.stderr.strip())
            return None
        return _git(repo, "rev-parse", "HEAD").stdout.strip() or None
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        logger.warning("git commit in %s failed: %s", repo, exc)
        return None
