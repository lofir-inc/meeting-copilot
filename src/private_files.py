"""Private permissions for local meeting files."""

import os
from pathlib import Path
from contextlib import contextmanager


def private_umask() -> None:
    """Set the process default before opening logs or starting child processes."""
    os.umask(0o077)


def make_session_directory(session_dir: Path | str) -> None:
    """Protect the sessions parent and new meetings without walking old contents."""
    session_dir = Path(session_dir)
    session_dir.parent.mkdir(parents=True, exist_ok=True)
    if session_dir.parent.name == "sessions":
        session_dir.parent.chmod(0o700)
    try:
        session_dir.mkdir(mode=0o700)
    except FileExistsError:
        if not session_dir.is_dir():
            raise
    else:
        session_dir.chmod(0o700)


@contextmanager
def private_writer(path: Path | str, *, binary: bool = False):
    """Create with 0600 even under a permissive umask, including overwritten files."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb" if binary else "w",
                   **({} if binary else {"encoding": "utf-8"})) as handle:
        os.fchmod(handle.fileno(), 0o600)
        yield handle
        os.fchmod(handle.fileno(), 0o600)
