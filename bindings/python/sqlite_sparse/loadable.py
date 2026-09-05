"""Locate and load the sparse0 extension. Search order: the copy bundled in
the package, $SQLITE_SPARSE_EXT, then the repo's build tree."""
import os
import sys
from pathlib import Path

_SUFFIX = {"darwin": ".dylib", "win32": ".dll"}.get(sys.platform, ".so")


def _candidates():
    here = Path(__file__).parent
    yield here / f"sparse0{_SUFFIX}"
    env = os.environ.get("SQLITE_SPARSE_EXT")
    if env:
        yield Path(env)
    yield here.parents[2] / "build" / f"sparse0{_SUFFIX}"


def loadable_path():
    """Absolute path of the extension binary."""
    tried = []
    for p in _candidates():
        tried.append(str(p))
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "sparse0 extension not found. Tried:\n  " + "\n  ".join(tried) +
        "\nBuild it with `make` at the repo root, or set SQLITE_SPARSE_EXT to the binary."
    )


def load(conn):
    """Load sparse0 into a connection. Returns the path used."""
    if not hasattr(conn, "enable_load_extension"):
        raise RuntimeError(
            "this Python's sqlite3 module was built without loadable-extension "
            "support (common with the python.org and GitHub Actions macOS builds). "
            "Use a Python from Homebrew or conda, or `pip install sqlean.py` and "
            "connect with `import sqlean as sqlite3`."
        )
    path = loadable_path()
    conn.enable_load_extension(True)
    conn.load_extension(path)
    conn.enable_load_extension(False)
    return path
