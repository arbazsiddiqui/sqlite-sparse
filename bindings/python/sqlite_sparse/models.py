"""Named models: download the GGUF and .sprs for an alias into
~/.cache/sqlite-sparse and register them. urllib only; HF_TOKEN honored."""
import os
import sys
import urllib.request
from pathlib import Path

HUB = "https://huggingface.co"

# alias -> (repo, file stem, encoder params)
MODELS = {
    "mini":         ("arbazsiddiqui/opensearch-neural-sparse-doc-v2-mini-GGUF", "mini", "23M"),
    "base":         ("arbazsiddiqui/opensearch-neural-sparse-doc-v3-distill-GGUF", "base", "67M"),
    "multilingual": ("arbazsiddiqui/opensearch-neural-sparse-multilingual-v1-GGUF", "multilingual", "168M"),
}


def cache_dir():
    root = os.environ.get("SQLITE_SPARSE_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "sqlite-sparse")
    Path(root).mkdir(parents=True, exist_ok=True)
    return root


def _download(url, dest, label):
    req = urllib.request.Request(url)
    tok = os.environ.get("HF_TOKEN")
    if tok:
        req.add_header("Authorization", f"Bearer {tok}")
    tmp = dest + ".part"
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total and sys.stderr.isatty():
                sys.stderr.write(f"\r{label}: {done * 100 // total}%")
        if total and sys.stderr.isatty():
            sys.stderr.write("\n")
    os.replace(tmp, dest)


def fetch(alias="mini", quant="q8"):
    """Download the GGUF and sidecar for `alias` if not cached. Returns (gguf, sprs)."""
    if alias not in MODELS:
        raise KeyError(f"unknown model alias {alias!r}; known: {sorted(MODELS)}")
    repo, stem, _ = MODELS[alias]
    if quant not in ("q8", "f16"):
        raise ValueError("quant must be 'q8' or 'f16'")
    d = os.path.join(cache_dir(), alias)
    Path(d).mkdir(parents=True, exist_ok=True)
    out = []
    for fname in (f"{stem}_{quant}.gguf", f"{stem}.sprs"):
        dest = os.path.join(d, fname)
        if not os.path.exists(dest):
            _download(f"{HUB}/{repo}/resolve/main/{fname}", dest, fname)
        out.append(dest)
    return tuple(out)


def register(conn, alias="mini", quant="q8", max_seq=512):
    """fetch() then sparse_register(). Returns the alias."""
    gguf, sprs = fetch(alias, quant)
    conn.execute("SELECT sparse_register(?, ?, ?, ?)", (alias, gguf, sprs, max_seq)).fetchone()
    return alias
