"""Model registry, no network."""
import os

import pytest

from sqlite_sparse import models

def test_unknown_alias_is_a_key_error():
    with pytest.raises(KeyError):
        models.fetch("nonexistent-model")

def test_bad_quant_is_rejected():
    with pytest.raises(ValueError):
        models.fetch("mini", quant="q4")

def test_cache_dir_honors_override(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_SPARSE_CACHE", str(tmp_path / "c"))
    assert models.cache_dir() == str(tmp_path / "c")
    assert os.path.isdir(tmp_path / "c")

def test_fetch_skips_download_when_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_SPARSE_CACHE", str(tmp_path))
    d = tmp_path / "mini"
    d.mkdir()
    (d / "mini_q8.gguf").write_bytes(b"x")
    (d / "mini.sprs").write_bytes(b"x")
    calls = []
    monkeypatch.setattr(models, "_download", lambda *a: calls.append(a))
    gguf, sprs = models.fetch("mini")
    assert calls == []
    assert gguf.endswith("mini_q8.gguf") and sprs.endswith("mini.sprs")
