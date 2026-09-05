import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "bindings" / "python"))

VOCAB_SIZE = 200
TOKENS = {"heart": 10, "attack": 11, "aspirin": 12, "drug": 13, "powai": 14,
          "mumbai": 15, "##s": 16, "prevent": 17}

@pytest.fixture
def vocab():
    v = [f"[unused{i}]" for i in range(VOCAB_SIZE)]
    for tok, i in TOKENS.items():
        v[i] = tok
    return v

@pytest.fixture
def qlut():
    q = np.zeros(VOCAB_SIZE)
    for i in TOKENS.values():
        q[i] = 1.5
    return q

class FakeEncoder:
    """Deterministic encoder: term weight = 2.0 for every vocab token present
    in the text, so tests can predict exact scores."""
    model_id = "fake-model"

    def __init__(self, vocab):
        self._v2i = {t: i for i, t in enumerate(vocab)}
        self._vocab = vocab

    def vocab(self):
        return self._vocab

    def qlut(self):
        q = np.zeros(len(self._vocab))
        for i in TOKENS.values():
            q[i] = 1.5
        return q

    def encode(self, texts, **kw):
        out = []
        for t in texts:
            words = t.lower().split()
            out.append({self._v2i[w]: 2.0 for w in words if w in self._v2i})
        return out

@pytest.fixture
def fake_encoder(vocab):
    return FakeEncoder(vocab)

@pytest.fixture
def index(tmp_path, vocab, fake_encoder, monkeypatch):
    from sqlite_sparse.api import SparseIndex
    monkeypatch.setattr(SparseIndex, "encoder", lambda self: fake_encoder)
    ix = SparseIndex.__new__(SparseIndex)
    from sqlite_sparse.store import SparseStore
    from sqlite_sparse.search import QueryEngine
    ix.store = SparseStore(str(tmp_path / "t.db"))
    ix._enc = fake_encoder
    ix._model = "fake-model"
    ix._max_seq, ix._device = 256, None
    ix.store.init_model("fake-model", vocab, fake_encoder.qlut())
    ix.engine = QueryEngine(ix.store.db)
    return ix
