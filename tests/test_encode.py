"""Encode path through the compiled extension. Downloads model files, so it
runs only with SQLITE_SPARSE_ENCODE_TEST=1 (CI sets it)."""
import os
import sqlite3

import pytest

sqlite_sparse = pytest.importorskip("sqlite_sparse")

pytestmark = pytest.mark.skipif(
    os.environ.get("SQLITE_SPARSE_ENCODE_TEST") != "1",
    reason="set SQLITE_SPARSE_ENCODE_TEST=1 (downloads ~75MB of model files)")

def test_register_insert_search(tmp_path):
    if not hasattr(sqlite3.Connection, "enable_load_extension"):
        pytest.skip("this Python's sqlite3 cannot load extensions")
    try:
        sqlite_sparse.loadable_path()
    except FileNotFoundError:
        pytest.skip("sparse0 extension not built")
    db = sqlite3.connect(str(tmp_path / "e.db"))
    sqlite_sparse.load(db)
    alias = sqlite_sparse.register(db, "mini")
    db.execute(f"CREATE VIRTUAL TABLE t USING sparse0(model='{alias}')")
    db.execute("INSERT INTO t(rowid, text) VALUES (1, 'Aspirin lowers the risk of heart attack and stroke.')")
    db.execute("INSERT INTO t(rowid, text) VALUES (2, 'Powai is a lakeside suburb of Mumbai.')")
    db.execute("INSERT INTO t(rowid, text) VALUES (3, 'SQLite is an embedded relational database engine.')")
    db.commit()
    rows = db.execute("SELECT rowid FROM t WHERE t MATCH 'what prevents cardiac arrest' AND k=2").fetchall()
    assert rows and rows[0][0] == 1, rows

    db.execute("INSERT INTO t(rowid, text) VALUES (4, ?)", (" ".join(["myocardial infarction"] * 400),))
    db.commit()
    assert db.execute("SELECT count(*) FROM t").fetchone()[0] == 4

def test_truncation_is_recorded_and_wrong_model_is_refused(tmp_path):
    if not hasattr(sqlite3.Connection, "enable_load_extension"):
        pytest.skip("this Python's sqlite3 cannot load extensions")
    try:
        sqlite_sparse.loadable_path()
    except FileNotFoundError:
        pytest.skip("sparse0 extension not built")
    db = sqlite3.connect(str(tmp_path / "t.db"))
    sqlite_sparse.load(db)
    sqlite_sparse.register(db, "mini", max_seq=64)
    db.execute("CREATE VIRTUAL TABLE t USING sparse0(model='mini')")
    db.execute("INSERT INTO t(rowid, text) VALUES (1, 'short note')")
    db.execute("INSERT INTO t(rowid, text) VALUES (2, ?)", (" ".join(["myocardial infarction"] * 200),))
    db.commit()
    rows = db.execute("SELECT id, ntokens, truncated FROM docs ORDER BY id").fetchall()
    assert rows[0][2] == 0 and rows[0][1] < 64
    assert rows[1][2] == 1 and rows[1][1] > 64
    meta = dict(db.execute("SELECT k, v FROM meta WHERE k IN ('encoder_sha256','sidecar_sha256','vocab_sha256','max_seq')"))
    assert len(meta["vocab_sha256"]) == 64 and meta["max_seq"] == "64"

    from sqlite_sparse.store import SparseStore
    import numpy as np
    other = SparseStore(str(tmp_path / "other.db"))
    other.init_model("fake", [f"tok{i}" for i in range(300)], np.ones(300))
    other.db.commit()
    db2 = sqlite3.connect(str(tmp_path / "other.db"))
    sqlite_sparse.load(db2)
    sqlite_sparse.register(db2, "mini")
    db2.execute("CREATE VIRTUAL TABLE temp.o USING sparse0(model='mini')")
    with pytest.raises(sqlite3.OperationalError, match="different model"):
        db2.execute("INSERT INTO temp.o(rowid, text) VALUES (1, 'x')")


def test_query_tokenizer_matches_hf(tmp_path):
    """Both tokenizers reproduce the HF tokenizer's ids, including accents,
    non-Latin scripts, CJK, punctuation, control characters and long words."""
    if not hasattr(sqlite3.Connection, "enable_load_extension"):
        pytest.skip("this Python's sqlite3 cannot load extensions")
    try:
        sqlite_sparse.loadable_path()
    except FileNotFoundError:
        pytest.skip("sparse0 extension not built")
    import json
    from sqlite_sparse.search import QueryEngine
    gold = json.load(open(os.path.join(os.path.dirname(__file__), "fixtures", "tokenizer_golden.json")))
    cases = gold["models"]["mini"]["cases"]
    db = sqlite3.connect(str(tmp_path / "tok.db"))
    sqlite_sparse.load(db)
    sqlite_sparse.register(db, "mini")
    db.execute("CREATE VIRTUAL TABLE t USING sparse0(model='mini')")
    db.commit()
    eng = QueryEngine(db)
    v2i = eng._v2i
    for c in cases:
        assert eng._wordpiece(c["text"]) == c["ids"], ("python", c["text"])
        toks = json.loads(db.execute("SELECT sparse_tokens(?)", (c["text"],)).fetchone()[0])
        assert [v2i[t] for t in toks] == c["ids"], ("extension", c["text"], toks)
