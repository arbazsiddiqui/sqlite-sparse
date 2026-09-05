"""Golden fixtures: stored term weights, query weights and top-k for a fixed model."""
import json, os, sqlite3
import pytest

sqlite_sparse = pytest.importorskip("sqlite_sparse")
GOLD = json.load(open(os.path.join(os.path.dirname(__file__), "fixtures", "golden.json")))

pytestmark = pytest.mark.skipif(os.environ.get("SQLITE_SPARSE_ENCODE_TEST") != "1",
                                reason="set SQLITE_SPARSE_ENCODE_TEST=1")

def test_golden(tmp_path):
    if not hasattr(sqlite3.Connection, "enable_load_extension"):
        pytest.skip("this Python's sqlite3 cannot load extensions")
    try:
        sqlite_sparse.loadable_path()
    except FileNotFoundError:
        pytest.skip("sparse0 extension not built")
    db = sqlite3.connect(str(tmp_path / "g.db")); sqlite_sparse.load(db)
    sqlite_sparse.register(db, GOLD["model"], max_seq=GOLD["max_seq"])
    db.execute(f"CREATE VIRTUAL TABLE t USING sparse0(model='{GOLD['model']}')")
    for d in GOLD["documents"]:
        db.execute("INSERT INTO t(rowid, text) VALUES (?, ?)", (d["rowid"], d["text"]))
    db.commit()
    meta = dict(db.execute("SELECT k, v FROM meta"))
    assert meta["sidecar_sha256"] == GOLD["sidecar_sha256"], "sidecar changed; regenerate fixtures deliberately"
    import numpy as np
    vocab = json.loads(meta["vocab"])
    stored = {d["rowid"]: {} for d in GOLD["documents"]}
    for t, dd, ws in db.execute("SELECT t, docs, ws FROM postings"):
        for di, wi in zip(np.frombuffer(dd, dtype="<i4"), np.frombuffer(ws, dtype=np.uint8)):
            stored[int(di)][vocab[t]] = int(wi)
    for d in GOLD["documents"]:
        got = stored[d["rowid"]]
        for term, w in d["top_terms_u8"].items():
            assert abs(got.get(term, 0) - w) <= 1, (d["rowid"], term, got.get(term), w)
        assert abs(len(got) - d["n_terms"]) <= max(3, d["n_terms"] // 20)
    for q in GOLD["queries"]:
        rows = [(r[0], round(r[1], 3)) for r in db.execute("SELECT rowid, score FROM t WHERE t MATCH ? AND k=3", (q["text"],))]
        assert [r[0] for r in rows] == [x[0] for x in q["topk"]], (q["text"], rows, q["topk"])
        for (rid, s), (grid, gs) in zip(rows, q["topk"]):
            assert abs(s - gs) < 0.02, (q["text"], rid, s, gs)
