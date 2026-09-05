"""Extension tests; skipped when no sparse0 binary is present."""
import sqlite3

import pytest

sqlite_sparse = pytest.importorskip("sqlite_sparse")

def _ext_or_skip():
    if not hasattr(sqlite3.Connection, "enable_load_extension"):
        pytest.skip("this Python's sqlite3 cannot load extensions")
    try:
        return sqlite_sparse.loadable_path()
    except FileNotFoundError:
        pytest.skip("sparse0 extension not built")

def test_extension_loads_and_reports_version():
    _ext_or_skip()
    db = sqlite3.connect(":memory:")
    sqlite_sparse.load(db)
    v = db.execute("SELECT sparse_version()").fetchone()[0]
    assert v.startswith("sqlite-sparse/1")

def test_extension_reads_a_python_built_index(tmp_path, index):
    """Format contract: the extension ranks a Python-written database identically."""
    _ext_or_skip()
    index.add(id="a", text="heart attack aspirin prevent")
    index.add(id="b", text="powai mumbai")
    index.commit()
    path = index.store.path
    py = [(h["id"], round(h["score"], 4)) for h in index.search("heart", k=5)]

    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    sqlite_sparse.load(db)
    db.execute("CREATE VIRTUAL TABLE temp.t USING sparse0()")
    c = [(r[0], round(r[1], 4)) for r in
         db.execute("SELECT rowid, score FROM temp.t WHERE t MATCH ? AND k=5", ("heart",))]
    assert c == py

def test_extension_breaks_ties_by_rowid(tmp_path, index):
    """Equal scores rank lower rowid first."""
    _ext_or_skip()
    for i in range(6):
        index.add(id=f"d{i}", text="powai mumbai")
    index.commit()
    db = sqlite3.connect(f"file:{index.store.path}?mode=ro", uri=True)
    sqlite_sparse.load(db)
    db.execute("CREATE VIRTUAL TABLE temp.t USING sparse0()")
    rows = db.execute("SELECT rowid, score FROM temp.t WHERE t MATCH ? AND k=4", ("mumbai",)).fetchall()
    assert [r[0] for r in rows] == [1, 2, 3, 4]
    assert len({round(r[1], 6) for r in rows}) == 1

def test_extension_delete_and_count(tmp_path, index):

    _ext_or_skip()
    index.add(id="a", text="heart attack aspirin")
    index.add(id="b", text="heart attack symptoms")
    index.commit()
    db = sqlite3.connect(index.store.path)
    sqlite_sparse.load(db)
    db.execute("CREATE VIRTUAL TABLE temp.t USING sparse0()")
    before = [r[0] for r in db.execute("SELECT rowid FROM temp.t WHERE t MATCH 'heart' AND k=5")]
    assert sorted(before) == [1, 2]
    assert db.execute("SELECT count(*) FROM temp.t").fetchone()[0] == 2
    db.execute("DELETE FROM temp.t WHERE rowid=1")
    after = [r[0] for r in db.execute("SELECT rowid FROM temp.t WHERE t MATCH 'heart' AND k=5")]
    assert after == [2]
    assert db.execute("SELECT count(*) FROM temp.t").fetchone()[0] == 1

def test_limit_and_order_by(tmp_path, index):

    _ext_or_skip()
    for i, t in enumerate(["heart attack aspirin", "heart", "heart heart attack", "powai mumbai", "attack"]):
        index.add(id=f"d{i}", text=t)
    index.commit()
    db = sqlite3.connect(f"file:{index.store.path}?mode=ro", uri=True)
    sqlite_sparse.load(db)
    db.execute("CREATE VIRTUAL TABLE temp.t USING sparse0()")
    by_k = db.execute("SELECT rowid FROM temp.t WHERE t MATCH 'heart attack' AND k=3").fetchall()
    by_limit = db.execute("SELECT rowid FROM temp.t WHERE t MATCH 'heart attack' LIMIT 3").fetchall()
    ordered = db.execute("SELECT rowid FROM temp.t WHERE t MATCH 'heart attack' ORDER BY score DESC LIMIT 3").fetchall()
    assert by_k == by_limit == ordered and len(by_k) == 3
    plan = db.execute("EXPLAIN QUERY PLAN SELECT rowid FROM temp.t WHERE t MATCH 'heart attack' ORDER BY score DESC LIMIT 3").fetchall()
    assert not any("TEMP B-TREE" in str(r) for r in plan), plan   # order consumed by the vtab, no sort step

def test_compact_removes_deleted_postings(tmp_path, index):

    _ext_or_skip()
    for i in range(6):
        index.add(id=f"d{i}", text="heart attack aspirin" if i % 2 == 0 else "powai mumbai")
    index.commit()
    db = sqlite3.connect(index.store.path)
    sqlite_sparse.load(db)
    db.execute("CREATE VIRTUAL TABLE temp.t USING sparse0()")
    before = db.execute("SELECT sum(length(docs))/4 FROM postings").fetchone()[0]
    db.execute("DELETE FROM temp.t WHERE rowid=1")
    db.execute("DELETE FROM temp.t WHERE rowid=2")
    db.commit()
    hits_before = db.execute("SELECT rowid FROM temp.t WHERE t MATCH 'heart' AND k=10").fetchall()
    removed = db.execute("SELECT sparse_compact()").fetchone()[0]
    db.commit()
    after = db.execute("SELECT sum(length(docs))/4 FROM postings").fetchone()[0]
    assert removed > 0 and after == before - removed
    assert db.execute("SELECT count(*) FROM docs WHERE deleted=1").fetchone()[0] == 0
    assert db.execute("SELECT rowid FROM temp.t WHERE t MATCH 'heart' AND k=10").fetchall() == hits_before
    assert db.execute("SELECT sparse_compact()").fetchone()[0] == 0
