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

def _load(path, ro=False):
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True) if ro else sqlite3.connect(path)
    sqlite_sparse.load(db)
    return db

def test_terms_match_equals_text_match_on_a_model_index(tmp_path, index):
    """A query vector with the file's own query weights scores like the text path."""
    _ext_or_skip()
    for i, t in enumerate(["heart attack aspirin", "heart", "powai mumbai", "attack heart heart"]):
        index.add(id=f"d{i}", text=t)
    index.commit()
    db = _load(index.store.path, ro=True)
    db.execute("CREATE VIRTUAL TABLE temp.t USING sparse0()")
    by_text = db.execute("SELECT rowid, round(score, 5) FROM temp.t WHERE t MATCH 'heart attack' AND k=10").fetchall()
    by_terms = db.execute("SELECT rowid, round(score, 5) FROM temp.t WHERE t.terms MATCH ? AND k=10",
                          ('{"heart": 1.5, "attack": 1.5}',)).fetchall()
    assert by_text == by_terms and len(by_text) == 3

def test_terms_insert_into_a_model_index_needs_no_model(tmp_path, index):
    """Postings written from a term vector match what the fixture encoder would write."""
    _ext_or_skip()
    index.add(id="a", text="heart attack")          # fixture encoder: weight 2.0 per token
    index.commit()
    db = _load(index.store.path)
    db.execute("CREATE VIRTUAL TABLE temp.t USING sparse0()")   # no model registered
    db.execute("INSERT INTO temp.t(rowid, terms) VALUES (2, ?)", ('{"heart": 2.0, "attack": 2.0}',))
    db.commit()
    rows = db.execute("SELECT rowid, round(score, 5) FROM temp.t WHERE t MATCH 'heart attack' AND k=5").fetchall()
    assert len(rows) == 2 and rows[0][1] == rows[1][1]
    assert db.execute("SELECT ntokens, truncated FROM docs WHERE id=2").fetchone() == (None, None)

def test_extension_creates_and_queries_an_external_index(tmp_path, vocab):
    _ext_or_skip()
    vpath = tmp_path / "vocab.txt"
    vpath.write_text("\n".join(vocab) + "\n")
    db = _load(str(tmp_path / "x.db"))
    db.execute(f"CREATE VIRTUAL TABLE t USING sparse0(vocab='{vpath}')")
    db.execute("INSERT INTO t(rowid, terms) VALUES (1, ?)", ('{"heart": 2.0, "attack": 1.5}',))
    db.execute("INSERT INTO t(rowid, terms) VALUES (2, ?)", ('{"heart": 0.5, "mumbai": 3.0}',))
    db.execute("INSERT INTO t(terms) VALUES (?)", ('{"powai": 1.0, "heart": 0.0, "attack": -1.0}',))
    db.commit()
    assert dict(db.execute("SELECT k, v FROM meta WHERE k IN ('model_id','ndocs')")) == {"model_id": "external", "ndocs": "3"}
    assert db.execute("SELECT COUNT(*) FROM qlut").fetchone()[0] == 0
    rows = db.execute("SELECT rowid, round(score, 4) FROM t WHERE t.terms MATCH ? AND k=5",
                      ('{"heart": 2.0, "attack": 1.0, "zzz": 9.0}',)).fetchall()
    assert rows == [(1, 5.5), (2, 1.0)]
    # the Python reference reads the same file and agrees
    py = sqlite_sparse.SparseIndex(str(tmp_path / "x.db")).search_terms({"heart": 2.0, "attack": 1.0}, k=5)
    assert [(int(h["ext_id"]), round(h["score"], 4)) for h in py] == rows
    # and the reverse direction: a Python-written external index read by the extension
    ix = sqlite_sparse.SparseIndex.create_external(str(tmp_path / "y.db"), vocab)
    ix.add_terms(1, {"heart": 2.0, "attack": 1.5})
    ix.add_terms(2, {"heart": 0.5, "mumbai": 3.0})
    db2 = _load(str(tmp_path / "y.db"), ro=True)
    db2.execute("CREATE VIRTUAL TABLE temp.y USING sparse0()")
    assert db2.execute("SELECT rowid, round(score, 4) FROM temp.y WHERE y.terms MATCH ? AND k=5",
                       ('{"heart": 2.0, "attack": 1.0}',)).fetchall() == [(1, 5.5), (2, 1.0)]

def test_external_index_errors_are_explicit(tmp_path, vocab):
    _ext_or_skip()
    vpath = tmp_path / "vocab.txt"
    vpath.write_text("\n".join(vocab))
    db = _load(str(tmp_path / "e.db"))
    db.execute(f"CREATE VIRTUAL TABLE t USING sparse0(vocab='{vpath}')")
    with pytest.raises(sqlite3.OperationalError, match="no model"):
        db.execute("INSERT INTO t(text) VALUES ('heart attack')")
    with pytest.raises(sqlite3.OperationalError, match="not in the index vocabulary"):
        db.execute("INSERT INTO t(terms) VALUES ('{\"heart\": 1.0, \"nope\": 1.0}')")
    with pytest.raises(sqlite3.OperationalError, match="JSON object"):
        db.execute("INSERT INTO t(terms) VALUES ('[1, 2]')")
    with pytest.raises(sqlite3.OperationalError, match="JSON object"):
        db.execute("INSERT INTO t(terms) VALUES ('not json')")
    with pytest.raises(sqlite3.OperationalError, match="must be a number"):
        db.execute("INSERT INTO t(terms) VALUES ('{\"heart\": \"high\"}')")
    with pytest.raises(sqlite3.OperationalError, match="not both"):
        db.execute("INSERT INTO t(text, terms) VALUES ('heart', '{\"heart\": 1.0}')")
    with pytest.raises(sqlite3.OperationalError, match="no query weight table"):
        db.execute("SELECT rowid FROM t WHERE t MATCH 'heart'").fetchall()
    assert db.execute("SELECT count(*) FROM t").fetchone()[0] == 0
    with pytest.raises(sqlite3.OperationalError, match="not both"):
        db.execute(f"CREATE VIRTUAL TABLE u USING sparse0(model='mini', vocab='{vpath}')")
    with pytest.raises(sqlite3.OperationalError, match="only applies when creating"):
        db.execute(f"CREATE VIRTUAL TABLE temp.v USING sparse0(vocab='{vpath}')")

def test_external_index_rejects_bad_vocab_files(tmp_path):
    _ext_or_skip()
    dup = tmp_path / "dup.txt"
    dup.write_text("a\nb\na\n")
    empty_line = tmp_path / "empty.txt"
    empty_line.write_text("a\n\nb\n")
    db = _load(str(tmp_path / "bad.db"))
    with pytest.raises(sqlite3.OperationalError, match="duplicate"):
        db.execute(f"CREATE VIRTUAL TABLE t USING sparse0(vocab='{dup}')")
    with pytest.raises(sqlite3.OperationalError, match="is empty"):
        db.execute(f"CREATE VIRTUAL TABLE t USING sparse0(vocab='{empty_line}')")
    with pytest.raises(sqlite3.OperationalError, match="cannot open"):
        db.execute("CREATE VIRTUAL TABLE t USING sparse0(vocab='/nonexistent/vocab.txt')")
    with pytest.raises(sqlite3.OperationalError, match="model='name' or vocab='file'"):
        db.execute("CREATE VIRTUAL TABLE t USING sparse0()")
