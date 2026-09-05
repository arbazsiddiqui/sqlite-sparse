import numpy as np
import pytest

from sqlite_sparse.store import SparseStore

@pytest.fixture
def store(tmp_path, vocab, qlut):
    st = SparseStore(str(tmp_path / "s.db"))
    st.init_model("m", vocab, qlut)
    return st

def test_init_is_idempotent(store, vocab, qlut):
    store.init_model("m", vocab, qlut)
    assert store.get_meta("format") == "sqlite-sparse/1"

def test_init_rejects_different_model(store, vocab, qlut):
    with pytest.raises(AssertionError):
        store.init_model("other", vocab, qlut)

def test_qlut_stores_nonzero_only(store):
    n = store.db.execute("SELECT COUNT(*) FROM qlut").fetchone()[0]
    assert n == 8

def test_queue_and_drain(store):
    store.queue("a", "text a")
    store.queue("b", "text b", title="B")
    assert store.pending_count() == 2
    rows = store.drain_pending(limit=1)
    assert len(rows) == 1 and rows[0][1] == "a"
    store.clear_pending([rows[0][0]])
    assert store.pending_count() == 1

def test_add_encoded_merges_postings(store):
    store.add_encoded([("d1", "", None, {10: 2.0})])
    store.add_encoded([("d2", "", None, {10: 3.0})])
    docs, ws = store.db.execute("SELECT docs, ws FROM postings WHERE t=10").fetchone()
    assert list(np.frombuffer(docs, dtype="<i4")) == [1, 2]
    assert len(np.frombuffer(ws, dtype=np.uint8)) == 2

def test_u8_quantization_bounds(store):
    store.add_encoded([("d1", "", None, {10: 0.001, 11: 99.0})])
    ws = np.frombuffer(store.db.execute(
        "SELECT ws FROM postings WHERE t=10").fetchone()[0], dtype=np.uint8)
    assert ws[0] == 1
    ws = np.frombuffer(store.db.execute(
        "SELECT ws FROM postings WHERE t=11").fetchone()[0], dtype=np.uint8)
    assert ws[0] == 255

def test_upsert_marks_old_deleted(store):
    store.add_encoded([("d1", "v1", None, {10: 2.0})])
    store.add_encoded([("d1", "v2", None, {10: 2.0})])
    live = store.db.execute("SELECT COUNT(*) FROM docs WHERE deleted=0").fetchone()[0]
    dead = store.db.execute("SELECT COUNT(*) FROM docs WHERE deleted=1").fetchone()[0]
    assert (live, dead) == (1, 1)

def test_delete_and_compact(store):
    store.add_encoded([("d1", "", None, {10: 2.0}), ("d2", "", None, {10: 1.0})])
    store.delete("d1")
    assert list(store.deleted_ids()) == [1]
    store.compact()
    docs = np.frombuffer(store.db.execute(
        "SELECT docs FROM postings WHERE t=10").fetchone()[0], dtype="<i4")
    assert list(docs) == [2]
    assert store.db.execute("SELECT COUNT(*) FROM docs").fetchone()[0] == 1

def test_f32_mode(tmp_path, vocab, qlut):
    st = SparseStore(str(tmp_path / "f.db"))
    st.init_model("m", vocab, qlut, weight_mode="f32")
    st.add_encoded([("d1", "", None, {10: 2.5})])
    ws = np.frombuffer(st.db.execute("SELECT ws FROM postings WHERE t=10").fetchone()[0], dtype="<f4")
    assert abs(ws[0] - 2.5) < 1e-6

def test_store_body(store):
    store.add_encoded([("d1", "t", "the body", {10: 1.0})], store_body=True)
    assert store.db.execute("SELECT body FROM docs WHERE ext_id='d1'").fetchone()[0] == "the body"
