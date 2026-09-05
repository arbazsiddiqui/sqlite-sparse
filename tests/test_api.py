import pytest

def test_add_commit_search(index):
    index.add(id="a1", text="aspirin prevent heart attack", title="A")
    assert index.store.pending_count() == 1
    assert index.commit() == 1
    r = index.search("aspirin heart")
    assert r[0]["ext_id"] == "a1" and r[0]["title"] == "A"

def test_sync_batches(index):
    for i in range(7):
        index.add(id=f"d{i}", text="heart drug")
    assert index.sync(batch_size=3) == 7
    assert index.store.pending_count() == 0

def test_auto_sync_on_search(index):
    index.add(id="x", text="powai mumbai")
    assert index.search("powai", auto_sync=False) == []
    assert index.search("powai", auto_sync=True)[0]["ext_id"] == "x"

def test_delete_via_api(index):
    index.add(id="x", text="heart")
    index.commit()
    index.delete("x")
    index.engine.reload()
    assert index.search("heart") == []

def test_attach_captures_inserts_updates_deletes(index):
    db = index.store.db
    db.execute("CREATE TABLE addresses(id INTEGER PRIMARY KEY, line1 TEXT, city TEXT)")
    db.execute("INSERT INTO addresses VALUES (7, 'aspirin lane', 'mumbai')")
    index.attach("addresses", ["line1", "city"], id_col="id")
    assert index.sync() == 1
    assert index.search("mumbai")[0]["ext_id"] == "7"

    db.execute("INSERT INTO addresses VALUES (8, 'drug street', 'powai')")
    db.commit()
    index.sync()
    assert index.search("powai")[0]["ext_id"] == "8"

    db.execute("UPDATE addresses SET city='heart' WHERE id=8")
    db.commit()
    index.sync()
    assert any(r["ext_id"] == "8" for r in index.search("heart"))

    db.execute("DELETE FROM addresses WHERE id=7")
    db.commit()
    index.engine.reload()
    assert all(r["ext_id"] != "7" for r in index.search("mumbai"))

def test_open_existing_without_model(index, tmp_path):
    from sqlite_sparse.api import SparseIndex
    index.add(id="a", text="heart")
    index.commit()
    path = index.store.path
    ix2 = SparseIndex(path)
    assert ix2.search("heart")[0]["ext_id"] == "a"

def test_new_index_requires_model(tmp_path):
    from sqlite_sparse.api import SparseIndex
    with pytest.raises(AssertionError):
        SparseIndex(str(tmp_path / "new.db"))

def test_commit_durably_clears_pending(index):
    index.add(id="p1", text="heart attack aspirin")
    index.add(id="p2", text="powai mumbai")
    assert index.commit() == 2
    index.store.db.rollback()
    assert index.store.pending_count() == 0
    assert len(index.search("heart", k=5)) == 1
