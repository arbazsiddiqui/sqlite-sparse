import pytest

from sqlite_sparse.search import QueryEngine
from sqlite_sparse.store import SparseStore

@pytest.fixture
def engine(tmp_path, vocab, qlut):
    st = SparseStore(str(tmp_path / "q.db"))
    st.init_model("m", vocab, qlut)
    st.add_encoded([
        ("d1", "aspirin study", None, {12: 2.9, 10: 2.3, 11: 2.1, 17: 1.4}),
        ("d2", "mumbai flats", None, {14: 2.5, 15: 2.2}),
        ("d3", "weak match", None, {10: 0.4}),
    ])
    return st, QueryEngine(st.db)

def test_rejects_foreign_file(tmp_path):
    import sqlite3
    db = sqlite3.connect(str(tmp_path / "x.db"))
    db.execute("CREATE TABLE meta(k TEXT PRIMARY KEY, v TEXT)")
    with pytest.raises(AssertionError):
        QueryEngine(db)

def test_wordpiece_greedy_longest_match(engine):
    _, eng = engine
    ids = eng._wordpiece("aspirins")
    assert ids == [12, 16]

def test_wordpiece_unknown_word_dropped(engine):
    _, eng = engine
    assert eng._wordpiece("zzzqqq") == []

def test_encode_query_sums_repeats(engine):
    _, eng = engine
    qw = eng.encode_query("heart heart")
    assert qw[10] == pytest.approx(3.0)

def test_search_ranks_by_dot_product(engine):
    _, eng = engine
    r = eng.search("aspirin heart attack", k=3)
    assert [x["ext_id"] for x in r][0] == "d1"
    assert r[0]["score"] > r[-1]["score"]

def test_search_zero_hit_query(engine):
    _, eng = engine
    assert eng.search("zzzqqq") == []

def test_search_empty_index(tmp_path, vocab, qlut):
    st = SparseStore(str(tmp_path / "e.db"))
    st.init_model("m", vocab, qlut)
    assert QueryEngine(st.db).search("heart") == []

def test_deleted_docs_filtered(engine):
    st, eng = engine
    st.delete("d1")
    eng.reload()
    assert all(x["ext_id"] != "d1" for x in eng.search("aspirin heart attack", k=3))

def test_k_limits_results(engine):
    _, eng = engine
    assert len(eng.search("heart mumbai aspirin", k=1)) == 1

def test_u8_score_close_to_f32(tmp_path, vocab, qlut):
    scores = {}
    for mode in ("u8", "f32"):
        st = SparseStore(str(tmp_path / f"{mode}.db"))
        st.init_model("m", vocab, qlut, weight_mode=mode)
        st.add_encoded([("d1", "", None, {12: 2.9, 10: 2.3})])
        scores[mode] = QueryEngine(st.db).search("aspirin heart")[0]["score"]
    assert scores["u8"] == pytest.approx(scores["f32"], rel=0.02)
