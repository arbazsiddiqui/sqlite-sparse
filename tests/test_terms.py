"""Caller-supplied term vectors in the Python reference implementation."""
import pytest

from sqlite_sparse.api import SparseIndex


@pytest.fixture
def external(tmp_path, vocab):
    ix = SparseIndex.create_external(str(tmp_path / "x.db"), vocab)
    ix.add_terms("d1", {"heart": 2.0, "attack": 1.5})
    ix.add_terms("d2", {"heart": 0.5, "mumbai": 3.0})
    ix.add_terms("d3", {"powai": 1.0})
    return ix

def test_external_index_has_no_query_table_and_external_model(external):
    assert external.store.get_meta("model_id") == "external"
    assert external.store.db.execute("SELECT COUNT(*) FROM qlut").fetchone()[0] == 0

def test_search_terms_scores_are_the_dot_product(external):
    r = external.search_terms({"heart": 2.0, "attack": 1.0}, k=5)
    assert [(h["ext_id"], round(h["score"], 4)) for h in r] == [("d1", 5.5), ("d2", 1.0)]

def test_search_terms_sums_repeats_and_drops_unknown_and_nonpositive(external):
    qw = external.engine.encode_terms({"heart": 1.0, "zzz": 4.0, "mumbai": -1.0, "powai": 0.0})
    assert qw == {10: 1.0}

def test_add_terms_rejects_unknown_tokens(external):
    with pytest.raises(ValueError, match="not in the index vocabulary"):
        external.add_terms("d4", {"heart": 1.0, "nope": 2.0})

def test_add_terms_drops_nonpositive_weights(external):
    external.add_terms("d4", {"heart": -3.0, "attack": 0.0, "drug": 1.0})
    assert [h["ext_id"] for h in external.search_terms({"drug": 1.0})] == ["d4"]
    assert not any(h["ext_id"] == "d4" for h in external.search_terms({"heart": 1.0, "attack": 1.0}, k=10))

def test_text_search_on_external_index_is_an_error(external):
    with pytest.raises(ValueError, match="no query weight table"):
        external.search("heart")

def test_search_terms_works_on_a_model_index_too(index):
    index.add(id="a", text="heart attack aspirin")
    index.add(id="b", text="powai mumbai")
    index.commit()
    by_text = index.search("heart", k=5)
    by_terms = index.search_terms({"heart": 1.5}, k=5)      # 1.5 is the fixture's query weight
    assert [(h["ext_id"], round(h["score"], 6)) for h in by_text] == \
           [(h["ext_id"], round(h["score"], 6)) for h in by_terms]

def test_reopen_external_index(tmp_path, external):
    again = SparseIndex(external.store.path)
    assert [h["ext_id"] for h in again.search_terms({"mumbai": 1.0})] == ["d2"]

def test_create_external_rejects_duplicate_tokens(tmp_path):
    with pytest.raises(AssertionError):
        SparseIndex.create_external(str(tmp_path / "dup.db"), ["a", "b", "a"])
