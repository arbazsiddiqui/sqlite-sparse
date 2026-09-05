"""User-facing API: SparseIndex with create/attach/add/commit/search/sync."""
import numpy as np

from .store import SparseStore
from .search import QueryEngine


class SparseIndex:
    def __init__(self, path, model=None, max_seq=256, device=None):
        self.store = SparseStore(path)
        self._enc = None
        self._model = model or self.store.get_meta("model_id")
        self._max_seq, self._device = max_seq, device
        if self.store.get_meta("format"):
            self.engine = QueryEngine(self.store.db)
        else:
            assert model, "new index needs model= ('mini' | 'base' | HF id)"
            enc = self.encoder()
            self.store.init_model(enc.model_id, enc.vocab(), enc.qlut())
            self.engine = QueryEngine(self.store.db)

    @classmethod
    def create(cls, path, model="mini", **kw):
        return cls(path, model=model, **kw)

    def encoder(self):
        if self._enc is None:
            from .encoder import TorchEncoder
            self._enc = TorchEncoder(self._model, max_seq=self._max_seq, device=self._device)
        return self._enc

    def add(self, id, text, title=""):
        self.store.queue(id, text, title)

    def commit(self, batch_size=256):
        return self.sync(batch_size=batch_size)

    def sync(self, batch_size=256):
        """Drain the pending queue through the encoder. Returns docs synced."""
        total = 0
        while True:
            rows = self.store.drain_pending(limit=batch_size)
            if not rows:
                break
            enc = self.encoder()
            terms = enc.encode([r[3] for r in rows])
            self.store.clear_pending([r[0] for r in rows])
            self.store.add_encoded(
                [(r[1], r[2], r[3], t) for r, t in zip(rows, terms)])
            total += len(rows)
        self.engine.reload()
        return total

    def delete(self, id):
        self.store.delete(id)

    def attach(self, table, columns, id_col="rowid"):
        cols = " || ' ' || ".join(f"COALESCE(NEW.{c},'')" for c in columns)
        db = self.store.db
        for ev in ("INSERT", "UPDATE"):
            db.execute(f"""
CREATE TRIGGER IF NOT EXISTS sparse_cap_{table}_{ev.lower()}
AFTER {ev} ON {table} BEGIN
  INSERT INTO pending(ext_id, title, body) VALUES (NEW.{id_col}, '', {cols});
END""")
        db.execute(f"""
CREATE TRIGGER IF NOT EXISTS sparse_cap_{table}_delete
AFTER DELETE ON {table} BEGIN
  UPDATE docs SET deleted=1 WHERE ext_id = CAST(OLD.{id_col} AS TEXT) AND deleted=0;
END""")
        sel = " || ' ' || ".join(f"COALESCE({c},'')" for c in columns)
        db.execute(f"INSERT INTO pending(ext_id, title, body) SELECT {id_col}, '', {sel} FROM {table}")
        db.commit()

    def search(self, text, k=10, auto_sync=False):
        if auto_sync and self.store.pending_count():
            self.sync()
        return self.engine.search(text, k=k)


def build_bulk(path, docs, ids, model="mini", batch_size=256, device=None,
               titles=None, log=None):
    """Bulk load: encode everything, accumulate postings in memory, write once."""
    from collections import defaultdict
    from .encoder import TorchEncoder
    from .store import SparseStore

    if log is None:
        def log(m):
            print(m, flush=True)
    enc = TorchEncoder(model, device=device)
    st = SparseStore(path)
    st.init_model(enc.model_id, enc.vocab(), enc.qlut())
    scale = float(st.get_meta("weight_scale", 40.0))
    titles = titles or ["" for _ in docs]
    acc_d, acc_w = defaultdict(list), defaultdict(list)
    st.db.executemany("INSERT INTO docs(id, ext_id, title) VALUES(?,?,?)",
                      [(i + 1, str(ids[i]), titles[i]) for i in range(len(docs))])
    import time
    t0 = time.time()
    for c0 in range(0, len(docs), batch_size * 8):
        chunk = docs[c0:c0 + batch_size * 8]
        for off, terms in enumerate(enc.encode(chunk, batch_size=batch_size)):
            did = c0 + off + 1
            for t, w in terms.items():
                acc_d[t].append(did)
                acc_w[t].append(w)
        if (c0 // (batch_size * 8)) % 10 == 0:
            log(f"[bulk] {min(c0 + batch_size*8, len(docs))}/{len(docs)} ({time.time()-t0:.0f}s)")
    for t in sorted(acc_d):
        d = np.array(acc_d[t], dtype="<i4")
        w = np.clip(np.rint(np.array(acc_w[t]) * scale), 1, 255).astype(np.uint8)
        st.db.execute("INSERT INTO postings VALUES(?,?,?)", (t, d.tobytes(), w.tobytes()))
    st.set_meta("ndocs", len(docs))
    st.db.commit()
    st.db.execute("VACUUM")
    log(f"[bulk] DONE {len(docs)} docs ({time.time()-t0:.0f}s)")
    return SparseIndex(path)
