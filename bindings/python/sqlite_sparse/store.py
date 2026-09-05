"""Storage layer for the sqlite-sparse/1 format: schema, incremental posting
merges, deletions, compaction. All writes go through SparseStore."""
import json
import sqlite3
import time
from collections import defaultdict

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS qlut(t INTEGER PRIMARY KEY, w REAL);
CREATE TABLE IF NOT EXISTS postings(t INTEGER PRIMARY KEY, docs BLOB, ws BLOB);
CREATE TABLE IF NOT EXISTS docs(id INTEGER PRIMARY KEY AUTOINCREMENT,
    ext_id TEXT UNIQUE, title TEXT, body TEXT, meta TEXT, deleted INTEGER DEFAULT 0,
    ntokens INTEGER, truncated INTEGER);
CREATE TABLE IF NOT EXISTS pending(id INTEGER PRIMARY KEY AUTOINCREMENT,
    ext_id TEXT, title TEXT, body TEXT NOT NULL);
"""


class SparseStore:
    def __init__(self, path):
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    def get_meta(self, k, default=None):
        r = self.db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return r[0] if r else default

    def set_meta(self, k, v):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (k, str(v)))

    def init_model(self, model_id, vocab, qlut_weights, weight_mode="u8", weight_scale=40.0):
        if self.get_meta("format"):
            assert self.get_meta("model_id") == model_id, \
                f"index built with {self.get_meta('model_id')}, not {model_id}"
            return
        for k, v in [("format", "sqlite-sparse/1"), ("model_id", model_id),
                     ("weight_mode", weight_mode), ("weight_scale", weight_scale),
                     ("vocab", json.dumps(vocab)),
                     ("created_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))]:
            self.set_meta(k, v)
        nz = np.nonzero(qlut_weights)[0]
        self.db.executemany("INSERT OR REPLACE INTO qlut VALUES(?,?)",
                            [(int(t), float(qlut_weights[t])) for t in nz])
        self.db.commit()

    def queue(self, ext_id, text, title=""):
        self.db.execute("INSERT INTO pending(ext_id, title, body) VALUES(?,?,?)",
                        (str(ext_id), title, text))
        self.db.commit()

    def pending_count(self):
        return self.db.execute("SELECT COUNT(*) FROM pending").fetchone()[0]

    def drain_pending(self, limit=1000):
        rows = self.db.execute(
            "SELECT id, ext_id, title, body FROM pending ORDER BY id LIMIT ?", (limit,)).fetchall()
        return rows

    def clear_pending(self, ids):
        # No commit here: sync() runs this inside add_encoded's transaction so
        # the queue drain and the postings write land atomically.
        self.db.executemany("DELETE FROM pending WHERE id=?", [(i,) for i in ids])

    def add_encoded(self, items, store_body=False):
        """items: list of (ext_id, title, body, {term: weight}). Merges into
        postings. Re-adding an existing ext_id marks the old row deleted."""
        mode = self.get_meta("weight_mode", "u8")
        scale = float(self.get_meta("weight_scale", 40.0))
        acc = defaultdict(lambda: ([], []))
        for ext_id, title, body, terms in items:
            old = self.db.execute("SELECT id FROM docs WHERE ext_id=? AND deleted=0",
                                  (str(ext_id),)).fetchone()
            if old:
                self.db.execute("UPDATE docs SET deleted=1, ext_id=ext_id||':del:'||id WHERE id=?", (old[0],))
            cur = self.db.execute("INSERT INTO docs(ext_id, title, body) VALUES(?,?,?)",
                                  (str(ext_id), title, body if store_body else None))
            did = cur.lastrowid
            for t, w in terms.items():
                d, ws = acc[int(t)]
                d.append(did)
                ws.append(w)
        for t, (d, ws) in acc.items():
            row = self.db.execute("SELECT docs, ws FROM postings WHERE t=?", (t,)).fetchone()
            nd = np.array(d, dtype="<i4")
            nw = np.array(ws, dtype=np.float32)
            if mode == "u8":
                nwb = np.clip(np.rint(nw * scale), 1, 255).astype(np.uint8)
            else:
                nwb = nw.astype("<f4")
            if row:
                nd = np.concatenate([np.frombuffer(row[0], dtype="<i4"), nd])
                old_w = np.frombuffer(row[1], dtype=np.uint8 if mode == "u8" else "<f4")
                nwb = np.concatenate([old_w, nwb])
            self.db.execute("INSERT OR REPLACE INTO postings VALUES(?,?,?)",
                            (t, nd.tobytes(), nwb.tobytes()))
        self.set_meta("ndocs", self.db.execute("SELECT MAX(id) FROM docs").fetchone()[0] or 0)
        self.db.commit()

    def delete(self, ext_id):
        self.db.execute("UPDATE docs SET deleted=1 WHERE ext_id=? AND deleted=0", (str(ext_id),))
        self.db.commit()

    def deleted_ids(self):
        return np.array([r[0] for r in self.db.execute("SELECT id FROM docs WHERE deleted=1")],
                        dtype=np.int64)

    def compact(self):
        """Rewrite postings without deleted docs and VACUUM."""
        dead = set(int(i) for i in self.deleted_ids())
        if dead:
            for t, db_, wb in self.db.execute("SELECT t, docs, ws FROM postings").fetchall():
                d = np.frombuffer(db_, dtype="<i4")
                keep = ~np.isin(d, list(dead))
                if keep.all():
                    continue
                mode = self.get_meta("weight_mode", "u8")
                w = np.frombuffer(wb, dtype=np.uint8 if mode == "u8" else "<f4")
                self.db.execute("UPDATE postings SET docs=?, ws=? WHERE t=?",
                                (d[keep].tobytes(), w[keep].tobytes(), t))
            self.db.execute("DELETE FROM docs WHERE deleted=1")
            self.db.commit()
        self.db.execute("VACUUM")
