"""Query engine: tokenizer + static table + exact scoring over an open
sqlite-sparse database. numpy only. No model, no network."""
import json
import unicodedata

import numpy as np

_MAX_WORD_CHARS = 100


def _is_punctuation(ch):
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(cp):
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF
            or 0x2A700 <= cp <= 0x2B73F or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF
            or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)


def _basic_tokenize(text):
    """BERT BasicTokenizer: clean, isolate CJK, split on whitespace, lowercase,
    strip combining marks, split punctuation."""
    cleaned = []
    for ch in text:
        cp = ord(ch)
        cat = unicodedata.category(ch)
        if cp == 0 or cp == 0xFFFD or (cat.startswith("C") and ch not in "\t\n\r"):
            continue
        if ch in " \t\n\r" or cat == "Zs":
            cleaned.append(" ")
        elif _is_cjk(cp):
            cleaned.append(f" {ch} ")
        else:
            cleaned.append(ch)
    words = []
    for w in "".join(cleaned).split():
        w = w.lower()
        w = "".join(c for c in unicodedata.normalize("NFD", w) if unicodedata.category(c) != "Mn")
        cur = []
        for c in w:
            if _is_punctuation(c):
                if cur:
                    words.append("".join(cur))
                    cur = []
                words.append(c)
            else:
                cur.append(c)
        if cur:
            words.append("".join(cur))
    return words


class QueryEngine:
    def __init__(self, db):
        self.db = db
        self.reload()

    def reload(self):
        meta = dict(self.db.execute("SELECT k, v FROM meta"))
        assert meta.get("format") == "sqlite-sparse/1", f"not sqlite-sparse: {meta.get('format')}"
        self.model_id = meta["model_id"]
        self.ndocs = int(meta.get("ndocs") or 0)
        self.weight_mode = meta.get("weight_mode", "u8")
        self.weight_scale = float(meta.get("weight_scale", 40.0))
        self._v2i = {t: i for i, t in enumerate(json.loads(meta["vocab"]))}
        self._unk = self._v2i.get("[UNK]")
        self._qlut = dict(self.db.execute("SELECT t, w FROM qlut"))
        self._dead = {r[0] for r in self.db.execute("SELECT id FROM docs WHERE deleted=1")}

    def _wordpiece_word(self, w):
        if len(w) > _MAX_WORD_CHARS:
            return [self._unk]
        ids, s = [], 0
        while s < len(w):
            e = len(w)
            while e > s:
                piece = w[s:e] if s == 0 else "##" + w[s:e]
                if piece in self._v2i:
                    ids.append(self._v2i[piece])
                    break
                e -= 1
            else:
                return [self._unk]
            s = e
        return ids

    def _wordpiece(self, text):
        out = []
        for w in _basic_tokenize(text):
            for t in self._wordpiece_word(w):
                if t is not None:
                    out.append(t)
            if len(out) >= 512:
                return out[:512]
        return out

    def encode_query(self, text):
        qw = {}
        for t in self._wordpiece(text):
            w = self._qlut.get(t)
            if w:
                qw[t] = qw.get(t, 0.0) + w
        return qw

    def search(self, text, k=10):
        qw = self.encode_query(text)
        if not qw or not self.ndocs:
            return []
        score = np.zeros(self.ndocs + 1, dtype=np.float64)
        for t, w in qw.items():
            row = self.db.execute("SELECT docs, ws FROM postings WHERE t=?", (t,)).fetchone()
            if not row:
                continue
            docs = np.frombuffer(row[0], dtype="<i4")
            if self.weight_mode == "u8":
                ws = np.frombuffer(row[1], dtype=np.uint8).astype(np.float32) / self.weight_scale
            else:
                ws = np.frombuffer(row[1], dtype="<f4")
            np.add.at(score, docs, ws * w)
        want = k + len(self._dead)
        if want >= len(score):
            top = np.argsort(-score)
        else:
            part = np.argpartition(score, -want)[-want:]
            top = part[np.argsort(-score[part])]
        out = []
        for d in top:
            d = int(d)
            if score[d] <= 0 or d in self._dead:
                continue
            out.append(d)
            if len(out) == k:
                break
        if not out:
            return []
        ph = ",".join("?" * len(out))
        rows = {r[0]: r for r in self.db.execute(
            f"SELECT id, ext_id, title FROM docs WHERE id IN ({ph})", out)}
        return [{"id": d, "ext_id": rows[d][1], "title": rows[d][2], "score": float(score[d])}
                for d in out if d in rows]
