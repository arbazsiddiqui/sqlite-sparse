"use strict";
// Smoke test on the query path, which needs no model: a vocabulary-only index, caller-supplied
// term vectors, and a terms MATCH whose scores are fixed by the format (see tests/test_loadable.py).
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { DatabaseSync } = require("node:sqlite");
const sparse = require("./index.js");

const dir = fs.mkdtempSync(path.join(os.tmpdir(), "sqlite-sparse-"));
const vocab = path.join(dir, "vocab.txt");
fs.writeFileSync(vocab, ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "heart", "attack", "mumbai", "powai"].join("\n") + "\n");
const file = path.join(dir, "x.db");

const db = new DatabaseSync(file, { allowExtension: true });
sparse.load(db);
const version = db.prepare("SELECT sparse_version() AS v").get().v;
assert.match(version, /^sqlite-sparse\/1 sparse0 /);

db.exec(`CREATE VIRTUAL TABLE t USING sparse0(vocab='${vocab}')`);
const ins = db.prepare("INSERT INTO t(rowid, terms) VALUES (?, ?)");
ins.run(1, JSON.stringify({ heart: 2.0, attack: 1.5 }));
ins.run(2, JSON.stringify({ heart: 0.5, mumbai: 3.0 }));
const q = JSON.stringify({ heart: 2.0, attack: 1.0, zzz: 9.0 });
const rows = db.prepare("SELECT rowid, round(score, 4) AS score FROM t WHERE t.terms MATCH ? AND k = 5").all(q);
assert.deepEqual(rows.map((r) => [Number(r.rowid), r.score]), [[1, 5.5], [2, 1.0]]);
db.close();

// A second handle attaches to the index already in the file, read-only.
const ro = new DatabaseSync(file, { allowExtension: true, readOnly: true });
sparse.load(ro);
ro.exec("CREATE VIRTUAL TABLE temp.y USING sparse0()");
const again = ro.prepare("SELECT rowid, round(score, 4) AS score FROM y WHERE y.terms MATCH ? AND k = 5").all(q);
assert.deepEqual(again.map((r) => [Number(r.rowid), r.score]), [[1, 5.5], [2, 1.0]]);
ro.close();

console.log(`ok ${process.platform}-${process.arch} ${version} ${path.basename(sparse.getLoadablePath())}`);
