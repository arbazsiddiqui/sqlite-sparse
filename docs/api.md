# API

## Loading

```
sqlite3> .load ./sparse0
```

```python
import sqlite3, sqlite_sparse
db = sqlite3.connect("notes.db")
sqlite_sparse.load(db)
```


The binary must be named `sparse0.so` or `sparse0.dylib`; SQLite derives the entry point
from the file name.

## Functions

`sparse_register(name, gguf_path, sprs_path [, max_seq])`
Registers a model for this process. `max_seq` is the number of tokens kept per document,
16 to 512, default 512. Returns `'ok'`. Calling it again with the same name replaces the
files and settings. The encoder loads on the first INSERT.

`sparse_compact()`
Removes deleted documents from every posting list and drops their rows. Returns the number
of posting entries removed.

`sparse_tokens(text)`
The query tokenizer's output for `text` as a JSON array of token strings, using the
vocabulary stored in this database. Useful to see what a query matched on.

`sparse_version()`
Returns `'sqlite-sparse/1 sparse0 1.1.0'`.

## Virtual table

```sql
CREATE VIRTUAL TABLE notes USING sparse0(model='mini');       -- new index from a registered model
CREATE VIRTUAL TABLE notes USING sparse0(vocab='vocab.txt');  -- new index from a vocabulary alone
CREATE VIRTUAL TABLE temp.notes USING sparse0();              -- existing file, no model needed
```

One database file holds one index. The first `CREATE` on a fresh file writes the named
model's vocabulary and query weight table, or, with `vocab=`, the vocabulary from a file
with one token per line (unique, non-empty) and an empty query weight table with
`model_id` `external`; later `CREATE`s in that file attach to the same index. Columns are
`text` (write), `score` (read) and the hidden `k` and `terms`.

```sql
INSERT INTO notes(rowid, text) VALUES (42, '...');
INSERT INTO notes(text) VALUES ('...');                    -- rowid assigned
```

Encodes the text and merges its terms into the posting lists. Documents beyond `max_seq`
tokens are truncated; `docs.ntokens` and `docs.truncated` record it. Requires a registered
model whose vocabulary and query weight table match the file; a mismatch is an error.

```sql
INSERT INTO notes(rowid, terms) VALUES (42, '{"heart": 0.95, "cardiac": 0.42}');
```

Stores a document encoded by the caller. `terms` is a JSON object of token to weight; every
token must be in the file's vocabulary, weights must be numbers, non-positive weights are
dropped and repeated tokens are summed. No model is needed, `ntokens` and `truncated` stay
NULL, and weights are quantized exactly as the text path does (one byte at the file's
scale, clamped to 1..255). `text` and `terms` cannot both be given.

```sql
SELECT rowid, score FROM notes WHERE notes MATCH ? LIMIT 10;
SELECT rowid, score FROM notes WHERE notes MATCH ? AND k = 10;
SELECT rowid, score FROM notes WHERE notes MATCH ? ORDER BY score DESC LIMIT 10;
```

`MATCH` takes plain text. `LIMIT n` or `k = n` sets the result count, default 10. Results
are ordered by score, ties by lower rowid; `ORDER BY score DESC` adds no sort step.

```sql
SELECT rowid, score FROM notes WHERE notes.terms MATCH '{"cardiac": 6.53, "arrest": 6.87}' LIMIT 10;
```

`terms MATCH` takes a query vector as a JSON object of token to weight and scores it
against the postings with the same scatter-add: tokens not in the vocabulary are ignored,
non-positive weights dropped, repeats summed. It works on any index. A text `MATCH` on an
index created with `vocab=` is an error, since that index has no query weight table.

```sql
DELETE FROM notes WHERE rowid = 42;                        -- flags the row; sparse_compact() reclaims it
SELECT count(*) FROM notes;                                -- live documents
```

`UPDATE` is not supported.

## Tables

| table | contents |
|---|---|
| `meta` | format version, model name, weight scale, vocabulary, SHA-256 of the model files, `max_seq`, document count |
| `qlut` | query weight table, one row per vocabulary word with a nonzero weight |
| `postings` | one row per word, document ids (`int32`) and weights (one byte each) |
| `docs` | one row per document, `ext_id`, `deleted`, `ntokens`, `truncated` |
| `pending` | write queue for the Python attach/sync path |

Layout in [FORMAT.md](../FORMAT.md).

## Python

```python
sqlite_sparse.load(conn)
sqlite_sparse.register(conn, alias, quant="q8", max_seq=512)   # downloads to ~/.cache/sqlite-sparse
sqlite_sparse.fetch(alias, quant="q8")                         # returns (gguf_path, sprs_path)
sqlite_sparse.loadable_path()
```

Aliases are `mini`, `base` and `multilingual`. `SQLITE_SPARSE_CACHE` sets the cache directory
and `HF_TOKEN` is used if set.

`sqlite_sparse.SparseIndex` is the pure-Python reference implementation of the file format
(indexes with torch).

SparseIndex.create_external(path, vocab)   # index from a token list, no model, no query table
ix.add_terms(id, {"heart": 0.95, ...})     # caller-encoded document; unknown tokens raise ValueError
ix.search_terms({"cardiac": 6.53, ...}, k=10)
```

`search()` on an external index raises `ValueError`, since there is no query weight table.

```
sqlite-sparse build <db> <file.jsonl> [--model mini]      # index {id, text[, title]} records
sqlite-sparse search <db> "<query>" [-k 10]
sqlite-sparse info <db>
sqlite-sparse convert <hf-model-id> out.sprs [--double-log]
```

Writes a sidecar for an inference-free OpenSearch-style checkpoint. `--double-log` for the
v3 models.
