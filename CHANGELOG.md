# Changelog

## 0.1.0 (2026-09-05)

First release.

- `sparse0` loadable SQLite extension for semantic search over a single database
  file with no model at query time. Documents are encoded in-process at INSERT
  by an embedded llama.cpp running OpenSearch's inference-free neural sparse
  encoders (GGUF). Queries are WordPiece tokenization plus a weight table stored
  in the file, scored by exact scatter-add over impact postings.
- Three named models (`mini` default, `base`, `multilingual`) as GGUF f16/Q8_0
  plus a `.sprs` sidecar carrying the MLM head, query weight table and
  vocabulary. Each conversion is differentially validated against the original
  SentenceTransformers implementation.
- Python package with the reference implementation of the same file format,
  `sqlite_sparse.load()` to load the extension, `sqlite_sparse.register()` to
  fetch and register a named model, and `sqlite-sparse convert` to write a
  sidecar for another checkpoint.
- `LIMIT n` works alongside `AND k = n`, and `ORDER BY score DESC` is consumed
  by the virtual table (results are already in that order, no sort step).
- Model identity in the file. The extension records SHA-256 hashes of the
  encoder, the sidecar and the vocabulary, and refuses to INSERT into a file
  built with a different vocabulary or query table.
- Per-document `ntokens` and `truncated` columns, so truncation at `max_seq`
  (default 512, configurable in `sparse_register`) is visible.
- `sparse_compact()` rewrites posting lists without soft-deleted documents.
- Query tokenization matches the models' BERT tokenizer exactly (Unicode
  lowercasing, accent stripping, punctuation and CJK handling, `[UNK]`), in the
  extension (via utf8proc) and the Python reference, checked against HF-produced
  golden ids. `sparse_tokens(text)` shows the tokens for a string.
- File format `sqlite-sparse/1`, shared by both implementations and enforced by
  the test suite (each ranks the other's databases identically).
