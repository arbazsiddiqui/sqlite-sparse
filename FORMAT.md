# sqlite-sparse file format v1

One SQLite file = one semantic search index. Built with a doc-side sparse
encoder (OpenSearch neural-sparse doc-* models); searched with a tokenizer
and a static weight table, with no model and no server at query time. Scoring is an
exact dot product over impact postings (no candidate stage, no recall loss).

## Tables
- meta(k TEXT PRIMARY KEY, v TEXT)
  keys are format=sqlite-sparse/1 · model_id · ndocs · weight_mode(u8|f32) ·
  weight_scale (u8 mode, score_weight = stored/scale) · vocab (JSON array of
  WordPiece tokens, index = term id) · created_utc (written by the Python
  implementation). Files written by the
  extension also carry encoder_sha256 (the GGUF), sidecar_sha256 (the .sprs),
  vocab_sha256 (sha256 of the newline-joined vocabulary) and max_seq (the
  truncation length used at insert). Before an INSERT the extension checks the
  registered model's vocabulary hash and every query-table entry against the
  file and refuses with an error if they differ, since postings are only
  meaningful with the vocabulary and query table they were built with.
- qlut(t INTEGER PRIMARY KEY, w REAL)
  the model's static query weight table, nonzero entries only
- postings(t INTEGER PRIMARY KEY, docs BLOB, ws BLOB)
  docs is a little-endian int32 array of doc rowids (ascending)
  ws is a uint8 array of quantized weights of the same length in u8 mode, or
  LE float32 in f32 mode
- docs(id INTEGER PRIMARY KEY, ext_id TEXT UNIQUE, title TEXT, body TEXT,
  meta TEXT, deleted INTEGER DEFAULT 0, ntokens INTEGER, truncated INTEGER)
  body/meta optional; ext_id is the caller's identifier. ntokens is the
  document's full token count and truncated is 1 when it exceeded max_seq, so
  a missed match on a long document can be diagnosed. Deletion is a soft
  flag. Postings keep the doc id and readers drop flagged ids after scoring
  (partial index docs_deleted ON docs(deleted) WHERE deleted=1), so deleted
  documents still cost scoring time until `SELECT sparse_compact()` rewrites
  the posting lists without them and removes their rows.
- One sparse index per database file. The tables above are shared by every
  sparse0 virtual table in the file, so a second CREATE VIRTUAL TABLE adopts
  the same index rather than creating another.
- pending(id INTEGER PRIMARY KEY, ext_id TEXT, title TEXT, body TEXT)
  write queue for the Python attach/sync path, empty in extension-built files

## Query algorithm (exact)
1. Tokenize the query as BERT does: drop control characters, isolate CJK
   ideographs, split on whitespace, lowercase, strip combining marks (NFD,
   category Mn), split punctuation, then greedy longest-match WordPiece with
   meta.vocab. A word that fails, or exceeds 100 characters, is [UNK].
2. qw[t] = sum of qlut[t] per occurrence; drop zeros.
3. For each t in qw, fetch the postings row and accumulate
   score[doc] += qw[t] * weight(ws[i]) (scatter-add / bincount).
4. Top-k by score, ties broken by lower doc id. No approximations anywhere.

## Design notes
- Blobs are raw little-endian typed arrays, so any language reads them
  directly (JavaScript with Int32Array/Uint8Array/Float32Array views).
- u8 quantization is w_q = round(w * weight_scale) clamped to 255, with a
  default scale of 40. The quality delta vs f32 is measured and published, never
  assumed; f32 mode exists for exactness.

## Versioning
Both this file format (`format=sqlite-sparse/1` in meta) and the `.sprs` sidecar
header carry version 1. Readers must refuse a version they do not know. Any
change that alters how an existing file is interpreted increments the version,
and the compatibility rules between versions will be stated here explicitly.
Format stability is not promised before 1.0; files written by 0.x releases may
need rebuilding.
