# sqlite-sparse

Run a sparse retrieval model on SQLite. No model required at query time.

*For small, read-heavy systems that need semantic search. Documents are encoded once at
insert and queries use only tokenization and a static weight table stored in the file,
scored by an exact inverted-index scatter-add. No vector database, no ANN index, no
query-time model.*

[![CI](https://github.com/arbazsiddiqui/sqlite-sparse/actions/workflows/ci.yml/badge.svg)](https://github.com/arbazsiddiqui/sqlite-sparse/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/sqlite-sparse)](https://pypi.org/project/sqlite-sparse/)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://github.com/arbazsiddiqui/sqlite-sparse/blob/master/LICENSE)

```sql
.load ./sparse0
CREATE VIRTUAL TABLE notes USING sparse0(model='mini');
INSERT INTO notes(rowid, text) VALUES (1, 'Aspirin lowers the risk of heart attack and stroke.');
SELECT rowid, score FROM notes WHERE notes MATCH 'what prevents cardiac arrest' LIMIT 5;
```

## Background

A learned sparse encoder (SPLADE, "sparse lexical and expansion") turns text into a
weighted bag of vocabulary terms. For the sentence above that is about 160 terms, among
them `prevents` and `cardiac`. The output can be stored and searched like a keyword index,
but the keywords were chosen by a transformer. OpenSearch's inference-free variants run
the encoder only on documents; each query token gets one learned weight from a lookup
table, and retrieval is an exact dot product.

Dense retrieval (vector search) instead runs an embedding model on every query. A learned
sparse index moves all model work to write time, and the representation is legible, since
you can see which terms matched and with what weight. The cost is quality against good
dense models of the same size and much slower indexing. `mini` (23M) averages 0.497
nDCG@10 on BEIR and mdbr-leaf-ir (23M dense) reports 0.5355 in its symmetric
configuration. The two do very different amounts of work at query time, so this is
context, not a controlled comparison. For read-heavy workloads where indexing is
amortized over many searches, the trade can be favourable.

This library ships OpenSearch's inference-free sparse models, which until now have lived
inside search clusters (OpenSearch, Elasticsearch, Vespa).
[sqlite-vec](https://github.com/asg017/sqlite-vec) provides a way to use embeddings in
SQLite; this does the same for learned sparse, whose posting lists are plain rows in the
database file. What this project contributes is the embedded
implementation, the file format with a reference implementation to test it against, and
the measurements.

The searchable index and the query weights live in the SQLite database. The encoder is
needed only when inserting documents. So the `.db` can be copied, shipped inside an app,
opened on any machine with no GPU, and queried with plain SQL from any language that
has SQLite. Prebuilt binaries cover Linux x86-64 and macOS arm64.

## Install

```
pip install sqlite-sparse
```

Or take the binary from the [releases page](https://github.com/arbazsiddiqui/sqlite-sparse/releases)
and use it from any language.

```
tar xzf sparse0-0.1.0-loadable-linux-x86_64.tar.gz    # or -macos-arm64
sqlite3 notes.db
sqlite> .load ./sparse0
```

Keep the filename `sparse0.so` / `sparse0.dylib`, since SQLite derives the entry point
from it. On macOS, the python.org installer's `sqlite3` module cannot load extensions.
Use Python from Homebrew or conda, or `pip install sqlean.py` and `import sqlean as
sqlite3`.

## Quickstart

```python
import sqlite3, sqlite_sparse

db = sqlite3.connect("notes.db")
sqlite_sparse.load(db)                       # loads the sparse0 extension
sqlite_sparse.register(db, "mini")           # downloads the model on first use
db.execute("CREATE VIRTUAL TABLE notes USING sparse0(model='mini')")
db.execute("INSERT INTO notes(rowid, text) VALUES (1, 'Aspirin lowers heart attack risk')")
db.commit()
db.execute("SELECT rowid, score FROM notes WHERE notes MATCH ? LIMIT 5",
           ("what prevents cardiac arrest",)).fetchall()
```

The model runs at INSERT only; MATCH never loads it. Searching an existing index needs
no model at all, on any machine.

```python
db = sqlite3.connect("notes.db")
sqlite_sparse.load(db)
db.execute("CREATE VIRTUAL TABLE temp.notes USING sparse0()")     # adopts the file
db.execute("SELECT rowid, score FROM temp.notes WHERE notes MATCH 'heart medication' LIMIT 5")
```

A database file holds one sparse index; another `sparse0` table in the same file
attaches to the same index rather than creating a second one.

Because results are rows, semantic search composes with plain SQL. Ask for extra
candidates with `k`, then filter and join like any other table.

```sql
SELECT n.rowid, n.score, d.title
FROM notes n JOIN documents d ON d.id = n.rowid
WHERE n.text MATCH 'heart medication' AND k = 50 AND d.folder = 'work'
ORDER BY n.score DESC LIMIT 10;
```

Indexing is the expensive half, so large corpora are best built once on a GPU machine
with the Python package (`sqlite-sparse build`, which writes the same file format) and
the `.db` then shipped to wherever the reads happen.

`LIMIT n` and `AND k = n` both work, and `ORDER BY score DESC` is honoured without a sort
step. `DELETE FROM notes WHERE rowid = ?` marks a document deleted; run
`SELECT sparse_compact()` now and then on an index with heavy churn to reclaim its
postings. Documents longer than `max_seq` tokens (default 512) are truncated at insert, and
each row in the `docs` table records `ntokens` and `truncated`. The full surface, including
the Python helpers, is in [docs/api.md](https://github.com/arbazsiddiqui/sqlite-sparse/blob/master/docs/api.md).

## How it works

![How sqlite-sparse indexes and searches](https://raw.githubusercontent.com/arbazsiddiqui/sqlite-sparse/master/docs/how-it-works.svg)

**INSERT.** The extension tokenizes the text with the registered model's tokenizer and
runs the encoder from the GGUF file through llama.cpp, one vector per token. It then
applies the scoring head from the `.sprs` sidecar, which scores every word in the model's
vocabulary against those vectors; positive scores are kept, compressed with a logarithm,
and become the document's weighted terms. For the aspirin sentence that is 157 terms
(`heart` 0.95, `stroke` 0.92, `risk` 0.78, `reduce` 0.70, `cardiac` 0.42, `prevents` 0.18,
and so on). Each term is appended to that word's posting list, a row in the file listing
the documents it scored and the weight, stored as one byte. The document's token count
and whether it was truncated go into the `docs` table.

**MATCH.** The extension tokenizes the query the same way and reads one number per token
from the query weight table stored in the file (`what` 2.77, `prevents` 6.72, `cardiac`
6.53, `arrest` 6.87). For each query word it walks that word's posting list and adds
query weight × stored weight into the running total of every document listed; that
accumulation is the scatter-add. `prevents` contributes 6.72 × 0.18 and `cardiac`
6.53 × 0.42 to document 1, total 3.95. The documents touched are sorted and the top k
returned. Scoring is exact over the stored weights, with no candidate stage and no
approximate index, and ties break on the lower rowid. Nothing from the GGUF or the
sidecar is read at query time.

The file layout is in [FORMAT.md](https://github.com/arbazsiddiqui/sqlite-sparse/blob/master/FORMAT.md). The format and the sidecar header carry
version 1; format stability is not promised before 1.0.

## Benchmarks

Three ways to search inside a SQLite file, each in its shipped form, on the same machine.
FTS5 is SQLite's built-in keyword search ranked by BM25. Dense brute-force is
[sqlite-vec](https://github.com/asg017/sqlite-vec) int8, which scans every vector, with
[mdbr-leaf-ir](https://huggingface.co/MongoDB/mdbr-leaf-ir) (23M) encoding queries on torch
CPU with 8 threads. Sparse is the `sparse0` extension with `mini` (23M, Q8_0 encoder, u8
postings).

Every number is end-to-end query latency, which for dense includes encoding the query,
because that is its real query path. This compares the brute-force vector path inside
SQLite, not an approximate nearest-neighbour index. The FTS5 query is the disjunction of
the query's tokens ranked by `bm25()`; a conjunction is faster but misses documents that
match only some of the terms.

| msmarco, 1M documents | FTS5 BM25 | dense brute-force | sqlite-sparse |
|---|---|---|---|
| query p50, warm | 582 ms | 735 ms | **3.1 ms** |
| query p99, warm | 1,305 ms | 739 ms | **7.2 ms** |
| cold process to first result | 88 ms | 6,989 ms | **62 ms** |
| peak RAM on the query path | 37 MB | 527 MB | **28 MB** |
| index size, bytes per document | **540** | 807 | 1,092 |
| indexing, documents per second (CPU) | **~43,000** | 167 | 20.5 |
| model at query time | none | 23M transformer | none |
| retrieval | lexical | semantic | semantic |

At 100K documents the p50s are 54 ms, 82 ms and 0.26 ms respectively.

### The extension does not lose the model's quality

The reference for quality is the model card. The compiled extension (Q8_0 encoder, u8
storage, 512-token truncation) reproduces it.

| nDCG@10 | OpenSearch doc-v2-mini, 23M (card) | same model through sqlite-sparse | FTS5 BM25, same corpora |
|---|---|---|---|
| SciFact | 0.699 | 0.6985 | 0.668 |
| NFCorpus | 0.336 | 0.3371 | 0.308 |
| SCIDOCS | 0.164 | 0.1633 | 0.151 |
| FiQA | 0.338 | 0.3387 | 0.234 |

The last column is what the semantic index buys over SQLite's built-in keyword search on
the same documents and queries. The gain ranges from small (SciFact) to large (FiQA).

The same model run in torch at fp32 agrees with the extension on 96 to 98 percent of
top-10 results on every dataset, and storing weights as one byte instead of fp32 changed
nDCG@10 by less than 0.001.

Venue was a dedicated GCE `c3-standard-8` (8 vCPU, 4 physical cores) running Debian 12,
with each lane in its own fresh process. 4,000 samples × 5 repetitions per lane (900 × 3
for dense and 1,000 × 3 for FTS5 at 1M), median of repetition medians. Cold start is the
second of three fresh-process runs. RAM is peak RSS after 50 warm queries. The corpus is
the first 100K and 1M passages of MS MARCO with its dev queries. Benchmark scripts and raw
results are attached to each release.

## Models

| alias | model | params | BEIR avg (card) | GGUF + sidecar |
|---|---|---|---|---|
| `mini` (default) | [doc-v2-mini](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini) | 23M | 0.497 | [arbazsiddiqui/opensearch-neural-sparse-doc-v2-mini-GGUF](https://huggingface.co/arbazsiddiqui/opensearch-neural-sparse-doc-v2-mini-GGUF) |
| `base` | [doc-v3-distill](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill) | 67M | 0.517 | [arbazsiddiqui/opensearch-neural-sparse-doc-v3-distill-GGUF](https://huggingface.co/arbazsiddiqui/opensearch-neural-sparse-doc-v3-distill-GGUF) |
| `multilingual` | [multilingual-v1](https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1) | 168M | multilingual | [arbazsiddiqui/opensearch-neural-sparse-multilingual-v1-GGUF](https://huggingface.co/arbazsiddiqui/opensearch-neural-sparse-multilingual-v1-GGUF) |

All three are in the [sqlite-sparse models](https://huggingface.co/collections/arbazsiddiqui/sqlite-sparse-models-6a929c8e0cb15b0e8ed47d43)
collection, and `sqlite_sparse.register(db, alias)` fetches one into `~/.cache/sqlite-sparse`.
Each conversion is validated against the original SentenceTransformers implementation
(encoder hidden states at cosine 0.9997 or better, term weights within 1.3e-3). Weights
are unmodified from the Apache-2.0 originals by the OpenSearch project.

### Bring your own model

Any inference-free OpenSearch-style sparse encoder on Hugging Face works. Two files are
needed, the encoder as GGUF and a sidecar with the scoring head, query weight table and
vocabulary.

```
git clone --depth 1 https://github.com/ggml-org/llama.cpp
python llama.cpp/convert_hf_to_gguf.py <hf-model-id> --outfile model_f16.gguf --outtype f16
llama.cpp/build/bin/llama-quantize model_f16.gguf model_q8.gguf q8_0      # optional
pip install "sqlite-sparse[build-torch]"
sqlite-sparse convert <hf-model-id> model.sprs                            # --double-log for v3 models
```

```sql
SELECT sparse_register('mine', 'model_q8.gguf', 'model.sprs');
CREATE VIRTUAL TABLE notes USING sparse0(model='mine');
```

The converter requires the checkpoint to be a BERT-family encoder with a masked-LM head
and a static query weight table. llama.cpp must support the encoder architecture; it does
not support GTE (`doc-v3-gte`).

## Development

```
git clone https://github.com/arbazsiddiqui/sqlite-sparse
make          # cmake fetches a pinned llama.cpp and builds build/sparse0.{so,dylib}
make test     # installs the Python binding and runs the suite
```

`src/` is the extension (`sparse0.c` virtual table, `wordpiece.c` tokenizer on utf8proc, `head.c`
scoring head on ggml, `scorer.c` scatter-add, `encoder.c` llama.cpp wrapper).
`bindings/python` is the reference implementation of the file format and the test oracle.
Component agreement is logged in [`tests/test_differential.md`](https://github.com/arbazsiddiqui/sqlite-sparse/blob/master/tests/test_differential.md).

## License

MIT. The model weights are unmodified Apache-2.0 work by the OpenSearch project and
llama.cpp is MIT; [NOTICE](https://github.com/arbazsiddiqui/sqlite-sparse/blob/master/NOTICE) lists every third-party artifact, its license and
its provenance.
