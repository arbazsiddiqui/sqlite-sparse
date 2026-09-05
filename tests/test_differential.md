# Differential test log

Every C component is checked against the Python reference, and the compiled
extension against the original torch pipeline. Dates are when each check ran.

- 2026-08-25. `src/wordpiece.c` vs the reference tokenizer on 6,980 real
  MS MARCO dev queries. Identical token ids on all of them (100.000%).
- 2026-08-25. llama.cpp encoder (GGUF) vs torch fp32 hidden states on
  43 tokens. Worst-token cosine 0.999999 at f16 and 0.999855 at Q8_0.
- 2026-08-25. Head formula in Python vs SparseEncoder. Max difference 0.
- 2026-08-25. `src/head.c` (ggml) vs torch. 154 of 154 terms, none missing or
  extra, max weight difference 6.93e-4.
- 2026-08-25. `src/scorer.c` vs the Python engine on a 100K index with 200 real
  queries. 200 of 200 rank-identical or tie-equivalent (37 tie-order
  permutations with identical score multisets, no real mismatches).
- 2026-08-26. Family conversion gates for base (doc-v3-distill) and
  multilingual. Worst-token encoder cosine at least 0.999994 (f16) and 0.999702
  (Q8_0); sidecar head term-weight delta vs SparseEncoder between 4.8e-4 and
  1.3e-3. The v3 models use log1p applied twice (activation flag 1 in the
  sidecar header).
- 2026-08-26. `src/head.c` on both activation paths vs torch. mini (flag 0)
  154 of 154 terms, max delta 6.93e-4. base (flag 1) 177 of 178 terms (the one
  missing term sits at 3.3e-4, below the 1e-3 emit threshold), max delta 6.97e-4.
- 2026-08-26. Extension vs Python reference four ways on the smoke corpus
  (C on C, Python on C, Python on Python, C on Python). Identical to 4 decimals.
- 2026-08-29. Extension (new scorer) vs Python reference on the same 200-document
  database with 50 real queries. 37 exact, 12 tie-reordered with identical score
  multisets, 1 tie at the k=10 boundary, no real mismatches. Tie order in the
  extension is deterministic (lower rowid first) and pinned in
  `tests/test_loadable.py`.
- 2026-08-29. Write visibility through the virtual table. INSERT then MATCH sees
  the document, DELETE then MATCH hides it, a second connection sees both, and a
  foreign write invalidates the reader's cache through `PRAGMA data_version`.
- 2026-08-29. Extension, mini Q8_0, on SciFact (5,183 documents). Per-document
  vectors vs a fresh torch run at 256 tokens are within cosine 0.9992 to 0.9996 on
  long and short abstracts. llama.cpp tokenization vs Hugging Face on 1,500
  documents differs on none.
- 2026-08-29. Extension end to end on SciFact at 512 tokens, the cards' setting.
  mini Q8_0 scores NDCG@10 0.6958 over 300 queries against 0.699 on the card.
  base Q8_0 (activation flag 1) scores 0.7066 against 0.708. The remaining gap
  is Q8_0 encoding plus u8 storage.
- 2026-08-30. Query-side cap removed. Both tokenizers truncated a query at 64
  words (inherited from the first reference); ArguAna queries are full
  arguments, so the extension agreed with the untruncated torch pipeline on
  only 64.5% of top-10 there while agreeing 96 to 98% elsewhere. The cap is
  now the scorer's 512-token bound in both implementations. C and Python agree
  exactly on a 160-word query; the 50-query short-query differential is
  unchanged at 50 of 50.

## 2026-09-03: query tokenizer vs the HF tokenizer

Found while re-examining the idea: both query tokenizers were ASCII-only
WordPiece (lowercase a-z0-9 runs, every other character its own token, no accent
stripping). Identical in C and Python, so the differential suite never saw it.
Against the real tokenizers: English matched; accented Latin diverged in both
models (`résumé` became `r`, `sum`); with the multilingual model, Russian had
0.00 token overlap with HF, Arabic 0.18, Hindi 0.27. The multilingual model's
query path was effectively broken for non-Latin, non-CJK scripts while its
document path (llama.cpp's tokenizer) was correct.

Fix: BERT BasicTokenizer semantics in both implementations (utf8proc in C,
unicodedata in Python) and `[UNK]` on failure as HF does. Golden ids from the HF
tokenizers for 14 strings × 2 vocabularies live in
`tests/fixtures/tokenizer_golden.json`; the Python reference and the extension
(via `sparse_tokens()`) reproduce all 28 exactly. Documents carry no postings on
special tokens, so the `[UNK]` query term is inert. English BEIR results are
unaffected (English tokenization was already identical).
