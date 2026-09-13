# sqlite-sparse

Semantic search in one SQLite file. No model, no server at query time.

This package carries the prebuilt `sparse0` SQLite extension for Node, Linux x86-64 and macOS
arm64, the same binaries as the [GitHub release](https://github.com/arbazsiddiqui/sqlite-sparse/releases).
How it works, benchmarks and the SQL reference are in the
[repository README](https://github.com/arbazsiddiqui/sqlite-sparse#readme) and
[docs/api.md](https://github.com/arbazsiddiqui/sqlite-sparse/blob/master/docs/api.md).

## Install

```
npm install sqlite-sparse
```

## Search

```js
const { DatabaseSync } = require("node:sqlite");   // Node 22.13+; better-sqlite3 works the same way
const sparse = require("sqlite-sparse");

const db = new DatabaseSync("notes.db", { allowExtension: true });
sparse.load(db);
db.exec("CREATE VIRTUAL TABLE temp.notes USING sparse0()");   // attaches to the index in the file
db.prepare("SELECT rowid, score FROM notes WHERE notes MATCH ? LIMIT 5").all("heart medication");
```

`sparse.getLoadablePath()` returns the binary's path for `loadExtension()` in any other driver.
Keep the filename `sparse0.so` / `sparse0.dylib`; SQLite derives the entry point from it.

## Index

Searching reads only the file. Inserting text runs the encoder, so register a model's GGUF and
`.sprs` sidecar once per process:

```js
db.prepare("SELECT sparse_register('mini', ?, ?)").get("/models/mini_q8.gguf", "/models/mini.sprs");
db.exec("CREATE VIRTUAL TABLE notes USING sparse0(model='mini')");
db.prepare("INSERT INTO notes(rowid, text) VALUES (?, ?)").run(1, "Aspirin lowers heart attack risk");
```

The files for `mini` (23M parameters) are `mini_q8.gguf` and `mini.sprs` in
[arbazsiddiqui/opensearch-neural-sparse-doc-v2-mini-GGUF](https://huggingface.co/arbazsiddiqui/opensearch-neural-sparse-doc-v2-mini-GGUF);
`base` (67M) and `multilingual` (168M) are in the sibling repositories. The Python package
(`pip install sqlite-sparse`) downloads them for you and has a bulk builder for large corpora;
an index built there is read here unchanged.

## License

MIT. Model weights are unmodified Apache-2.0 work by the OpenSearch project; llama.cpp is MIT.
