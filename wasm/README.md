# sparse0 in WebAssembly

`make -C wasm` builds the official SQLite WebAssembly bundle (`sqlite3.mjs`, `sqlite3.wasm`)
with `sparse0` compiled in, using SQLite's `sqlite3_wasm_extra_init.c` hook. The build is
query-only: llama.cpp and ggml are left out, so `sparse_register` and text `INSERT` are
refused, while `MATCH`, `terms` INSERT, `vocab=` creation, `sparse_tokens` and
`sparse_compact` run the same C code as the native extension.

Needs Emscripten (`emcc`), wabt, GNU sed, `tclsh`, `curl` and `git`. `make -C wasm check` serves
`check.html`, which loads the bundle in your browser, verifies the extension and lets you run a
`MATCH` against any sqlite-sparse `.db` you pick.
The SQLite version is pinned in the Makefile; utf8proc is pinned to the same commit the
native build uses.

Use it like the official bundle:

```js
import init from "./sqlite3.mjs";
const sqlite3 = await init();
const db = new sqlite3.oo1.DB();
// deserialize a .db you fetched, then
db.exec("CREATE VIRTUAL TABLE temp.notes USING sparse0()");
db.selectArrays("SELECT rowid, score FROM notes WHERE notes MATCH ? AND k = 10", ["heart medication"]);
```
