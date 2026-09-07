/* Compiles sparse0 into the SQLite WebAssembly build, query-only: no llama.cpp,
 * no ggml, so no sparse_register and no text INSERT, but every other operation
 * on an index (MATCH, terms INSERT, vocab= creation, compaction) runs the same
 * C code as the native extension. Built by wasm/Makefile; see wasm/README.md. */
#define SQLITE_CORE 1
#define SPARSE0_QUERY_ONLY 1
#include "sqlite3.h"
#include "utf8proc.c"
#include "../src/sparse0.c"

int sqlite3_wasm_extra_init(const char *unused) {
    (void)unused;
    return sqlite3_auto_extension((void (*)(void))sqlite3_sparse_init);
}
