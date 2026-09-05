/* Document encoder: llama.cpp with POOLING_TYPE_NONE, per-token hidden states. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "llama.h"

typedef struct {
    struct llama_model *model;
    struct llama_context *ctx;
    const struct llama_vocab *vocab;
    int n_embd;
    int max_seq;
} SparseEncoder;

static void sparse_log_quiet(enum ggml_log_level level, const char *text, void *ud) {
    (void)level; (void)text; (void)ud;
}

static SparseEncoder *sparse_encoder_load(const char *gguf_path, int n_threads, int max_seq) {
    static int backend_ready = 0;
    if (!backend_ready) {
        llama_log_set(sparse_log_quiet, NULL);
        llama_backend_init();
        backend_ready = 1;
    }

    struct llama_model_params mp = llama_model_default_params();
    struct llama_model *model = llama_model_load_from_file(gguf_path, mp);
    if (!model) return NULL;

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = max_seq;
    cp.n_batch = max_seq;
    cp.n_ubatch = max_seq;
    cp.embeddings = 1;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); return NULL; }

    SparseEncoder *e = calloc(1, sizeof(SparseEncoder));
    e->model = model;
    e->ctx = ctx;
    e->vocab = llama_model_get_vocab(model);
    e->n_embd = llama_model_n_embd(model);
    e->max_seq = max_seq;
    return e;
}

static void sparse_encoder_free(SparseEncoder *e) {
    if (!e) return;
    llama_free(e->ctx);
    llama_model_free(e->model);
    free(e);
}

/* Returns n_tokens (<= max_seq) or -1. *out_hs is malloc'd [n_tokens * n_embd],
 * caller frees. *out_total is the untruncated token count. */
static int sparse_encoder_encode(SparseEncoder *e, const char *text, float **out_hs, int *out_total) {
    int cap = 8192;
    llama_token *toks = malloc(cap * sizeof(llama_token));
    int n = llama_tokenize(e->vocab, text, (int32_t)strlen(text), toks, cap,
                           /*add_special=*/1, /*parse_special=*/0);
    if (n < 0) {                      /* text longer than cap: retokenize exactly */
        cap = -n;
        toks = realloc(toks, cap * sizeof(llama_token));
        n = llama_tokenize(e->vocab, text, (int32_t)strlen(text), toks, cap, 1, 0);
    }
    if (n <= 0) { free(toks); return -1; }
    if (out_total) *out_total = n;
    if (n > e->max_seq) {             /* truncate, keeping the terminal [SEP] */
        toks[e->max_seq - 1] = toks[n - 1];
        n = e->max_seq;
    }

    struct llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i] = toks[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    batch.n_tokens = n;

    llama_memory_clear(llama_get_memory(e->ctx), 1);
    int rc = llama_encode(e->ctx, batch);
    llama_batch_free(batch);
    free(toks);
    if (rc != 0) return -1;

    const float *emb = llama_get_embeddings(e->ctx);
    if (!emb) return -1;
    size_t bytes = (size_t)n * e->n_embd * sizeof(float);
    *out_hs = malloc(bytes);
    memcpy(*out_hs, emb, bytes);
    return n;
}
