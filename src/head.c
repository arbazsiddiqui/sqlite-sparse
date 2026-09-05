/* MLM head on ggml: dense -> GELU -> LayerNorm -> tied decoder + bias ->
 * max-pool over tokens -> log1p(relu), applied twice when the sidecar's
 * activation flag is 1. Weights come from the .sprs sidecar. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ggml.h"
#include "ggml-cpu.h"

typedef struct {
    uint32_t hidden, vocab_n;
    float *dense_w;   /* H*H row-major (out,in) */
    float *dense_b;   /* H */
    float *ln_g;      /* H */
    float *ln_b;      /* H */
    float *dec_w;     /* vocab_n*H row-major */
    float *dec_b;     /* vocab_n */
    float *qlut;      /* vocab_n */
    char  *vocab_blob;
    uint32_t vocab_blob_len;
    void  *raw;
    uint32_t act;   /* 0 = log1p(relu) (v1/v2 models), 1 = log1p(log1p(relu)) (v3) */
} SparseHead;

SparseHead *sparse_head_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[4];
    uint32_t hdr[5];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "SPRS", 4) != 0
        || fread(hdr, 4, 5, f) != 5 || fseek(f, 8, SEEK_CUR) != 0) { fclose(f); return NULL; }
    uint32_t ver = hdr[0], H = hdr[1], V = hdr[2], fmt = hdr[3], act = hdr[4];
    if (ver != 1 || fmt != 0 || act > 1
        || H < 8 || H > 16384 || V < 8 || V > (1u << 24)) { fclose(f); return NULL; }
    SparseHead *h = calloc(1, sizeof(SparseHead));
    h->hidden = H; h->vocab_n = V; h->act = act;
    size_t sizes[7] = {(size_t)H*H, H, H, H, (size_t)V*H, V, V};
    float **dst[7] = {&h->dense_w, &h->dense_b, &h->ln_g, &h->ln_b, &h->dec_w, &h->dec_b, &h->qlut};
    for (int i = 0; i < 7; i++) {
        *dst[i] = malloc(sizes[i] * 4);
        if (!*dst[i] || fread(*dst[i], 4, sizes[i], f) != sizes[i]) goto fail;
    }
    if (fread(&h->vocab_blob_len, 4, 1, f) != 1) goto fail;
    h->vocab_blob = malloc((size_t)h->vocab_blob_len + 1);
    if (!h->vocab_blob || fread(h->vocab_blob, 1, h->vocab_blob_len, f) != h->vocab_blob_len) goto fail;
    h->vocab_blob[h->vocab_blob_len] = 0;
    fclose(f);
    return h;
fail:
    fclose(f);
    free(h->dense_w); free(h->dense_b); free(h->ln_g); free(h->ln_b);
    free(h->dec_w); free(h->dec_b); free(h->qlut); free(h->vocab_blob);
    free(h);
    return NULL;
}

/* hs is [n_tokens, H] row-major. Returns the number of terms with weight > wmin. */
int sparse_head_apply(const SparseHead *h, const float *hs, int n_tokens,
                      float wmin, int32_t *out_terms, float *out_weights, int max_terms) {
    const uint32_t H = h->hidden, V = h->vocab_n;
    size_t ctx_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead()
        + (size_t)n_tokens * H * 4 * 8 + (size_t)n_tokens * V * 4 * 2
        + (size_t)(H*H + V*H) * 4 + (4u<<20);
    struct ggml_init_params ip = { ctx_size, NULL, 0 };
    struct ggml_context *ctx = ggml_init(ip);

    struct ggml_tensor *x  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, n_tokens);
    memcpy(x->data, hs, (size_t)n_tokens * H * 4);
    struct ggml_tensor *dw = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, H);
    memcpy(dw->data, h->dense_w, (size_t)H * H * 4);
    struct ggml_tensor *db = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    memcpy(db->data, h->dense_b, H * 4);
    struct ggml_tensor *lg = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    memcpy(lg->data, h->ln_g, H * 4);
    struct ggml_tensor *lb = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    memcpy(lb->data, h->ln_b, H * 4);
    struct ggml_tensor *ww = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, V);
    memcpy(ww->data, h->dec_w, (size_t)V * H * 4);

    /* rows of dense_w are output neurons, so ggml_mul_mat(dw, x) is [H, n_tokens] */
    struct ggml_tensor *t = ggml_mul_mat(ctx, dw, x);
    t = ggml_add(ctx, t, db);
    t = ggml_gelu(ctx, t);
    t = ggml_norm(ctx, t, 1e-12f);
    t = ggml_add(ctx, ggml_mul(ctx, t, lg), lb);
    struct ggml_tensor *logits = ggml_mul_mat(ctx, ww, t);   /* [V, n_tokens] */

    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, logits);
    struct ggml_cplan plan = ggml_graph_plan(gf, 3, NULL);
    uint8_t *work = plan.work_size ? malloc(plan.work_size) : NULL;
    plan.work_data = work;
    ggml_graph_compute(gf, &plan);
    free(work);

    const float *L = (const float *)logits->data;
    int n_out = 0;
    for (uint32_t v = 0; v < V; v++) {
        float mx = -1e30f;
        for (int tk = 0; tk < n_tokens; tk++) {
            float val = L[(size_t)tk * V + v];
            if (val > mx) mx = val;
        }
        mx += h->dec_b[v];
        if (mx <= 0) continue;
        float w = log1pf(mx);
        if (h->act == 1) w = log1pf(w);
        if (w < wmin) continue;
        if (n_out < max_terms) { out_terms[n_out] = (int32_t)v; out_weights[n_out] = w; n_out++; }
    }
    ggml_free(ctx);
    return n_out;
}

#ifdef HEAD_MAIN
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s head.sprs hidden.bin\n", argv[0]); return 1; }
    SparseHead *h = sparse_head_load(argv[1]);
    if (!h) { fprintf(stderr, "bad sidecar\n"); return 1; }
    FILE *f = fopen(argv[2], "rb");
    int32_t n_tokens;
    fread(&n_tokens, 4, 1, f);
    float *hs = malloc((size_t)n_tokens * h->hidden * 4);
    fread(hs, 4, (size_t)n_tokens * h->hidden, f);
    fclose(f);
    int32_t terms[4096]; float ws[4096];
    int n = sparse_head_apply(h, hs, n_tokens, 0.0001f, terms, ws, 4096);
    for (int i = 0; i < n; i++) printf("%d %.6f\n", terms[i], ws[i]);
    return 0;
}
#endif
